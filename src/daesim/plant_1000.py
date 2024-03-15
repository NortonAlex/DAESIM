"""
Plant model class: Includes differential equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
from daesim.climate import *
from daesim.biophysics_funcs import func_TempCoeff, growing_degree_days_DTT_nonlinear, growing_degree_days_DTT_linear1, growing_degree_days_DTT_linear2, growing_degree_days_DTT_linear3
from daesim.plantgrowthphases import PlantGrowthPhases
from daesim.management import ManagementModule
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.leafgasexchange import LeafGasExchangeModule

@define
class PlantModuleCalculator:
    """
    Calculator of plant biophysics
    """

    CUE: float = field(default=0.5)  ## Plant carbon-use-efficiency (CUE=NPP/GPP)
    LMA: float = field(default=200)  ## Leaf mass per leaf area (g m-2)
    hc: float = field(default=0.6)   ## Canopy height (m) TODO: make this a dynamic variable at some point. 
    SAI: float = field(default=0.1)   ## Stem area index (m2 m-2) TODO: make this a dynamic variable at some point. 
    clumping_factor: float = field(default=0.7)   ## Foliage clumping index (-) TODO: Place this parameter in a more suitable spot/module
    albsoib: float = field(default=0.2)   ## Soil background albedo for beam radiation (-) TODO: Place this parameter in a more suitable spot/module
    albsoid: float = field(default=0.2)   ## Soil background albedo for diffuse radiation (-) TODO: Place this parameter in a more suitable spot/module

    GDD_method: str = field(
        default="nonlinear"
        ) ## method used to calculate daily thermal time and hence growing degree days. Options are: "nonlinear", "linear1", "linear2", "linear3"
    GDD_Tbase: float = field(
        default=5.0
    )  ## Base temperature (minimum threshold) used for calculating the growing degree days
    GDD_Topt: float = field(
        default=25.0
    )  ## Optimum temperature used for calculating the growing degree days (non-linear method only)
    GDD_Tupp: float = field(
        default=40.0
    )  ## Upper temperature (maximum threshold) used for calculating the growing degree days

    ## TODO: Update management module with these parameters later on
    propHarvestSeed: float = field(default=1.0)  ## proportion of seed carbon pool removed at harvest
    propHarvestLeaf: float = field(default=0.9)  ## proportion of seed carbon pool removed at harvest
    propHarvestStem: float = field(default=0.7)  ## proportion of seed carbon pool removed at harvest
    
    def calculate(
        self,
        Cleaf,
        Cstem,
        Croot,
        Cseed,
        Bio_time,
        solRadswskyb,        ## Atmospheric direct beam solar radiation (W/m2)
        solRadswskyd,        ## Atmospheric diffuse solar radiation (W/m2)
        airTempCMin,   ## minimum air temperature (degrees Celsius)
        airTempCMax,   ## maximum air temperature (degrees Celsius)
        airPressure,   ## atmospheric pressure (Pa)
        airRH,         ## relative humidity (%)
        airCO2,        ## partial pressure CO2 (bar)
        airO2,         ## partial pressure O2 (bar)
        _doy,
        _year,
        Site=ClimateModule(),   ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.
        Management=ManagementModule(),   ## It is optional to define Management for this method. If no argument is passed in here, then default setting for Management is the default ManagementModule()
        PlantDev=PlantGrowthPhases(),   ## It is optional to define PlantDev for this method. If no argument is passed in here, then default setting for Management is the default PlantGrowthPhases()
        Leaf=LeafGasExchangeModule(),   ## It is optional to define PlantDev for this method. If no argument is passed in here, then default setting for Leaf is the default LeafGasExchangeModule()
        Canopy=CanopyLayers(),   ## It is optional to define PlantDev for this method. If no argument is passed in here, then default setting for Canopy is the default CanopyLayers()
        CanopyRad=CanopyRadiation(),   ## It is optional to define PlantDev for this method. If no argument is passed in here, then default setting for CanopyRad is the default CanopyRadiation()
    ) -> Tuple[float]:

        ## Solar calculations
        eqtime, houranglesunrise, theta = Site.solar_calcs(_year,_doy)

        ## Climate calculations
        airTempC = (airTempCMin+airTempCMax)/2

        ## Calculate leaf area index
        LAI = Cleaf/self.LMA

        BioPlanting = self.calculate_BioPlanting(_doy,Management.plantingDay,Management.plantingRate,Management.plantWeight) ## Modification: using a newly defined parameter in this function instead of "frequPlanting" as used in Stella, considering frequPlanting was being used incorrectly, as its use didn't match with the units or definition.

        BioHarvestLeaf = self.calculate_BioHarvest(Cleaf,_doy,Management.harvestDay,self.propHarvestLeaf,Management.PhHarvestTurnoverTime)
        BioHarvestStem = self.calculate_BioHarvest(Cstem,_doy,Management.harvestDay,self.propHarvestStem,Management.PhHarvestTurnoverTime)
        BioHarvestSeed = self.calculate_BioHarvest(Cseed,_doy,Management.harvestDay,self.propHarvestSeed,Management.PhHarvestTurnoverTime)

        sunrise, solarnoon, sunset = Site.solar_day_calcs(_year,_doy)
        DTT = self.calculate_dailythermaltime(airTempCMin,airTempCMax,sunrise,sunset)
        GDD_reset = self.calculate_growingdegreedays_reset(Bio_time,_doy,Management.plantingDay)
        dGDDdt = DTT - GDD_reset


        ## Canopy radiative transfer
        swleaf, fracsun = self.calculate_canopy_radiative_transfer(LAI, theta, solRadswskyb, solRadswskyd, Canopy, CanopyRad)
        
        dlai = Canopy.cast_parameter_over_layers_betacdf(LAI,Canopy.beta_lai_a,Canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)

        An, gs, Rd = self.calculate_gas_exchange_ml(swleaf, airTempC, airCO2, airO2, airRH, Canopy, CanopyRad, Leaf)
        
        GPP = self.calculate_total_canopy_gpp(dlai, fracsun, An, Rd, Canopy)
        
        # Calculate NPP
        NPP = self.calculate_NPP(GPP)

        # Development phase index
        idevphase = PlantDev.get_active_phase_index(Bio_time)
        # Allocation fractions per pool
        alloc_coeffs = PlantDev.allocation_coeffs[idevphase]
        # Turnover rates per pool
        tr_ = PlantDev.turnover_rates[idevphase]
        
        # ODE for plant carbon pools
        dCleafdt = alloc_coeffs[PlantDev.ileaf]*NPP - tr_[PlantDev.ileaf]*Cleaf - BioHarvestLeaf
        dCstemdt = alloc_coeffs[PlantDev.istem]*NPP - tr_[PlantDev.istem]*Cstem - BioHarvestStem
        dCrootdt = alloc_coeffs[PlantDev.iroot]*NPP - tr_[PlantDev.iroot]*Croot + BioPlanting
        dCseeddt = alloc_coeffs[PlantDev.iseed]*NPP - tr_[PlantDev.iseed]*Cseed - BioHarvestSeed

        return (dCleafdt, dCstemdt, dCrootdt, dCseeddt, dGDDdt)

    def calculate_canopy_radiative_transfer(self, LAI, theta, solRadswskyb, solRadswskyd, Canopy, CanopyRad):
        # Calculate radiative transfer properties
        (fracsun, kb, omega, avmu, betab, betad, tbi) = CanopyRad.calculateRTProperties(
            LAI, self.SAI, self.clumping_factor, self.hc, theta, Canopy=Canopy)
        
        # Cast parameters over layers using beta CDF
        dlai = Canopy.cast_parameter_over_layers_betacdf(LAI, Canopy.beta_lai_a, Canopy.beta_lai_b)
        dsai = Canopy.cast_parameter_over_layers_betacdf(self.SAI, Canopy.beta_sai_a, Canopy.beta_sai_b)
        dpai = dlai + dsai  # Canopy layer plant area index (m2/m2)
        clump_fac = Canopy.cast_parameter_over_layers_uniform(self.clumping_factor)
        
        # Calculate two-stream absorption per leaf area index
        swleaf = CanopyRad.calculateTwoStream(solRadswskyb, solRadswskyd, dpai, fracsun, kb, clump_fac, omega, avmu, betab, betad, tbi, self.albsoib, self.albsoid, Canopy=Canopy)
        
        return swleaf, fracsun
    
    def calculate_gas_exchange_ml(self, swleaf, airTempC, airCO2, airO2, airRH, Canopy, CanopyRad, Leaf):
        # Initialize arrays for An, gs, and Rd
        _An = np.zeros((Canopy.nlevmlcan, Canopy.nleaf))
        _gs = np.zeros((Canopy.nlevmlcan, Canopy.nleaf))
        _Rd = np.zeros((Canopy.nlevmlcan, Canopy.nleaf))
        
        # Loop through leaves and canopy layers to calculate gas exchange
        for ileaf in range(Canopy.nleaf):
            for ic in range(Canopy.nbot, Canopy.ntop+1):
                Q = 1e-6 * swleaf[ic, ileaf] * CanopyRad.J_to_umol  # Absorbed PPFD, mol PAR m-2 s-1
                An, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q, airTempC, airCO2, airO2, airRH)
                _An[ic, ileaf] = An
                _gs[ic, ileaf] = gs
                _Rd[ic, ileaf] = Rd
                
        return _An, _gs, _Rd

    def calculate_total_canopy_gpp(self, dlai, fracsun, An, Rd, Canopy):
        # Calculate total GPP 
        # - also convert leaf level molar fluxes per second (mol m-2 s-1) to mass fluxes per ground area per day (g C m-2 d-1)
        GPP = 12.01 * (60*60*24) * (np.sum(dlai * fracsun * (An[:, Canopy.isun] + Rd[:, Canopy.isun])) + 
                                    np.sum(dlai * (1 - fracsun) * (An[:, Canopy.isha] + Rd[:, Canopy.isha])))
        return GPP

    def calculate_NPP(self,GPP):
        return self.CUE*GPP

    def calculate_dailythermaltime(self,Tmin,Tmax,sunrise,sunset):
        if self.GDD_method == "nonlinear":
            _vfunc = np.vectorize(growing_degree_days_DTT_nonlinear)
            DTT = _vfunc(Tmin,Tmax,sunrise,sunset,self.GDD_Tbase,self.GDD_Tupp,self.GDD_Topt)
        elif self.GDD_method == "linear1":
            _vfunc = np.vectorize(growing_degree_days_DTT_linear1)
            DTT = _vfunc(Tmin,Tmax,self.GDD_Tbase,self.GDD_Tupp)
        elif self.GDD_method == "linear2":
            _vfunc = np.vectorize(growing_degree_days_DTT_linear2)
            DTT = _vfunc(Tmin,Tmax,self.GDD_Tbase,self.GDD_Tupp)
        elif self.GDD_method == "linear3":
            _vfunc = np.vectorize(growing_degree_days_DTT_linear3)
            DTT = _vfunc(Tmin,Tmax,self.GDD_Tbase,self.GDD_Tupp)
        return DTT

    def calculate_growingdegreedays_reset(self,GDD,_doy,plantingDay):
        _vfunc = np.vectorize(self.calculate_growingdegreedays_reset_conditional)
        GDD_reset_flux = _vfunc(GDD,_doy,plantingDay)
        return GDD_reset_flux

    def calculate_growingdegreedays_reset_conditional(self,GDD,_doy,plantingDay):
        """
        When it is planting/sowing time (plantingTime == 1), subtract some arbitrarily large 
        number from the growing degree days (GDD) state. It just has to be a large enough 
        number to trigger a zero-crossing "event" for Scipy solve_ivp to recognise. 
        """
        PlantingTime = self.calculate_plantingtime_conditional(_doy,plantingDay)
        GDD_reset_flux = PlantingTime * 1e9
        return GDD_reset_flux

    def calculate_BioPlanting(self,_doy,plantingDay,plantingRate,plantWeight):
        """
        _doy = ordinal day of year
        propBMPlanting = the proportion of planting that applies to this live biomass pool (e.g. if sowing seeds, calculation of the the non-photosynthetic planting flux will require propBMPlanting=1). Modification: The Stella code uses a parameter "frequPlanting" which isn't the correct use, given its definition. 

        returns:
        BioPlanting = the flux of carbon planted
        """
        _vfunc = np.vectorize(self.calculate_BioPlanting_conditional,otypes=[float])
        BioPlanting = _vfunc(_doy,plantingDay,plantingRate,plantWeight)
        return BioPlanting

    def calculate_BioPlanting_conditional(self,_doy,plantingDay,plantingRate,plantWeight):
        # Modification: I have modified the variables/parameters used in this function as the definitions and units in the Stella code didn't match up (see previous parameters maxDensity and frequPlanting vs new parameters plantingRate and propPhPlanting).
        PlantingTime = self.calculate_plantingtime_conditional(_doy,plantingDay)
        BioPlanting = PlantingTime * plantingRate * plantWeight
        return BioPlanting

    def calculate_plantingtime_conditional(self,_doy,plantingDay):
        if plantingDay is None:
            return 0
        elif (plantingDay <= _doy < plantingDay+1):
            return 1
        else:
            return 0

    def calculate_BioHarvest(self,Biomass,_doy,harvestDay,propHarvest,HarvestTurnoverTime):
        _vfunc = np.vectorize(self.calculate_harvesttime_conditional,otypes=[float])
        HarvestTime = _vfunc(_doy,harvestDay)
        BioHarvest = HarvestTime*propHarvest*Biomass/HarvestTurnoverTime
        return BioHarvest

    def calculate_harvesttime_conditional(self,_doy,harvestDay):
        if harvestDay is None:
            return 0
        elif (harvestDay <= _doy < harvestDay+3):  ## assume harvest happens over a single day
            return 1
        else:
            return 0


"""
Differential equation solver implementation for plant model
"""

@define
class PlantModelSolver:

    """
    Plant model solver implementation
    """

    calculator: PlantModuleCalculator
    """Calculator of plant model"""

    site: ClimateModule
    """Site and climate details"""

    management: ManagementModule
    """Management details"""

    plantdev: PlantGrowthPhases
    """"Plant Development Phases details"""

    leaf: LeafGasExchangeModule
    """"Leaf gas exchange details"""

    canopy: CanopyLayers
    """"Canopy structure details"""

    canopyrad: CanopyRadiation
    """"Canopy Radiative Transfer details"""

    state1_init: float
    """
    Initial value for state 1
    """

    state2_init: float
    """
    Initial value for state 2
    """

    state3_init: float
    """
    Initial value for state 3
    """

    state4_init: float
    """
    Initial value for state 4
    """

    state5_init: float
    """
    Initial value for state 5
    """

    time_start: float
    """
    Time at which the initialisation values apply.
    """
    
    def run(
        self,
        solRadswskyb: Callable[[float], float],
        solRadswskyd: Callable[[float], float],
        airTempCMin: Callable[[float], float],
        airTempCMax: Callable[[float], float],
        airPressure: Callable[[float], float],
        airRH: Callable[[float], float],
        airCO2: Callable[[float], float],
        airO2: Callable[[float], float],
        _doy: Callable[[float], float],
        _year: Callable[[float], float],
        time_axis: float,
    ) -> Tuple[float]:
        func_to_solve = self._get_func_to_solve(
            self.site,
            self.management,
            self.plantdev,
            self.leaf,
            self.canopy,
            self.canopyrad,
            solRadswskyb,
            solRadswskyd,
            airTempCMin,
            airTempCMax,
            airPressure,
            airRH,
            airCO2,
            airO2,
            _doy,
            _year,
        )

        t_eval = time_axis
        t_span = (self.time_start, t_eval[-1])
        start_state = (
            self.state1_init,
            self.state2_init,
            self.state3_init,
            self.state4_init,
            self.state5_init,
        )

        ## When the state y[4] goes below zero we trigger the solver to terminate and restart at the next time step
        def zero_crossing_event(t, y):
            return y[4]
        zero_crossing_event.terminal = True
        zero_crossing_event.direction = -1

        solve_kwargs = {
            "t_span": t_span,
            "t_eval": t_eval,
            "y0": start_state,
            "events": zero_crossing_event,
        }

        res_raw = self._solve_ivp(
            func_to_solve,
            **solve_kwargs,
        )

        ## if a termination event occurs, we restart the solver at the next time step 
        ## with new initial values and continue until the status changes
        if res_raw.status == 1:
            res_next = res_raw
        while res_raw.status == 1:
            t_restart = np.ceil(res_next.t_events[0][0])
            t_eval = time_axis[time_axis >= t_restart]
            t_span = (t_restart, t_eval[-1])
            start_state_restart = res_next.y_events[0][0]
            start_state_restart[4] = 0
            solve_kwargs = {
                "t_span": t_span,
                "t_eval": t_eval,
                "y0": tuple(start_state_restart),
                "events": zero_crossing_event,
            }
            res_next = self._solve_ivp(
                func_to_solve,
                **solve_kwargs,
                )
            res_raw.t = np.append(res_raw.t,res_next.t)
            res_raw.y = np.append(res_raw.y,res_next.y,axis=1)
            if res_next.status == 1:
                res_raw.t_events = res_raw.t_events + res_next.t_events
                res_raw.y_events = res_raw.y_events + res_next.y_events
            res_raw.nfev += res_next.nfev
            res_raw.njev += res_next.njev
            res_raw.nlu += res_next.nlu
            res_raw.sol = res_next.sol
            res_raw.message = res_next.message
            res_raw.success = res_next.success
            res_raw.status = res_next.status

        return res_raw

    def _get_func_to_solve(
        self,
        Site,
        Management,
        PlantDev,
        Leaf,
        Canopy,
        CanopyRad,
        solRadswskyb: Callable[float, float],
        solRadswskyd: Callable[float, float],
        airTempCMin: Callable[float, float],
        airTempCMax: Callable[float, float],
        airPressure: Callable[float, float],
        airRH: Callable[float, float],
        airCO2: Callable[float, float],
        airO2: Callable[float, float],
        _doy: Callable[float, float],
        _year: Callable[float, float],
    ) -> Callable[float, float]:
        def func_to_solve(t: float, y: np.ndarray) -> np.ndarray:
            """
            Function to solve i.e. f(t, y) that goes on the RHS of dy/dt = f(t, y)

            Parameters
            ----------
            t
                time

            y
                State vector

            Returns
            -------
                dy / dt (also as a vector)
            """
            solRadswskybh = solRadswskyb(t).squeeze()
            solRadswskydh = solRadswskyd(t).squeeze()
            airTempCMinh = airTempCMin(t).squeeze()
            airTempCMaxh = airTempCMax(t).squeeze()
            airPressureh = airPressure(t).squeeze()
            airRHh = airRH(t).squeeze()
            airCO2h = airCO2(t).squeeze()
            airO2h = airO2(t).squeeze()
            _doyh = _doy(t).squeeze()
            _yearh = _year(t).squeeze()

            dydt = self.calculator.calculate(
                Cleaf=y[0],
                Cstem=y[1],
                Croot=y[2],
                Cseed=y[3],
                Bio_time=y[4],
                solRadswskyb=solRadswskybh,
                solRadswskyd=solRadswskydh,
                airTempCMin=airTempCMinh,
                airTempCMax=airTempCMaxh,
                airPressure=airPressureh,
                airRH=airRHh,
                airCO2=airCO2h,
                airO2=airO2h,
                _doy=_doyh,
                _year=_yearh,
                Site=Site,
                Management=Management,
                PlantDev=PlantDev,
                Leaf=Leaf,
                Canopy=Canopy,
                CanopyRad=CanopyRad,
            )

            # TODO: Use this python magic when we have more than one state variable in dydt
            # dydt = [v for v in dydt]

            return dydt

        return func_to_solve

    def _solve_ivp(
        self, func_to_solve, t_span, t_eval, y0, rtol=1e-6, atol=1e-6, **kwargs
    ) -> OptimizeResult:
        raw = solve_ivp(
            func_to_solve,
            t_span=t_span,
            t_eval=t_eval,
            y0=y0,
            atol=atol,
            rtol=rtol,
            **kwargs,
        )
        if not raw.success:
            info = "Your model failed to solve, perhaps there was a runaway feedback?"
            error_msg = f"{info}\n{raw}"
            raise SolveError(error_msg)

        return raw