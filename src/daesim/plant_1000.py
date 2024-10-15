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
from daesim.plantcarbonwater import PlantModel as PlantCH2O
from daesim.plantallocoptimal import PlantOptimalAllocation

@define
class PlantModuleCalculator:
    """
    Calculator of plant biophysics
    """
    
    ## Module dependencies
    Site: Callable = field(default=ClimateModule())    ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.
    Management: Callable = field(default=ManagementModule())    ## It is optional to define Management for this method. If no argument is passed in here, then default setting for Management is the default ManagementModule()
    PlantDev: Callable = field(default=PlantGrowthPhases())    ## It is optional to define PlantDev for this method. If no argument is passed in here, then default setting for Management is the default PlantGrowthPhases()
    PlantCH2O: Callable = field(default=PlantCH2O())    ## It is optional to define Plant for this method. If no argument is passed in here, then default setting for Plant is the default PlantModel().
    PlantAlloc: Callable = field(default=PlantOptimalAllocation())    ## It is optional to define PlantOptimalAllocation for this method. If no argument is passed in here, then default setting for PlantAlloc is the default PlantOptimalAllocation().


    ## Module parameter attributes
    f_C: float = field(default=0.45)  ## Fraction of carbon in dry structural biomass (g C g d.wt-1)
    alpha_Rg: float = field(default=0.2)   ## Proportion of assimilates that are respired during growth, assumed to be fixed fraction of (GPP - Rm), usually 0.1-0.3
    SAI: float = field(default=0.1)   ## Stem area index (m2 m-2) TODO: make this a dynamic variable at some point. 
    clumping_factor: float = field(default=0.7)   ## Foliage clumping index (-) TODO: Place this parameter in a more suitable spot/module
    albsoib: float = field(default=0.2)   ## Soil background albedo for beam radiation (-) TODO: Place this parameter in a more suitable spot/module
    albsoid: float = field(default=0.2)   ## Soil background albedo for diffuse radiation (-) TODO: Place this parameter in a more suitable spot/module
    hc_max: float = field(default=0.6)   ## Maximum canopy height (m)
    hc_max_GDDindex: float = field(default=0.75)    ## Relative growing degree day index at peak canopy height (ranges between 0-1). A rule of thumb is that canopy height peaks at anthesis (see Kukal and Irmak, 2019, doi:10.2134/agronj2019.01.0017)
    d_r_max: float = field(default=2.0)    ## Maximum potential rooting depth (m)

    ## Seed germination and emergence parameters
    germination_phase: str = field(default="germination")  ## Name of developmental/growth phase in which germination to first emergence occurs. N.B. phase must be defined in in PlantDev.phases in the PlantDev() module
    k_sowdepth: float = field(default=1.0)        ## Sowing depth factor that determines how sowing depth impacts the germination and emergence developmental rate (-)
    sowingDepthMax: float = field(default=0.10)   ## Maximum potential sowing depth, below which germination and emergence are not possible (m)

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
    
    VD_method: str = field(
        default="nonlinear"
        ) ## method used to calculate daily thermal time for vernalization days. Options are: "nonlinear", "linear", "APSIM-Wheat-O"
    VD_Tbase: float = field(
        default=-1.3
    )  ## Base temperature (minimum threshold) used for calculating the vernalization days
    VD_Topt: float = field(
        default=4.9
    )  ## Optimum temperature used for calculating the vernalization days (non-linear method only)
    VD_Tupp: float = field(
        default=15.7
    )  ## Upper temperature (maximum threshold) used for calculating the vernalization days
    VD50: float = field(default=25,)  ## Vernalization days where vernalization fraction is 50% (for "nonlinear" and "APSIM-Wheat-O" methods)
    VD_n: float = field(default=5,)  ## Vernalization sigmoid function shape parameter (for "nonlinear" method only)
    VD_Vnd: float = field(default=46.0,)  ## Vernalization requirement i.e. vernalization state where total vernalization is achieved (for "linear" method only)
    VD_Vnb: float = field(default=9.2,)  ## Base vernalization days (for "linear" method only)
    VD_Rv: float = field(default=0.6,)  ## Vernalization sensitivity factor (for "APSIM-Wheat-O" method only)

    HTT_T_b: float = field(default=0.0)  ## Germination hydrothermal time base temperature threshold (degrees Celsius)
    HTT_T_c: float = field(default=30.0)  ## Germination hydrothermal time ceiling temperature threshold (degrees Celsius)
    HTT_psi_b: float = field(default=-3.0)  ## Germination hydrothermal time base water potential threshold (MPa)
    HTT_k: float = field(default=0.07)  ## Germination slope of linear increase in psi_b when temperature increases (MPa degrees Celsius-1)
    HTT_Theta_HT: float = field(default=60.0)  ## HTT requirement for germination (MPa degrees Celcius d)

    remob_phase: str = field(default="spike")  ## Developmental phase when stem remobilization occurs. N.B. phase must be defined in in PlantDev.phases in the PlantDev() module
    Vmaxremob: float = field(default=0.5)  ## Maximum potential remobilization rate (analogous to phloem loading rate) for Michaelis-Menten function (g C m-2 d-1)
    Kmremob: float = field(default=0.4)  ## Michaelis-Menten kinetic parameter (unitless; same units as substrate in Michaelis-Menten equation which, in this case, is a unitless ratio)

    specified_phase: str = field(default="spike")  ## Developmental phase when accumulation of a defined carbon flux occurs. Usually used to support the grain production module. N.B. phase must be defined in in PlantDev.phases in the PlantDev() module

    ## Grain production module parameters for wheat (triticum)
    grainfill_phase: str = field(default="fruiting")  ## Name of developmental/growth phase in which grain filling occurs. N.B. phase must be defined in in PlantDev.phases in the PlantDev() module
    W_seedTKW0: float = field(default=35.0)  ## Wheat: Thousand kernel weight of grain (g thousand grains-1), acceptable range 28-48
    GY_FE: float = field(default=0.1)  ## Wheat: Reproductive (fruiting) efficiency (thousand grains g d.wt of spike-1), acceptable range 80-210
    GY_GN_max: float = field(default=20)  ## Wheat: Maximum potential grain number per ground area (default 20 thousand grains m-2), acceptable range 18-22
    GY_SDW_50: float = field(default=100)  ## Wheat: Spike dry weight at anthesis (SDW_a) at which grain number is half of GY_GN_max (default 100 g d.wt m-2), acceptable range 80-150
    GY_k: float = field(default=0.02)  ## Wheat: Growth rate controlling steepness of grain number sigmoid function (default 0.02), acceptable range 0.01-0.03

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
        VRN_time,      ## vernalization state
        HTT_time,      ## hydrothermal time state, for germination (MPa degrees Celcius d)
        Cstate,        ## Ambiguous carbon state to track, provides an input to the grain production module
        Cseedbed,      ## seed bed carbon
        solRadswskyb,  ## atmospheric direct beam solar radiation (W/m2)
        solRadswskyd,  ## atmospheric diffuse solar radiation (W/m2)
        airTempCMin,   ## minimum air temperature (degrees Celsius)
        airTempCMax,   ## maximum air temperature (degrees Celsius)
        airP,          ## air pressure (Pa)
        airRH,         ## relative humidity (%)
        airCO2,        ## partial pressure CO2 (bar)
        airO2,         ## partial pressure O2 (bar)
        soilTheta,     ## volumetric soil water content (m3 m-3), dimensions (soil layer,)
        _doy,
        _year,
    ) -> Tuple[float]:

        ## Solar calculations
        eqtime, houranglesunrise, theta = self.Site.solar_calcs(_year,_doy)

        ## Climate calculations
        airTempC = self.Site.compute_mean_daily_air_temp(airTempCMin,airTempCMax)

        F_C_sowing = self.calculate_sowingrate_conditional(_doy)

        BioHarvestLeaf = self.calculate_BioHarvest(Cleaf,_doy,self.Management.harvestDay,self.propHarvestLeaf,self.Management.PhHarvestTurnoverTime)
        BioHarvestStem = self.calculate_BioHarvest(Cstem,_doy,self.Management.harvestDay,self.propHarvestStem,self.Management.PhHarvestTurnoverTime)
        BioHarvestSeed = self.calculate_BioHarvest(Cseed,_doy,self.Management.harvestDay,self.propHarvestSeed,self.Management.PhHarvestTurnoverTime)

        # Germination: hydrothermal time state
        dHTTdt = self.calculate_dailyhydrothermaltime(airTempC, soilTheta[0])  ## assume only the uppermost soil layer controls germination
        # TODO: Need trigger to say, when HTT_time >= self.HTT_Theta_HT, we update the PlantDev to the next development phase

        # Development phase index
        idevphase = self.PlantDev.get_active_phase_index(Bio_time)
        # Determine canopy height
        relative_gdd = self.PlantDev.calc_relative_gdd_index(Bio_time)
        hc = self.calculate_canopy_height(relative_gdd)
        relative_gdd_anthesis = self.PlantDev.calc_relative_gdd_to_anthesis(Bio_time)
        d_r = self.calculate_root_depth(relative_gdd_anthesis)
        # Vernalization state
        self.PlantDev.update_vd_state(VRN_time,Bio_time)    # Update vernalization state information to track developmental phase changes
        VD = self.PlantDev.get_phase_vd()    # Get vernalization state for current developmental phase
        # Update vernalization days requirement for current developmental phase
        self.VD50 = 0.5 * self.PlantDev.vd_requirements[idevphase]

        sunrise, solarnoon, sunset = self.Site.solar_day_calcs(_year,_doy)
        DTT = self.calculate_dailythermaltime(airTempCMin,airTempCMax,sunrise,sunset)
        fV = self.vernalization_factor(VD)
        fGerm = self.calculate_sowingdepth_factor(Bio_time)
        dGDDdt = fGerm*fV*DTT

        deltaVD = self.calculate_vernalizationtime(airTempCMin,airTempCMax,sunrise,sunset)
        dVDdt = deltaVD

        W_L = Cleaf/self.f_C
        W_R = Croot/self.f_C

        if (W_L == 0) or (W_R == 0):
            GPP = 0
            Rm = 0
        else:
            _GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = self.PlantCH2O.calculate(W_L,W_R,soilTheta,airTempC,airTempC,airRH,airCO2,airO2,airP,solRadswskyb,solRadswskyd,theta,hc,d_r)
            GPP = _GPP * 12.01 * (60*60*24) / 1e6  ## converts native PlantCH2O units (umol C m-2 s-1) to units needed in this module (g C m-2 d-1)
            Rm = (Rml+Rmr) * 12.01 * (60*60*24) / 1e6  ## Maintenance respiration. Converts native PlantCH2O units (umol C m-2 s-1) to units needed in this module (g C m-2 d-1)
        
        # Calculate NPP
        NPP = self.calculate_NPP_RmRgpropto(GPP,Rm)

        # Allocation fractions per pool
        alloc_coeffs = self.PlantDev.allocation_coeffs[idevphase]
        # Turnover rates per pool (days-1)
        tr_ = self.PlantDev.turnover_rates[idevphase]

        # Grain production module calculations
        # Calculate potential grain number
        GN_pot = self.calculate_wheat_grain_number(Cstate/self.f_C)
        # Calculate stem remobilization to grain
        F_C_stem2grain = self.calculate_nsc_stem_remob(Cstem, Cleaf, Cseed/self.f_C, GN_pot*self.W_seedTKW0, Bio_time)
        # Calculate allocation fraction to grain
        u_Seed = self.calculate_grain_alloc_coeff(alloc_coeffs[self.PlantDev.iseed], Cseed/self.f_C, GN_pot*self.W_seedTKW0, Bio_time)
        u_Stem = alloc_coeffs[self.PlantDev.istem]

        # Set any constant allocation coefficients for optimal allocation
        self.PlantAlloc.u_Stem = u_Stem
        self.PlantAlloc.u_Seed = u_Seed
        # Set pool turnover rates for optimal allocation
        self.PlantAlloc.tr_L = tr_[self.PlantDev.ileaf]    #1 if tr_[self.PlantDev.ileaf] == 0 else max(1, 1/tr_[self.PlantDev.ileaf])
        self.PlantAlloc.tr_R = tr_[self.PlantDev.iroot]    #1 if tr_[self.PlantDev.iroot] == 0 else max(1, 1/tr_[self.PlantDev.ileaf])
        if (W_L == 0) or (W_R == 0):
            u_L = alloc_coeffs[self.PlantDev.ileaf]
            u_R = alloc_coeffs[self.PlantDev.iroot]
        else:
            u_L, u_R, _, _, _, _ = self.PlantAlloc.calculate(W_L,W_R,soilTheta,airTempC,airTempC,airRH,airCO2,airO2,airP,solRadswskyb,solRadswskyd,theta,hc,d_r)

        # Fluxes from seedbed (i.e. sowed seeds)
        F_C_seed2leaf, F_C_seed2stem, F_C_seed2root = self.calculate_emergence_fluxes(Bio_time, Cseedbed)

        # ODE for plant carbon pools
        dCseedbeddt = F_C_sowing - F_C_seed2leaf - F_C_seed2stem - F_C_seed2root
        dCleafdt = F_C_seed2leaf + u_L*NPP - tr_[self.PlantDev.ileaf]*Cleaf - BioHarvestLeaf
        dCstemdt = F_C_seed2stem + u_Stem*NPP - tr_[self.PlantDev.istem]*Cstem - BioHarvestStem - F_C_stem2grain
        dCrootdt = F_C_seed2root + u_R*NPP - tr_[self.PlantDev.iroot]*Croot
        dCseeddt = u_Seed*NPP - tr_[self.PlantDev.iseed]*Cseed - BioHarvestSeed + F_C_stem2grain
        dCStatedt = self.calculate_devphase_Cflux(u_Stem*NPP - tr_[self.PlantDev.istem]*Cstem - BioHarvestStem - F_C_stem2grain, Bio_time)

        return (dCleafdt, dCstemdt, dCrootdt, dCseeddt, dGDDdt, dVDdt, dHTTdt, dCStatedt, dCseedbeddt)

    def calculate_NPP_RmRgpropto(self,GPP,R_m):
        R_g = self.alpha_Rg * (GPP - R_m)
        R_a = R_g+R_m
        NPP = GPP - R_a
        return NPP

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

    def calculate_vernalizationtime(self,Tmin,Tmax,sunrise,sunset):
        _vfunc = np.vectorize(growing_degree_days_DTT_nonlinear)
        deltaVD = _vfunc(Tmin,Tmax,sunrise,sunset,self.VD_Tbase,self.VD_Tupp,self.VD_Topt,normalise=True)
        return deltaVD

    def calculate_dailyhydrothermaltime(self,airTempC,soilTheta):
        """
        Calculates the daily increment in hydrothermal time
        """
        T = airTempC
        Psi_s = self.PlantCH2O.soil_water_potential(soilTheta)
        bool_multiplier = (T > self.HTT_T_b) & (T < self.HTT_T_c) & (Psi_s > self.HTT_psi_b + self.HTT_k * (T - self.HTT_T_b))  # this constrains the equation to be within the temperature and soil water potential limits
        deltaHTT_d = np.maximum(0, (T - self.HTT_T_b) * (Psi_s - self.HTT_psi_b - self.HTT_k*(T - self.HTT_T_b))) * bool_multiplier
        return deltaHTT_d

    def vernalization_factor(self,VD):
        """
        
        Parameters
        ----------
        VD : float or array_like
            Vernalization days
        VD50 : float
            Vernalization days where vernalization fraction is 50%
        n : float
            Vernalization sigmoid function shape parameter for "nonlinear" model
        Vnd : float
            Vernalization requirement (vernalization state where total vernalization is achieved) for the "linear" model
        Vnb : float
            Base vernalization days for the "linear" model
        Rv : float 
            Vernalization sensitivity factor for the "APSIM-Wheat-O" model
        method : str
            One of "linear", "nonlinear", or "APSIM-Wheat-O"

        Notes
        -----
        The default parameters Vnd and Vnb are set according to Wang and Engel (1998), where Vnb is 20% of Vnd. 
        Typical range for the Rv parameter in the APSIM-Wheat-O model is 0.2-2.3 (Zheng et al., 2013). 

        References
        ----------
        Streck et al., 2003, doi:10.1016/S0168-1923(02)00228-9
        Wang and Engel, 1998, doi:10.1016/S0308-521X(98)00028-6
        Zheng et al., 2013, doi:10.1093/jxb/ert209
        """
        if self.VD50 == 0:
            return np.ones_like(VD)
        elif self.VD_method == "linear":
            fV = np.minimum(1, np.maximum(0, (VD - self.VD_Vnb)/(self.VD_Vnd - self.VD_Vnb)))
        elif self.VD_method == "nonlinear":
            fV = (VD**self.VD_n)/(self.VD50**self.VD_n + VD**self.VD_n)
        elif self.VD_method == "APSIM-Wheat-O":
            fV = 1 - (0.0054545*self.VD_Rv + 0.0003)*((2*self.VD50)-VD)
        return fV

    def calculate_sowingtime_conditional(self,_doy):
        if self.Management.sowingDay is None:
            return 0
        elif (self.Management.sowingDay <= _doy < self.Management.sowingDay+1):
            return 1
        else:
            return 0

    def calculate_sowingrate_conditional(self,_doy):
        """
        Calculates the sowing rate based on the day of year (DOY) and management parameters. 
        The sowing rate is converted from the standard units of kg ha-1 to g C m-2 and adjusted 
        based on the conditional sowing time.

        Parameters
        ----------
        _doy : int
            The day of year (DOY) used to calculate the sowing time conditionally, which is 
            factored into the final sowing rate calculation.

        Returns
        -------
        sowingrate : float
            The sowing rate in g C m-2

        Notes
        -----
        The function first calculates the conditional sowing time based on the day of year (DOY), 
        using `calculate_sowingtime_conditional`. The management-defined sowing rate (kg ha-1) 
        is then multiplied by the conditional factor `f_C` and converted to g C m-2.
        The formula:
            sowingrate = sowingTime * (sowingRate * f_C * 1000 / 10000)
        adjusts the sowing rate, where the constant factors convert from kg ha-1 to g C m-2.
        """
        sowingTime = self.calculate_sowingtime_conditional(_doy)
        sowingrate = sowingTime * (self.Management.sowingRate * self.f_C * 1000 / 10000)    # includes conversion of sowing rate units from kg ha-1 to g C m-2
        return sowingrate

    def calculate_sowingdepth_factor(self, Bio_time):
        """
        Calculates the sowing depth scaling factor (f_d) for GDD accumulation during the germination phase.
        The scaling factor decreases linearly with increasing sowing depth and ensures no germination 
        occurs if the sowing depth exceeds a specified maximum.

        Parameters
        ----------
        Bio_time : float
            The current growing degree days (thermal time), used to determine the developmental phase of the plant.

        Returns
        -------
        f_d : float
            The sowing depth scaling factor, which modifies the GDD accumulation rate. 
            If the sowing depth is greater than the specified maximum, f_d is set to 0 (i.e., no germination).
            If the sowing depth is within the permissible range, f_d scales linearly with depth and is clamped to be non-negative.

        Notes
        -----
        This function only modifies the GDD accumulation rate during the germination phase of plant development. 
        Outside of the germination phase, the scaling factor is set to 1.0, meaning no modification is applied.
        """
        if self.PlantDev.is_in_phase(Bio_time, self.germination_phase):
            f_d = (1 - self.k_sowdepth*self.Management.sowingDepth)*(self.Management.sowingDepth <= self.sowingDepthMax) + (self.Management.sowingDepth > self.sowingDepthMax)*0.0
            return np.maximum(f_d, 0.0)
        else:
            # no sowing depth factor modifier when outside germination developmental phase
            return 1.0

    def calculate_emergence_fluxes(self, Bio_time, Cseedbed):
        """
        Calculates the carbon fluxes from the seedbed pool to the leaf, stem, and root pools during the emergence phase.
        The transfer of carbon depends on whether the growing degree day (GDD) requirement for emergence has been met.

        Parameters
        ----------
        Bio_time : float
            The current biological time or thermal time, used to determine if the GDD requirement for emergence has been satisfied (deg C)
            
        Cseedbed : float
            The amount of available carbon in the seedbed pool that can be allocated to plant organs (g C m-2)

        Returns
        -------
        F_C_seed2leaf : float
            The flux of carbon from the seedbed pool to the leaf pool. This value is non-zero only after the GDD requirement 
            for emergence has been reached (g C m-2 d-1)
            
        F_C_seed2stem : float
            The flux of carbon from the seedbed pool to the stem pool. This value is non-zero only after the GDD requirement 
            for emergence has been reached (g C m-2 d-1)
            
        F_C_seed2root : float
            The flux of carbon from the seedbed pool to the root pool. This value is non-zero only after the GDD requirement 
            for emergence has been reached (g C m-2 d-1)

        Notes
        -----
        The function models carbon allocation based on thermal time, ensuring that carbon transfer from the seedbed pool 
        to the plant organs (leaf, stem, root) occurs only once the growing degree day requirement for emergence is met. 
        Before this threshold, no carbon is transferred from the seedbed.
        """
        idevphase_germ = self.PlantDev.phases.index(self.germination_phase)
        u_L = self.PlantDev.allocation_coeffs[idevphase_germ][self.PlantDev.ileaf]
        u_Stem = self.PlantDev.allocation_coeffs[idevphase_germ][self.PlantDev.istem]
        u_R = self.PlantDev.allocation_coeffs[idevphase_germ][self.PlantDev.iroot]

        GDD_requirement_emerg = self.PlantDev.gdd_requirements[idevphase_germ]
        
        if (Bio_time >= GDD_requirement_emerg):
            # immediate and complete transfer of carbon from seed bed to leaf, stem and root pools
            F_C_seed2leaf = u_R * Cseedbed
            F_C_seed2stem = u_Stem * Cseedbed
            F_C_seed2root = u_R * Cseedbed
        else:
            # have not reached growing-degree-day requirement yet, no input fluxes
            F_C_seed2leaf = 0.0
            F_C_seed2stem = 0.0
            F_C_seed2root = 0.0

        return F_C_seed2leaf, F_C_seed2stem, F_C_seed2root

    def calculate_BioHarvest(self,Biomass,_doy,harvestDay,propHarvest,HarvestTurnoverTime):
        _vfunc = np.vectorize(self.calculate_harvesttime_conditional,otypes=[float])
        HarvestTime = _vfunc(_doy,harvestDay)
        BioHarvest = HarvestTime*np.maximum(propHarvest*Biomass/HarvestTurnoverTime,0)
        return BioHarvest

    def calculate_harvesttime_conditional(self,_doy,harvestDay):
        if harvestDay is None:
            return 0
        elif (harvestDay <= _doy < harvestDay+3):  ## assume harvest happens over a single day
            return 1
        else:
            return 0

    def calculate_canopy_height(self, relative_gdd_index):
        """
        Calculate canopy height of an annual herbaceous plants. 
        Canopy height is determined as a linear function of growing degree days up 
        to a point in the season where canopy height peaks at its defined maximum. 

        Parameters
        ----------
        relative_gdd_index : float or array_like
            Relative growing degree day index, ranges between 0 and 1, indicating the relative development growth phase from germination to the end of seed/fruit filling period. 

        Returns
        -------
        hc : float or array_like
            Canopy height (m)

        Notes
        -----
        This model is most applicable to annual crops. 
        A rule of thumb is that canopy height peaks at anthesis (Kakul and Irmak, 2019)

        References
        ----------
        Kukal and Irmak, 2019, doi:10.2134/agronj2019.01.0017
        """
        hc = self.hc_max * np.minimum(1, (1/self.hc_max_GDDindex)*relative_gdd_index)
        return hc

    def calculate_root_depth(self, relative_gdd_index):
        """
        Calculate root depth of an annual herbaceous plants. 
        Root depth (d_r) is determined as a linear function of growing degree days up 
        to a point in the season where it reaches a defined maximum. 

        Parameters
        ----------
        relative_gdd_index : float or array_like
            Relative growing degree day index, ranges between 0 and 1, indicating the relative development growth phase from germination to the end of a defined period e.g. anthesis

        Returns
        -------
        d_r : float or array_like
            Root depth (m)

        Notes
        -----

        References
        ----------
        """
        d_r = self.d_r_max * np.minimum(1, relative_gdd_index)
        return d_r

    def calculate_nsc_stem_remob(self, Cstem, Cleaf, W_seed, W_seed_pot, current_gdd):
        """
        Calculates the non-structural carbohydrate (NSC) remobilization from the stem during a specific growth phase.

        Parameters:
        ----------
        Cstem : float or array_like
            Carbon content in the stem (gC m-2)
        Cleaf : float or array_like
            Carbon content in the leaves (gC m-2)
        W_seed : float
            Current seed biomass pool size (g d.wt m-2)
        W_seed_pot : float
            Maximum potential seed biomass pool size (g d.wt m-2)
        current_gdd : float
            The current growing degree days (GDD), used to determine if the plant is in the specified remobilization phase.

        Returns:
        -------
        float or array_like
            The rate of stem remobilization based on the stem-to-leaf carbon ratio if the plant is in the remobilization phase, otherwise returns 0.
        """

        if self.PlantDev.is_in_phase(current_gdd, self.remob_phase):
            if W_seed/W_seed_pot <= 1:
                # if the seed dry weight is below the maximum seed dry weight
                # Calculate the actual stem:leaf ratio
                R_stem_leaf_actual = Cstem / Cleaf
                # C_organ = np.minimum(R_stem_leaf_actual/R_stem_leaf_opt,1)
                C_organ = R_stem_leaf_actual #/R_stem_leaf_opt
                L = self.Vmaxremob * C_organ/(self.Kmremob + C_organ)
            else:
                # if the seed dry weight has reached the maximum seed dry weight
                L = 0
            return L
        else:
            return 0

    def calculate_devphase_Cflux(self, Cflux, current_gdd):
        """
        Returns the Cflux if the plant is in a specified growth phase, otherwise it returns 0. 
        This is used to determine things like accumulated NPP during spike development in wheat to 
        input to the grain production module. 

        Parameters:
        ----------
        Cflux : float or array_like
            Carbon flux (gC m-2 d-1).
        current_gdd : float
            The current growing degree days (GDD), used to determine if the plant is in the specified remobilization phase.

        Returns:
        -------
        float or array_like
            The carbon flux if the plant is in the specified phase, otherwise returns 0.
        """
        
        if self.PlantDev.is_in_phase(current_gdd, self.specified_phase):
            return Cflux
        else:
            return 0

    def calculate_wheat_grain_number(self, SDW_a):
        """
        Calculate the grain number using a sigmoid function based on spike dry weight at anthesis 
        and the fruiting efficiency. Applicable to wheat (triticum).
        
        Parameters
        ----------
        SDW_a : float
            Spike dry weight at anthesis (g d.wt m-2 ground area)
        FE : float
            Fruiting efficiency (thousand grains g d.wt of spike-1)
        GN_max : float
            Maximum potential grain number (default 20 thousand grains m-2)
        SDW_50 : float 
            SDW_a at which grain number is half of GN_max (default 10.0 g d.wt m-2)
        k : float
            Growth rate controlling steepness (default 0.02)
        
        Returns
        -------
        float: Calculated grain number, GN (thousand grains m-2 ground area)

        Notes
        -----
        This formulation was designed to capture the relationship of grain number to spike dry weight and 
        fruiting efficiency as described in Pretini et al. (2021), Terrile et al. (2017) Fischer et al. 
        (2024) and references therein. The formulation produces a near-linear relationship between SDW_a 
        and GN when there is adequate levels of SDW_a, while it is non-linear (positive curvi-linear) 
        as SDW_a nears zero, to capture the potential ability of grain production from new assimilates, 
        even if spike mass is low after anthesis. It also produces the observed linear relationship 
        between FE and GN (Fig. 7, Terrile et al., 2017).

        Sensitivity tests and comparison against the literature (e.g. Pretini et al., 2021) show that 
        the parameters k, SDW_50 and GN_max should be fairly conservative and thus kept relatively constant
        for wheat, with acceptable ranges of k=[0.01-0.03], SDW_50=[80-150] and GN_max=[18000-20000] for 
        wheat. The parameter FE typically ranges between 80-210 grains g d.wt-1
        (FE=[0.08-0.21] thousand grains g d.wt-1), which should be specified per genotype and perhaps 
        modified for environmental conditions in future (e.g. frost will reduce FE).

        References
        ----------
        Pretini et al., 2021, doi:10.1093/jxb/erab080
        Fischer et al., 2024, doi:10.1016/j.fcr.2024.109497
        Terrile et al., 2017, doi:10.1016/j.fcr.2016.09.026
        """
        # Logistic function for gradual transitions
        sigmoid_factor = 1 / (1 + np.exp(-self.GY_k * (SDW_a - self.GY_SDW_50)))
        GN = self.GY_FE * SDW_a * sigmoid_factor
        return np.minimum(GN, self.GY_GN_max)

    def calculate_grain_alloc_coeff(self, u_S_max, W_seed, W_seed_pot, current_gdd):
        """
        Calculates the allocation coefficient to the seed pool given the current and maximum potential
        seed pool size and developmental state. 

        Parameters
        ----------
        u_S_max : float
            Maximum potential allocation coefficient (-)
        W_seed : float
            Current seed biomass pool size (g d.wt m-2)
        W_seed_pot : float
            Maximum potential seed biomass pool size (g d.wt m-2)

        Returns
        -------
        u_S : float 
            Allocation coefficient to seed pool

        Notes
        -----
        The actual seed biomass pool, W_seed, will be determined by the amount of biomass flowing into the 
        seed pool over a given time interval i.e. grain filling. This flux should continue until the maximum 
        potential seed pool size, W_seed_pot, is reached. The allocation coefficient from assimilates to the 
        seed pool, u_S, is assumed to occur at the maximum until the maximum potential seed pool size is 
        reached. So, as long as the current seed biomass pool is below the potential, then the flux of carbon 
        continues at the maximum potential rate.

        """
        grainfill = self.PlantDev.is_in_phase(current_gdd, self.grainfill_phase)
        if grainfill:
            # if plant is in the grain filling phase
            if W_seed/W_seed_pot <= 1:
                # if the seed dry weight is below the maximum seed dry weight
                u_S = u_S_max
            else:
                # if the seed dry weight has reached the maximum seed dry weight
                u_S = 0
        else:
            # if the plant is not in the grain filling phase
            u_S = 0
        return u_S


