"""
Plant model class: Includes differential equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
from daesim.climate import *
from daesim.biophysics_funcs import func_TempCoeff, growing_degree_days_DTT_nonlinear, growing_degree_days_DTT_linear1, growing_degree_days_DTT_linear2, growing_degree_days_DTT_linear3, growing_degree_days_DTT_linear4, growing_degree_days_DTT_linearpeaked
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


    ## Module parameter attributes
    
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
    
    def calculate(
        self,
        Bio_time,
        VRN_time,      ## vernalization state
        solRadswskyb,  ## atmospheric direct beam solar radiation (W/m2)
        solRadswskyd,  ## atmospheric diffuse solar radiation (W/m2)
        airTempCMin,   ## minimum air temperature (degrees Celsius)
        airTempCMax,   ## maximum air temperature (degrees Celsius)
        airP,          ## air pressure (Pa)
        airRH,         ## relative humidity (%)
        airCO2,        ## partial pressure CO2 (bar)
        airO2,         ## partial pressure O2 (bar)
        airU,          ## wind speed (m s-1) at measurement height specified in Site module
        soilTheta,     ## volumetric soil water content (m3 m-3), dimensions (soil layer,)
        _doy,
        _year,
        return_diagnostics: bool = False  # This flag is set and used by ODEModelSolver
    ) -> Tuple[float]:

        # Solar calculations
        eqtime, houranglesunrise, theta = self.Site.solar_calcs(_year,_doy)
        sunrise, solarnoon, sunset = self.Site.solar_day_calcs(_year,_doy)

        # Climate calculations
        airTempC = self.Site.compute_mean_daily_air_temp(airTempCMin,airTempCMax)

        # Plant development
        
        # Plant development phase index
        idevphase = self.PlantDev.get_active_phase_index(Bio_time)
        
        # Vernalization state
        self.PlantDev.update_vd_state(VRN_time,Bio_time)    # Update vernalization state information to track developmental phase changes
        VD = self.PlantDev.get_phase_vd()    # Get vernalization state for current developmental phase
        self.VD50 = (0.5*self.PlantDev.vd_requirements[idevphase] if idevphase is not None else 0)    # Update vernalization days requirement for current developmental phase
        deltaVD = self.calculate_vernalizationtime(airTempCMin,airTempCMax,sunrise,sunset)
        dVDdt = deltaVD

        # Growing degree days (thermal time)
        DTT = self.calculate_dailythermaltime(airTempCMin,airTempCMax,sunrise,sunset)
        fV = self.vernalization_factor(VD)
        fGerm = self.calculate_sowingdepth_factor(Bio_time)
        dGDDdt = fGerm*fV*DTT

        # Prepare diagnostics if requested (N.B. diagnostics must always be the last item in the returned output)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                'fV': fV,
                'fGerm': fGerm,
                'DTT': DTT,
                'idevphase': idevphase,
            }

        return (dGDDdt, dVDdt, diagnostics)

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
        elif self.GDD_method == "linear4":
            _vfunc = np.vectorize(growing_degree_days_DTT_linear4)
            DTT = _vfunc(Tmin,Tmax,self.GDD_Tbase,self.GDD_Tupp,self.GDD_Topt)
        elif self.GDD_method == "linearpeaked":
            _vfunc = np.vectorize(growing_degree_days_DTT_linearpeaked)
            DTT = _vfunc(Tmin,Tmax,self.GDD_Tbase,self.GDD_Tupp,self.GDD_Topt)
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
            Vernalization days (deg C d)
        VD50 : float
            Vernalization days where vernalization fraction is 50% (deg C d)
        n : float
            Vernalization sigmoid function shape parameter for "nonlinear" model (-)
        Vnd : float
            Vernalization requirement (vernalization state where total vernalization is achieved) for the "linear" model (deg C d)
        Vnb : float
            Base vernalization days for the "linear" model (deg C d)
        Rv : float 
            Vernalization sensitivity factor for the "APSIM-Wheat-O" model (-)
        method : str
            One of "linear", "nonlinear", or "APSIM-Wheat-O"

        Returns
        -------
        fV : float or array_like
            Vernalization developmental rate factor (-)

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
