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
from daesim.leafgasexchange2 import LeafGasExchangeModule2
from daesim.canopygasexchange import CanopyGasExchange
from daesim.plantcarbonwater import PlantModel as PlantCH2O

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

    ## Module parameter attributes
    f_C: float = field(default=0.45)  ## Fraction of carbon in dry structural biomass (g C g d.wt-1)
    CUE: float = field(default=0.5)  ## Plant carbon-use-efficiency (CUE=NPP/GPP)
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
        solRadswskyb,  ## atmospheric direct beam solar radiation (W/m2)
        solRadswskyd,  ## atmospheric diffuse solar radiation (W/m2)
        airTempCMin,   ## minimum air temperature (degrees Celsius)
        airTempCMax,   ## maximum air temperature (degrees Celsius)
        airP,          ## air pressure (Pa)
        airRH,         ## relative humidity (%)
        airCO2,        ## partial pressure CO2 (bar)
        airO2,         ## partial pressure O2 (bar)
        soilTheta,     ## volumetric soil water content (m3 m-3)
        _doy,
        _year,
    ) -> Tuple[float]:

        ## Solar calculations
        eqtime, houranglesunrise, theta = self.Site.solar_calcs(_year,_doy)

        ## Climate calculations
        airTempC = self.Site.compute_mean_daily_air_temp(airTempCMin,airTempCMax)

        BioPlanting = self.calculate_BioPlanting(_doy,self.Management.plantingDay,self.Management.plantingRate,self.Management.plantWeight) ## Modification: using a newly defined parameter in this function instead of "frequPlanting" as used in Stella, considering frequPlanting was being used incorrectly, as its use didn't match with the units or definition.

        BioHarvestLeaf = self.calculate_BioHarvest(Cleaf,_doy,self.Management.harvestDay,self.propHarvestLeaf,self.Management.PhHarvestTurnoverTime)
        BioHarvestStem = self.calculate_BioHarvest(Cstem,_doy,self.Management.harvestDay,self.propHarvestStem,self.Management.PhHarvestTurnoverTime)
        BioHarvestSeed = self.calculate_BioHarvest(Cseed,_doy,self.Management.harvestDay,self.propHarvestSeed,self.Management.PhHarvestTurnoverTime)

        sunrise, solarnoon, sunset = self.Site.solar_day_calcs(_year,_doy)
        DTT = self.calculate_dailythermaltime(airTempCMin,airTempCMax,sunrise,sunset)
        dGDDdt = DTT

        W_L = Cleaf/self.f_C
        W_R = Croot/self.f_C

        GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = self.PlantCH2O.calculate(W_L,W_R,soilTheta,airTempC,airTempC,airRH,airCO2,airO2,airP,solRadswskyb,solRadswskyd,theta)
        GPP_gCm2d = GPP * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
        
        # Calculate NPP
        NPP = self.calculate_NPP(GPP)

        # Development phase index
        idevphase = self.PlantDev.get_active_phase_index(Bio_time)
        # Allocation fractions per pool
        alloc_coeffs = self.PlantDev.allocation_coeffs[idevphase]
        # Turnover rates per pool
        tr_ = self.PlantDev.turnover_rates[idevphase]
        
        # ODE for plant carbon pools
        dCleafdt = alloc_coeffs[self.PlantDev.ileaf]*NPP - tr_[self.PlantDev.ileaf]*Cleaf - BioHarvestLeaf
        dCstemdt = alloc_coeffs[self.PlantDev.istem]*NPP - tr_[self.PlantDev.istem]*Cstem - BioHarvestStem
        dCrootdt = alloc_coeffs[self.PlantDev.iroot]*NPP - tr_[self.PlantDev.iroot]*Croot + BioPlanting
        dCseeddt = alloc_coeffs[self.PlantDev.iseed]*NPP - tr_[self.PlantDev.iseed]*Cseed - BioHarvestSeed

        return (dCleafdt, dCstemdt, dCrootdt, dCseeddt, dGDDdt)

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
        BioHarvest = HarvestTime*np.maximum(propHarvest*Biomass/HarvestTurnoverTime,0)
        return BioHarvest

    def calculate_harvesttime_conditional(self,_doy,harvestDay):
        if harvestDay is None:
            return 0
        elif (harvestDay <= _doy < harvestDay+3):  ## assume harvest happens over a single day
            return 1
        else:
            return 0

