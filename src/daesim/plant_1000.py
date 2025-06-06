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
    PlantCH2O: Callable = field(default=PlantCH2O())    ## It is optional to define Plant for this method. If no argument is passed in here, then default setting for Plant is the default PlantModel().
    PlantAlloc: Callable = field(default=PlantOptimalAllocation())    ## It is optional to define PlantOptimalAllocation for this method. If no argument is passed in here, then default setting for PlantAlloc is the default PlantOptimalAllocation().


    ## Module parameter attributes
    alpha_Rg: float = field(default=0.2)   ## Proportion of assimilates that are respired during growth, assumed to be fixed fraction of (GPP - Rm), usually 0.1-0.3
    SAI: float = field(default=0.1)   ## Stem area index (m2 m-2) TODO: make this a dynamic variable at some point. 
    CI: float = field(default=0.5)   ## Foliage clumping index (-)
    hc_max: float = field(default=0.6)   ## Maximum canopy height (m)
    hc_max_GDDindex: float = field(default=0.75)    ## Relative growing degree day index at peak canopy height (ranges between 0-1). A rule of thumb is that canopy height peaks at anthesis (see Kukal and Irmak, 2019, doi:10.2134/agronj2019.01.0017)
    d_r_max: float = field(default=2.0)    ## Maximum potential rooting depth (m)
    d_r_maxphase: str = field(default="anthesis")  ## Developmental phase when peak rooting depth is achieved (i.e. the start of the defined phase). Usually assumed to be start of anthesis/flowering. N.B. phase must be defined in in PlantDev.phases in the PlantDev() module

    ## Seed germination and emergence parameters
    germination_phase: str = field(default="germination")  ## Name of developmental/growth phase in which germination to first emergence occurs. N.B. phase must be defined in in PlantDev.phases in the PlantDev() module
    k_sowdepth: float = field(default=1.0)        ## Sowing depth factor that determines how sowing depth impacts the germination and emergence developmental rate (-)
    sowingDepthMax: float = field(default=0.10)   ## Maximum potential sowing depth, below which germination and emergence are not possible (m)

    GDD_method: str = field(
        default="nonlinear"
        ) ## method used to calculate daily thermal time and hence growing degree days. Options are: "nonlinear", "linear1", "linear2", "linear3", "linear4", "linearpeaked"
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

    remob_phase: list[str] = field(factory=lambda: ["grainfill"])  ## Developmental phase(s) when stem remobilization occurs (list of strings). N.B. phase(s) must be defined in in PlantDev.phases in the PlantDev() module
    Vmaxremob: float = field(default=0.5)  ## Maximum potential remobilization rate (analogous to phloem loading rate) for Michaelis-Menten function (g C m-2 d-1)
    Kmremob: float = field(default=0.4)  ## Michaelis-Menten kinetic parameter (unitless; same units as substrate in Michaelis-Menten equation which, in this case, is a unitless ratio)

    specified_phase: str = field(default="spike")  ## Developmental phase when accumulation of a defined carbon flux occurs. Usually used to support the grain production module. N.B. phase must be defined in in PlantDev.phases in the PlantDev() module
    downreg_phase: str = field(default="maturity")  ## Developmental phase when down-regulation of selected physiological parameters occurs. Usually occurs during maturity or senescence. N.B. phase must be defined in in PlantDev.phases in the PlantDev() module
    downreg_proportion: float = field(default=0.99)  ## Fractional down-regulation of selected physiological parameters over defined downreg_phase e.g. 0.99 means a total of 99% down-regulation of parameters by the end of the downreg_phase

    ## Grain production module parameters
    grainfill_phase: list[str] = field(factory=lambda: ["grainfill"])  ## Name of developmental/growth phase(s) in which grain filling occurs (list of strings). N.B. phase(s) must be defined in in PlantDev.phases in the PlantDev() module
    
    ## Grain production module parameters for wheat (triticum)
    W_seedTKW0: float = field(default=35.0)  ## Wheat: Thousand kernel weight of grain (g d.wt thousand grains-1), acceptable range 28-48
    GY_FE: float = field(default=0.1)  ## Wheat: Reproductive (fruiting) efficiency (thousand grains g d.wt of spike-1), acceptable range 80-210
    GY_GN_max: float = field(default=20)  ## Wheat: Maximum potential grain number per ground area (default 20 thousand grains m-2), acceptable range 18-22
    GY_SDW_50: float = field(default=100)  ## Wheat: Spike dry weight at anthesis (SDW_a) at which grain number is half of GY_GN_max (default 100 g d.wt m-2), acceptable range 80-150
    GY_k: float = field(default=0.02)  ## Wheat: Growth rate controlling steepness of grain number sigmoid function (default 0.02), acceptable range 0.01-0.03
    
    ## Grain production module parameters for canola (Brassica napus L.)
    GY_B: float = field(default=100)  ## Rate constant (g C m-2 ground area) describing the rate of increase in potential seed density relative to cumulative NPP during anthesis
    GY_S_dmin: float = field(default=30)  ## Minimum potential seed density (thousand seeds m-2 ground area)
    GY_S_dmax: float = field(default=130)  ## Maximum potential seed density (thousand seeds m-2 ground area)
    GY_W_TKWseed_base: float = field(default=4.0)  ## Base thousand kernel weight of seed when S_dpot = S_dmax or when k_comp=0 (g thousand seeds-1)
    GY_k_comp: float = field(default=0.01)  ## Compensation factor that controls the rate of increase in seed weight as potential seed density decreases from GY_S_dmax (g m2 thousand seeds-1)

    ## Parameters that are set and need storage
    p1: float = field(default=None)
    p2: float = field(default=None)
    p3: float = field(default=None)
    
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
        leafTempC = self.Site.compute_skin_temp(airTempC, solRadswskyb+solRadswskyd)

        # Sowing event and rate
        F_C_sowing = self.calculate_sowingrate_conditional(_doy,_year)
        F_C_seed2leaf, F_C_seed2stem, F_C_seed2root = self.calculate_emergence_fluxes(Bio_time, Cseedbed)  # Fluxes from seedbed (i.e. sowed seeds) to plant pools

        # Harvest event and rates
        BioHarvestLeaf = self.calculate_BioHarvest(Cleaf,_doy,_year,self.Management.propHarvestLeaf,self.Management.PhHarvestTurnoverTime)
        BioHarvestStem = self.calculate_BioHarvest(Cstem,_doy,_year,self.Management.propHarvestStem,self.Management.PhHarvestTurnoverTime)
        BioHarvestSeed = self.calculate_BioHarvest(Cseed,_doy,_year,self.Management.propHarvestSeed,self.Management.PhHarvestTurnoverTime)

        # Germination: hydrothermal time state
        dHTTdt = self.calculate_dailyhydrothermaltime(airTempC, soilTheta)
        # TODO: Need trigger to say, when HTT_time >= self.HTT_Theta_HT, we update the PlantDev to the next development phase

        # Plant development
        
        # Plant development phase index
        idevphase = self.PlantDev.get_active_phase_index(Bio_time)
        
        # Calculate canopy height
        relative_gdd = self.PlantDev.calc_relative_gdd_index(Bio_time)
        hc = self.calculate_canopy_height(relative_gdd)
        
        # Calculate root depth
        relative_gdd_d_r = self.PlantDev.calc_relative_gdd_to_phase(Bio_time,self.d_r_maxphase)
        d_rpot = self.calculate_root_depth(relative_gdd_d_r)

        # Down-regulate selected physiological parameters during senescence/maturity phase
        # TODO: This is not good coding practice, need to find a better way to handle this
        if self.downreg_phase is not None:
            # Before any modification of module attributes (i.e. parameters), we must store their initial values
            if self.p1 == None:
                self.p1 = self.PlantCH2O.k_rl
                self.p2 = self.PlantCH2O.CanopyGasExchange.Leaf.Vcmax_opt
                self.p3 = self.PlantCH2O.CanopyGasExchange.Leaf.g1

            # Down-regulation is developmental phase specific
            if self.PlantDev.is_in_phase(Bio_time, self.downreg_phase):
                # if development is within the downreg_phase then physiological down-regulation occurs
                scaling_factor = self.PlantDev.index_progress_through_phase(Bio_time, self.downreg_phase)
                self.PlantCH2O.k_rl = self.scale_parameter(self.p1, self.downreg_proportion, scaling_factor)
                self.PlantCH2O.CanopyGasExchange.Leaf.Vcmax_opt = self.scale_parameter(self.p2, self.downreg_proportion, scaling_factor)
                self.PlantCH2O.CanopyGasExchange.Leaf.g1 = self.scale_parameter(self.p3, self.downreg_proportion, scaling_factor)
            elif self.PlantDev.calc_relative_gdd_to_phase(Bio_time, self.downreg_phase, to_phase_start=False) == 1:
                # development is past the downreg_phase, so maintain the down-regulated physiological parameter values
                self.PlantCH2O.k_rl = self.scale_parameter(self.p1, self.downreg_proportion, 1.0)
                self.PlantCH2O.CanopyGasExchange.Leaf.Vcmax_opt = self.scale_parameter(self.p2, self.downreg_proportion, 1.0)
                self.PlantCH2O.CanopyGasExchange.Leaf.g1 = self.scale_parameter(self.p3, self.downreg_proportion, 1.0)
            elif (self.PlantDev.calc_relative_gdd_to_phase(Bio_time, self.downreg_phase) > 0) and (self.PlantDev.calc_relative_gdd_to_phase(Bio_time, self.downreg_phase) < 1):
                self.PlantCH2O.k_rl = self.p1  # after downreg_phase and return to growing season we reset the parameter back to its original value
                self.PlantCH2O.CanopyGasExchange.Leaf.Vcmax_opt = self.p2  # after downreg_phase and return to growing season we reset the parameter back to its original value
                self.PlantCH2O.CanopyGasExchange.Leaf.g1 = self.p3  # after downreg_phase and return to growing season we reset the parameter back to its original value
        
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

        # Live leaf and root biomass pools
        W_L = Cleaf/self.PlantCH2O.f_C
        W_R = Croot/self.PlantCH2O.f_C

        # Photosynthesis and plant respiration rates
        if (W_L <= 0) or (W_R <= 0):
            # If leaf or root biomass is zero, do not perform plant ecophysiology (carbon and water) calculations
            LAI = 0
            _GPP, _Rml, _Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            d_r = 0
        else:
            # Calculate wind speed at top-of-canopy
            LAI = self.PlantCH2O.calculate_LAI(W_L)
            airUhc = self.calculate_wind_speed_hc(airU,hc,LAI+self.SAI)
            # Calculate canopy carbon and water dynamics
            _GPP, _Rml, _Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = self.PlantCH2O.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,solRadswskyb,solRadswskyd,theta,self.SAI,self.CI,hc,d_rpot)
            d_r = self.PlantCH2O.calculate_root_depth(W_R, d_rpot)
        
        GPP = _GPP * 12.01 * (60*60*24) / 1e6  ## converts native PlantCH2O units (umol C m-2 s-1) to units needed in this module (g C m-2 d-1)
        Rml = _Rml * 12.01 * (60*60*24) / 1e6  ## converts native PlantCH2O units (umol C m-2 s-1) to units needed in this module (g C m-2 d-1)
        Rmr = _Rmr * 12.01 * (60*60*24) / 1e6  ## converts native PlantCH2O units (umol C m-2 s-1) to units needed in this module (g C m-2 d-1)
        Rm = (Rml+Rmr) #* 12.01 * (60*60*24) / 1e6  ## Maintenance respiration. Converts native PlantCH2O units (umol C m-2 s-1) to units needed in this module (g C m-2 d-1)
        
        # Calculate NPP
        NPP, Rg = self.calculate_NPP_RmRgpropto(GPP,Rm)

        # Plant carbon allocation. N.B. allocation coefficients are zero when idevphase=None (i.e. outside of life cycle)
        alloc_coeffs = (self.PlantDev.allocation_coeffs[idevphase] if idevphase is not None else [0]*len(self.PlantDev.phases))
        # Plant turnover rates. N.B. turnover remains active when idevphase=None (i.e. outside of life cycle)
        tr_ = (self.PlantDev.turnover_rates[idevphase] if idevphase is not None else self.PlantDev.turnover_rates[-1]) 

        # Grain production module calculations
        if self.Management.cropType == "Wheat":
            S_d_pot = self.calculate_wheat_grain_number(Cstate/self.PlantCH2O.f_C)    # Calculate potential seed density (often called grain number density for wheat)
            W_seed_pot = S_d_pot*self.W_seedTKW0  # Calculate potential grain mass
        if self.Management.cropType == "Canola":
            S_d_pot = self.calculate_canola_seed_density(Cstate)   # Calculate potential seed density
            W_TKWseed_pot = self.seed_mass_compensation(S_d_pot)   # Calculate potential seed thousand kernel weight, considering compensation
            W_seed_pot = W_TKWseed_pot * S_d_pot    # Calculate potential seed mass

        F_C_stem2grain = self.calculate_nsc_stem_remob(Cstem, Cleaf+Cstem+Croot, Cseed/self.PlantCH2O.f_C, W_seed_pot, Bio_time)    # Calculate stem remobilization to seed
        u_Seed = self.calculate_grain_alloc_coeff(alloc_coeffs[self.PlantDev.iseed], Cseed/self.PlantCH2O.f_C, W_seed_pot, Bio_time)    # Calculate allocation fraction to seed/grain pool
        u_Stem = alloc_coeffs[self.PlantDev.istem]

        # Dynamic, optimal trajectory carbon allocation to leaves and roots
        self.PlantAlloc.u_Stem = u_Stem    # Set constant allocation coefficients for optimal allocation
        self.PlantAlloc.u_Seed = u_Seed    # Set constant allocation coefficients for optimal allocation
        self.PlantAlloc.tr_L = tr_[self.PlantDev.ileaf]    # Set pool turnover rates for optimal allocation
        self.PlantAlloc.tr_R = tr_[self.PlantDev.iroot]    # Set pool turnover rates for optimal allocation
        if (W_L <= 0) or (W_R <= 0):
            # If leaf or root biomass is zero, do not perform plant ecophysiology (carbon and water) calculations, assume allocation coefficients are fixed
            u_L = alloc_coeffs[self.PlantDev.ileaf]
            u_R = alloc_coeffs[self.PlantDev.iroot]
            dGPPRmdWleaf, dGPPRmdWroot, dSdWleaf, dSdWroot = 0, 0, 0, 0
        else:
            u_L, u_R, dGPPRmdWleaf, dGPPRmdWroot, dSdWleaf, dSdWroot = self.PlantAlloc.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,solRadswskyb,solRadswskyd,theta,self.SAI,self.CI,hc,d_rpot)

        # If there is no net benefit for allocating to leaves or roots, allocate instead to stem reserves
        if (u_L <= 0) and (u_R <= 0):
            u_LR_unused = 1 - u_Stem - u_Seed   # this is a useful diagnostic
            u_Stem += u_LR_unused
        else:
            u_LR_unused = 0

        # ODE for plant carbon pools
        # N.B. the NPP allocation fluxes are constrained to be greater than 0 to ensure there is no allocation when the plant has a negative carbon balance (e.g. if Rm > GPP)
        dCseedbeddt = F_C_sowing - F_C_seed2leaf - F_C_seed2stem - F_C_seed2root
        dCleafdt = F_C_seed2leaf + max(u_L*NPP,0) - tr_[self.PlantDev.ileaf]*Cleaf - BioHarvestLeaf
        dCstemdt = F_C_seed2stem + max(u_Stem*NPP,0) - tr_[self.PlantDev.istem]*Cstem - BioHarvestStem - F_C_stem2grain
        dCrootdt = F_C_seed2root + max(u_R*NPP,0) - tr_[self.PlantDev.iroot]*Croot
        dCseeddt = max(u_Seed*NPP,0) - tr_[self.PlantDev.iseed]*Cseed - BioHarvestSeed + F_C_stem2grain
        if self.Management.cropType == "Wheat":
            dCStatedt = self.calculate_devphase_Cflux(dCstemdt, Bio_time)
        elif self.Management.cropType == "Canola":
            dCStatedt = self.calculate_devphase_Cflux(NPP, Bio_time)

        # Prepare diagnostics if requested (N.B. diagnostics must always be the last item in the returned output)
        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                'LAI': LAI,
                'E': E, 
                'GPP': GPP,
                'NPP': NPP,
                'Rm': Rm,
                'Rg': Rg,
                'Rml': Rml,
                'Rmr': Rmr,
                'trflux_total': tr_[self.PlantDev.ileaf]*Cleaf + tr_[self.PlantDev.istem]*Cstem + tr_[self.PlantDev.iroot]*Croot + tr_[self.PlantDev.iseed]*Cseed,
                'u_Leaf': u_L,
                'u_Stem': u_Stem,
                'u_Root': u_R,
                'u_Seed': u_Seed,
                'h_c': hc,
                'd_rpot': d_rpot,
                'd_r': d_r,
                'fV': fV,
                'fGerm': fGerm,
                'DTT': DTT,
                'S_d_pot': S_d_pot,
                'F_C_stem2grain': F_C_stem2grain,
                'idevphase': idevphase,
                'fPsil': fPsil,
                'Psil': Psil,
                'Psir': Psir,
                'Psis': Psis,
                'K_s': K_s,
                'K_sr': K_sr,
                'k_srl': k_srl,
                'dGPPRmdWleaf': dGPPRmdWleaf,
                'dGPPRmdWroot': dGPPRmdWroot,
                'dSdWleaf': dSdWleaf,
                'dSdWroot': dSdWroot,
                'u_LR_unused': u_LR_unused,
            }

        # Return down-regulated physiological parameters to original values
        # TODO: This is not good coding practice, need to find a better way to handle this
        self.PlantCH2O.k_rl = self.p1  # after downreg_phase we reset the parameter back to its original value
        self.PlantCH2O.CanopyGasExchange.Leaf.Vcmax_opt = self.p2  # after downreg_phase we reset the parameter back to its original value
        self.PlantCH2O.CanopyGasExchange.Leaf.g1 = self.p3  # after downreg_phase we reset the parameter back to its original value

        return (dCleafdt, dCstemdt, dCrootdt, dCseeddt, dGDDdt, dVDdt, dHTTdt, dCStatedt, dCseedbeddt, diagnostics)   # N.B. diagnostics must always be the last item in the returned output

    def calculate_NPP_RmRgpropto(self,GPP,R_m):
        R_g = self.alpha_Rg * np.maximum((GPP - R_m),0)  # There can't be negative growth respiration
        R_a = R_g+R_m
        NPP = GPP - R_a   # There can be negative NPP, if R_a is greater than GPP
        return NPP, R_g

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
        Psi_s_z = self.PlantCH2O.soil_water_potential(soilTheta)
        Psi_s = Psi_s_z[self.PlantCH2O.SoilLayers.ntop]   ## assume only the uppermost soil layer controls germination
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

    def calculate_sowingtime_conditional(self,_doy,_year):
        sowingDays = self.Management.sowingDays
        sowingYears = self.Management.sowingYears
        if sowingDays is None or sowingYears is None:
            return 0
        
        # Convert to lists if they are single integers
        if isinstance(sowingDays, int):
            sowingDays = [sowingDays]
        if isinstance(sowingYears, int):
            sowingYears = [sowingYears]
        
        # Check each sowing event
        for day, year in zip(sowingDays, sowingYears):
            if day <= _doy < day + 1 and year == _year:
                return 1
        
        # If no match is found
        return 0

    def calculate_sowingrate_conditional(self,_doy,_year):
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
        sowingTime = self.calculate_sowingtime_conditional(_doy,_year)
        sowingrate = sowingTime * (self.Management.sowingRate * self.PlantCH2O.f_C * 1000 / 10000)    # includes conversion of sowing rate units from kg ha-1 to g C m-2
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

    def calculate_BioHarvest(self,Biomass,_doy,_year,propHarvest,HarvestTurnoverTime):
        _vfunc = np.vectorize(self.calculate_harvesttime_conditional,otypes=[float])
        HarvestTime = _vfunc(_doy,_year)
        BioHarvest = HarvestTime*np.maximum(propHarvest*Biomass/HarvestTurnoverTime,0)
        return BioHarvest

    def calculate_harvesttime_conditional(self,_doy,_year):
        harvestDays = self.Management.harvestDays
        harvestYears = self.Management.harvestYears
        if harvestDays is None or harvestYears is None:
            return 0
        
        # Convert to lists if they are single integers
        if isinstance(harvestDays, int):
            harvestDays = [harvestDays]
        if isinstance(harvestYears, int):
            harvestYears = [harvestYears]
        
        # Check each sowing event
        for day, year in zip(harvestDays, harvestYears):
            if day <= _doy < day + 1 and year == _year:
                return 1
        
        # If no match is found
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
        remob = any(self.PlantDev.is_in_phase(current_gdd, phase) for phase in self.remob_phase)
        if remob:
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

    def calculate_canola_seed_density(self, NPP_anthesis):
        """
        NPP_anthesis : float 
            Total assimilate supply during anthesis period (g C m-2 ground area)
        B : float
            Rate constant (g C m-2 ground area)
        S_dmax : float
            Maximum potential seed density (thousand seeds m-2 ground area)
        S_dmin : float
            Minimum potential seed density (thousand seeds m-2 ground area)
        """
        S_dpot = self.GY_S_dmax * (1 - np.exp(-NPP_anthesis/self.GY_B))
        return np.maximum(S_dpot,self.GY_S_dmin)

    def seed_mass_compensation_conditional(self, S_dpot):
        """
        Calculates the compensated seed weight (W_seed) based on potential seed density (S_dpot). This 
        describes a linear increase in seed weight to compensate for low seed density, constrained to be 
        within typical limits. 
        
        Parameters
        ----------
        S_dpot : float
            Potential seed density (thousand seeds m-2 ground area), determined during anthesis.
        S_dmin : float
            Minimum potential seed density (thousand seeds m-2 ground area).
        S_dmax : float
            Maximum potential seed density (thousand seeds m-2 ground area).
        W_TKWseed_base : float
            Base thousand kernel weight of seed when S_dpot = S_dmax or when k_comp=0 (g thousand seeds-1)
        k_comp : float
            Compensation factor that controls the rate of increase in W_seed as S_dpot decreases from S_dmax (g m2 thousand seeds-2).
        
        Returns
        -------
        float: Compensated seed weight, W_seed (g thousand seeds-1)

        Notes
        -----
        For canola, the typical range for k_comp is likely between 0-0.02. 

        References
        ----------
        Kirkegaard et al. (2018) The critical period for yield and quality determination in canola (Brassica napus L.), doi: 10.1016/j.fcr.2018.03.018
        Labra et al. (2017) Plasticity of seed weight compensates reductions in seed number of oilseed rape in response to shading at flowering, doi: 10.1016/j.eja.2016.12.011
        Zhang and Flottmann (2018) Source-sink manipulations indicate seed yield in canola is limited by source availability, doi: 10.1016/j.eja.2018.03.005
        """
        # When S_dpot >= S_dmax, W_seed = W_seed_base
        if S_dpot >= self.GY_S_dmax:
            return self.GY_W_TKWseed_base
        # When S_dmin <= S_dpot < S_dmax, linearly increase W_seed with slope k_comp
        elif self.GY_S_dmin <= S_dpot < self.GY_S_dmax:
            W_seed = self.GY_W_TKWseed_base + self.GY_k_comp * (self.GY_S_dmax - S_dpot)
            return W_seed
        # When S_dpot < S_dmin, keep W_seed constant at the value when S_dpot = S_dmin
        else:
            W_seed_min = self.GY_W_TKWseed_base + self.GY_k_comp * (self.GY_S_dmax - self.GY_S_dmin)
            return W_seed_min

    def seed_mass_compensation(self, S_dpot):
        _vfunc = np.vectorize(self.seed_mass_compensation_conditional)
        W_seed = _vfunc(S_dpot)
        return W_seed

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
            Maximum potential grain number (thousand grains m-2)
        SDW_50 : float 
            SDW_a at which grain number is half of GN_max (g d.wt m-2)
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
        grainfill = any(self.PlantDev.is_in_phase(current_gdd, phase) for phase in self.grainfill_phase)
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

    def calculate_wind_speed_hc(self, airU, hc, PAI):
        """
        Calculates wind speed at the top-of-canopy based on wind speed at a given measurement height.

        Parameters
        ----------
        airU : float
            Wind speed at measurement height (m s-1)
        hc : float
            Canopy height (m)
        PAI : float
            Plant area index (m2 m-2) i.e. leaf-area index plus stem-area index

        Returns
        -------
        Wind speed at top-of-canopy (m s-1)
        """

        # Calculate zero-plane displacement height
        d = hc * self.PlantCH2O.BoundaryLayer.calculate_R_d_h(PAI)
        # Calculate the ratio of friction velocity to mean velocity at height hc
        R_ustar_Uh = self.PlantCH2O.BoundaryLayer.calculate_R_ustar_Uh(PAI)
        # Calculate the roughness length (z0)
        z0 = hc * self.PlantCH2O.BoundaryLayer.calculate_R_z0_h(PAI,R_ustar_Uh)
        # Calculate wind speed at top-of-canopy based on derived wind speed profile above canopy
        airUhc = self.PlantCH2O.BoundaryLayer.estimate_wind_profile_log(airU, self.Site.met_z_meas, hc, d, z0)

        return airUhc

    def scale_parameter(self, original_value, fractional_reduction, scaling_factor):
        """
        Scale a parameter based on a fractional reduction and a scaling factor.

        Parameters
        ----------
        original_value : float
            The original value of the parameter (e.g., p0).
        fractional_reduction : float
            The fraction of the parameter's value to reduce by when scaling_factor is 1 (e.g., 0.1 for 10% reduction).
        scaling_factor : float
            A value between 0 and 1 that determines the degree of scaling (e.g., 0 means no scaling, 1 means full scaling).

        Returns:
        float : The scaled parameter value.
        """
        return original_value * (1 - fractional_reduction * scaling_factor)
