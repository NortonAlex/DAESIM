"""
Plant carbon and water model class: Includes equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from functools import partial
from scipy.optimize import OptimizeResult, bisect
from scipy.integrate import solve_ivp
from daesim.biophysics_funcs import MinQuadraticSmooth, fT_Q10
from daesim.climate import ClimateModule
from daesim.canopygasexchange import CanopyGasExchange
from daesim.boundarylayer import BoundaryLayerModule
from daesim.soillayers import SoilLayers

@define
class PlantModel:
    """
    Calculator of plant ecophysiology, particularly the carbon and water fluxes
    """

    ## Module dependencies
    Site: Callable = field(default=ClimateModule())    ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.
    CanopyGasExchange: Callable = field(default=CanopyGasExchange())     ## It is optional to define CanopyGasExchange for this method. If no argument is passed in here, then default setting for CanopyGasExchange is the default CanopyGasExchange().
    BoundaryLayer: Callable = field(default=BoundaryLayerModule())    ## It is optional to define BoundaryLayer for this method. If no argument is passed in here, then default setting for BoundaryLayer is the default BoundaryLayer().
    SoilLayers: Callable = field(default=SoilLayers())     ## It is optional to define SoilLayers for this method. If no argument is passed in here, then default setting for SoilLayers is the default SoilLayers().

    ## Class parameters
    f_C: float = field(default=0.45)  ## Fraction of carbon in dry structural biomass (g C g d.wt-1) TODO: Ensure this is only defined in one module, it is currently defined in plant_1000 and this module.
    m_r_r_opt: float = field(default=0.01)  ## Maintenance respiration coefficient for roots at optimum temperature i.e. 25 deg C (d-1)
    m_r_r_Q10: float = field(default=1.4)  ## Root maintenance respiration temperature response Q10 coefficient N.B. because we use air temperature as input, we dampen the Q10 coefficient slightly to account for this, considering actual soil (or root) temperature won't vary as significantly as air temperature
    SLA: float = field(default=0.020)  ## Specific leaf area (m2 g d.wt-1), fresh leaf area per dry leaf mass
    maxLAI: float = field(default=3)  ## Maximum potential leaf area index (m2 m-2)
    CI: float = field(default=0.5)    ## Foliage clumping index (-)  TODO: Double check default values of clumping index that is suitable for crops
    d_leaf: float = field(default=0.015)     ## Leaf dimension parameter (m), defined as the mean length of the leaf in the downwind direction, used to determine leaf boundary layer resistance
    
    soilThetaMax: float = field(default=0.5) ## Volumetric soil water content at saturation (m3 water m-3 soil)
    b_soil: float = field(default=5.0)       ## Empirical soil-specific parameter relating volumetric water content to hydraulic conductivity (-)
    Psi_e: float = field(default=-0.05)      ## Air-entry value of (hydrostatic) soil water potential; this is the soil water potential at the transition of saturated to unsaturated soil (MPa)
    K_sat: float = field(default=12)         ## Saturated value of soil hydraulic conductivity, K_s (mol m-1 s-1 MPa-1)
    
    ksr_coeff: float = field(default=500)    ## scale factor for soil-to-root conductivity/conductance (TODO: Check units and definition); conversion factor for soil hydraulic conductivity and root biomass density to a soil-to-root conductivity/conductance. In some models this is represented by a single root occupying a cylinder of soil. In principle, it considers the distance water travels from the bulk soil to the root surface, root geometry (e.g. radius) and conducting propoerties of the root. Typical values range from approx 100-18000
    d_soil: float = field(default=1.0)       ## depth of soil layer (m)
    f_r: float = field(default=1.0)          ## fraction of roots in soil layer (unitless)
    root_distr_d50: float = field(default=0.15)  ## Soil depth at which 50% of total root amount is accumulated (m) i.e. 50% of roots occur above this depth
    root_distr_c: float = field(default=-1.2)     ## A dimensionless shape-parameter to describe root distribution in soil profile (-)
    SRD: float = field(default=0.01)         ## Specific root depth, represents the ratio of root depth to total root dry biomass (m g.dwt-1)
    
    k_rl: float = field(default=0.10)     ## Leaf-area specific root-to-leaf (from inside the root to bulk leaf) hydraulic conductance (mol m-2 s-1 MPa-1) TODO: should increase with distance i.e. leaf height above root node
    Psi_f: float = field(default=-2.3)   ## Leaf water potential at which half of stomatal conductance occurs (MPa), see Drewry et al. (2010, doi:10.1029/2010JG001340)
    sf: float = field(default=3.5)     ## Stomatal sensitivity parameter between stomatal conductance and leaf water potential (MPa-1), see Drewry et al. (2010, doi:10.1029/2010JG001340)

    def calculate(
        self,
        W_L,         ## leaf structural dry biomass (g d.wt m-2)
        W_R,         ## root structural dry biomass (g d.wt m-2)
        soilTheta,   ## volumetric soil water content per layer (m3 m-3), dimensions (soil layer,)
        leafTempC,   ## leaf temperature (deg C)
        airTempC,    ## air temperature (deg C), outside leaf boundary layer 
        airRH,      ## relative humidity of air (%), outside leaf boundary layer
        airCO2,  ## leaf surface CO2 partial pressure, bar, (corrected for boundary layer effects)
        airO2,   ## leaf surface O2 partial pressure, bar, (corrected for boundary layer effects)
        airP,    ## air pressure, Pa, (in leaf boundary layer)
        airUhc,    ## wind speed at top-of-canopy, m s-1
        swskyb, ## Atmospheric direct beam solar radiation, W/m2
        swskyd, ## Atmospheric diffuse solar radiation, W/m2
        sza,    ## Solar zenith angle, degrees
        SAI,    ## stem area index, m2 m-2  TODO: Make sure this is passed properly to each method
        hc,     ## canopy height, m
        d_rpot,     ## potential root depth, m
    ) -> Tuple[float]:

        ## Make sure to run set_index which assigns the canopy layer indexes for the given canopy structure
        self.SoilLayers.set_index()
        z_soil, d_soil = self.SoilLayers.discretise_layers()
        if soilTheta.size != self.SoilLayers.nlevmlsoil:
            raise ValueError("Size of the soil moisture input must be the same size as the defined number of soil layers.")

        if W_L < 0 or W_R < 0:
            raise ValueError(f"States W_L or W_R cannot be negative")
        elif np.isnan(W_L) or np.isnan(W_R):
            raise ValueError(f"States W_L or W_R cannot be NaN")
            import pdb; pdb.set_trace()
        
        LAI = self.calculate_LAI(W_L)

        ## Calculate wind speed profile within canopy, given canopy properties
        dlai = self.CanopyGasExchange.Canopy.cast_parameter_over_layers_betacdf(LAI, self.CanopyGasExchange.Canopy.beta_lai_a, self.CanopyGasExchange.Canopy.beta_lai_b)   # Canopy layer leaf area index (m2/m2)
        dsai = self.CanopyGasExchange.Canopy.cast_parameter_over_layers_betacdf(SAI, self.CanopyGasExchange.Canopy.beta_sai_a, self.CanopyGasExchange.Canopy.beta_sai_b)   # Canopy layer stem area index (m2/m2)
        dpai = dlai+dsai    # Canopy layer plant area index (m2/m2)
        ntop, nbot = self.CanopyGasExchange.Canopy.index_canopy()
        airUz = self.BoundaryLayer.calculate_wind_profile_exp(airUhc,dpai,ntop)   # Wind speed at mid-point of each canopy layer

        ## Calculate soil water potential
        Psi_s_z = self.soil_water_potential(soilTheta)
        
        ## Calculate soil properties
        K_s_z = self.soil_hydraulic_conductivity(Psi_s_z)

        ## Calculate actual root depth
        d_r = self.calculate_root_depth(W_R, d_rpot)

        ## Cumulative root fraction for all layers (assume d_soil is a vector for soil layer thickness)
        fc_r_z = self.calculate_root_distribution(d_r, d_soil)  # Calculate the cumulative root distribution for all layers

        ## Calculate actual root fraction per layer (by difference, no loops needed)
        f_r_z = np.diff(np.insert(fc_r_z, 0, 0, axis=0), axis=0)  # Fractional root density per layer

        ## Calculate soil-to-root conductivity/conductance (TODO: Check definition and units of conductivity vs conductance)
        K_sr_z = self.soil_root_hydraulic_conductivity(W_R, K_s_z, f_r_z, d_soil)

        ## Determine weighting in root zone soil layers, used to determine a single, bulk soil-root zone value for both Psi_s and K_sr
        ## Notes: Because we use the soil-to-root hydraulic conductivity to calculate the layer weights, any layers without root biomass will have a weight of zero
        z_weights = K_sr_z/K_sr_z.sum()
        Psi_s = np.sum(Psi_s_z*z_weights)  # Average soil water potential over the soil profile TODO: Fix this and its use below in other functions
        K_sr = np.sum(K_sr_z*z_weights)   # Average soil-to-root hydraulic conductivity over the root profile
        # Average soil hydraulic conductivity over the root profile (no weighting for this). Note: Because this is a plant module, we determine the soil hydraulic conductivity over the root zone only
        K_s = self.calculate_root_profile_mean(K_s_z, d_r, d_soil)

        ## Convert soil-to-root conductance to leaf-area specific soil-to-root conductance (TODO: Check definition and units of conductivity vs conductance)
        k_srl = self.soil_root_hydraulic_conductance_l(K_sr,LAI)

        ## Initial estimate of GPP without leaf water potential limitation
        GPP, E, Rd = self.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, airUz, 1.0, LAI, SAI, hc, sza, swskyb, swskyd)

        ## Determine the total leaf-area specific conductance from soil-to-root-to-leaf
        ## - assumes a one-dimensional pathway (in series) and Ohm's law for the hydraulic conductances i.e. the relationship 1/k_tot = 1/k_srl + 1/k_rl
        k_tot = (k_srl*self.k_rl)/(self.k_rl+k_srl)

        ## Calculate leaf water potential
        Psi_l = self.leaf_water_potential_solve(Psi_s, k_tot, airTempC, leafTempC, airCO2, airO2, airRH, airP, airUz, LAI, SAI, hc, sza, swskyb, swskyd)

        ## Calculate actual leaf water potential scaling factor on photosynthesis/dry-matter production
        f_Psi_l = self.tuzet_fsv(Psi_l)

        ## Calculate actual gpp and stomatal conductance
        GPP, E, Rd = self.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, airUz, f_Psi_l, LAI, SAI, hc, sza, swskyb, swskyd)

        ## Calculate root water potential
        Psi_r = self.root_water_potential(Psi_s,E,k_srl) # TODO: Calculate per layer but write out as a bulk average (analagous to bulk average Psi_l)

        ## Calculate canopy total transpiration
        E_c = LAI * E

        ## Calculate maintenance respiration of leaf and root pools
        Rm_l = Rd   # Total leaf maintenance respiration is assumed to equal total leaf mitochondrial respiration
        Rm_r = self.calculate_Rm_k(W_R*self.f_C,airTempC,self.m_r_r_opt)

        return (GPP, Rm_l, Rm_r, E_c, f_Psi_l, Psi_l, Psi_r, Psi_s, K_s, K_sr, k_srl)


    def leaf_water_potential_solve(self, Psi_s, k_tot, airTempC, leafTempC, airCO2, airO2, airRH, airP, airU, LAI, SAI, hc, sza, swskyb, swskyd):
        """
        Calculate leaf water potential that balances water supply (root uptake) and water loss (transpiration) using the bisection method.
    
        Parameters
        ----------
        LAI : float
            Leaf area index (m2 m-2).
        SAI : float
            Stem area index (m2 m-2).
        hc : float
            Canopy height (m).
        Psi_s : float
            Soil water potential (MPa).
        k_tot : float
            Total leaf-area specific soil-to-leaf hydraulic conductance (mol m-2 s-1 MPa-1).
        Q : float
            Absorbed photosynthetically active radiation (APAR) (mol m-2 s-1).
        leafTempC : float
            Leaf temperature (°C).
        airTempC : float
            Air temperature (°C) outside leaf boundary layer.
        airCO2 : float
            CO2 concentration in the air (bar).
        airO2 : float
            O2 concentration in the air (bar).
        airRH : float
            Relative humidity of air (%) outside leaf boundary layer.
        airU : float
            Wind speed within canopy air space (m s-1).
        leaf : LeafGasExchangeModule
            Optional module defining leaf-specific properties and functions.
    
        Returns
        -------
        Psi_l : float
            Calculated leaf water potential (MPa).
    
        Notes
        -----
        This function uses the bisection method to find the leaf water potential that balances
        water supply from roots and water loss from stomata. It internally defines functions for
        calculating the components of the leaf water potential balance equation based on given
        environmental and leaf-specific parameters.

        It is assumed that the rate of water loss from the canopy is balanced by the rate of water 
        uptake from the roots (Meinzer, 2002, doi:10.1046/j.1365-3040.2002.00781.x). Also see 
        Bonan et al. (2014, doi:10.5194/gmd-7-2193-2014).
        """
        
        ## First function
        def f1(Psi_l): #,Psi_s,k_tot):
            E = k_tot*(Psi_s - Psi_l)
            return E

        ## Second function
        def f2(Psi_l):
            f_Psi_l = self.tuzet_fsv(Psi_l)
            GPP, E, Rd = self.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, airU, f_Psi_l, LAI, SAI, hc, sza, swskyb, swskyd)
            return E

        # Define the function for which we want to find the root
        def f(Psi_l, f1, f2):
            left_side = f1(Psi_l)
            right_side = f2(Psi_l)
            return left_side - right_side
        fpartial = partial(f, f1=f1, f2=f2)

        # Find the root
        # Initial interval [a, b] for the bisection method
        a = -100.0
        b = 0.0
        Psi_l = bisect(fpartial,a,b,xtol=1e-4)
        
        return Psi_l

    def soil_water_potential_conditional(self,soilTheta):
        """
        Parameters
        ----------
        soilTheta: volumteric soil water content (m3 water m-3 soil)
        
        soilThetaMax: volumetric soil water content at saturation (m3 water m-3 soil)
        Psi_e: saturating value of soil water potential (MPa)
        b_soil: empirical soil-water retention curve parameter (-)
        
        Returns
        -------
        Psi_s: soil water potential (MPa)
    
        References
        ----------
        Campbell (1974) A simple method for determining unsaturated conductivity from moisture retention data, Soil Science 117(6), p 311-314
        Duursma et al. (2008) doi:10.1093/treephys/28.2.265
        """
        if soilTheta < self.soilThetaMax:
            Psi_s = self.Psi_e*(soilTheta/self.soilThetaMax)**(-self.b_soil)
        else:
            Psi_s = self.Psi_e
        return Psi_s
    
    def soil_water_potential(self,soilTheta):
        """
        Parameters
        ----------
        soilTheta: volumteric soil water content (m3 water m-3 soil)
        
        soilThetaMax: volumteric soil water content at saturation (m3 water m-3 soil)
        Psi_e: saturating value of soil water potential (MPa)
        b_soil: empirical soil-water retention curve parameter (-)
        
        Returns
        -------
        Psi_s: soil water potential (MPa)
    
        References
        ----------
        Campbell (1974) A simple method for determining unsaturated conductivity from moisture retention data, Soil Science 117(6), p 311-314
        Duursma et al. (2008) doi:10.1093/treephys/28.2.265
        """
        _vfunc = np.vectorize(self.soil_water_potential_conditional,otypes=[float])
        Psi_s = _vfunc(soilTheta)
        return Psi_s

    def soil_hydraulic_conductivity(self,Psi_s):
        """
        Parameters
        ----------
        Psi_s: soil water potential (MPa)
        
        Psi_e: saturating value of soil water potential (MPa)
        K_sat: saturating value of soil hydraulic conductance, K_s (mol m-1 s-1 MPa-1; mol H2O per m per second per MPa pressure difference)
        b_soil: empirical soil-water retention curve parameter (-)
        
        Returns
        -------
        K_s: soil hydraulic conductivity (mol m-1 s-1 MPa-1; mol H2O per m per second per MPa pressure difference)
    
        Notes
        -----
        An alternative, useful approach is to use the soil hydraulic conductivity at saturation ($K_{sat}$, mol m-1 s-1 MPa-1) rather than the leaf-area specific soil hydraulic conductance, $k_sat$, which is useful because there are established
        datasets that have measured these soil properties (Cosby et al., 1984). If using this approach, one must then scale it by the root, xylem, sapwood or leaf area (e.g. see Duursma et al., 2008). 
        
        References
        ----------
        Dewar et al. (2022) doi:10.1111/nph.17795
        Duursma et al. (2008) doi:10.1093/treephys/28.2.265
        Cosby, B.J., G.M. Hornberger, R.B. Clapp and T.R. Ginn. 1984. A statistical exploration of the relationships of soil-moisture characteristics to the physical properties of soils. Water Resour. Res. 20:682–690
        
        """
        K_s = self.K_sat*(self.Psi_e/Psi_s)**(2+3/self.b_soil)
        return K_s

    def soil_root_hydraulic_conductivity_conditional(self,W_R,K_s,f_r,d_soil):
        """
        Parameters
        ----------
        W_r: total root structural biomass (g d.wt m-2 ground area)
        K_s: soil hydraulic conductivity (mol m-1 s-1 MPa-1)
        d_soil: depth of soil layer (m)
        f_r: fraction of roots in soil layer (-)
        ksr_coeff: constant of proportionality (g d.wt-1 m-1) TODO: Check units here
    
        Returns
        -------
        K_sr: soil-to-root radial hydraulic conductivity (mol m-2 s-1 MPa-1)
    
        Notes
        -----
        A note on units:
        The soil hydraulic conductivity, K_s, represents the molar flux density per hydrostatic pressure drop
        K_s = 1e-5    ## mol m-1 s-1 MPa-1  =  (mol m-2 s-1)/(MPa m-1)
        This can be converted to a volume flux density per hydrostatic pressure drop by multiplying by the molar mass of water and dividing by the density 
        of water e.g.
        K_s_vol = K_s * M_w / rho_w   ## m3 m-1 s-1 MPa-1  =  (m2 s-1 MPa-1) = (m s-1)/(MPa m-1)

        A note on ksr_coeff:
        This parameter is represented in various ways in the literature. In most cases, it is represented by a single root cylinder model which represents the plant roots as 
        a fixed radius cylinder that draws water from a larger cylinder of homogenous soil. This is useful but assumes roots involved in water uptake are all homogenous in properties, 
        shape, orientation in the soil and distribution in the soil. It also ignores various factors such as root hairs, tapering, clustering, shrinkage during drying, etc. Rather 
        than define fixed parameters for things like root radius, root length index, root area index, and distance from root surface to bulk soil water, we opt to reduce the number of 
        free parameters to a single empirical parameter that represents these combined properties. For single root cylinder models see Katul et al. 
        (2003, doi: 10.1046/j.1365-3040.2003.00965.x), Duursma et al. (2008, doi: 10.1093/treephys/28.2.265), Duursma and Medlyn (2012, doi: 10.5194/gmd-5-919-2012) and Nobel 
        (2009, Physicochemical and Environmental Plant Physiology, Ch. 9.3D) for a more fundamental desciption. Furthermore, previous modelling has shown that defining the plant root 
        water uptake function based on root dry mass, root length or surface area density all produce very similar results in water uptake (Himmelbauer et al., 2008, Journal of 
        Hydrology and Hydromechanics 56(1)). 
        """
        L_v = (W_R * f_r) / d_soil    ## calculate fine root density in the soil layer (g d.wt root m-3 soil)
        K_sr = K_s * L_v/self.ksr_coeff
        return K_sr
    
    def soil_root_hydraulic_conductivity(self,W_R,K_s,f_r,d_soil):
        """
        Parameters
        ----------
        W_r: total root structural biomass (g d.wt m-2 ground area)
        K_s: soil hydraulic conductivity (mol m-1 s-1 MPa-1)
        r_r: root radius (m)
        SRL: specific root length (m g-1 d.wt)
        d_soil: depth of soil layer (m)
        f_r: fraction of roots in soil layer (-)
        ksr_coeff: constant of proportionality
    
        Returns
        -------
        K_sr: soil-to-root radial hydraulic conductivity (mol m-2 s-1 MPa-1)
        
        """
        _vfunc = np.vectorize(self.soil_root_hydraulic_conductivity_conditional,otypes=[float])
        K_sr = _vfunc(W_R,K_s,f_r,d_soil)
        return K_sr
    
    def soil_root_hydraulic_conductance_l(self,K_sr,LAI):
        """
        Leaf-area specific soil-to-root radial hydraulic conductance
    
        Parameters
        ----------
        K_sr: soil-to-root radial hydraulic conductivity (mol m-1 s-1 MPa-1)
        LAI: leaf area index (m2 m-2)
    
        Returns
        -------
        k_srl: leaf-area specific soil-to-root radial hydraulic conductance (mol m-2 s-1 MPa-1)
        
        """
        k_srl = K_sr * (1/LAI)
        return k_srl

    def root_water_potential(self,Psi_s,E,k_srl):
        """
        Parameters
        ----------
        Psi_s: soil water potential (MPa)
        E: transpiration rate (mol H2O m-2 s-1)
        k_srl: leaf-area specific soil-to-root hydraulic conductance (mol m-2 s-1 MPa-1)
    
        Returns
        -------
        Psi_r: root water potential (MPa)
        """
        Psi_r = Psi_s - E/k_srl
        return Psi_r
    
    def leaf_water_potential(self,Psi_r,E):
        """
        Parameters
        ----------
        Psi_r: root water potential (MPa)
        E: transpiration rate (mol H2O m-2 s-1)
        k_rl: leaf-area specific root-to-leaf (plant) hydraulic conductance (mol m-2 s-1 MPa-1)
    
        Returns
        -------
        Psi_l: leaf water potential (MPa)
        """
        Psi_l = Psi_r - E/self.k_rl
        return Psi_l
        
    def leaf_transpiration(self,gsw,leafTempC,airTempC,airP,airRH,airU):
        """
        Calculates the transpiration rate

        Parameters
        ----------
        gsw: stomatal conductance to water vapor (mol m-2 s-1)
        leafTempC: leaf temperature (degrees Celsius)
        airTempC: air temperature (degrees Celsius)
        airP: air pressure (Pa)
        airRH: relative humidity of canopy air space (%)
        airU: wind speed in canopy air space (m s-1)
    
        Returns
        -------
        E: transpiration rate (mol H2O m-2 s-1)

        Notes
        -----
        The returned transpiration rate is given on a per area basis, this area is the same as the units of gsw, usually per leaf area.

        Calculated variables:
        Wi = concentration of water vapour in leaf air spaces (mol m-3), assumed to be saturated air water vapour concentration at leaf temperature
        Wa = concentration of water vapour outside the leaf boundary layer (mol m-3)
        r_wa = resistance to water vapor flux across leaf boundary layer (s m-1)
        r_ws = resistance to water vapor flux across leaf surface (s m-1)
        r_w = total resistance to water vapor flux across the leaf boundary layer and leaf surface, in series (s m-1)

        For details on the conversion of conductance to resistance see Nobel (2009) Section 8.1F and Table 8-1

        Also see Section 7.2B in Nobel 2009 for air boundary layer description

        References
        ----------
        Nobel (2009) Section 8.1F and Table 8-1, doi:10.1016/B978-0-12-374143-1.X0001-4, ISBN:978-0-12-374143-1
        """
        
        ## Calculate water vapor concentration gradient between the inside of the leaf (assumed to be saturated) and the air
        Wi = self.Site.compute_sat_vapor_pressure(leafTempC)   ## Pa
        Wa = self.Site.compute_actual_vapor_pressure(airTempC,airRH)   ## Pa
        Wi_molconc = Wi/(self.Site.R_w_mol * (leafTempC+273.15))   ## converts Pa to mol m-3
        Wa_molconc = Wa/(self.Site.R_w_mol * (airTempC+273.15))   ## converts Pa to mol m-3
        
        ## Resistance for H2O across leaf surface
        r_ws = (airP/(self.Site.R_w_mol*(leafTempC+273.15)))/gsw    ## converts stomatal conductance (mol H2O m-2 s-1) to stomatal resistance (s m-1)  (see Nobel (2009) Section 8.1F) TODO: Check this and cross-check it with calculations of stomatal resistance in leafgasexchange modules and perhaps also SCOPE model code
        ## Resistance for H2O across leaf boundary layer
        r_wa = self.BoundaryLayer.calculate_leaf_boundary_resistance(airTempC,airP,airU,self.d_leaf)
        ## Total resistance to water vapour across stomata and boundary layer, acting in series
        r_w = r_wa + r_ws
        E = (Wi_molconc - Wa_molconc)/r_w

        return E

    def factor_leaf_water_potential_conditional(self,Psi_l):
        """
        Parameters
        ----------
        Psi_l: leaf water potential (J kg-1)
        
        Psi_crit: critical leaf water potential below which there is no dry matter production (J kg-1)
        
        Returns
        -------
        f_Psi_l: leaf water potential scaling factor on dry matter production (-)
        """
        f_Psi_l = max(0,1 - (Psi_l/self.Psi_crit))
        return f_Psi_l
    
    def factor_leaf_water_potential(self,Psi_l):
        """
        Parameters
        ----------
        Psi_l: leaf water potential (J kg-1)
        
        Psi_crit: critical leaf water potential below which there is no dry matter production (J kg-1)
    
        Returns
        -------
        f_Psi_l: leaf water potential scaling factor on dry matter production (-)
        """
        _vfunc = np.vectorize(self.factor_leaf_water_potential_conditional,otypes=[float])
        f_Psi_l = _vfunc(Psi_l)
        return f_Psi_l
        
    def calculate_LAI(self,W_L):
        """
        Parameters
        ----------
        W_L: leaf dry structural biomass (g d.wt m-2)
        
        SLA: specific leaf area (m2 g d.wt-1)
    
        Returns
        -------
        LAI: leaf area index (m2 m-2)
        """
        LAI = MinQuadraticSmooth(W_L*self.SLA,self.maxLAI)
        return LAI

    def calculate_canopygasexchange(self, airTempC, leafTempC, airCO2, airO2, airRH, airP, airU, f_Psi_l, LAI, SAI, hc, sza, swskyb, swskyd):
        """
        Parameters
        ----------
        airTempC : Air temperature (degree Celcius)
        leafTempC : Leaf temperature (degree Celcius)
        airCO2 : Leaf surface CO2 partial pressure (bar), corrected for boundary layer effects
        airO2 : Leaf surface O2 partial pressure (bar), corrected for boundary layer effects
        airRH : Relative humidity (%)
        airP : Air pressure (Pa)
        airU : Wind speed (m s-1)
        f_Psi_l : Leaf water potential limitation factor on stomatal conductance (-)
        LAI : Leaf area index (m2/m2)
        SAI : Stem area index (m2/m2)
        CI : Foliage clumping index (-)
        hc : Canopy height (m)
        sza : Solar zenith angle (degrees)
        swskyb : Atmospheric direct beam solar radiation (W/m2)
        swskyd : Atmospheric diffuse solar radiation (W/m2)
        Leaf : Leaf gas exchange module class
        Canopy : Canopy module class
        CanopyRad : Canopy radiation exchange module class
        CanopyGasExchange : Canopy gas exchange module class
        Site : Climate module class

        Returns
        -------
        GPP : Canopy total gross primary productivity (umol m-2 s-1; on a ground area basis)
        E : Canopy total transpiration rate (mol m-2 s-1; on a ground area basis)
        Rd : Canopy total leaf mitochondrial respiration (umol m-2 s-1; on a ground area basis)

        """

        An_ml, gs_ml, Rd_ml = self.CanopyGasExchange.calculate(leafTempC,airCO2,airO2,airRH,f_Psi_l,LAI,SAI,self.CI,hc,sza,swskyb,swskyd)  # TODO: Modify this so SAI is an input not a module parameter

        GPP = np.sum(An_ml + Rd_ml)*1e6

        Rd = np.sum(Rd_ml)*1e6

        E_ml = self.leaf_transpiration(gs_ml,leafTempC,airTempC,airP,airRH,airU)
        E = np.sum(E_ml)

        return GPP, E, Rd
        
    def calculate_Rm_k(self,C_k,TempC,m_r_25):
        """
        Parameters
        ----------
        C_k: carbon pool mass k (g C m-2)
        TempC : temperature (degree Celcius)        
        m_r: specific maintenance respiration rate at standard temperature (d-1)
    
        Returns
        -------
        R_m: maintenance respiration (g C m-2 d-1)
        """
        m_r = fT_Q10(m_r_25,TempC,Q10=self.m_r_r_Q10)
        R_m = m_r * C_k
        return R_m

    def tuzet_fsv(self,Psi_l):
        """
        Parameters
        ----------
        Psi_l: Leaf water potential (MPa)
        
        Psi_f: Leaf water potential at which half of stomatal conductance occurs (MPa)
        sf: Stomatal sensitivity parameter between stomatal conductance and leaf water potential (MPa-1)
        
        Returns
        -------
        f_sv: Stomatal conductance scaling factor (-)
    
        Notes
        -----
        From Drewry et al. (2010): "This represents stomatal sensitivity to leaf water potential (Psi_l) and varies from 1 (no change 
        in stomatal conductance) to 0 (total loss of conductance)... as Psi_l decreases to the critical water potential (Tuzet et al., 
        2003; Sperry et al., 1998). This function depends on a species specific reference potential (Psi_f) and sensitivity parameter 
        (sf) (Tuzet et al., 2003)."
    
        References
        ----------
        Tuzet et al. (2003) doi:10.1046/j.1365-3040.2003.01035.x
        Drewry et al. (2010) doi:10.1029/2010JG001340
        """
        f_sv = (1 + np.exp(self.sf*self.Psi_f))/(1 + np.exp(self.sf*(self.Psi_f-Psi_l)))
        return f_sv

    def calculate_root_distribution_conditional(self, d_r, d_soil):
        """
        

        Parameters
        ----------
        d_r : Root depth (m)
        d_soil : Depth from soil surface to bottom of each soil layer (m)

        Returns
        -------
        Cumulative amount of roots from surface to each soil layer (-)

        Notes
        -----
        Multiple studies have demonstrated that the distribution of roots in the soil profile, measured as 
        root mass density (g root m-3 soil) or root length density (m root m-3 soil), declines exponentially 
        or near-linearly with depth from the soil surface. See Fan et al. (2016) and references therein. 
        Also see Haberle and Svoboda (2014) and refereinces therein (Gerwitz & Page 1974; Rowse 1974; 
        Dwyer et al. 1996; Himmelbauer & Novák 2008; Bingham & Wu 2011; Zhang et al. 2012; Zuo et al. 2013). 

        Schenk and Jackson (2002) described a logistic dose–response curve to represent the cumulative root 
        distribution. Fan et al. (2016) adapted and further evaluated this model for applications to common 
        temperate crops:

        $\frac{R}{R_{max}} = \frac{1}{1+(d/d_{50})^c}$

        where $R$ is the cumulative amount (i.e., biomass mass or root length) of roots to soil depth $d$ 
        (cm; i.e. the amount of roots above profile depth $d$), $R_{max}$ is the total amount of roots, 
        $d_{50}$ is depth at which 50\% of total root amount was accumulated, $c$ is a dimensionless shape-
        parameter.

        The above equation does not define a maximum rooting depth, as the function continues to be positive 
        definite to infinite soil depth. To modify this so that all roots are confined to a maximum rooting 
        depth, the authors do the following:

        $Y(d) = \frac{1}{1+(d/d_{50})^c} + \left(1 - \frac{1}{1+(d_{max}/d_{50})^c} \right) \times \frac{d}{d_{max}} $

        Where $Y(d)$ is the root distribution at depth $d$ (m), $d_a$ is another fitting parameter (m). This 
        function is constrained to values where $d \leq d_{max}$. 

        Fan et al. (2016) provide values for the parameters above for eleven different crop types, including wheat and canola. 

        References
        ----------
        Fan et al. (2016, doi:10.1016/j.fcr.2016.02.013)
        Haberle and Svoboda (2014, doi:10.1080/03650340.2014.903560)
        Schenk and Jackson (2002, doi:10.1890/0012-9615(2002)072[0311:TGBOR]2.0.CO;2)
        """

        max_rooting_depth = min(d_r, self.SoilLayers.z_max)  # rooting depth cannot exceed total soil depth

        if d_soil <= max_rooting_depth:
            Y = 1/(1+(d_soil/self.root_distr_d50)**self.root_distr_c) + (1 - 1/(1+(max_rooting_depth/self.root_distr_d50)**self.root_distr_c))*d_soil/max_rooting_depth
        else:
            Y = 1
        return Y

    def calculate_root_distribution(self, d_r, d_soil):
        """
        Parameters
        ----------
        d_r : Root depth (m)
        d_soil : Depth from soil surface to bottom of each soil layer (m)

        Returns
        -------
        Cumulative amount of roots from surface to each soil layer (-)
        """
        _vfunc = np.vectorize(self.calculate_root_distribution_conditional,otypes=[float])
        Y = _vfunc(d_r,d_soil)
        return Y

    def calculate_root_profile_mean(self, param_z, rooting_depth, d_soil):
        """
        Calculate the average of a given parameter over the rooting depth (without weighting by layer depth)

        TODO: Modify function so that it takes in actual rooting depth, rather than just assuming the roots 
        are at their maximum (defined by self.root_depth_max). 

        Parameters
        ----------
        param_z : array_like
            Parameter that you want to average over the root profile
        rooting_depth : scalar
            Depth of roots (m)
        d_soil : array_like
            Depth from soil surface to bottom of each soil layer (m)

        Returns
        -------
        Mean of the input parameter over the root profile

        Notes
        -----
        Each layer that has roots present is included with an equal weight (i.e., if the roots are present 
        in the layer, the weight is 1 regardless of how much of the layer is within the rooting depth)

        """
        if len(d_soil) <= 1:
            # if there is only one soil layer, we just return the parameter
            return param_z
        else:
            # if there are multiple soil layers, we take the average over the rooting depth
            # Calculate depth to the top of each soil layer by prepending 0 (soil surface)
            d0_soil = np.array([0] + d_soil[:-1])  # Depth from soil surface to top of each layer (m)
            param_mean = np.nanmean(param_z[d0_soil < rooting_depth])
            return param_mean

    def calculate_root_depth(self, W_R, d_rpot):
        """
        Calculate the actual root depth as the minimum of the potential root depth (based on 
        plant developmental rate) and biomass-based root depth (based on root biomass). 

        Parameters
        ----------
        W_R : float or array_like
            Root dry structural biomass (g d.wt m-2)
        d_rpot : float or array_like
            Potential root depth (m)

        Returns
        -------
        d_r : float or array_like
            Actual root depth (m)
        """
        d_r_srd = W_R * self.SRD    # root depth based on root biomass and specific root depth
        d_r = np.minimum(d_r_srd, d_rpot) 
        return d_r
