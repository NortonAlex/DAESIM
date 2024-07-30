"""
Plant carbon and water model class: Includes equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from functools import partial
from scipy.optimize import OptimizeResult, bisect
from scipy.integrate import solve_ivp
from daesim.biophysics_funcs import MinQuadraticSmooth
from daesim.climate import ClimateModule
from daesim.leafgasexchange2 import LeafGasExchangeModule2
from daesim.canopygasexchange import CanopyGasExchange
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation

@define
class PlantModel:
    """
    Calculator of plant ecophysiology, particularly the carbon and water fluxes
    """

    ## Module dependencies
    Site: Callable = field(default=ClimateModule())    ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.
    CanopyGasExchange: Callable = field(default=CanopyGasExchange())     ## It is optional to define CanopyGasExchange for this method. If no argument is passed in here, then default setting for CanopyGasExchange is the default CanopyGasExchange().

    ## Class parameters
    m_r_l: float = field(default=0.02)  ## Maintenance respiration coefficient for leaves (d-1)
    m_r_r: float = field(default=0.01)  ## Maintenance respiration coefficient for roots (d-1)
    SLA: float = field(default=0.020)  ## Specific leaf area (m2 g d.wt-1), fresh leaf area per dry leaf mass
    maxLAI: float = field(default=3)  ## Maximum potential leaf area index (m2 m-2)
    SAI: float = field(default=0.0)    ## Stem area index, m2/m2
    CI: float = field(default=0.5)    ## Foliage clumping index (-)  TODO: Double check default values of clumping index that is suitable for crops
    z: float = field(default=1.0)    ## Canopy height, m TODO: make this dynamic, perhaps a function of total biomass or growth development stage of plant
    r_wa: float = field(default=20.0)   ## Resistance to water vapor across the leaf boundary layer (s m-1), see Table 8-1 in Nobel 2009 for typical range of values
    
    soilThetaMax: float = field(default=0.5) ## Volumetric soil water content at saturation (m3 water m-3 soil)
    b_soil: float = field(default=5.0)       ## Empirical soil-specific parameter relating volumetric water content to hydraulic conductivity (-)
    Psi_e: float = field(default=-0.05)      ## Air-entry value of (hydrostatic) soil water potential; this is the soil water potential at the transition of saturated to unsaturated soil (MPa)
    K_sat: float = field(default=12)         ## Saturated value of soil hydraulic conductivity, K_s (mol m-1 s-1 MPa-1)
    
    ksr_coeff: float = field(default=500)    ## scale factor for soil-to-root conductivity/conductance (TODO: Check units and definition); conversion factor for soil hydraulic conductivity and root biomass density to a soil-to-root conductivity/conductance. In some models this is represented by a single root occupying a cylinder of soil. In principle, it considers the distance water travels from the bulk soil to the root surface, root geometry (e.g. radius) and conducting propoerties of the root. Typical values range from approx 100-18000
    d_soil: float = field(default=1.0)       ## depth of soil layer (m)
    f_r: float = field(default=1.0)          ## fraction of roots in soil layer (unitless)
    
    k_rl: float = field(default=0.10)     ## Leaf-area specific root-to-leaf (from inside the root to bulk leaf) hydraulic conductance (mol m-2 s-1 MPa-1) TODO: should increase with distance i.e. leaf height above root node
    Psi_f: float = field(default=-2.3)   ## Leaf water potential at which half of stomatal conductance occurs (MPa), see Drewry et al. (2010, doi:10.1029/2010JG001340)
    sf: float = field(default=3.5)     ## Stomatal sensitivity parameter between stomatal conductance and leaf water potential (MPa-1), see Drewry et al. (2010, doi:10.1029/2010JG001340)

    def calculate(
        self,
        W_L,         ## leaf structural dry biomass (g d.wt m-2)
        W_R,         ## root structural dry biomass (g d.wt m-2)
        soilTheta,   ## volumetric soil water content (m3 m-3)
        leafTempC,   ## leaf temperature (deg C)
        airTempC,    ## air temperature (deg C), outside leaf boundary layer 
        airRH,      ## relative humidity of air (%), outside leaf boundary layer
        airCO2,  ## leaf surface CO2 partial pressure, bar, (corrected for boundary layer effects)
        airO2,   ## leaf surface O2 partial pressure, bar, (corrected for boundary layer effects)
        airP,    ## air pressure, Pa, (in leaf boundary layer)
        swskyb, ## Atmospheric direct beam solar radiation, W/m2
        swskyd, ## Atmospheric diffuse solar radiation, W/m2
        sza,    ## Solar zenith angle, degrees
    ) -> Tuple[float]:

        if W_L < 0 or W_R < 0:
            raise ValueError(f"States W_L or W_R cannot be negative")
        
        LAI = self.calculate_LAI(W_L)

        ## Calculate soil water potential
        Psi_s = self.soil_water_potential(soilTheta)
        
        ## Calculate soil properties
        K_s = self.soil_hydraulic_conductivity(Psi_s)

        ## Calculate soil-to-root conductivity/conductance (TODO: Check definition and units of conductivity vs conductance)
        K_sr = self.soil_root_hydraulic_conductivity(W_R,K_s)

        ## Convert soil-to-root conductance to leaf-area specific soil-to-root conductance (TODO: Check definition and units of conductivity vs conductance)
        k_srl = self.soil_root_hydraulic_conductance_l(K_sr,LAI)

        ## Initial estimate of GPP without leaf water potential limitation
        GPP, E = self.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, 1.0, LAI, sza, swskyb, swskyd)

        ## Determine the total leaf-area specific conductance from soil-to-root-to-leaf
        ## - assumes a one-dimensional pathway (in series) and Ohm's law for the hydraulic conductances i.e. the relationship 1/k_tot = 1/k_srl + 1/k_rl
        k_tot = (k_srl*self.k_rl)/(self.k_rl+k_srl)

        ## Calculate leaf water potential
        Psi_l = self.leaf_water_potential_solve(Psi_s, k_tot, airTempC, leafTempC, airCO2, airO2, airRH, airP, LAI, sza, swskyb, swskyd)

        ## Calculate actual leaf water potential scaling factor on photosynthesis/dry-matter production
        f_Psi_l = self.tuzet_fsv(Psi_l)

        ## Calculate actual gpp and stomatal conductance
        GPP, E = self.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, f_Psi_l, LAI, sza, swskyb, swskyd)

        ## Calculate actual transpiration
        E = LAI * E

        ## Calculate root water potential
        Psi_r = self.root_water_potential(Psi_s,E,k_srl)

        ## Calculate maintenance respiration of leaf and root pools
        Rm_l = self.calculate_Rm_k(W_L,self.m_r_l)
        Rm_r = self.calculate_Rm_k(W_R,self.m_r_r)

        return (GPP, Rm_l, Rm_r, E, f_Psi_l, Psi_l, Psi_r, Psi_s, K_s, K_sr, k_srl)


    def leaf_water_potential_solve(self, Psi_s, k_tot, airTempC, leafTempC, airCO2, airO2, airRH, airP, LAI, sza, swskyb, swskyd):
        """
        Calculate leaf water potential that balances water supply (root uptake) and water loss (transpiration) using the bisection method.
    
        Parameters
        ----------
        LAI : float
            Leaf area index (m2 m-2).
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
            GPP, E = self.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, f_Psi_l, LAI, sza, swskyb, swskyd)
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

    def soil_root_hydraulic_conductivity_conditional(self,W_R,K_s):
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
        (2009, Physicochemical and Environmental Plant Physiology, Ch. 9.3D) for a more fundamental desciption. 
        """
        L_v = (W_R * self.f_r) / self.d_soil    ## calculate fine root density in the soil layer (g d.wt root m-3 soil)
        K_sr = K_s * L_v/self.ksr_coeff
        return K_sr
    
    def soil_root_hydraulic_conductivity(self,W_R,K_s):
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
        K_sr = _vfunc(W_R,K_s)
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
        
    def leaf_transpiration(self,gsw,leafTempC,airTempC,airP,airRH):
        """
        Calculates the transpiration rate

        Parameters
        ----------
        gsw: stomatal conductance to water vapor (mol m-2 s-1)
        leafTempC: leaf temperature (degrees Celsius)
        airTempC: air temperature (degrees Celsius)
        airP: air pressure (Pa)
        airRH: relative humidity of canopy air space (%)
    
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

        References
        ----------
        Nobel (2009) Section 8.1F and Table 8-1, doi:10.1016/B978-0-12-374143-1.X0001-4, ISBN:978-0-12-374143-1
        """
        
        ## Calculate water vapor concentration gradient between the inside of the leaf (assumed to be saturated) and the air
        Wi = self.Site.compute_sat_vapor_pressure(leafTempC)   ## Pa
        Wa = self.Site.compute_actual_vapor_pressure(airTempC,airRH)   ## Pa
        Wi_molconc = Wi/(self.Site.R_w_mol * (leafTempC+273.15))   ## converts Pa to mol m-3
        Wa_molconc = Wa/(self.Site.R_w_mol * (airTempC+273.15))   ## converts Pa to mol m-3
        
        ## Resistances for H2O across leaf surface and leaf boundary layer
        r_ws = (airP/(self.Site.R_w_mol*(leafTempC+273.15)))/gsw    ## converts stomatal conductance (mol H2O m-2 s-1) to stomatal resistance (s m-1)  (see Nobel (2009) Section 8.1F) TODO: Check this and cross-check it with calculations of stomatal resistance in leafgasexchange modules and perhaps also SCOPE model code
        r_wa = self.r_wa    ## resistance to water vapor across the leaf boundary layer (s m-1), see Table 8-1 in Nobel 2009 for typical range of values
        r_w = r_wa + r_ws   ## total resistance to water vapour across stomata and boundary layer, acting in series
        E = (Wi_molconc - Wa_molconc)/r_w   ## 

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

    def calculate_canopygasexchange(self, airTempC, leafTempC, airCO2, airO2, airRH, airP, f_Psi_l, LAI, sza, swskyb, swskyd):
        """
        Parameters
        ----------
        airTempC : Air temperature (degree Celcius)
        leafTempC : Leaf temperature (degree Celcius)
        airCO2 : Leaf surface CO2 partial pressure (bar), corrected for boundary layer effects
        airO2 : Leaf surface O2 partial pressure (bar), corrected for boundary layer effects
        airRH : Relative humidity (%)
        airP : Air pressure (Pa)
        f_Psi_l : Leaf water potential limitation factor on stomatal conductance (-)
        LAI : Leaf area index (m2/m2)
        SAI : Stem area index (m2/m2)
        CI : Foliage clumping index (-)
        z : Canopy height (m)
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

        """

        An_ml, gs_ml, Rd_ml = self.CanopyGasExchange.calculate(leafTempC,airCO2,airO2,airRH,f_Psi_l,LAI,self.SAI,self.CI,self.z,sza,swskyb,swskyd)

        GPP = np.sum(An_ml + Rd_ml)*1e6

        E_ml = self.leaf_transpiration(gs_ml,leafTempC,airTempC,airP,airRH)
        E = np.sum(E_ml)

        return GPP, E
        
    def calculate_Rm_k(self,C_k,m_r):
        """
        Parameters
        ----------
        C_k: carbon pool mass k (kg C m-2)
        
        m_r: specific maintenance respiration rate (d-1)
    
        Returns
        -------
        R_m: maintenance respiration (kg C m-2 d-1)
        """
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