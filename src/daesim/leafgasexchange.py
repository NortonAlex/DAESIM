"""
Leaf gas exchange model class: Includes equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
from daesim.biophysics_funcs import fT_arrhenius, fT_arrheniuspeaked, fT_Q10
from daesim.management import ManagementModule
from daesim.climate import ClimateModule
from daesim.climate_funcs import solar_day_calcs

@define
class LeafGasExchangeModule:
    """
    Calculator of leaf gas exchange including photosynthesis and stomatal conductance
    """

    # Module dependencies
    Site: Callable = field(default=ClimateModule())    ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.

    # Class parameters

    ## Biochemical constants
    Kc_opt: float = field(
        default=405.0*1e-06
    )  ## Michaelis-Menten kinetic coefficient for CO2 (VonCaemmerer and Furbank, 1999), bar
    Ko_opt: float = field(
        default=278.0*1e-03
    )  ## Michaelis-Menten kinetic coefficient for O2 (VonCaemmerer and Furbank, 1999), bar
    spfy_opt: float = field(
        default=2600.0
    )  ## Specificity (tau in Collatz e.a. 1991). This is, in theory, Vcmax/Vomax.*Ko./Kc, but used as a separate parameter.

    Vcmax_opt: float = field(
        default=100.0*1e-6
    )  ## Maximum Rubisco activity at optimum temperature, mol CO2 m-2 s-1
    Vqmax_opt: float = field(
        default=350.0*1e-6
        ) ## Maximum Cyt b6f activity at optimum temperature, mol e-1 m-2 s-1
    TPU_opt_rVcmax: float = field(default=0.1666)  ## TPU as a ratio of Vcmax_opt (from Bonan, 2019, Chapter 11, p. 171)
    ## Temperature dependence parameters
    Kc_Ea: float = field(default=79.43)  ## activation energy of Kc, kJ mol-1 (Bernacchi et al., 2001; Medlyn et al., 2002, Eq. 5)
    Ko_Ea: float = field(default=36.28)  ## activation energy of Ko, kJ mol-1 (Bernacchi et al., 2001; Medlyn et al., 2002, Eq. 6)
    Vcmax_Ea: float = field(default=70.0)  ## activation energy of Vcmax, kJ mol-1 (Medlyn et al., 2002)
    Vcmax_Hd: float = field(default=200.0)  ## deactivation energy of Vcmax, kJ mol-1 (Medlyn et al., 2002)
    Vcmax_DeltaS: float = field(default=0.65)  ## entropy of process for Vcmax, kJ mol-1 K-1 (Medlyn et al., 2002)
    Vqmax_Ea: float = field(default=80.0)  ## activation energy of Vqmax, kJ mol-1 (it is assumed that derived Jmax temperature response parameters can apply to Vqmax; Medlyn et al., 2002)
    Vqmax_Hd: float = field(default=200.0)  ## deactivation energy of Vqmax, kJ mol-1 (it is assumed that derived Jmax temperature response parameters can apply to Vqmax; Medlyn et al., 2002)
    Vqmax_DeltaS: float = field(default=0.65)  ## entropy of process for Vqmax, kJ mol-1 K-1 (it is assumed that derived Jmax temperature response parameters can apply to Vqmax; Medlyn et al., 2002)
    Rd_Q10: float = field(default=1.8)  ## Q10 coefficient for the temperature response of Rd
    TPU_Q10: float = field(default=1.8)  ## Q10 coefficient for the temperature response of TPU
    spfy_Ea: float = field(default=-29.0)  ## activation energy for the specificity factor (Medlyn et al., 2002, p. 1170)

    Abs: float = field(default=0.85)  ## Total leaf absorptance to PAR, mol PPFD absorbed mol-1 PPFD incident
    beta: float = field(default=0.52)  ## PSII fraction of total leaf absorptance, mol PPFD absorbed by PSII mol-1 PPFD absorbed
    Rds: float = field(default=0.01)  ## Scalar for dark respiration, dimensionless

    nl: float = field(default=0.75) ## ATP per e- in linear flow, ATP/e-
    nc: float = field(default=1.00) ## ATP per e- in cyclic flow, ATP/e-
    effcon: float = field(default=0.25)  ## Efficiency of conversion. TODO: Add better notes here
    atheta: float = field(default=1)  ## Empirical smoothing parameter to allow for co-limitation of Vc and Ve. In Johnson and Berry (2021) model this must equal 1 (i.e. no smoothing). 

    ## Stomatal conductance constants
    g0: float = field(default=0.0)   ## g0, see Medlyn et al. (2011, doi: 10.1111/j.1365-2486.2012.02790.x) 
    g1: float = field(default=3.0)   ## g1, see Medlyn et al. (2011, doi: 10.1111/j.1365-2486.2012.02790.x) 

    ## Mesophyll conductance constants
    gm_opt: float = field(default=1e6)   ## mesophyll conductance to CO2 diffusion, mol m-2 s-1 bar-1 

    alpha_opt: str = field(default='static') ## Model choice of static or dynamic absorption cross-section calculations

    ## Photochemical constants
    Kf: float = field(default=0.05e09) ## Rate constant for fluoresence at PSII and PSI, s-1
    Kd: float = field(default=0.55e09) ## Rate constant for constitutive heat loss at PSII and PSI, s-1
    Kp1: float = field(default=14.5e09) ## Rate constant for photochemistry at PSI, s-1
    Kn1: float = field(default=14.5e09) ## Rate constant for regulated heat loss at PSI, s-1
    Kp2: float = field(default=4.5e09) ## Rate constant for photochemistry at PSII, s-1
    Ku2: float = field(default=0e09) ## Rate constant for exciton sharing at PSII, s-1

    ## Transfer functions for fluorescence
    eps1: float = field(default=0) ## PS I transfer function, mol PSI F to detector mol-1 PSI F emitted
    eps2: float = field(default=1) ## PS II transfer function, mol PSII F to detector mol-1 PSII F emitted

    ## Constants
    rhoa: float = field(default=1.2047)  ## Specific mass of air, kg m-3
    Mair: float = field(default=28.96)  ## Molecular mass of dry air, g mol-1

    def calculate(
        self,
        Q,    ## Absorbed PPFD, mol PAR m-2 s-1
        T,    ## Leaf temperature, degrees Celsius
        Cs,   ## Leaf surface CO2 partial pressure, bar, (corrected for boundary layer effects)
        O,    ## Leaf surface O2 partial pressure, bar, (corrected for boundary layer effects)
        RH,   ## Relative humidity, %
        fgsw, ## Leaf water potential limitation factor on stomatal conductance, unitless
    ) -> Tuple[float]:

        # Calculate derived variables from constants
        Vqmax = fT_arrheniuspeaked(self.Vqmax_opt,T,E_a=self.Vqmax_Ea,H_d=self.Vqmax_Hd,DeltaS=self.Vqmax_DeltaS)       # Maximum Cyt b6f activity, mol e-1 m-2 s-1
        Vcmax = fT_arrheniuspeaked(self.Vcmax_opt,T,E_a=self.Vcmax_Ea,H_d=self.Vcmax_Hd,DeltaS=self.Vcmax_DeltaS)       # Maximum Rubisco activity, mol CO2 m-2 s-1
        TPU = fT_Q10(self.TPU_opt_rVcmax*self.Vcmax_opt,T,Q10=self.TPU_Q10) 
        Rd = fT_Q10(Vcmax*self.Rds,T,Q10=self.Rd_Q10)
        S = fT_arrhenius(self.spfy_opt,T,E_a=self.spfy_Ea)
        Kc = fT_arrhenius(self.Kc_opt,T,E_a=self.Kc_Ea)
        Ko = fT_arrhenius(self.Ko_opt,T,E_a=self.Ko_Ea)
        phi1P_max = self.Kp1/(self.Kp1+self.Kd+self.Kf)  # Maximum photochemical yield PS I
        Gamma_star   = 0.5 / S * O      # compensation point in absence of Rd

        # g1 and g0 are input ALWAYS IN UNITS OF H20
        # G0 must be converted to CO2 (but not G1, see below)
        g0 = self.g0/1.6

        ## Establish PSII and PS I cross-sections, mol PPFD abs PS2/PS1 mol-1 PPFD
        ## TODO: consider moving this to a separate function
        if self.alpha_opt == 'static':
            a2 = self.Abs*self.beta
            a1 = self.Abs - a2
        elif self.alpha_opt == 'dynamic':
            print("dynamic absorption cross-section not implemented yet")
            return np.nan

        VPD = self.Site.compute_VPD(T,RH)*1e-3

        # Compute stomatal conductance and Ci based on optimal stomatal theory (Medlyn et al., 2011)
        A, gs, Ci = self.solve_Ci(Cs,Q,O,VPD,Vqmax,a1,phi1P_max,S,fgsw)    ## TODO: Check the units of A, gs, and Ci here. Is it in ppm (umol mol-1?)? or bar? 
        #A, gs, Ci = self.solve_Cc(Cs,Q,O,VPD,Vqmax,a1,phi1P_max,S,max_iterations=20)    ## TODO: Check the units of A, gs, and Ci here. Is it in ppm (umol mol-1?)? or bar? 

        # Update photosynthetic rate with optimal Ci
        (A, Ag, Vc, Ve, Vs) = self.compute_A(Ci, O, Q, Vcmax, Vqmax, TPU, Gamma_star, Rd, S, Kc, Ko, a1, phi1P_max)

        # Actual electron transport rate
        Ja = Ag / ((Ci - Gamma_star) / (Ci + 2 * Gamma_star)) / self.effcon

        # Stomatal resistance
        rcw    = ( self.rhoa / (self.Mair*1.0e3) )/gs

        return (A, gs, Ci, Vc, Ve, Vs, Rd)

    def JB21_A_c(self, C, O, Vcmax, S, Kc, Ko):  #Vcmax=100e-6, S=2600, Kc=405.0*1e-06, Ko=278.0*1e-03):
        """
        Calculates the potential rate of (gross) CO2 assimilation under Rubisco limitation. This is synonymous 
        with the ‘light-saturated’ or 'Rubisco limited' metabolic state where carbon metabolism is 
        limiting electron transport.
        
        Parameters
        ----------
        C = CO2 partial pressure in the chloroplasts, bar, (often denoted $C_c$ when mesophyll conductance is considered and $C_i$ when mesophyll conductance is ignored)
        O = O2 partial pressure in the chloroplasts, bar
        Vcmax = Maximum carboxylase activity of Rubisco, mol CO2 m-2 s-1
        S = Rubisco specificity for CO2/O2, dimensionless

        Returns
        -------
        A_c = Potential rate of gross CO2 assimilation, mol CO2 m-2 s-1

        References
        ----------
        Johnson and Berry (2021) The role of Cytochrome b6f in the control of steady-state photosynthesis: a conceptual and quantitative model, doi: 10.1007/s11120-021-00840-4
        """
        
        Gamma_star = O/(2*S)  # Calculate CO2 compensation point in the light, $\Gamma^*$
        
        eta = (1-(self.nl/self.nc)+(3+7*Gamma_star/C)/((4+8*Gamma_star/C)*self.nc))  # Calculate PS I/II ETR

        # Expressions for potential Rubisco-limited rates (_c)
        # N.B., see Eqns. 32-33
        Vc_c = C*Vcmax/(Kc*(1+O/Ko + C))
        Vo_c = Vc_c*2*Gamma_star/C
        Ag_c = Vc_c - Vo_c/2
        return Ag_c

    def JB21_A_j(self, Q, C, O, Vqmax, a1, phi1P_max, S):
        """
        Calculates the potential rate of CO2 assimilation under Cyt b6f limitation. This is synonymous 
        with the ‘light-limited’ or 'RuBP-regeneration limited' metabolic state where electron transport is 
        limiting carbon metabolism.
        
        Parameters
        ----------
        Q = absorbed photosynthetically active radiation (PPFD), mol PAR m-2 s-1
        C = CO2 partial pressure in the chloroplasts, bar, (often denoted $C_c$ when mesophyll conductance is considered and $C_i$ when mesophyll conductance is ignored)
        O = O2 partial pressure in the chloroplasts, bar
        Vqmax = Maximum Cyt b6f activity, mol e-1 m-2 s-1
        a1 = PS I cross-section, mol PPFD abs PS2/PS1 mol-1 PPFD
        phi1P_max = Maximum photochemical yield PS I, dimensionless
        S = Rubisco specificity for CO2/O2, dimensionless

        Returns
        -------
        A_j = Potential rate of gross CO2 assimilation, mol CO2 m-2 s-1

        References
        ----------
        Johnson and Berry (2021) The role of Cytochrome b6f in the control of steady-state photosynthesis: a conceptual and quantitative model, doi: 10.1007/s11120-021-00840-4
        """
        
        Gamma_star = O/(2*S)  # Calculate CO2 compensation point in the light, $\Gamma^*$
        
        eta = (1-(self.nl/self.nc)+(3+7*Gamma_star/C)/((4+8*Gamma_star/C)*self.nc))  # Calculate PS I/II ETR

        #% Expressions for potential Cytochrome b6f-limited rates
        JP700_j = (Q*Vqmax)/(Q+Vqmax/(a1*phi1P_max))
        JP680_j = JP700_j/eta
        Vc_j = JP680_j/(4*(1+2*Gamma_star/C))
        Vo_j = Vc_j*2*Gamma_star/C
        Ag_j = Vc_j - Vo_j/2
        return Ag_j

    def net_co2_assimilation(self,Ag,Rd):
        """
        Calculates the net CO2 assimilation rate as the difference between gross assimilation and mitochondrial respiration

        Parameters
        ----------
        Ag: float
            Gross photosynthetic CO2 assimilation, mol CO2 m-2 s-1
        Rd: float
            Mitochondrial respiration, mol CO2 m-2 s-1

        Returns
        -------
        An: float
            Net photosynthetic CO2 assimation, mol CO2 m-2 s-1
        """
        return Ag - Rd
        
    def gs_Medlyn(self,Cs,D,An,fgsw):
        """
        Model for the optimal stomatal conductance to water vapour following Medlyn et al. (2011)
        
        Parameters
        ----------
        Cs: float
            Atmospheric carbon dioxide partial pressure at the leaf surface, umol mol-1
        D: float
            Leaf-to-air vapour pressure deficit, kPa
        An: float
            Net assimilation rate of CO2, umol CO2 m-2 s-1
        fgsw: float
            Leaf water potential limitation factor on stomatal conductance, unitless, e.g. soil moisture limitation, Tuzet et al. (2003, doi: 10.1046/j.1365-3040.2003.01035.x)
        g1, g0: float
            Fitted parameters according to the Medlyn et al. (2011) model

        Returns
        -------
        gs: float
            Optimal stomatal conductance, mol H2O m-2 s-1

        References
        ----------
        See original paper and corrigendum: Medlyn et al., 2011, doi: 10.1111/j.1365-2486.2012.02790.x
        """
        gs = self.g0 + fgsw*1.6*(1 + self.g1/np.sqrt(D))*An/Cs
        return gs

    def Ficks_diffusion_Ci(self,Cs,An,gs):
        Ci = Cs - 1.6*An/gs
        return Ci

    def Ficks_diffusion_Cc(self,Ci,An,gm):
        Cc = Ci - 1.6*An/gm
        return Cc

    def Ficks_diffusion_An(self,Cs,Ci,gs):
        An = gs/1.6 * (Cs - Cs)
        return An
        
    def solve_Ci_conditional(self,Cs,Q,O,D,Vqmax,a1,phi1P_max,S,fgsw,max_iterations=50,rtol_Ci=0.01):
        """

        Notes
        -----
        To determine Ci the photosynthetic rate is represented as the RuBP-regeneration limited (light-limited)
        rate, following Medlyn et al. (2011, doi:10.1111/j.1365-2486.2010.02375.x and Corrigendum doi:10.1111/j.1365-2486.2012.02790.x)
        and Medlyn et al. (2013, doi:10.1016/j.agrformet.2013.04.019).
        """
        #rtol_Ci: tolerance limit for Ci convergence expressed as a fraction of Cs
        # Step 1 - Initial estimate of Ci
        Ci =  0.7*Cs
        tol = rtol_Ci*Cs
        
        for i in range(max_iterations):
            A = self.JB21_A_j(Q,Ci,O,Vqmax,a1,phi1P_max,S)      # Step 2 - Calculate A with estimate of Ci
            gs = self.gs_Medlyn(Cs,D,A,fgsw)    # Step 3 - Update gs using the Medlyn formulation
            Ci_new = self.Ficks_diffusion_Ci(Cs,A,gs)   # Step 4 - Update Ci using the formulation
            if abs(Ci - Ci_new) < tol:   # Convergence criterion
                break
            Ci = Ci_new
        return (A, gs, Ci)

    def solve_Ci(self,Cs,Q,O,D,Vqmax,a1,phi1P_max,S,fgsw,max_iterations=50,rtol_Ci=0.01):
        _vfunc = np.vectorize(self.solve_Ci_conditional)
        A, gs, Ci = _vfunc(Cs,Q,O,D,Vqmax,a1,phi1P_max,S,fgsw)
        return (A, gs, Ci)

    def solve_Cc_conditional(self,Cs,Q,O,D,Vqmax,a1,phi1P_max,S,max_iterations=10,rtol_Cc=0.01):
        """

        Notes
        -----
        To determine Cc the photosynthetic rate is represented as the RuBP-regeneration limited (light-limited)
        rate, following Medlyn et al. (2011, doi:10.1111/j.1365-2486.2010.02375.x and Corrigendum doi:10.1111/j.1365-2486.2012.02790.x)
        and Medlyn et al. (2013, doi:10.1016/j.agrformet.2013.04.019).
        """
        #rtol_Cc: tolerance limit for Cc convergence expressed as a fraction of Cs
        # Step 1 - Initial estimate of Cc
        Cc =  0.7*Cs
        tol = rtol_Cc*Cs
        
        for i in range(max_iterations):
            A = self.JB21_A_j(Q,Cc,O,Vqmax,a1,phi1P_max,S)      # Step 2 - Calculate A with estimate of Ci
            gs = self.gs_Medlyn(Cs,D,A)    # Step 3 - Update gs using the Medlyn formulation
            Ci_new = self.Ficks_diffusion_Ci(Cs,A,gs)   # Step 4 - Update Ci using the formulation
            Cc_new = self.Ficks_diffusion_Cc(Ci_new,A,self.gm_opt)  ## Step 5 - Update Cc using the formulation
            if abs(Cc - Cc_new) < tol:   # Convergence criterion
                break
            Cc = Cc_new
        return (A, gs, Cc)

    def solve_Cc(self,Cs,Q,O,D,Vqmax,a1,phi1P_max,S,max_iterations=10,rtol_Cc=0.01):
        _vfunc = np.vectorize(self.solve_Cc_conditional)
        A, gs, Cc = _vfunc(Cs,Q,O,D,Vqmax,a1,phi1P_max,S)
        return (A, gs, Cc)

    def compute_A(self, C, O, Q, Vcmax, Vqmax, TPU, Gamma_star, Rd, S, Kc, Ko, a1, phi1P_max):
        """NOTES HERE"""
        
        eta = (1-(self.nl/self.nc)+(3+7*Gamma_star/C)/((4+8*Gamma_star/C)*self.nc))  # Calculate PS I/II ETR

        Ve  = self.JB21_A_j(Q, C, O, Vqmax, a1, phi1P_max, S)

        # Expressions for potential Rubisco-limited rates (_c)
        #   N.B., see Eqns. 32-33
        Vc  = self.JB21_A_c(C, O, Vcmax, S, Kc, Ko)

        JP680_c = Vc*4*(1+2*Gamma_star/C)/(1-Gamma_star/C)
        JP700_c = JP680_c*eta

        Vs  = 3*TPU

        # Here we allow for co-limitation of Vc and Ve, providing a smooth transition between the 
        # two potentially limiting rates. This takes the smaller root of a quadratic equation,  
        # where 0.98 is the empirical smoothing parameter. See Bonan (2019) Eq. 11.31-11.33
        V    = self.hyperbolic_min_Ac_Aj(Vc, Ve)

        Ag   = self.hyperbolic_min_Ap_Am(Vs, V)

        A    = Ag - Rd

        # Select minimum PS1 ETR
        #JP700_a = self.hyperbolic_min_Ac_Aj(JP700_j, JP700_c)

        # Select minimum PS2 ETR 
        #JP680_a = self.hyperbolic_min_Ac_Aj(JP680_j, JP680_c)

        return (A, Ag, Vc, Ve, Vs) # , JP680_a, JP700_a)

    def hyperbolic_min_Ac_Aj(self, Ac, Aj):
        """Hyperbolic minimum between Ac and Aj"""
        _vfunc = np.vectorize(self.quadp)
        Am = -_vfunc(self.atheta, Ac+Aj, Ac*Aj)
        return Am

    def hyperbolic_min_Ap_Am_conditional(self, Ap, Am):
        """Hyperbolic minimum with TPU limitation, applied only if Ap is less than Am"""
        if Ap < Am:
            _vfunc = np.vectorize(self.quadp)
            Am = -_vfunc(1 - 1E-07, Am+Ap, Am*Ap)
            return Am
        else: 
            return Am

    def hyperbolic_min_Ap_Am(self, Ap, Am):
        """Hyperbolic minimum with TPU limitation, applied only if Ap is less than Am"""
        _vfunc = np.vectorize(self.hyperbolic_min_Ap_Am_conditional)
        Am = _vfunc(Ap,Am)
        return Am

    def quadp(self, a, b, c):
        """
        Returns the larger root of the quadratic equation ax^2 + bx + c = 0.
        If the roots are imaginary, prints a warning and returns 0.
        Handles cases when a or b are zero.
        """
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            print("IMAGINARY ROOTS IN QUADRATIC")
            return 0
        
        if a == 0:
            if b == 0:
                return 0
            else:
                return -c / b
        else:
            return (-b + np.sqrt(discriminant)) / (2 * a)

    def set_Vcmax_for_layer(self, Vcmax_opt, adjustment_factor):
        self.Vcmax_opt =  Vcmax_opt * adjustment_factor

    def set_Vqmax_for_layer(self, Vqmax_opt, adjustment_factor):
        self.Vqmax_opt =  Vqmax_opt * adjustment_factor