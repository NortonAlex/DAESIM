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
class LeafGasExchangeModule2:
    """
    Calculator of leaf gas exchange including photosynthesis and stomatal conductance
    """

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
    Jmax_opt: float = field(
        default=150.0*1e-6
        ) ## Maximum electron transport rate at optimum temperature, mol e-1 m-2 s-1
    theta: float = field(default=0.85)  ## Empirical curvature parameter for the shape of light response curve
    alpha: float = field(default=0.24)  ## Quantum yield of electron transport (mol mol-1)
    TPU_opt_rVcmax: float = field(default=0.1666)  ## TPU as a ratio of Vcmax_opt (from Bonan, 2019, Chapter 11, p. 171)
    alphag: float = field(default=1.0)  ## Fraction of glycolate not returned to the chloroplast; parameter in TPU-limited photosynthesis (optional, only to be used when TPU is provided) (range from 0-1). Following Ellsworth et al. (2015) Eq. 7.
    ## Temperature dependence parameters
    Kc_Ea: float = field(default=79.43)  ## activation energy of Kc, kJ mol-1 (Bernacchi et al., 2001; Medlyn et al., 2002, Eq. 5)
    Ko_Ea: float = field(default=36.28)  ## activation energy of Ko, kJ mol-1 (Bernacchi et al., 2001; Medlyn et al., 2002, Eq. 6)
    Vcmax_Ea: float = field(default=70.0)  ## activation energy of Vcmax, kJ mol-1 (Medlyn et al., 2002)
    Vcmax_Hd: float = field(default=200.0)  ## deactivation energy of Vcmax, kJ mol-1 (Medlyn et al., 2002)
    Vcmax_DeltaS: float = field(default=0.65)  ## entropy of process for Vcmax, kJ mol-1 K-1 (Medlyn et al., 2002)
    Jmax_Ea: float = field(default=80.0)  ## activation energy of Jmax, kJ mol-1 (Medlyn et al., 2002)
    Jmax_Hd: float = field(default=200.0)  ## deactivation energy of Jmax, kJ mol-1 (Medlyn et al., 2002)
    Jmax_DeltaS: float = field(default=0.632)  ## entropy of process for Jmax, kJ mol-1 K-1 (Medlyn et al., 2002)
    Rd_Q10: float = field(default=1.8)  ## Q10 coefficient for the temperature response of Rd
    TPU_Q10: float = field(default=1.8)  ## Q10 coefficient for the temperature response of TPU
    spfy_Ea: float = field(default=-29.0)  ## activation energy for the specificity factor (Medlyn et al., 2002, p. 1170)

    Rds: float = field(default=0.01)  ## Scalar for dark respiration, dimensionless

    effcon: float = field(default=0.25)  ## Efficiency of conversion. TODO: Add better notes here
    atheta: float = field(default=1-1e-04)  ## Empirical smoothing parameter to allow for co-limitation of Vc and Ve. In Johnson and Berry (2021) model this must equal 1 (i.e. no smoothing). 

    ## Stomatal conductance constants
    g0: float = field(default=0.0)   ## g0, see Medlyn et al. (2011, doi: 10.1111/j.1365-2486.2012.02790.x) 
    g1: float = field(default=3.0)   ## g1, see Medlyn et al. (2011, doi: 10.1111/j.1365-2486.2012.02790.x) 
    VPDmin: float = field(default=0.5)  ## Below vpdmin, VPD=vpdmin, to avoid very high gs, see Duursma (2015, doi: 10.1371/journal.pone.0143346)
    whichA: str = field(default="Ah")  ## Photosynthetic limiting rate that stomatal conductance responds to. One of the options "Ah", "Aj", "Ac". TODO: Add check on initialisation to ensure whichA is defined as one these options. 
    GCtoGW: float = field(default=1.57)  ## Conversion factor from conductance to CO2 to H2O

    ## Mesophyll conductance constants
    gm_opt: float = field(default=1e6)   ## mesophyll conductance to CO2 diffusion, mol m-2 s-1 bar-1 

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
        Site=ClimateModule(),   ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.
    ) -> Tuple[float]:

        # Calculate derived variables from constants
        Jmax = fT_arrheniuspeaked(self.Jmax_opt,T,E_a=self.Jmax_Ea,H_d=self.Jmax_Hd,DeltaS=self.Jmax_DeltaS)       # Maximum electron transport rate, mol e-1 m-2 s-1
        Vcmax = fT_arrheniuspeaked(self.Vcmax_opt,T,E_a=self.Vcmax_Ea,H_d=self.Vcmax_Hd,DeltaS=self.Vcmax_DeltaS)       # Maximum Rubisco activity, mol CO2 m-2 s-1
        TPU = fT_Q10(self.TPU_opt_rVcmax*self.Vcmax_opt,T,Q10=self.TPU_Q10) 
        Rd = fT_Q10(Vcmax*self.Rds,T,Q10=self.Rd_Q10)
        S = fT_arrhenius(self.spfy_opt,T,E_a=self.spfy_Ea)
        Kc = fT_arrhenius(self.Kc_opt,T,E_a=self.Kc_Ea)
        Ko = fT_arrhenius(self.Ko_opt,T,E_a=self.Ko_Ea)
        Km = Kc*(1+O/Ko)
        Gamma_star   = 0.5 / S * O      # compensation point in absence of Rd

        # g1 and g0 are input ALWAYS IN UNITS OF H20
        # G0 must be converted to CO2 (but not G1, see below)
        g0 = self.g0/1.6

        VPD = Site.compute_VPD(T,RH)*1e-3
        VPDuse = np.maximum(VPD, self.VPDmin)    ## Set VPD values below lower limit to VPDmin to ensure gs doesn't go wacky

        GsDIVA = (1 + fgsw*self.g1/(VPDuse**0.5))/Cs

        J = self.Jfun(Q,Jmax)
        Vj = J/4

        # Solve for Ci under both limiting rate conditions
        Ci_c = self.getCi_c(Vj,GsDIVA,Q,Cs,Rd,Vcmax,Km,Gamma_star)
        Ci_j = self.getCi_j(Vj,GsDIVA,Q,Cs,Rd,Vcmax,Km,Gamma_star)

        ## Calculate Rubisco and RuBP-regen photosynthetic rates (without mesophyll conductance)
        Ac = Vcmax*(Ci_c - Gamma_star)/(Ci_c + Km)
        Aj = Vj*(Ci_j - Gamma_star)/(Ci_j + 2*Gamma_star)

        ## When below the light-compensation points, assume Ci=Cs
        Aj, Ci = self.adjust_for_lcp(Aj,Rd,Cs,Gamma_star,Vj,Ci_c,Ci_j)

        ## Limitation by triose-phosphate utilization
        Ap = self.compute_A_TPU_rate(TPU,Ci,Gamma_star)

        ## Hyperbolic minimum of Ac and Aj
        Am = self.hyperbolic_min_Ac_Aj(Ac, Aj)

        ## Hyperbolic minimum with the transition to TPU
        Am = self.hyperbolic_min_Ap_Am(Ap, Am)

        ## Net photosynthesis
        An = Am - Rd

        ## Stomatal conductance to CO2
        gsc = self.conductance_to_CO2(An, Ac, Aj, Rd, GsDIVA)

        ## Stomatal conductance to H2O
        gsw = gsc*self.GCtoGW

        # Actual electron transport rate
        #Ja = Ag / ((Ci - Gamma_star) / (Ci + 2 * Gamma_star)) / self.effcon

        # Stomatal resistance
        rcw    = ( self.rhoa / (self.Mair*1.0e3) )/gsw

        return (An, gsw, Ci, Ac, Aj, Ap, Rd)

    def Jfun(self, Q, Jmax):
        """
        Electron transport rate from non-rectangular hyperbola

        References
        ----------
        von Caemmerer, 2000, S. Biochemical models of leaf photosynthesis. CSIRO Publishing, Australia
        """
        J = (self.alpha*Q + Jmax - np.sqrt((self.alpha*Q + Jmax)**2 - 4*self.alpha*self.theta*Q*Jmax))/(2*self.theta)
        return J

    def getCi_c_conditional(self,Vj,GsDIVA,Q,Ca,Rd,Vcmax,Km,Gamma_star):
        if (Vj == 0) or (Q == 0):
            return (Ca, Ca)
        
        # Solution when Rubisco activity is limiting
        # a, b and c are coefficients to a quadratic equation
        a = self.g0 + GsDIVA * (Vcmax - Rd)
        b = (1 - Ca*GsDIVA) * (Vcmax - Rd) + self.g0*(Km - Ca) - GsDIVA*(Vcmax*Gamma_star + Km*Rd)
        c =  -(1 - Ca*GsDIVA) * (Vcmax*Gamma_star + Km*Rd) - self.g0*Km*Ca
        Ci_c = self.quadp(a,b,c)
        
        return Ci_c

    def getCi_c(self,Vj,GsDIVA,Q,Ca,Rd,Vcmax,Km,Gamma_star):
        _vfunc = np.vectorize(self.getCi_c_conditional)
        Ci_c = _vfunc(Vj,GsDIVA,Q,Ca,Rd,Vcmax,Km,Gamma_star)
        return Ci_c

    def getCi_j_conditional(self,Vj,GsDIVA,Q,Ca,Rd,Vcmax,Km,Gamma_star):
        if (Vj == 0) or (Q == 0):
            return (Ca, Ca)
        
        # Solution when electron transport is limiting
        a = self.g0 + GsDIVA * (Vj - Rd)
        b = (1 - Ca*GsDIVA) * (Vj - Rd) + self.g0 * (2.*Gamma_star - Ca) - GsDIVA * (Vj*Gamma_star + 2.*Gamma_star*Rd)
        c = -(1 - Ca*GsDIVA) * Gamma_star * (Vj + 2*Rd) - self.g0*2*Gamma_star*Ca
        Ci_j = self.quadp(a,b,c)

        return Ci_j

    def getCi_j(self,Vj,GsDIVA,Q,Ca,Rd,Vcmax,Km,Gamma_star):
        _vfunc = np.vectorize(self.getCi_j_conditional)
        Ci_j = _vfunc(Vj,GsDIVA,Q,Ca,Rd,Vcmax,Km,Gamma_star)
        return Ci_j

    def adjust_for_lcp_conditional(self, Aj, Rd, Ca, Gamma_star, Vj, Ci_c, Ci_j):
        """
        Adjust the intercellular CO2 concentration (Ci) and electron transport limited photosynthetic rate 
        if below the light-compensation point. 

        Parameters
        ----------
        Aj : float
            Electron transport rate-limited photosynthesis.
        Rd : float
            Mitochondrial respiration rate.
        Ca : float
            Ambient CO2 concentration.
        Gamma_star : float)
            CO2 compensation point in the absence of mitochondrial respiration.
        Vj : float 
            Electron transport rate.
        Ci_c : float 
            Ci when Rubisco activity is limiting.
        Ci_j : float
            Ci when electron transport is limiting.

        Returns
        -------
        Ci : float
            The calculated intercellular CO2 concentration (Ci).

        Notes
        -----
        As noted in Duursma's plantecophys package, when below light-compensation points, assume Ci=Ca. 

        References
        ----------
        Duursma, 2015, doi: 10.1371/journal.pone.0143346
        """
        if Aj <= Rd + 1e-09:
            Ci_j = Ca
            Aj = max(0,Vj * (Ci_j - Gamma_star) / (Ci_j + 2 * Gamma_star))

        # Choose between Ci_j and Ci_c based on photosynthetic limitations
        Ci = Ci_j if Aj < Vj * (Ci_c - Gamma_star) / (Ci_c + 2 * Gamma_star) else Ci_c

        return Aj, Ci

    def adjust_for_lcp(self, Aj, Rd, Ca, Gamma_star, Vj, Ci_c, Ci_j):
        """
        Adjust the intercellular CO2 concentration (Ci) and electron transport limited photosynthetic rate 
        if below the light-compensation point. 

        Parameters
        ----------
        Aj : float or ndarray
            Electron transport rate-limited photosynthesis.
        Rd : float or ndarray
            Mitochondrial respiration rate.
        Ca : float or ndarray
            Ambient CO2 concentration.
        Gamma_star : float or ndarray
            CO2 compensation point in the absence of mitochondrial respiration.
        Vj : float or ndarray
            Electron transport rate.
        Ci_c : float or ndarray
            Ci when Rubisco activity is limiting.
        Ci_j : float or ndarray
            Ci when electron transport is limiting.

        Returns
        -------
        Ci : float or ndarray
            The calculated intercellular CO2 concentration (Ci).

        Notes
        -----
        As noted in Duursma's plantecophys package, when below light-compensation points, assume Ci=Ca. 

        References
        ----------
        Duursma, 2015, doi: 10.1371/journal.pone.0143346
        """
        _vfunc = np.vectorize(self.adjust_for_lcp_conditional,otypes=[float,float])
        Aj, Ci = _vfunc(Aj, Rd, Ca, Gamma_star, Vj, Ci_c, Ci_j)
        return Aj, Ci

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

    def conductance_to_CO2(self, An, Ac, Aj, Rd, GsDIVA):
        """
        Calculates conductance to CO2 based on the chosen limiting rate that stomata respond to. 
        
        Parameters
        ----------
        An : float or ndarray
            Net photosynthetic rate (umol m-2 s-1)
        Ac : float or ndarray
            Rubisco-limited photosynthetic rate (umol m-2 s-1)
        Aj : float or ndarray
            Electron transport limited photosynthetic rate (umol m-2 s-1)
        Rd : float or ndarray
            Mitochondrial respiration rate (umol m-2 s-1)
        whichA : str
            String to define which assimilation rate stomatal conductance responds to

        Returns
        -------
        gs : float or ndarray
            Stomatal conductance to CO2

        Notes
        -----
        The conductance can be negative (e.g. Am-Rd < 0), so we take the maximum of g0 and the 
        chosen stomatal conductance equation.
        """
        gs = np.maximum(self.g0, (self.whichA == "Ah")*(self.g0+GsDIVA*An) + (self.whichA == "Ac")*(self.g0+GsDIVA*(Ac-Rd)) + (self.whichA == "Aj")*(self.g0+GsDIVA*(Aj-Rd)))
        return gs
    
    def compute_A_TPU_rate(self,TPU,Ci,Gamma_star):
        Ap = (Ci < 400)*1000 + (Ci >= 400)*3 * TPU * (Ci - Gamma_star)/(Ci - (1 + 3*self.alphag)*Gamma_star)
        return Ap

    def set_Vcmax_for_layer(self, Vcmax_opt, adjustment_factor):
        self.Vcmax_opt =  Vcmax_opt * adjustment_factor

    def set_Vqmax_for_layer(self, Vqmax_opt, adjustment_factor):
        self.Vqmax_opt =  Vqmax_opt * adjustment_factor

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

    def quadm(self, a, b, c):
        """
        Returns the smaller root of the quadratic equation ax^2 + bx + c = 0.
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
            return (-b - np.sqrt(discriminant)) / (2 * a)