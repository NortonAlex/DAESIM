"""
Plant optimal trajectory carbon allocation model class: Includes equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from daesim.plantcarbonwater import PlantModel as PlantCH2O


@define 
class PlantOptimalAllocation:
    """
    Calculator of plant allocation based on optimal trajectory principle
    """

    ## Module dependencies
    PlantCH2O: Callable = field(default=PlantCH2O())    ## It is optional to define Plant for this method. If no argument is passed in here, then default setting for Plant is the default PlantModel().

    ## Class parameters

    ## Allocation coefficients that are not optimal
    u_Stem: float = field(default=0.0)    ## Carbon allocation coefficient to stem
    u_Seed: float = field(default=0.0)    ## Carbon allocation coefficient to seed

    ## Turnover rates (carbon pool lifespan)
    tr_L: float = field(default=0.01)    ## Turnover rate of leaf biomass (days-1)
    tr_R: float = field(default=0.01)    ## Turnover rate of root biomass (days-1)

    ## Method for calculating the gradient: For evaluating marginal gain and marginal cost with chosen method
    gradient_method: str = field(default="fd_adaptive")  ## Gradient calculation method
    min_step_rel_WL: float = field(default=0.01)   ## minimum relative step size (relative to function input 'x' i.e. W_L)
    max_step_rel_WL: float = field(default=0.10)   ## maximum relative step size (relative to function input 'x' i.e. W_L)
    min_step_rel_WR: float = field(default=0.01)   ## minimum relative step size (relative to function input 'x' i.e. W_R)
    max_step_rel_WR: float = field(default=0.10)   ## maximum relative step size (relative to function input 'x' i.e. W_R)
    gradient_threshold_WL: float = field(default=1e-7)  ## when the calculated gradient (with respect to leaf biomass, WL) is less than this threshold, it is effectively assumed to be zero and the allocation coefficient is set to zero
    gradient_threshold_WR: float = field(default=1e-7)  ## when the calculated gradient (with respect to root biomass, WR) is less than this threshold, it is effectively assumed to be zero and the allocation coefficient is set to zero

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
        airUhc,  ## wind speed at top-of-canopy, m s-1
        swskyb, ## Atmospheric direct beam solar radiation, W/m2
        swskyd, ## Atmospheric diffuse solar radiation, W/m2
        sza,    ## Solar zenith angle, degrees
        SAI,    ## Stem area index, m2 m-2
        CI,     ## Foliage clumping index, -
        hc,     ## Canopy height, m
        d_r,    ## Root depth, m
    ) -> Tuple[float]:

        ## Define functions to differentiate
        def GPPRm_WL(W_L_var):
            GPP, Rml, Rmr, *_ = self.PlantCH2O.calculate(
                W_L_var, W_R, soilTheta, leafTempC, airTempC, airRH,
                airCO2, airO2, airP, airUhc, swskyb, swskyd, sza,
                SAI, CI, hc, d_r
            )
            return GPP - (Rml + Rmr)

        def GPPRm_WR(W_R_var):
            GPP, Rml, Rmr, *_ = self.PlantCH2O.calculate(
                W_L, W_R_var, soilTheta, leafTempC, airTempC, airRH,
                airCO2, airO2, airP, airUhc, swskyb, swskyd, sza,
                SAI, CI, hc, d_r
            )
            return GPP - (Rml + Rmr)

        ## Compute gradients of net carbon profit with respect to (i) leaf biomass, and (ii) root biomass
        if self.gradient_method == "fd_forward":
            dGPPRmdWleaf = self.forward_finite_difference(GPPRm_WL, W_L, init_step=self.min_step_rel_WR*W_L)
            dGPPRmdWroot = self.forward_finite_difference(GPPRm_WR, W_R, init_step=self.min_step_rel_WR*W_R)
        elif self.gradient_method == "fd_adaptive":
            dGPPRmdWleaf = self.adaptive_finite_difference(GPPRm_WL, W_L, init_step=self.min_step_rel_WR*W_L, max_step=self.max_step_rel_WR*W_L)
            dGPPRmdWroot = self.adaptive_finite_difference(GPPRm_WR, W_R, init_step=self.min_step_rel_WR*W_R, max_step=self.max_step_rel_WR*W_R)
        elif self.gradient_method == "richardson":
            dGPPRmdWleaf = self.richardson_extrapolation(GPPRm_WL, W_L, step=self.min_step_rel_WL*W_L)
            dGPPRmdWroot = self.richardson_extrapolation(GPPRm_WR, W_R, step=self.min_step_rel_WR*W_R)
        
        ## Calculate marginal change in cost per unit change in biomass pool - proportional to pool instantaneous turnover rate (inverse of mean lifespan)
        dSdWleaf = self.tr_L
        dSdWroot = self.tr_R

        ## Calculate allocation coefficients
        if dGPPRmdWleaf <= self.gradient_threshold_WL:
            # If marginal gain is >= 0, set coefficient to zero to avoid division with zeros
            u_L_prime = 0
        else:
            u_L_prime = (dGPPRmdWleaf/dSdWleaf)/((dGPPRmdWleaf/dSdWleaf)+np.maximum(0,dGPPRmdWroot/dSdWroot))
        if dGPPRmdWroot <= self.gradient_threshold_WR:
            # If marginal gain is >= 0, set coefficient to zero to avoid division with zeros
            u_R_prime = 0
        else:
            u_R_prime = (dGPPRmdWroot/dSdWroot)/(np.maximum(0,dGPPRmdWleaf/dSdWleaf)+(dGPPRmdWroot/dSdWroot))

        ## Scale optimal allocation coefficients so that the sum of all allocation coefficients is equal to 1
        u_L = (1 - (self.u_Stem+self.u_Seed))*u_L_prime
        u_R = (1 - (self.u_Stem+self.u_Seed))*u_R_prime

        return u_L, u_R, dGPPRmdWleaf, dGPPRmdWroot, dSdWleaf, dSdWroot

    def forward_finite_difference(self, func, x, init_step=1e-2):
        """
        Compute the derivative of func at x using an forward difference method.
        """
        step = init_step
        f_0 = func(x)
        f_plus = func(x + step)
        gradient = (f_plus - f_0) / (step)
        return gradient  # Return the gradient

    def adaptive_finite_difference(self, func, x, init_step=1e-2, max_step=1e-1, tol=1e-6):
        """
        Compute the derivative of func at x using an adaptive central difference method.
        If the gradient is too small, increase the step size.
        """
        step = init_step  # Start with an initial step
        prev_gradient = None

        for _ in range(10):  # Limit number of adjustments
            f_plus = func(x + step)
            f_minus = func(x - step)
            gradient = (f_plus - f_minus) / (2 * step)

            # If the gradient is too small, increase the step size
            if prev_gradient is not None and np.abs(gradient) < tol:
                step *= 2  # Increase step size
            else:
                return gradient  # Return valid gradient

            prev_gradient = gradient

        print("Warning: Gradient may still be small.")
        return prev_gradient  # Return the last valid gradient

    def richardson_extrapolation(self, func, x, step=1e-4):
        """
        Compute the derivative of func at x using Richardson extrapolation.
        """
        f1 = (func(x + step) - func(x - step)) / (2 * step)  # Standard central difference
        f2 = (func(x + step/2) - func(x - step/2)) / (step)  # Smaller step size
        return (4 * f2 - f1) / 3  # Extrapolate toward zero step size