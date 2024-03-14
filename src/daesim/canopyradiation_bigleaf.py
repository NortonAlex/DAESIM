"""
Canopy radiative transfer model class: Includes equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp, quad
from daesim.climate import ClimateModule
from daesim.climate_funcs import solar_day_calcs

@define
class CanopyRadiation:
    """
    Calculator of canopy radiative transfer including absorption of photosynthetically active radiation
    """

    # Class parameters

    LAD_type: str = field(default="spherical")  ## Leaf angle distribution function type. One of 'spherical', 'planophile', 'erectophile', 'plagiophile', 'extremophile', 'uniform'.

    rho: float = field(default=0.12)  ## canopy/leaf reflectance coefficient

    J_to_umol: float = field(default=4.6)  ## Conversion factor of shortwave irradiance (W/m2) to PPFD (umol photons/m2/s) (umol/J)

    def calculate(
        self,
        S,    ## Incoming solar irradiance, units (TODO: add units)
        L,    ## Leaf area index, m2 m-2
        theta,    ## Solar zenith angle, radians
        Site=ClimateModule(),   ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.
    ) -> Tuple[float]:

        ## Canopy gap fraction
        P = self.canopy_gap_fraction(theta,L,LAD_type=self.LAD_type)

        ## Absorbed irradiance
        Sabs = S * self.canopy_fraction_absorbed_irradiance(theta,L,LAD_type=self.LAD_type)

        return Sabs


    def g_L(self,theta_L,LAD_type='spherical'):
        """
        Function to characterise the leaf angle distribution based on de Wit (1965). 
        Provides a mathematical approach to "characterize the positions of the leaves 
        of a canopy only by the cumulative frequency distribution of the inclination 
        of the leaves" (de Wit, 1965). 
        All distribution types are assumed independent of the azimuthal direction i.e. 
        the azimuthal distribution of the leaves is random.

        Parameters
        ----------
        theta_L: float
            Leaf inclination angle (radians). Leaf surface normal relative to zenith direction. 
            Ranges from 0 degrees for a horizontal leaf to 90 degree for a vertical one.
        LAD_type: str
            Leaf angle distribution function type. One of 'spherical', 'planophile', 'erectophile', 'plagiophile', 'extremophile', 'uniform'.

        Returns
        -------
        Probability density for occurrence of leaf inclination angle (leaf angle distribution function). 

        Notes
        -----
        On the LAD_type 'spherical': The relative frequency of leaf angle is the same as for surface elements of a sphere.
        On the LAD_type 'planophile': Horizontal leaves are most frequent
        On the LAD_type 'erectophile': Vertical leaves are most frequent
        On the LAD_type 'plagiophile': Oblique leaves are most frequent
        On the LAD_type 'extremophile': Oblique leaves are least frequent
        On the LAD_type 'uniform': Proportion of leaf angle is the same at any angle

        The equations for 'planophile' and 'erectophile' used here are swapped around compared to those reported in 
        Wang et al. (2007). It is believed that the Wang et al. (2007) equations 15 and 16 should be swapped around. 
        Doing so produces the expected leaf angle distribution function that matches Figure 4a in de Wit (1965). 

        References
        ----------
        de Wit, C.T., 1965. Photosynthesis of Leaf Canopies: Centre for Agricultural Publications and Documentation.
        Wang et al., 2007, https://doi.org/10.1016/j.agrformet.2006.12.003
        
        """
        if LAD_type == 'spherical':
            g_L = np.sin(theta_L)
        elif LAD_type == 'planophile':
            g_L = (2/np.pi)*(1+np.cos(2*theta_L))
        elif LAD_type == 'erectophile':
            g_L = (2/np.pi)*(1-np.cos(2*theta_L))
        elif LAD_type == 'plagiophile':
            g_L = (2/np.pi)*(1-np.cos(4*theta_L))
        elif LAD_type == 'extremophile':
            g_L = (2/np.pi)*(1+np.cos(4*theta_L))
        elif LAD_type == 'uniform':
            g_L = 2/np.pi
        
        return g_L

    def A(self,theta, theta_L):
        """
        Calculate the A(theta, theta_L) based on the given condition.

        Parameters
        ----------
        theta: float
            Solar zenith angle (radians)
        
        theta_L: float
            Leaf inclination angle (radians). Leaf surface normal relative to zenith direction. 
            Ranges from 0 degrees for a horizontal leaf to 90 degree for a vertical one.

        Returns
        -------
        Calculated A(theta, theta_L) value.
        """
        
        # Handling edge cases where the zenith angle is at the extreme values (0 or 90 degrees) 
        # without encountering mathematical issues like division by zero or undefined trigonometric functions
        if np.isclose(theta, 0):  # Directly overhead (zenith)
            return np.cos(theta)*np.cos(theta_L)   # Simplified assumption
        elif np.isclose(theta, np.pi/2):  # Horizontal (90 degrees)
            return 0   # No direct sunlight projection for horizontal light
        elif np.isclose(theta_L, np.pi/2):
            theta_L = np.pi/2 - 1e-8   # set theta_L to a fraction less than 90 degrees to prevent undefined value
            cot_theta = 1 / np.tan(theta)
            cot_theta_L = 1 / np.tan(theta_L)
            psi = np.arccos(cot_theta * cot_theta_L)
            return np.cos(theta) * np.cos(theta_L) * (1 + (2 / np.pi) * (np.tan(psi) - psi))

        cot_theta = 1 / np.tan(theta)
        cot_theta_L = 1 / np.tan(theta_L)
        
        if abs(cot_theta * cot_theta_L) > 1:
            return np.cos(theta) * np.cos(theta_L)
        else:
            psi = np.arccos(cot_theta * cot_theta_L)
            return np.cos(theta) * np.cos(theta_L) * (1 + (2 / np.pi) * (np.tan(psi) - psi))

    def G(self,theta,LAD_type="spherical"):
        """
        Calculate the fraction of leaf area projected in the direction of the sun.
        It represents a dimensionless geometry factor for "the leaf area projected 
        to the direction theta by a unit leaf area in the canopy" on a plane perpendicular
        to the direction, "whose distribution function of leaf normal orientation is 
        specified by" g_L (Myneni et al., 1989). This assumes the canopy is symmetrical 
        in the azimuth direction i.e. the azimuthal distribution of the leaves is random.

        Parameters
        ----------
        theta_L: float
            Leaf inclination angle (radians). Leaf surface normal relative to zenith direction. 
            Ranges from 0 degrees for a horizontal leaf to 90 degree for a vertical one.
        LAD_type: str
            Leaf angle distribution function type. One of 'spherical', 'planophile', 'erectophile', 'plagiophile', 'extremophile', 'uniform'.

        Returns
        -------
        Calculated G(theta) value. 

        Notes
        -----
        Leaf azimuth angle is assumed to be random. 

        References
        ----------
        Myneni et al., 1989, A Review of the Theory of Photon Transport in Leaf Canopies, Ag. and Forest Met.
        Pisek et al., 2011, doi: 10.1007/s00468-011-0566-6
        Stenberg et al., 2006, A note on the G-function for needle leaf canopies, doi: 10.1016/j.agrformet.2006.01.009
        """
            
        def integrand(theta_L):
            return self.A(theta, theta_L) * self.g_L(theta_L,LAD_type=LAD_type)

        # Integrate over theta_L from 0 to pi/2
        result, _ = quad(integrand, 0, np.pi/2)
        return result

    def canopy_gap_fraction(self,theta,L,LAD_type="spherical"):
        """
        The probability of the transmission of a beam of light through the canopy (P) with a defined
        dispersion of the infinitesimal size of the leaves as described by a ‘Beer-Lambert law’ function.

        Parameters
        ----------
        theta: float
            view zenith angle (radians)
        L: float
            cumulative leaf area index from top-of-canopy downward (m2 m-2). At the top-of-canopy, L=0.
        LAD_type: str
            Leaf angle distribution function type. One of 'spherical', 'planophile', 'erectophile', 'plagiophile', 'extremophile', 'uniform'.

        Returns
        -------
        Gap fraction probability: float
       
        """
        G_theta = self.G(theta,LAD_type=LAD_type)
        return np.exp(-G_theta*L/np.cos(theta))

    def canopy_fraction_absorbed_irradiance(self,theta,L,LAD_type="spherical"):
        """
        The probability of the transmission of a beam of light through the canopy (P) with a defined
        dispersion of the infinitesimal size of the leaves as described by a ‘Beer-Lambert law’ function.

        Parameters
        ----------
        P: float
            gap fraction probability
        rho: float
            canopy/leaf reflectance coefficient

        Returns
        -------
        Fraction of absorbed irradiance: float
            Fraction of incident radiation absorbed, accounting for some proportion that is reflected. 
        
        """
        P = self.canopy_gap_fraction(theta,L,LAD_type=LAD_type)
        return (1-self.rho)*(1-P)