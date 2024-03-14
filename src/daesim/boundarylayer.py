"""
Boundary layer class: Includes module parameters calculations to determine boundary layer characteristics.
"""

import numpy as np
from attrs import define, field
from datetime import datetime, date, timedelta
from daesim.climate_funcs import *

@define
class BoundaryLayerModule:
    """
    Boundary layer module
    """

    ## Class parameters
    kappa: float = field(default=0.41)  ## von Karman's constant (-)
    beta: float = field(default=0.5)  ## empirical coefficient for the exponential decay model (-)

    def calculate_wind_profile_exp_L(self, Uh, L_cumulative, L_total):
        """
        Calculate the wind speed at a specific layer within the canopy using the exponential decay model 
        assuming an exponential decay model based on the cumulative plant area index from the top of the
        canopy. 
        
        Parameters
        ----------
        Uh: float
            wind speed at canopy top (m s-1)
        L_cumulative: float
            cumulative plant area index from top of canopy to a specific canopy layer (m2 m-2)
        L_total: float
            plant area index of canopy (m2 m-2)
        beta: float
            empirical coefficient for the exponential decay model (-)
        
        Returns
        -------
        u_L: float
            wind speed at specified canopy layer (m s-1)

        Notes
        -----
        Assume beta is a given constant for simplicity, adjust based on empirical data or literature as necessary.
        """
        u_L = Uh * np.exp(-self.beta * (L_cumulative / L_total))
        return u_L

    def calculate_wind_profile_exp(self, Uh, dpai, ntop):
        """
        Calculate the wind speed profile over the defined canopy layers based on the wind speed at the canopy top and profile of plant area index.
        
        Parameters
        ----------
        Uh: float
            wind speed at canopy top (m s-1)
        dpai: array-like
            array of plant area index per canopy layer (m2 m-2)
        ntop: int
            index of the top of the canopy
        beta: float
            empirical coefficient for the exponential decay model (-)
        
        Returns
        -------
        u_z: array-like 
            wind speed profile across the canopy layers (m s-1)
        """
        # assuming the number of canopy layers is equal to the size of dpai input
        nlevmlcan = dpai.size
        
        # Determine the order of the PAI array based on the top of the canopy
        if ntop == 0:
            L = dpai
        else:
            L = dpai[::-1]
        
        # Calculate total canopy PAI
        Ltotal = dpai.sum()
        
        # Initialize the wind speed profile array
        _u_z = np.zeros(nlevmlcan)
        
        # Calculate the wind speed for each layer
        for ic in range(nlevmlcan):
            _u_z[ic] = self.calculate_wind_profile_exp_L(Uh, L[:ic+1].sum(), Ltotal)
        
        # Adjust the wind speed profile if the canopy top is not the first index
        if ntop != 0:
            _u_z = _u_z[::-1]
        
        return _u_z

    def calculate_R_ustar_Uh(self,PAI, C_R=0.3, C_S=0.003, R_ustar_Uh_max=0.3):
        """
        Calculate the ratio of friction velocity to mean velocity at height h.

        Parameters
        ----------
        PAI: float
            plant area index (m2 m-2)
        C_R: float
            drag coefficient of an isolated roughness element mounted on the surface (-)
        C_S: float
            substrate-surface drag coefficient (-)
        R_ustar_Uh_max: float
            maximum ratio of friction velocity (ustar) to mean velocity at height h (Uh) (-)

        Returns
        -------
        R_ustar_Uh: float
            ratio of friction velocity (ustar) to mean velocity at height h (Uh) (-)
        
        References
        ----------
        Raupach, Simplified expressions for vegetation roughness length and zero-plane displacement 
        as functions of canopy height and area index, Boundary-Layer Meteorology, 71, p. 211-216, 1994.
        """
        return min((C_S + C_R*PAI/2)**0.5, R_ustar_Uh_max)

    def calculate_R_d_h(self,PAI, c_d1=7.5):
        """
        Calculate the ratio of zero-plane displacement height (d) to canopy height (h)

        Parameters
        ----------
        PAI: float
            plant area index (m2 m-2)
        c_d1: float
            empirical parameter for relation between displacement height and canopy height (-)

        Returns
        -------
        R_d_h: float
            ratio of zero-plane displacement height (d) to canopy height (h) (-)

        References
        ----------
        Raupach, Simplified expressions for vegetation roughness length and zero-plane displacement 
        as functions of canopy height and area index, Boundary-Layer Meteorology, 71, p. 211-216, 1994.
        """
        return 1 - (1 - np.exp(-np.sqrt(c_d1*PAI)))/(np.sqrt(c_d1*PAI))

    def calculate_R_z0_h(self,PAI, R_ustar_Uh, psi_h=0.193):
        """
        Calculate the ratio of roughness length (z0) to canopy height

        Parameters
        ----------
        PAI: float
            plant area index (m2 m-2)
        R_ustar_Uh: float
            ratio of friction velocity (ustar) to mean velocity at height h (Uh) (-)
        kappa: float
            Von Kármán constant (-)
        psi_h: float
            roughness-sublayer influence function, describing the departure of the velocity profile just 
            above the roughness from the inertial-sublayer logarithmic law (-)

        Returns
        -------
        R_z0_h: float
            ratio of roughness length (z0) to canopy height (h) (-)

        References
        ----------
        Raupach, Simplified expressions for vegetation roughness length and zero-plane displacement 
        as functions of canopy height and area index, Boundary-Layer Meteorology, 71, p. 211-216, 1994.
        """
        return (1 - self.calculate_R_d_h(PAI)) * np.exp(-self.kappa * 1/R_ustar_Uh - psi_h)

    def calculate_z0_and_d(self,h, PAI):
        """
        Calculate roughness length (z0) and zero-plane displacement height (d) based on canopy height and 
        plant area index (PAI)

        Parameters
        ----------
        h: float
            canopy height (m)
        PAI: float
            plant area index (m2 m-2)

        Returns
        -------
        R_z0_d: float
            ratio of roughness length (z0) to zero-plane displacement height (h) (-)

        References
        ----------
        Raupach, Simplified expressions for vegetation roughness length and zero-plane displacement 
        as functions of canopy height and area index, Boundary-Layer Meteorology, 71, p. 211-216, 1994.
        """
        R_ustar_Uh = self.calculate_R_ustar_Uh(PAI)
        R_d_h = self.calculate_R_d_h(PAI)
        R_z0_h = self.calculate_R_z0_h(PAI, R_ustar_Uh)
        d = R_d_h * h
        z0 = R_z0_h * h
        return z0, d


    def estimate_wind_profile_log_conditional(self,u_z_meas, z_meas, z, d, z0):
        """
        Estimate the wind speed at any height z using a logarithmic wind speed profile

        Parameters
        ----------
        u_z_meas: float
            Measured wind speed at height z_meas (m s-1)
        z_meas: float
            Height at which the wind speed is measured (m)
        z: float
            Height at which to estimate the wind speed (m)
        d: float
            Zero-plane displacement height (m)
        z0: float
            Roughness length (m)
        kappa: float
            Von Kármán constant (-)

        Returns
        -------
        u_z: float
            Estimated wind speed at height z (m s-1)
        """
        if (z-d) <= z0:
            return 0
        # Calculate friction velocity u_*
        u_star = u_z_meas * self.kappa / np.log((z_meas - d) / z0)
        # Estimate wind speed at height z
        u_z = u_star / self.kappa * np.log((z - d) / z0)
        return u_z

    def estimate_wind_profile_log(self,u_z_meas, z_meas, z, d, z0):
        """
        Estimate the wind speed at any height z using a logarithmic wind speed profile

        Parameters
        ----------
        u_z_meas: float
            Measured wind speed at height z_meas (m s-1)
        z_meas: float
            Height at which the wind speed is measured (m)
        z: float or array_like
            Height at which to estimate the wind speed (m)
        d: float
            Zero-plane displacement height (m)
        z0: float
            Roughness length (m)
        kappa: float
            Von Kármán constant (-)

        Returns
        -------
        u_z: float
            Estimated wind speed at height z (m s-1)
        """
        _vfunc = np.vectorize(self.estimate_wind_profile_log_conditional,otypes=[float])
        u_z = _vfunc(u_z_meas,z_meas,z,d,z0)
        return u_z

    def calculate_leaf_boundary_resistance(self,T,p,u,d,Site):
        """
        Calculates the leaf boundary layer layer resistance for water vapour.

        Parameters
        ----------
        T: float
            air temperature (degrees Celsius)
        p: float
            atmospheric pressure (Pa)
        u: float
            wind speed (m s-1)
        d: float
            mean length of the leaf in the direction of the wind (m), typically ranges 0.001-0.030
        Site: Class
            ClimateModule class which defines constants and location-specific attributes

        Returns
        -------
        r_bl: float
            leaf boundary layer resistance (s m-1)

        References
        ----------
        Nobel (2009) Chapters 7, 8, doi:10.1016/B978-0-12-374143-1.X0001-4, ISBN:978-0-12-374143-1
        """
        #D_H2O_std = 2.42e-5  ## reference binary diffusion coefficient for water vapor in air at reference temperature and std pressure (m2 s-1)
        #D_H2O_Tref = 20.0    ## reference temperature (degrees Celsius) for the reference binary diffusion coefficient for water vapor in air
        delta_bl = 0.0040 * np.sqrt(d/u)  ## boundary layer thickness (m), see Nobel (2009) p. 337, Eq. 7.10
        D_H2O = Site.diffusion_coefficient_powerlaw(Site.D_H2O_T20,T+273.15,p,Tref=273.15+20.0)  ## diffusion coefficient for water vapor in air
        r_bl = delta_bl/D_H2O   ## boundary layer resistance (s m-1)
        return r_bl

    def calculate_aerodynamic_resistance(self,u_z,z_meas,h,k=0.41):
        """
        Calculates the aerodynamic resistance by applying the Monin-Obukhov similarity theory
        assuming neutral atmospheric stability conditions. 

        Parameters
        ----------
        u_z: float
            wind speed at specified height (m s-1)
            
        z_meas: float
            measurement height for wind speed (m)
            
        h: float
            canopy height (m)
    
        k: float
            von Karman's constant (-)
    
        Returns
        -------
        r_a: float
            aerodynamic resistance (s m-1)
    
        Notes
        -----
        This assumes that the measurement height for humidity measurements is the same as those for wind speed. 
        See details on p. 20-21 in Allen et al. (1998)
        """
        d = h*2/3  ## zero-plane displacement height (m)
        if z_meas <= d:
            raise ValueError("Specified meteorological measurement height (z_meas=%1.1f) must be greater than zero-plane displacement height (d=h*2/3=%1.1f)" % (z_meas,d))
        z_om = 0.123 * h  ## roughness length governing momentum transfer (m)
        z_oh = 0.1 * z_om  ## roughness length governing heat and vapor transfer (m)
        if (z_meas - d)/z_om <= 1:
            raise ValueError("The ratio (z_meas - d):z_om must be greater than 1. It equals %1.2f. Check defined meteorological measurement height (z_meas=%1.1fm) and canopy height (h=%1.1fm)" % ((z_meas - d)/z_om,z_meas,h))
        elif (z_meas - d)/z_oh <= 1:
            raise ValueError("The ratio (z_meas - d):z_oh must be greater than 1. It equals %1.2f. Check defined meteorological measurement height (z_meas=%1.1fm) and canopy height (h=%1.1fm)" % ((z_meas - d)/z_oh,z_meas,h))
        r_a = (np.log((z_meas - d)/z_om) * np.log((z_meas - d)/z_oh))/(k**2 * u_z)
        return r_a
    
    def calculate_surface_resistance(self,r_1,LAI_active):
        """
        Calculates the surface (canopy) resistance using a simple empirical approach outlined by 
        Allen et al. (1998) 
        
        Parameters
        ----------
        r_1: float
            bulk stomatal resistance of the well-illuminated leaf (s m-1)
            
        LAI_active: float
            active (sunlit) leaf area index (m2 m-2)
    
        Returns
        -------
        r_s: float
            (bulk) surface resistance (s m-1)
        
        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        """
        r_s = r_1 / LAI_active
        return r_s