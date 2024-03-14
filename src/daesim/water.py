"""
Water class: Includes module parameters and calculations involving the water cycle.
"""

import numpy as np
from attrs import define, field
from daesim.climate_funcs import *
from daesim.climate import ClimateModule

@define
class WaterModule:
    """
    Calculator, parameters and methods for the water cycle.
    """

    ## Module class parameters

    def calculate_PenmanMonteith_ET0_FAO56(self,year,doy,T,Tmin,Tmax,RH,u_2,P,G,albedo,fsunhrs,Site):
        """
        Penman-Monteith equation for potential evapotranspiration assuming simplified parameters 
        that represent a hypothetical grass reference crop with fixed aerodynamic and canopy properties. 

        Parameters
        ----------
        year: int or ndarray
            Year in format YYYY

        doy: float or ndarray
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        T: float or ndarray
            Mean daily air temperature (degrees Celsius)

        Tmin: float or ndarray
            Minimum daily air temperature (degrees Celsius)
        
        Tmax: float or ndarray
            Maximum daily air temperature (degrees Celsius)

        RH: float or ndarray
            Relative humidity (%)

        u_2: float or ndarray
            Wind speed at 2 m height (m s-1)

        P: float or ndarray
            Atmospheric pressure (Pa)

        G: float or ndarray
            Ground heat flux (MJ m-2 d-1)

        albedo: float or ndarray
            albedo or canopoy reflection coefficient (-)

        fsunhrs: float or ndarray
            Fraction of sunshine within the day (hours), i.e. number of hours without cloud or overcast conditions divided by total daylight hours; fsunhrs=1 on a completely clear-sky day

        Site: Class
            ClimateModule class which defines constants and location-specific attributes

        Returns
        -------
        ET0: float or ndarray
            Potential evapotranspiration (mm d-1)

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        
        Also see: Zotarelli et al., 2009, AE459: Step by Step Calculation of the Penman-Monteith Evapotranspiration 
        (FAO-56 Method), University of Florida
        """
        # calculate net radiation using simplified
        Rnet = Site.calculate_radiation_net(year,doy,Tmin,Tmax,RH,albedo,fsunhrs)
        VPD_mean = 1e-3*Site.compute_VPD_daily(Tmin,Tmax,RH)
        gamma = Site.compute_psychometric_constant(P)
        Delta = Site.compute_slope_sat_vapor_press_curve(T)
        ET0 = ( (1/(Site.L/1000)) * Delta * (Rnet - G) + gamma * 900 / (T+273) * u_2 * (VPD_mean) )/( Delta + gamma*(1 + 0.34*u_2))
        return ET0

    def calculate_Hargreaves_ET0(self,year,doy,T,Tmin,Tmax,Site):
        """
        Penman-Monteith equation for potential evapotranspiration assuming simplified parameters to represent a 
        hypothetical grass reference crop. 

        Parameters
        ----------
        year: int or ndarray
            Year in format YYYY

        doy: float or ndarray
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        T: float or ndarray
            Mean daily air temperature (degrees Celsius)

        Tmin: float or ndarray
            Minimum daily air temperature (degrees Celsius)
        
        Tmax: float or ndarray
            Maximum daily air temperature (degrees Celsius)

        Site: Class
            ClimateModule class which defines constants and location-specific attributes

        Returns
        -------
        ET0: float or ndarray
            Potential evapotranspiration (mm d-1)

        References
        ----------
        Hargreaves and Samani, Reference Crop Evapotranspiration from Temperature, Applied Engineering in Agriculture. 
        1(2): 96-99, 1985, doi:10.13031/2013.26773
        
        Hargreaves and Allen, History and Evaluation of the Hargreaves Evapotranspiration Equation, Journal of 
        Irrigation and Drainage Engineering, 129(1), 2003, doi:10.1061/(ASCE)0733-9437(2003)129:1(53)
        
        """
        Ra = Site.calculate_solarradiation_Ra(year,doy)
        ET0 = (1/(Site.L/1000)) * (0.0023 * Ra * (T + 17.8) * np.sqrt((Tmax-Tmin)))
        return ET0

    def calculate_PenmanMonteith_ET(self,year,doy,T,Tmin,Tmax,RH,u_2,P,G,albedo,fsunhrs,r_a,r_s,Site):
        """
        Penman-Monteith equation for potential evapotranspiration assuming simplified parameters 
        that represent a hypothetical grass reference crop with fixed aerodynamic and canopy properties. 

        Parameters
        ----------
        year: int or ndarray
            Year in format YYYY

        doy: float or ndarray
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        T: float or ndarray
            Mean daily air temperature (degrees Celsius)

        Tmin: float or ndarray
            Minimum daily air temperature (degrees Celsius)
        
        Tmax: float or ndarray
            Maximum daily air temperature (degrees Celsius)

        RH: float or ndarray
            Relative humidity (%)

        u_2: float or ndarray
            Wind speed at 2 m height (m s-1)

        P: float or ndarray
            Atmospheric pressure (Pa)

        G: float or ndarray
            Ground heat flux (MJ m-2 d-1)

        albedo: float or ndarray
            albedo or canopoy reflection coefficient (-)

        fsunhrs: float or ndarray
            Fraction of sunshine within the day (hours), i.e. number of hours without cloud or overcast conditions divided by total daylight hours; fsunhrs=1 on a completely clear-sky day

        r_a: float or ndarray
            Aerodynamic resistance (s m-1)

        r_s: float or ndarray
            Surface (bulk canopy) resistance (s m-1)

        Site: Class
            ClimateModule class which defines constants and location-specific attributes

        Returns
        -------
        ET: float or ndarray
            Actual evapotranspiration (mm d-1)

        Notes
        -----
        The Penman-Monteith equation is a single-layer model where the resistances for vegetation and soil are 
        assumed to reside in parallel. 

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        
        Also see: Zotarelli et al., 2009, AE459: Step by Step Calculation of the Penman-Monteith Evapotranspiration 
        (FAO-56 Method), University of Florida
        """
        # calculate net radiation using simplified, temperature-based method
        Rnet = Site.calculate_radiation_net(year,doy,Tmin,Tmax,RH,albedo,fsunhrs)
        VPD_mean = 1e-3*Site.compute_VPD_daily(Tmin,Tmax,RH)
        gamma = Site.compute_psychometric_constant(P)
        Delta = Site.compute_slope_sat_vapor_press_curve(T)
        ET = (1/(Site.L/1e3))*(Delta * (Rnet - G) + Site.rho_air * Site.cp_air * ((VPD_mean)/r_a))/(Delta + gamma*(1 + r_s/r_a))
        return ET
