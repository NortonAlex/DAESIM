"""
Climate class: Includes module parameters and solar calculations to specify and initialise a site/grid-cell over at time-domain
"""

import numpy as np
from attrs import define, field
from daesim.climate_funcs import *

@define
class ClimateModule:
    """
    Climate (location specs, calendar, solar) module
    """

    ## Location/site details
    CLatDeg: float = field(
        default=-33.715
    )  ## latitude of site in degrees; Beltsville = 39.0; min = 64.; max = 20.
    Elevation: float = field(
        default=70.74206543
    )  ## IF there is sediment surface BELOW MSL (e.g., tidal creeks) then use the bathimetry data (depth below MSL) to determine elevation of sediments above the base datum.; ELSE use the land elevation data above MSL (indicating distance from MSL to the soil surface) plus the distance from the datum to mean sea level; ALL VALUES ARE POSITIVE (m) above base datum.
    degSlope: float = field(
        default=4.62
    )  ## angle of the slope of the site (degrees)
    slopeLength: float = field(
        default=97.2
    )  ## slope length, the distance from the point of origin of overland flow to the point where either the slope gradient decreases enough for deposition to start, or runoff waters are streamed into a channel
    iniSoilDepth: float = field(
        default=0.09
    )  ## initial soil depth (m)
    cellArea = 1  ## area of the unit considered m2


    ## Unit conversion factors
    rainConv: float = 0.001  ## conversion factor for mm/day to m/day
    T_K0: float = 273.15 ## conversion factor for degrees Celcius to Kelvin

    ## Constants
    L: float = 2.25e6 ## latent heat of vaporization (approx. 2250 kJ kg-1, temperature-dependent!)
    R_w_mol: float = 8.31446 ## specific gas constant for water vapor, J mol-1 K-1
    R_w_mass: float = 0.4615 ## specific gas constant for water vapor, J g-1 K-1 (=>R_w_mol * M_H2O = 8.31446/18.01)

    def time_discretisation(self, t, dt=1):
        """
        t  = array of consecutive time steps (days) e.g. a 2 year run with a 1-day time-step would require t=np.arange(1,2*365+1,1)
        dt = time step size (days). Default is dt=1 day. TODO: Convert all time dimension units to seconds (t, dt)

        """

        ## TODO: DayJul and DayJulPrev are really the "ordinal date" variables, not the Julian day. Rename them.
        DayJul = (
            t - dt
        ) % 365 + 1  # Modification: Changed this equation so that Jan 1st (UTC 00:00:00) is represented by 1 (not 0). December 31st (UTC 00:00:00) is represented by 365.
        DayJulPrev = (t - 2 * dt) % 365 + 1

        #         Climate_ampl = np.exp(7.42 + 0.045 * Climate_CLatDeg) / 3600   ## ErrorCheck: Where does this equation come from? Is it globally applicable?
        #         Climate_dayLength = Climate_ampl * np.sin((Climate_DayJul - 79) * 0.01721) + 12  ## ErrorCheck: This formulation seems odd. It doesn't return expected behaviour of a day-length calculator. E.g. it gives a shorter day length amplitude (annual min to annual max) at higher latitudes (e.g. -60o compared to -30o), it should be the other way around! I am going to replace it with my own solar calculations
        #         Climate_dayLengthPrev = Climate_ampl * np.sin((Climate_DayJulPrev - 79) * 0.01721) + 12

        dayLength = sunlight_duration(self.CLatDeg, DayJul - 1)
        dayLengthPrev = sunlight_duration(self.CLatDeg, DayJulPrev - 1)

        return (DayJul, DayJulPrev, dayLength, dayLengthPrev)

    def compute_mean_daily_air_temp(self,airTempMin,airTempMax):
        """
        Computes the actual vapor pressure from relative humidity and temperature.

        Parameters
        ----------
        airTempMin : scalar or ndarray
            Array containing daily minimum air temperature (degC).
        airTempMax : scalar or ndarray
            Array containing daily maximum air temperature (degC).

        Returns
        -------
        airTempC : scalar or ndarray (see dtype of parameters)
            Array of daily mean air temperature (degC)
        """
        return (airTempMax+airTempMin)/2

    def compute_absolute_humidity(self,airTempC,relativeHumidity):
        """
        Computes the absolute humidity. That is the actual mass of water vapor in a specified volume of air.

        $AH = \frac{RH \times e_s}{R_w \times T \times 100}$

        where: $RH$ is relative humidity (%), $e_s$ is the saturation vapor pressure of water (Pa), $R_w$ is the specific gas constant for water vapor, $T$ is the temperature (K) and $AH$ is the absolute humidity (g/m3)

        T = temperature (K)
        relativeHumidity = RH = relative humidity (%)
        e_s = saturation vapor pressure (Pa)
        R_w = specific gas constant for water vapor (J g-1 K-1)

        returns
        AH = absolute humidity (g m-3)
        """

        e_s = self.compute_sat_vapor_pressure(airTempC)
        airTempK = airTempC + self.T_K0
        AH = relativeHumidity*e_s/(self.R_w_mass*airTempK*100)  ## Modification: Slightly different formula than that used in Stella code, same result to within <0.1%
        return AH

    def compute_sat_vapor_pressure(self,T):
        """
        Computes the saturation vapor pressure using Tetens' formula

        T = air temperature (degC)
        e_s = saturation vapor pressure of water (Pa)
        """
        e_s = 1000 * 0.61078 * np.exp( (17.269*T) / (237.3+T) )  ## Modification: Slightly different formula than that used in Stella code, same result to within <0.1%
        return e_s

    def compute_relative_humidity(self,e_a,e_s):
        """
        Computes the relative humidity (RH) in units of percent (%).

        e_a = actual vapor pressure or density (units must be same as e_s, e.g. kPa)
        e_s = saturation vapor pressure or density (units must be same as e_a, e.g. kPa)

        """
        RH = e_a/e_s * 100
        return RH

    def compute_actual_vapor_pressure(self,T,RH):
        """
        Computes the actual vapor pressure from relative humidity and temperature.

        Parameters
        ----------
        T : scalar or ndarray
            Array containing air temperature (degC).
        RH : scalar or ndarray
            Array containing relative humidity (%).

        Returns
        -------
        e_a : scalar or ndarray (see dtype of parameters)
            Array of actual vapor pressure (Pa)
        """
        e_s = self.compute_sat_vapor_pressure(T)
        e_a = e_s * RH/100  ## Modification: This way of calculating e_a is correct but it differs to the formula used in Stella code
        return e_a

    def compute_VPD(self,T,RH):
        """
        Computes the vapor pressure deficit.

        Parameters
        ----------
        T : scalar or ndarray
            Array containing air temperature (degC).
        RH : scalar or ndarray
            Array containing relative humidity (%).

        Returns
        -------
        VPD : scalar or ndarray (see dtype of parameters)
            Array of vapor pressure deficit (Pa)
        """
        e_s = self.compute_sat_vapor_pressure(T)
        e_a = e_s * RH/100
        VPD = e_s - e_a
        return VPD

    def compute_Cloudy(self,precipM,vapPress):
        """
        Computes a cloudy factor. 
        Question: Unclear what this variable means and where the equation comes from. 

        Parameters
        ----------
        precipM : scalar or ndarray
            Array containing precipitation rate (m day-1).  ## TODO: change units to mm day-1 i.e. remove use of "precipM" everywhere
        vapPress : scalar or ndarray
            Array containing actual vapor pressure (Pa).

        Returns
        -------
        Cloudy : scalar or ndarray
            Array of cloudy factor (-)
        """
        def _func(p,v):
            if p > 0:
                return max(0, 10 - 1.155 * (v / (p * 1000 * 30)) ** 0.5)
            else:
                return 0

        _vfunc = np.vectorize(_func)
        Cloudy = _vfunc(precipM,vapPress)  ## TODO: change units to mm day-1 i.e. remove use of "precipM" everywhere
        return Cloudy
