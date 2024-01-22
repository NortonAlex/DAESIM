"""
Climate class: Includes module parameters and solar calculations to specify and initialise a site/grid-cell over at time-domain
"""

import numpy as np
from attrs import define, field
from datetime import datetime, date, timedelta
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
    CLonDeg: float = field(
        default=-76.922
    )  ## longitude of site in degrees
    timezone: float = field(
        default=-5
        ) ## Time zone in hours relative to UTC (positive to the East). Must be Local Standard Time – Daylight Savings Time is not used. 
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

    def time_discretisation(self, start_doy, start_year, nrundays=None, end_doy=None, end_year=None, dt=1):
        """
        Initialise the time dimension of a simulation. Calculates the consecutive simulation day (nday), 
        the ordinal day-of-year (doy) and the corresponding year (year). You must define either the 
        number of simulation days (nrundays) or the end date (by setting the ordinal day-of-year, end_doy 
        and year, end_year). 

        Parameters
        ----------
        start_doy: int or float
            Ordinal day-of-year of the start of the simulation period.

        start_year: int or float
            Year for the start of the simulation period

        nrundays: int or float
            Number of consecutive simulation days (e.g. if you want to run for 100 days, nrundays=100; if you want to run for two non-leap years, nrundays=730)

        end_doy: int or float
            Ordinal day-of-year of the end of the simulation period (inclusive).

        end_year: int or float
            Year for the end of the simulation period.

        dt: int or float
            Time step size (days). Default is 1 day. TODO: Convert all time dimension units to seconds (t, dt)

        Returns
        -------
        time_nday: array_like
            Array containing the consecutive simulation days.

        time_doy: array_like
            Array containing the ordinal day-of-year values.

        time_year: array_like
            Array containing the corresponding year values.

        """
        if (nrundays != None):
            start_date = datetime.strptime(str(int(start_year)) + "-" + str(int(start_doy)), "%Y-%j")
            date_list = [start_date + timedelta(days=x) for x in range(nrundays)]
            time_year = [d.year for d in date_list]
            time_doy = [float(d.strftime('%j')) for d in date_list]
        elif (nrundays == None) and (end_doy != None):
            start_date = datetime.strptime(str(int(start_year)) + "-" + str(int(start_doy)), "%Y-%j")
            end_date = datetime.strptime(str(int(end_year)) + "-" + str(int(end_doy)), "%Y-%j")
            date_list = []
            while start_date <= end_date:
                date_list.append(start_date)
                start_date += timedelta(days=1)    
            time_year = [d.year for d in date_list]
            time_doy = [float(d.strftime('%j')) for d in date_list]

        time_nday = np.arange(1, len(time_doy)+dt)

        return (time_nday, time_doy, time_year)

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

    def solar_day_calcs(self,year,doy):
        """
        Parameters
        ----------
        year: int or ndarray
            Year in format YYYY

        doy: float or ndarray
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        latitude: float or ndarray
            Latitude in degrees (north is positive)

        longitude: float or ndarray
            Longitude in degrees (east is positive)

        timezone: int or ndarray
            Time zone in hours relative to UTC (positive to the East). Must be Local Standard Time – Daylight Savings Time is not used. 

        Returns
        -------
        (sunrise_t, solarnoon_t, sunset_t): tuple of floats
            Returned values are the sunrise, solar noon, and sunset times given in units of 24 hour time (e.g. midday = 12)
            Note: This is given as the Local Standard Time – Daylight Savings Time is not used
        """
        _vfunc = np.vectorize(self._solar_day_calcs)
        (sunrise_t,solarnoon_t,sunset_t) = _vfunc(year,doy)
        return (sunrise_t,solarnoon_t,sunset_t)

    def _solar_day_calcs(self,year,doy):
        """
        Calculate sunrise, solar noon and sunset time based on equations from NOAA: http://www.srrb.noaa.gov/highlights/sunrise/calcdetails.html
        Also see Python code implementation code at: https://michelanders.blogspot.com/2010/12/calulating-sunrise-and-sunset-in-python.html

        Parameters
        ----------
        year: int
            Year in format YYYY

        doy: float
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        latitude: float
            Latitude in degrees (north is positive)

        longitude: float
            Longitude in degrees (east is positive)

        timezone: int
            Time zone in hours relative to UTC (positive to the East). Must be Local Standard Time – Daylight Savings Time is not used. 

        Returns
        -------
        (sunrise_t, solarnoon_t, sunset_t): tuple of floats
            Returned values are the sunrise, solar noon, and sunset times given in units of 24 hour time (e.g. midday = 12).
            Note: This is given as the Local Standard Time – Daylight Savings Time is not used
        """

        #print(year,type(year))
        #print(doy,type(doy))
        s = str(int(year))+str(int(doy))  # convert year and fractional doy to string format
        #print(s,type(s))
        try:
            xdate = datetime.strptime(s, '%Y%j').date()
        except ValueError:
            import pdb; pdb.set_trace()
        # xdate = datetime.strptime(s, '%Y%j').date()

        time = doy % 1 # time  # percentage past midnight, i.e. noon  is 0.5
        dt = xdate - date(1900,1,1)
        day = dt.days + 1  #day     # daynumber 1=1/1/1900

        Jday = day + 2415018.5 + time - self.timezone / 24  # Julian day
        Jcent = (Jday - 2451545) / 36525    # Julian century

        Manom = 357.52911 + Jcent * (35999.05029 - 0.0001537 * Jcent)
        Mlong = 280.46646 + Jcent * (36000.76983 + Jcent * 0.0003032) % 360
        Eccent = 0.016708634 - Jcent * (0.000042037 + 0.0001537 * Jcent)
        Mobliq = 23 + (26 + ((21.448 - Jcent * (46.815 + Jcent * \
                       (0.00059 - Jcent * 0.001813)))) / 60) / 60
        obliq = Mobliq + 0.00256 * np.cos(np.deg2rad(125.04 - 1934.136 * Jcent))
        vary = np.tan(np.deg2rad(obliq / 2)) * np.tan(np.deg2rad(obliq / 2))
        Seqcent = np.sin(np.deg2rad(Manom)) * (1.914602 - Jcent * (0.004817 + 0.000014 * Jcent)) + \
            np.sin(np.deg2rad(2 * Manom)) * (0.019993 - 0.000101 * Jcent) + np.sin(np.deg2rad(3 * Manom)) * 0.000289
        Struelong = Mlong + Seqcent
        Sapplong = Struelong - 0.00569 - 0.00478 * \
            np.sin(np.deg2rad(125.04 - 1934.136 * Jcent))
        declination = np.rad2deg(np.arcsin(np.sin(np.deg2rad(obliq)) * np.sin(np.deg2rad(Sapplong))))

        eqtime = 4 * np.rad2deg(vary * np.sin(2 * np.deg2rad(Mlong)) - 2 * Eccent * np.sin(np.deg2rad(Manom)) + 4 * Eccent * vary * np.sin(np.deg2rad(Manom))
                         * np.cos(2 * np.deg2rad(Mlong)) - 0.5 * vary * vary * np.sin(4 * np.deg2rad(Mlong)) - 1.25 * Eccent * Eccent * np.sin(2 * np.deg2rad(Manom)))

        hourangle = np.rad2deg(np.arccos(np.cos(np.deg2rad(90.833)) /
                             (np.cos(np.deg2rad(self.CLatDeg)) *
                              np.cos(np.deg2rad(declination))) -
                             np.tan(np.deg2rad(self.CLatDeg)) *
                             np.tan(np.deg2rad(declination))))

        solarnoon_t = (
            720 - 4 * self.CLonDeg - eqtime + self.timezone * 60) / 1440
        sunrise_t = solarnoon_t - hourangle * 4 / 1440
        sunset_t = solarnoon_t + hourangle * 4 / 1440

        return (24*sunrise_t,24*solarnoon_t,24*sunset_t)


    def sunlight_duration(self,year,doy):
        """
        Sunlight duration (hours of daylight between sunrise and sunset) for the given day-of-year and location.

        Parameters
        ----------
        year: int or array_like
            Year in format YYYY

        doy: float or array_like
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        latitude: float or array_like
            Latitude in degrees (north is positive)

        longitude: float or array_like
            Longitude in degrees (east is positive)

        timezone: int or array_like
            Time zone in hours relative to UTC (positive to the East). Must be Local Standard Time – Daylight Savings Time is not used. 

        Returns
        -------
        sunlight_duration_hrs : scalar or ndarray (see dtype of parameters)
            Array of sunlight duration in hours.

        """
        sunrise_t,solarnoon_t,sunset_t = self.solar_day_calcs(year,doy)
        sunlight_duration_hrs = sunset_t-sunrise_t
        return sunlight_duration_hrs
