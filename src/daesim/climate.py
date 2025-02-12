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
    )  ## latitude of site (degrees)
    CLonDeg: float = field(
        default=-76.922
    )  ## longitude of site (degrees)
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
    cellArea = 1  ## area of the unit considered (m2)
    met_z_meas: float = field(default=10.0)   ## Measurement height for wind speed forcing (m)


    ## Unit conversion factors
    rainConv: float = 0.001  ## conversion factor for mm/day to m/day
    T_K0: float = 273.15 ## conversion factor for degrees Celsius to Kelvin

    ## Constants
    g: float = 9.80665 ## acceleration due to gravity (m s-2)
    L: float = 2450 ## latent heat of vaporization (J g-1) (technically this is temperature-dependent)
    R_w_mol: float = 8.31446 ## specific gas constant for water vapor (J mol-1 K-1)
    R_w_mass: float = 0.4615 ## specific gas constant for water vapor (J g-1 K-1) (=>R_w_mol * M_H2O = 8.31446/18.01)
    MW_ratio_H2O: float = 0.622  ## ratio molecular weight of water vapor to dry air
    rho_air: float = 1.293  ## mean dry air density at constant pressure (kg m-3)
    cp_air: float = 1.013 ## specific heat of air at constant pressure (J g-1 K-1)
    StefanBoltzmannConstant: float = 5.6704e-8  ## Stefan-Boltzmann constant (W m-2 K-4)
    S0_Wm2: float = 1370  ## Solar constant (W m-2), incoming solar radiation at the top of Earth's atmosphere
    S0_MJm2min: float = 0.0822  ## Solar constant (MJ m-2 min-1), incoming solar radiation at the top of Earth's atmosphere

    airPressSeaLevel: float = 101325 # Sea-level standard atmospheric pressure (Pa)
    TempSeaLevel: float = 288.15 ## Standard sea-level temperature (K)
    tropos_lapse_rate: float = 0.0065 ## Temperature lapse rate in lower troposphere (K m-1)
    M: float = 0.0289644 ## Molar mass of Earth's air (kg/mol).

    D_H2O_T20: float = 2.42e-5  ## diffusion coefficient for water vapor at 20 degrees Celsius (m2 s-1)
    D_CO2_T20: float = 1.51e-5  ## diffusion coefficient for carbon dioxide at 20 degrees Celsius (m2 s-1)

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

    def validate_event_dates(self, event_dates, forcing_index, event_name):
        """
        Validates that all event dates are within the range of the forcing data time index.
        
        Parameters
        ----------
        event_dates (list of pd.Timestamp): List of sowing or harvest dates.
        forcing_index (pd.DatetimeIndex): Forcing data time index.
        event_name (str): 'Sowing' or 'Harvest' for error messages.
        
        Raises
        ------
        ValueError: If any event date is outside the forcing data range.
        """
        min_forcing_date = forcing_index.min()
        max_forcing_date = forcing_index.max()
        
        for date in event_dates:
            if not (min_forcing_date <= date <= max_forcing_date):
                raise ValueError(
                    f"{event_name} date {date} is outside the forcing data range "
                    f"({min_forcing_date} to {max_forcing_date})."
                )

    def find_event_steps(self, event_dates, time_index):
        """
        Find the closest time steps in time_index (DatetimeIndex array) for a list of event dates.

        Returns
        -------
        Indexes of event dates for the given time index array
        """
        event_steps = []
        for date in event_dates:
            try:
                step = time_index.get_loc(date)
            except KeyError:
                step = (time_index >= date).argmax()
            event_steps.append(step)
        return event_steps

    def time_index_growing_season(self, time_index, idevphase_array, Management, PlantDev, iseason=0):
        """
        Support for post-processing model outputand time indexing

        Parameters
        ----------
        time_index : DatetimeIndex array
            DatetimeIndex array for the model simulation time period i.e. time_axis
        idevphase_array : np.array
            Numpy array of the developmental phase index values over the model simulation period
        Management : class
            DAESIM2 Management class, which must include the attributes sowingDays, sowingYears, harvestDays and harvestYears
        PlantDev : class
            DAESIM2 PlantDev class, which must include the attribute phases
        iseason : int
            Index for selected growing season, if there is more than one growing season simulated, you must select which one to get time indexes for

        Returns
        -------
        itax_sowing, itax_harvest, itax_phase_transitions
            Time array indexes for sowing, harvest, and phase transition events
        """
        # ngrowing_seasons = (len(Management.sowingDays) if (isinstance(Management.sowingDays, int) == False) else 1)
        # if ngrowing_seasons > 1:
        #     print("Multiple sowing and harvest events occur. Returning results for growing season %d" % (iseason+1))
        # else:
        #     print("Just one sowing event and one harvest event occurs. Returning results for first (and only) growing season.")

        sowingDay, sowingYear = Management.sowingDays[iseason], Management.sowingYears[iseason]
        harvestDay, harvestYear = Management.harvestDays[iseason], Management.harvestYears[iseason]

        sowingTimestamp = datetime(int(sowingYear), 1, 1) + timedelta(days=int(sowingDay) - 1)
        harvestTimestamp = datetime(int(harvestYear), 1, 1) + timedelta(days=int(harvestDay) - 1)

        # Index for sowing and harvest dates along the time_axis
        itax_sowing, itax_harvest = self.find_event_steps([sowingTimestamp, harvestTimestamp], time_index)

        # Developmental phase index array, handling mixed int and float types
        valid_mask = ~np.isnan(idevphase_array)

        # Identify all transitions (number-to-NaN, NaN-to-number, or number-to-different-number)
        itax_phase_transitions = np.where(
            ~valid_mask[:-1] & valid_mask[1:] |  # NaN-to-number
            valid_mask[:-1] & ~valid_mask[1:] |  # Number-to-NaN
            (valid_mask[:-1] & valid_mask[1:] & (np.diff(idevphase_array) != 0))  # Number-to-different-number
        )[0] + 1

        # Find time index for the end of the maturity phase (that occurs before the given seasons harvest date)
        if PlantDev.phases.index('maturity') in idevphase_array[itax_sowing:itax_harvest+1]:
            itax_mature = np.where(idevphase_array[itax_sowing:itax_harvest+1] == PlantDev.phases.index('maturity'))[0][-1] + itax_sowing     # Index for end of maturity phase
        elif Management.harvestDays is not None:
            itax_mature = itax_harvest    # Maturity developmental phase not completed, so take harvest as the end of growing season
        else:
            itax_mature = -1    # if there is no harvest day specified and there is no maturity in idevphase, we just take the last day of the simulation.

        maturityTimestamp = time_index[itax_mature]

        # Filter out transitions that occur on or before the sowing day
        itax_phase_transitions = [t for t in itax_phase_transitions if t > int(itax_sowing+1)]
        # Filter out transitions that occur after the end of the maturity phase
        itax_phase_transitions = [t for t in itax_phase_transitions if t <= itax_mature]

        return itax_sowing, itax_mature, itax_harvest, itax_phase_transitions

    def compute_mean_daily_air_temp(self,airTempMin,airTempMax):
        """
        Computes the daily mean air temperature from the daily minimum and daily maximum temperatures.

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

    def compute_sat_vapor_pressure_daily(self,Tmin,Tmax):
        """
        Computes the saturation vapor pressure using Tetens' formula and minimum and maximum temperature. 
        This is necessary due to the non-linearity of the saturation vapor pressure calculation.
        This formula can be applied to any averaging period that includes minimum and maximum temperatures
        e.g. daily, weekly, monthly.

        Parameters
        ----------
        Tmin : scalar or ndarray
            Array containing minimum air temperature (degC).
        Tmax : scalar or ndarray
            Array containing maximum air temperature (degC).

        Returns
        -------
        e_a : scalar or ndarray (see dtype of parameters)
            Array of actual vapor pressure (Pa)
        """
        e_s_min = self.compute_sat_vapor_pressure(Tmin)
        e_s_max = self.compute_sat_vapor_pressure(Tmax)
        e_s = (e_s_min+e_s_max)/2
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

    def compute_actual_vapor_pressure_daily(self,Tmin,Tmax,RH):
        """
        Computes the actual vapor pressure from relative humidity, minimum and maximum temperature. 
        This is necessary due to the non-linearity of the saturation vapor pressure calculation.
        This formula can be applied to any averaging period that includes minimum and maximum temperatures
        e.g. daily, weekly, monthly.

        Parameters
        ----------
        Tmin : scalar or ndarray
            Array containing minimum air temperature (degC).
        Tmax : scalar or ndarray
            Array containing maximum air temperature (degC).
        RH : scalar or ndarray
            Array containing relative humidity (%).

        Returns
        -------
        e_a : scalar or ndarray (see dtype of parameters)
            Array of actual vapor pressure (Pa)
        """
        e_s_min = self.compute_sat_vapor_pressure(Tmin)
        e_s_max = self.compute_sat_vapor_pressure(Tmax)
        e_s = (e_s_min + e_s_max)/2
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

    def compute_VPD_daily(self,Tmin,Tmax,RH):
        """
        Computes the mean vapor pressure deficit given minimum and maximum air temperatures 
        and relative humidity. This is necessary due to the non-linearity of the 
        saturation vapor pressure calculation. This formula can be applied to any averaging 
        period that includes minimum and maximum temperatures e.g. daily, weekly, monthly.

        Parameters
        ----------
        Tmin : scalar or ndarray
            Array containing minimum air temperature (degC).
        Tmax : scalar or ndarray
            Array containing maximum air temperature (degC).
        RH : scalar or ndarray
            Array containing relative humidity (%).

        Returns
        -------
        VPD : scalar or ndarray (see dtype of parameters)
            Array of vapor pressure deficit (Pa)
        """
        e_s_min = self.compute_sat_vapor_pressure(Tmin)
        e_s_max = self.compute_sat_vapor_pressure(Tmax)
        e_s = (e_s_min + e_s_max)/2
        e_a = e_s * RH/100
        VPD = e_s - e_a
        return VPD

    def compute_psychometric_constant(self,P):
        """
        Computes the psychometric constant. This relates the partial pressure of water in the air 
        to the air temperature. Another way to describe the psychrometric constant is the ratio of 
        specific heat of moist air at constant pressure (Cp) to latent heat of vaporization. 

        Parameters
        ----------
        P: float
            atmospheric pressure, Pa

        Returns
        -------
        gamma: float
            psychometric constant, kPa K-1 (=kPa oC-1)

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56

        Zotarelli et al., 2009, AE459: Step by Step Calculation of the Penman-Monteith Evapotranspiration 
        (FAO-56 Method), University of Florida

        """
        gamma = 1000 * (self.cp_air*1e-3 * P/1e3)/(self.L * self.MW_ratio_H2O)
        return gamma

    def compute_slope_sat_vapor_press_curve(self,T):
        """
        Computes the slope of the relationship between the saturation vapor pressure and temperature. 

        Parameters
        ----------
        T: float
            air temperature, degrees Celsius

        Returns
        -------
        Delta: float
            slope of saturation vapor pressure curve, kPa K-1 (=kPa oC-1)

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56

        Zotarelli et al., 2009, AE459: Step by Step Calculation of the Penman-Monteith Evapotranspiration 
        (FAO-56 Method), University of Florida

        """
        e_s = 1e-3 * self.compute_sat_vapor_pressure(T)
        Delta = (4098 * e_s) / (T + 237.3)**2
        return Delta

    def compute_wind_speed_height_z(self,u_zmeas,z_meas,z_height=2):
        """
        Estimate the wind speed at a specified height given the speed at a defined measurement height and 
        assuming the wind speed profile is logarithmic.

        Parameters
        ----------
        u_zmeas: float
            wind speed at specified measurement height, m s-1

        z_meas: float
            measurement height for wind speed, m

        z_height: float
            target height to estimate wind speed, m

        Returns
        -------
        u_zheight: float
            wind speed at specified target height, m s-1

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56

        """
        u_zheight = u_zmeas * 4.87 / (np.log(67.8 * z_meas - 5.42))
        return u_zheight

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

        eqtime, houranglesunrise, theta = self._solar_calcs(year,doy)

        solarnoon_t = (
            720 - 4 * self.CLonDeg - eqtime + self.timezone * 60) / 1440
        sunrise_t = solarnoon_t - houranglesunrise * 4 / 1440
        sunset_t = solarnoon_t + houranglesunrise * 4 / 1440

        return (24*sunrise_t,24*solarnoon_t,24*sunset_t)


    def _solar_calcs(self,year,doy,return_declination=False):
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

        # hour angle sunrise
        houranglesunrise = np.rad2deg(np.arccos(np.cos(np.deg2rad(90.833)) /
                             (np.cos(np.deg2rad(self.CLatDeg)) *
                              np.cos(np.deg2rad(declination))) -
                             np.tan(np.deg2rad(self.CLatDeg)) *
                             np.tan(np.deg2rad(declination))))

        solarnoon = (720-4*self.CLonDeg-eqtime+self.timezone*60)/1440

        truesolartime = (time*1440+eqtime+4*self.CLonDeg-60*self.timezone) % 1440

        hourangle = (truesolartime / 4 + 180) if (truesolartime / 4 < 0) else (truesolartime / 4 - 180)

        theta = np.rad2deg(np.arccos(np.sin(np.deg2rad(self.CLatDeg))*np.sin(np.deg2rad(declination))+np.cos(np.deg2rad(self.CLatDeg))*np.cos(np.deg2rad(declination))*np.cos(np.deg2rad(hourangle))))  # solar zenith angle

        if return_declination:
            return (eqtime, houranglesunrise, theta, declination)
        else:
            return (eqtime, houranglesunrise, theta)

    def solar_calcs(self,year,doy,return_declination=False):
        _vfunc = np.vectorize(self._solar_calcs)
        if return_declination:
            (eqtime, houranglesunrise, theta, declination) = _vfunc(year,doy,return_declination=return_declination)
            return (eqtime, houranglesunrise, theta, declination)
        else:
            (eqtime, houranglesunrise, theta) = _vfunc(year,doy,return_declination=return_declination)
            return (eqtime, houranglesunrise, theta)

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

    def calculate_solarradiation_Ra(self,year,doy):
        """
        Daily extraterrestrial radiation, Ra, for a given time (day of the year, year) and for a given latitude. Estimated
        from the solar constant, the solar declination and the time of the year according to Allen et al. (1998).

        Parameters
        ----------
        year: int
            Year in format YYYY

        doy: float or ndarray
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        Returns
        -------
        Ra: float
            Local extraterrestrial radiation (MJ m-2 d-1)

        Notes
        -----
        For sub-daily periods (e.g. hourly) a different formulation is required. See p. 47 of Allen et al. (1998).

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        """
        # compute solar declination angle
        eqtime, houranglesunrise, theta, declination = self.solar_calcs(year,doy,return_declination=True)

        # compute inverse relative distance Earth-Sun
        d_r = 1 + 0.033 * np.cos(2*np.pi/365 * doy)

        # compute hour angle at sunset (radians)
        houranglesunset = np.arccos(-np.tan(np.deg2rad(self.CLatDeg) * np.tan(np.deg2rad(declination))))

        # compute daily extraterrestrial radiation
        Ra = (24 * 60) / np.pi * self.S0_MJm2min * d_r * (houranglesunset * np.sin(np.deg2rad(self.CLatDeg)) * np.sin(np.deg2rad(declination)) + np.cos(np.deg2rad(self.CLatDeg)) * np.cos(np.deg2rad(declination)) * np.sin(houranglesunset))

        return Ra

    def calculate_solarradiation_clearsky(self,Ra):
        """
        Calculates the incoming surface solar radiation assuming clear-sky conditions.
        This ignores atmospheric water vapor and turbity effects.

        Parameters
        ----------
        Ra: float
            Daily extraterrestrial incoming solar radiation (MJ m-2 d-1)

        Returns
        -------
        Rso: float
            Clear-sky downward shortwave radiation (MJ m-2 d-1)

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        """
        Rso = (0.75 + 2 * 10e-5 * self.Elevation) * Ra
        return Rso

    def calculate_solarradiation_Rs(self,year,doy,Ra,fsunhrs,a_s=0.25,b_s=0.50):
        """
        Calculate daily incoming surface solar radiation based on the Angstrom formula with empirical adjustment for
        sunny/cloudy conditions. This relates solar radiation to extraterrestrial radiation and relative daily sunshine
        duration.

        Parameters
        ----------
        year: int
            Year in format YYYY

        doy: float or ndarray
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        Ra: float
            Daily extraterrestrial incoming solar radiation (MJ m-2 d-1)

        fsunhrs: float
            Fraction of sunshine within the day (hours), i.e. number of hours without cloud or overcast conditions divided by total daylight hours; fsunhrs=1 on a completely clear-sky day

        a_s: float
            Angstrom parameter (see Notes)

        b_s: float
            Angstrom parameter (see Notes)

        Returns
        -------
        Rs: float
            Incoming surface solar radiation (MJ m-2 d-1)

        Notes
        -----
        a_s is a regression constant expressing the fraction of extraterrestrial radiation reaching the earth on an overcast day (n=0).
        a_s + b_s is the fraction of extraterrestrial radiation reaching the earth on a clear-sky day (i.e. n=N).
        According to Allen et al. (1998) "Depending on atmospheric conditions (humidity, dust) and solar declination (latitude and month),
        the Angstrom values a_s and b_s will vary. Where no actual solar radiation data are available and no calibration has been carried
        out for improved as and bs parameters, the values a_s = 0.25 and b_s = 0.50 are recommended."

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        """
        a_s = 0.25
        b_s = 0.5

        Rs = (a_s + b_s * fsunhrs) * Ra
        return Rs

    def calculate_radiation_netshortwave(self,Rs,albedo):
        """
        Calculates the daily surface net shortwave radiation.

        Parameters
        ----------
        Rs: float
            Incoming surface shortwave radiation (MJ m-2 d-1)

        albedo: float
            albedo or canopoy reflection coefficient (-)

        Returns
        -------
        Rns: float
            Surface net shortwave radiation, (MJ m-2 d-1)

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        """
        Rns = (1 - albedo) * Rs
        return Rns

    def calculate_radiation_netlongwave(self, Rs, Rso, Tmin, Tmax, RH):
        """
        Calculates the daily surface net longwave radiation based on temperature and relative humidity.

        Parameters
        ----------
        Rs: float
            Incoming surface shortwave radiation (MJ m-2 d-1)

        Rso: float
            Incoming surface shortwave radiation under clear-sky conditions (MJ m-2 d-1)

        Tmin: float
            Minimum daily air temperature (degrees Celsius)

        Tmax: float
            Maximum daily air temperature (degrees Celsius)

        RH: float
            Relative humidity (%)

        Returns
        -------
        Rnl: float
            Surface net longwave radiation, (MJ m-2 d-1)

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        """
        e_a = self.compute_actual_vapor_pressure((Tmax-Tmin)/2,RH)/1e3  ## actual vapor pressure (kPa)
        Rnl = (self.StefanBoltzmannConstant / 1e6 * 60 * 60 * 24) * (((Tmax+273.15)**4 + (Tmin+273.15)**4)/2) * (0.34 - 0.14 * np.sqrt(e_a)) * (1.35 * Rs/Rso - 0.35)
        return Rnl

    def calculate_radiation_net(self,year,doy,Tmin,Tmax,RH,albedo,fsunhrs):
        """
        Calculates daily surface net radiation according to time, location, meteorology, and surface albedo.

        Parameters
        ----------
        year: int or ndarray
            Year in format YYYY

        doy: float or ndarray
            Ordinal day of year plus fractional day (e.g. midday on Jan 1 = 1.5; 6am on Feb 1 = 32.25)

        Tmin: float or ndarray
            Minimum daily air temperature (degrees Celsius)

        Tmax: float or ndarray
            Maximum daily air temperature (degrees Celsius)

        RH: float or ndarray
            Relative humidity (%)

        albedo: float or ndarray
            albedo or canopoy reflection coefficient (-)

        fsunhrs: float or ndarray
            Fraction of sunshine within the day (hours), i.e. number of hours without cloud or overcast conditions divided by total daylight hours; fsunhrs=1 on a completely clear-sky day  #Actual duration of sunshine within the day (hours), i.e. number of hours without cloud or overcast conditions; n=N on a completely clear-sky day

        Returns
        -------
        Rnet: float
            Net radiation (MJ m-2 d-1)

        References
        ----------
        Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
        """
        Ra = self.calculate_solarradiation_Ra(year,doy)  ## extraterrestrial incoming solar radiation
        Rso = self.calculate_solarradiation_clearsky(Ra)  ## clear-sky surface incoming solar radiation
        Rs = self.calculate_solarradiation_Rs(year,doy,Ra,fsunhrs)  ## surface incoming solar radiation
        Rns = self.calculate_radiation_netshortwave(Rs,albedo)  ## net surface shortwave radiation
        Rnl = self.calculate_radiation_netlongwave(Rs, Rso, Tmin, Tmax, RH)  ## net surface longwave radiation
        Rnet = Rns - Rnl  ## net surface radiation
        return Rnet

    def diffusion_coefficient_powerlaw(self,D_ref,T,p,Tref=298.15,pref=101300,alpha=1.8):
        """
        Empirical equation for determining the binary diffusion coefficient of a gas species in air. 

        Parameters
        ----------
        D_ref: float
            Reference diffusivity value of gas species (m2 s-1)

        T: float
            Air temperature (K)

        p: float
            Atmospheric pressure (Pa)

        Tref: float
            Reference air temperature for corresponding D_ref value (K)

        pref: float
            Reference atmospheric pressure for corresponding D_ref value (Pa)

        Returns
        -------
        D_Tp: float
            Diffusivity value corrected for temperature and pressure (m2 s-1)

        Notes
        -----

        References
        ----------
        Nobel (2009) Physicochemical and Environmental Plant Physiology, Ch. 8, pp. 379
        Massman (1998) Atmospheric Environment Vol. 32, No. 6, pp. 1111—1127, doi:10.1016/S1352-2310(97)00391-9
        """
        D_Tp = D_ref * (pref/p) * (T/Tref)**alpha
        return D_Tp

    def compute_skin_temp(self, airTempC, solRadswskyg):
        """
        Empirically-derived equation based on the Yanco eddy covariance flux tower site, part of the Australia's 
        TERN OzFlux network. 

        Parameters
        ----------
        airTempC : float or array_like
            Air temperature (deg C)

        solRadswskyg : float or array_like
            Atmospheric global solar radiation (W/m2)

        Returns
        -------
        Tskin : float or array_like
            Surface skin temperature (deg C)

        Notes
        -----
        The Yanco OzFlux eddy covariance tower is located in an agricultural site near Morundah, NSW. We can use 
        the measurements from this station to determine the difference between air temperature and surface (skin) 
        temperature. To do this, we can invert the Stefan-Boltzmann law, assuming a surface emissivity value of 
        0.965 (typical range for agricultural settings with a mix of vegetation and bare soil is 0.95-0.98), using 
        the measured upwelling longwave radiation as the input. Figure 1 shows that there are is up to 10 deg C 
        temperature difference between the surface and the air above, with strong season variation showing a peak 
        in summer and minimum in winter. We can assume a very simple relationship between this temperature 
        difference and the downwelling shortwave radiation (Figure 2) which explains 45% of the variability in the 
        skin–air temperature difference using a linear regression. Downwelling shortwave radiation is available in 
        the standard DAESIM2 forcing data, so we can use this to assume a surface skin temperature in the model.
        """
        Tskin_air_diff = 0.0117 * solRadswskyg - 0.0566
        Tskin = airTempC + Tskin_air_diff
        return Tskin

    def air_pressure_at_elevation(self):
        """
        Calculate air pressure at a given elevation using the simplified relationship for the troposphere.

        Parameters
        ----------
        h (float): Elevation above sea level (in meters).
        P0 (float): Sea-level standard atmospheric pressure (Pa). Default is 101325 Pa.
        T0 (float): Standard sea-level temperature (K). Default is 288.15 K.
        lapse_rate (float): Temperature lapse rate (K/m). Default is 0.0065 K/m.
        g (float): Acceleration due to gravity (m/s^2). Default is 9.80665 m/s^2.
        M (float): Molar mass of Earth's air (kg/mol). Default is 0.0289644 kg/mol.
        R (float): Universal gas constant (J/(mol·K)). Default is 8.3144598 J/(mol·K).

        Returns
        -------
        float: Air pressure at the specified elevation (in Pa).
        """
        # Calculate air pressure at elevation
        pressure = self.airPressSeaLevel * (1 - (self.tropos_lapse_rate * self.Elevation) / self.TempSeaLevel) ** (self.g * self.M / (self.R_w_mol * self.tropos_lapse_rate))
        return pressure
