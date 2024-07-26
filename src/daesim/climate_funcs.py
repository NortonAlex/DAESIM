"""
Helper functions for the climate module
"""

import numpy as np
from datetime import datetime, date, timedelta
from scipy.interpolate import interp1d
from functools import partial

def solar_day_calcs(year,doy,latitude,longitude,timezone):
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
    _vfunc = np.vectorize(_solar_day_calcs)
    (sunrise_t,solarnoon_t,sunset_t) = _vfunc(year,doy,latitude,longitude,timezone)
    return (sunrise_t,solarnoon_t,sunset_t)

def _solar_day_calcs(year,doy,latitude,longitude,timezone):
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

    s = str(int(year))+str(int(doy))  # convert year and fractional doy to string format
    xdate = datetime.strptime(s, '%Y%j').date()

    time = doy % 1 # time  # percentage past midnight, i.e. noon  is 0.5
    dt = xdate - date(1900,1,1)
    day = dt.days + 1  #day     # daynumber 1=1/1/1900

    Jday = day + 2415018.5 + time - timezone / 24  # Julian day
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
                         (np.cos(np.deg2rad(latitude)) *
                          np.cos(np.deg2rad(declination))) -
                         np.tan(np.deg2rad(latitude)) *
                         np.tan(np.deg2rad(declination))))
    
    solarnoon_t = (
        720 - 4 * longitude - eqtime + timezone * 60) / 1440
    sunrise_t = solarnoon_t - hourangle * 4 / 1440
    sunset_t = solarnoon_t + hourangle * 4 / 1440
    
    return (24*sunrise_t,24*solarnoon_t,24*sunset_t)


def sunlight_duration(year,doy,latitude,longitude,timezone):
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
    sunrise_t,solarnoon_t,sunset_t = solar_day_calcs(year,doy,latitude,longitude,timezone)
    sunlight_duration_hrs = sunset_t-sunrise_t
    return sunlight_duration_hrs

def interp_nearest_lower_neighbour(x,y,xt,fill_value=(np.nan,np.nan)):
    """
    One-dimensional piecewise constant interpolation, with given discrete data points (x, y), 
    evaluated at xt.

    Given an array of x and y coordinates and a point xt anywhere within the x domain, it will
    index to the nearest x coordinate below xt and return the corresponding y value.

    Parameters
    ----------
    x: array_like
        The x-coordinates of the data points
    y: array_like
        The y-coordinates of the data points, length must match x
    xt: float
        The x-coordinate at which to evaluate the interpolated value
    fill_value: two-element tuple
        The first value of this tuple will be used to fill in for requested points below the data range.
        The second value of this tuple will be used to fill in for requested points above the data range.

    Returns
    -------
    yt: float
        The interpolated value at the new x-coordinate, xt.

    Examples
    -------- 
    
        # Create some discrete data 
        x = np.array([0,1.5,3,5])
        y = np.array([12,10,15,20])
        plt.scatter(x,y,marker="o",label="y(x)")
        
        # Generate the interpolation by using partial and vectorize
        f = partial(interp_nearest_lower_neighbour,x,y,fill_value=(y[0],y[-1]))
        y_interped_f = np.vectorize(f)
        
        # Interpolate new y values across x dimension
        xt_array = np.linspace(0,6,1000)
        y_interped = y_interped_f(xt_array)
        plt.plot(xt_array,y_interped,label="interpolation",c="C1")
        plt.legend()

    """
    x = np.asarray(x)
    y = np.asarray(y)
    xt = np.asarray(xt)
    
    def interpolate_single_value(xt_single):
        if xt_single < x.min():
            return fill_value[0]
        elif xt_single > x.max():
            return fill_value[1]
        x_xt = np.max(x[x <= xt_single])
        i_xt = (x == x_xt)
        y_xt = y[i_xt]
        return y_xt[0]
    
    if xt.ndim == 0:
        return np.array(interpolate_single_value(xt))
    else:
        return np.array([interpolate_single_value(xi) for xi in xt])

def interp_forcing(x,y,kind="linear",fill_value=(np.nan,np.nan)):
    """
    This method returns a function whose call method uses the interpolation to find the value of new points.
    """

    if (kind == "linear") or (kind == "quadratic"):
        ## Use the scipy.interpolate.interp1d method
        interp_f = interp1d(x,y,kind=kind,bounds_error=False,fill_value=(fill_value[0],fill_value[-1]))
    elif kind == "pconst":
        ## Use the piecewise constant interpolation method (interp_nearest_lower_neighbour)
        f = partial(interp_nearest_lower_neighbour,x,y,fill_value=(fill_value[0],fill_value[-1]))
        interp_f = np.vectorize(f,otypes=[float])
    else:
        raise Exception("Interpolation method %s not accepted. Please choose one of 'linear', 'quadratic' or 'pconst'" % kind)

    return interp_f
