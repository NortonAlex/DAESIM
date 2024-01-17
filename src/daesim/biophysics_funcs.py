"""
Biophysics helper functions used across more than one DAESim module
"""

import numpy as np

def func_TempCoeff(airTempC,optTemperature=20):
        """
        Function to calculate the temperature coefficient.

        Errorcheck: This function seems okay for temperatures below 40 degC but it goes whacky above 40 degC. This is a problem that we'll have to correct.
        TODO: Correct the whacky values from the calculate_TempCoeff functiono when airTempC > 40 degC.
        """
        TempCoeff = np.exp(0.20 * (airTempC - optTemperature)) * np.abs(
            ((40 - airTempC) / (40 - optTemperature))
        ) ** (
            0.2 * (40 - optTemperature)
        )  ## See Stella docs
        return TempCoeff

def _diurnal_temperature(Tmin,Tmax,t_sunrise,t_sunset,tstep=1):
    """
    Parameters
    ----------
    Tmin: float
        Minimum daily air temperature (degrees Celcius)

    Tmax: float
        Maximum daily air temperature (degrees Celcius)

    t_sunrise: float
        Time of sunrise (24 hour time, e.g. at 6:30 am, t = 6.5)

    t_sunrise: float
        Time of sunset (24 hour time, e.g. at 8:15 pm, t = 20.25)
        
    tstep: float
        Time step of diurnal cycle (fractional hour e.g. for 1 hour, tstep = 1; for 30 min, tstep = 0.5)

    Returns
    -------

    T_hr: array_like (length depends on tstep)
        Temperature derived from Tmin and Tmax daily temperatures. 
    """
    ## Time array for diurnal cycle (default it 1 hourly starting from 00:00:00)
    ## - 24 hour time, e.g. at 3 pm, t = 15.0
    t = np.arange(0,24,tstep)

    T_average = (Tmin+Tmax)/2
    T_amplitude = (Tmax-Tmin)/2 
    ## Equation 1
    T_H_1 = T_average - T_amplitude * (np.cos(np.pi * (t-t_sunrise)/(14-t_sunrise)))
    ## Equation 2
    H_prime = (t>14.0)*(t-14.0) + (t<t_sunrise)*(t+10)
    T_H_2 = T_average + T_amplitude * (np.cos(np.pi * H_prime/(10+t_sunrise)))

    ## Set output to Equation 2
    T_H = T_H_2
    ## get indexes of time array where we use Equation 1
    t_Trise_1 = (t >= t_sunrise) * (t <= 14.0)
    ## replace the corresponding values in the output array with Equation 1
    T_H[t_Trise_1] = T_H_1[t_Trise_1]
    return T_H

def diurnal_temperature(Tmin,Tmax,t_sunrise,t_sunset,tstep=1):
    """
    Calculates a synthetic diurnal temperature profile based on the minimum and maximum daily temperature. 
    The model describes the daily temperature curve using a combination of two formulas: 
    a cosine curve to describe daytime warming, and a separate cosine curve for nighttime cooling. 
    
    The transition points between the two formulas are determined from the sunrise time and the time assumed 
    to reach the maximum temperature (14:00). The first formula is used for daytime warming between sunrise 
    and 14:00. The second formula is used for nighttime cooling from 14:00 to sunrise on the next day.

    References: See "WAVE Model" in Bal et al. (2023, doi:10.1038/s41598-023-34194-9)

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celcius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celcius)

    t_sunrise: float or ndarray
        Time of sunrise (24 hour time, e.g. at 6:30 am, t = 6.5)

    t_sunrise: float or ndarray
        Time of sunset (24 hour time, e.g. at 8:15 pm, t = 20.25)

    tstep: float
        Time step of diurnal cycle (fractional hour e.g. for 1 hour, tstep = 1; for 30 min, tstep = 0.5)

    Returns
    -------

    T_hr: float or ndarray
        Temperature derived from Tmin and Tmax daily temperatures
    """
    if Tmin.size != Tmax.size:
        raise ValueError("Size of Tmin and Tmax inputs must be the same")

    Tmin_ = list(np.array(Tmin))  #np.asarray(Tmin)
    Tmax_ = list(np.array(Tmax))  #np.asarray(Tmax)
    iday = 0

    _vfunc = np.vectorize(_diurnal_temperature,otypes=[float])
    # T_hr = 
    
    for (xTmin,xTmax) in zip(Tmin_,Tmax_):
        _vfunc = np.vectorize(_diurnal_temperature,otypes=[float])
        if iday == 0:
            T_hr = _vfunc(t,xTmin,xTmax,t_sunrise,t_sunset)
        else:
            T_hr = np.append(T_hr,_vfunc(t,xTmin,xTmax,t_sunrise,t_sunset))
        iday += 1
    return np.reshape(T_hr,(Tmin.size,t.size))

def growing_degree_days_HTT(Th,Tb,Tu,Topt):
    """
    Calculates the hourly thermal time (HTT) or 'heat units' according to a peaked temperature response model. 
    The temperature response model is based on that of Yan and Hunt (1999, doi:10.1006/anbo.1999.0955). 
    Also see description in Zhou and Wang (2018, doi:10.1038/s41598-018-28392-z). 

    Parameters
    ----------
    Th : float
        Hourly temperature (degrees Celcius)
    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celcius)
    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celcius)
    Topt : float
        Thermal optimum temperature (degrees Celcius)

    Returns
    -------
    Hourly thermal time (HTT): float
    """
    if Th < Tb:
        return 0
    elif (Th >= Tb) and (Th <= Tu):
        return ((Tu-Th)/(Tu-Topt)) * ((Th-Tb)/(Topt-Tb))**((Topt-Tb)/(Tu-Topt)) * (Topt-Tb)
    elif Tu < Th:
        return 0

def growing_degree_days_DTT_from_HTT(HTT,tstep=1):
    """
    Calculates the daily thermal time (DTT) from the hourly thermal time (HTT) by 
    taking the average of the HTT values. 
    """
    t = np.arange(0,24,tstep)
    n = t.size
    return np.sum(HTT)/n

def growing_degree_days_DTT_nonlinear(Tmin,Tmax,t_sunrise,t_sunset,Tb,Tu,Topt):
    """
    Calculates the daily thermal time from the minimum daily temperature, maximum daily 
    temperature, sunrise time, sunset time, and the cardinal temperatures that describe 
    the hourly thermal time temperature response model. 

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celcius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celcius)

    t_sunrise: float or ndarray
        Time of sunrise (24 hour time, e.g. at 6:30 am, t = 6.5)

    t_sunrise: float or ndarray
        Time of sunset (24 hour time, e.g. at 8:15 pm, t = 20.25)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celcius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celcius)

    Topt : float
        Thermal optimum temperature (degrees Celcius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celcius)
    """
    T_diurnal_profile = _diurnal_temperature(Tmin,Tmax,t_sunrise,t_sunset)
    _vfunc = np.vectorize(growing_degree_days_HTT,otypes=[float])
    HTT_ = _vfunc(T_diurnal_profile,Tb,Tu,Topt)
    DTT = growing_degree_days_DTT_from_HTT(HTT_)
    return DTT
