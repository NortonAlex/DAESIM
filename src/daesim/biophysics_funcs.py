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

def fT_arrheniuspeaked(k_25, T_k, E_a=70.0, H_d=200, DeltaS=0.650):
    """
    Applies a peaked Arrhenius-type temperature scaling function to the given parameter.
    
    Parameters
    ----------
    k_25: float
        Rate constant at 25oC

    T_k: float
        Temperature, degrees Celsius

    E_a: float
        Activation energy, kJ mol-1. Describes the rate of exponential increase of the function below the optimum
    
    H_d: float
        Deactivation energy, kJ mol-1. Describes the rate of decrease of the function above the optimum
    
    DeltaS: float
        Entropy of the process, kJ mol-1 K-1. Also known as an entropy factor but is not readily interpreted.

    Returns
    -------
    Temperature adjusted rate constant at the given temperature.

    References
    ----------
    From Medlyn et al. (2002, doi: 10.1046/j.1365-3040.2002.00891.x) Equation 17. Note that this ignores
    the correction as described in Murphy and Stinziano (2020, doi: 10.1111/nph.16883) Equation 10 because 
    the biochemical parameters were calibrated using the Medlyn formulation. 
    """
    T_k = T_k + 273.15
    R   = 8.314      # universal gas constant J mol-1 K-1
    E_a = E_a * 1e3  # convert kJ mol-1 to J mol-1
    H_d = H_d * 1e3  # convert kJ mol-1 to J mol-1
    DeltaS = DeltaS * 1e3  # convert kJ mol-1 K-1 to J mol-1 K-1
    
    exponential_term1 = (E_a*(T_k - 298.15))/(298.15*R*T_k)
    exponential_term2 = (298.15 * DeltaS - H_d)/(289.15*R)
    exponential_term3 = (T_k*DeltaS - H_d)/(T_k*R)
    
    k_scaling = np.exp(exponential_term1) * ((1.0 + np.exp(exponential_term2))/(1.0 + np.exp(exponential_term3)))

    return k_25*k_scaling

def fT_arrhenius(k_25, T_k, E_a=70.0, T_opt=298.15):
    """
    Applies an Arrhenius-type temperature scaling function to the given parameter.
    
    Parameters
    ----------
    k_25: float
        Rate constant at 25oC

    T_k: float
        Temperature, degrees Celsius

    E_a: float
        Activation energy, kJ mol-1, gives the rate of exponential increase of the function

    T_opt: float
        Optimum temperature for rate constant, K

    Returns
    -------
    Temperature adjusted rate constant at the given temperature.

    References
    ----------
    Medlyn et al. (2002) Equation 16
    """
    T_k = T_k + 273.15
    R   = 8.314      # universal gas constant J mol-1 K-1
    E_a = E_a * 1e3  # convert kJ mol-1 to J mol-1

    k_scaling = np.exp( (E_a * (T_k - T_opt))/(T_opt*R*T_k) ) 

    return k_25*k_scaling

def fT_Q10(k_25, T_k, Q10=2.0):
    """
    Applies a Q10 temperature scaling function to the given parameter (e.g. a rate constant).
    
    Parameters
    ----------
    k_25: float
        Rate constant at 25oC

    T_k: float
        Temperature, degrees Celsius

    Q10: float
        Q10 coefficient (factor change per 10oC increments), unitless

    Returns
    -------
    Temperature adjusted rate constant at the given temperature

    """
    T_k = T_k + 273.15
    k_scaling = Q10**((T_k - 298.15)/10)

    return k_25*k_scaling

def _diurnal_temperature(Tmin,Tmax,t_sunrise,t_sunset,tstep=1):
    """
    Parameters
    ----------
    Tmin: float
        Minimum daily air temperature (degrees Celsius)

    Tmax: float
        Maximum daily air temperature (degrees Celsius)

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
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

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

def growing_degree_days_HTT(Th,Tb,Tu,Topt,normalise):
    """
    Calculates the hourly thermal time (HTT) or 'heat units' according to a peaked temperature response model.
    The temperature response model is based on that of Yan and Hunt (1999, doi:10.1006/anbo.1999.0955).
    Also see description in Zhou and Wang (2018, doi:10.1038/s41598-018-28392-z).

    Parameters
    ----------
    Th : float
        Hourly temperature (degrees Celsius)
    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)
    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)
    Topt : float
        Thermal optimum temperature (degrees Celsius)
    normalise : str
        Normalize the thermal time function to range between 0-1. 

    Returns
    -------
    Hourly thermal time (HTT): float
    """
    if Th < Tb:
        return 0
    elif (Th >= Tb) and (Th <= Tu):
        if normalise:
            return ((Tu-Th)/(Tu-Topt)) * ((Th-Tb)/(Topt-Tb))**((Topt-Tb)/(Tu-Topt))
        else:
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

def growing_degree_days_DTT_nonlinear(Tmin,Tmax,t_sunrise,t_sunset,Tb,Tu,Topt,normalise=False):
    """
    Calculates the daily thermal time from the minimum daily temperature, maximum daily
    temperature, sunrise time, sunset time, and the cardinal temperatures that describe
    the hourly thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    t_sunrise: float or ndarray
        Time of sunrise (24 hour time, e.g. at 6:30 am, t = 6.5)

    t_sunrise: float or ndarray
        Time of sunset (24 hour time, e.g. at 8:15 pm, t = 20.25)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celcsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Topt : float
        Thermal optimum temperature (degrees Celsius)

    normalise : str
        Normalize the thermal time function to range between 0-1. 

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    T_diurnal_profile = _diurnal_temperature(Tmin,Tmax,t_sunrise,t_sunset)
    _vfunc = np.vectorize(growing_degree_days_HTT,otypes=[float])
    HTT_ = _vfunc(T_diurnal_profile,Tb,Tu,Topt,normalise=normalise)
    DTT = growing_degree_days_DTT_from_HTT(HTT_)
    return DTT

def growing_degree_days_DTT_linear1(Tmin,Tmax,Tb,Tu):
    """
    Calculates the daily thermal time (DTT) using the linear "Method 1" in McMaster and 
    Wilhelm (1997, doi:10.1016/S0168-1923(97)00027-0). This function requires the 
    minimum daily temperature, maximum daily temperature, the base and upper threshold 
    temperatures that describe the hourly thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    Tavg = (Tmin+Tmax)/2
    if Tavg < Tb:
        return 0
    elif (Tavg > Tb) and (Tavg < Tu):
        return Tavg - Tb
    elif Tavg > Tu:
        return Tu - Tb

def growing_degree_days_DTT_linear2(Tmin,Tmax,Tb,Tu):
    """
    Calculates the daily thermal time (DTT) using the linear "Method 2" in McMaster and 
    Wilhelm (1997, doi:10.1016/S0168-1923(97)00027-0). This function requires the 
    minimum daily temperature, maximum daily temperature, the base and upper threshold 
    temperatures that describe the hourly thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    if Tmax < Tb:
        Tmax = Tb
    elif Tmax > Tu:
        Tmax = Tu
    if Tmin < Tb:
        Tmin = Tb
    elif Tmin > Tu:
        Tmin = Tu
    Tavg = (Tmin+Tmax)/2
    if Tavg < Tb:
        return 0
    elif (Tavg > Tb) and (Tavg < Tu):
        return Tavg - Tb
    elif Tavg > Tu:
        return Tu - Tb

def growing_degree_days_DTT_linear3(Tmin,Tmax,Tb,Tu):
    """
    Calculates the daily thermal time (DTT) using the linear "Method 2" in Zhou and 
    Wang (2018, doi:10.1038/s41598-018-28392-z). This function requires the minimum 
    daily temperature, maximum daily temperature, the base and upper threshold 
    temperatures that describe the hourly thermal time temperature response model.

    Parameters
    ----------
    Tmin: float or ndarray
        Minimum daily air temperature (degrees Celsius)

    Tmax: float or ndarray
        Maximum daily air temperature (degrees Celsius)

    Tb : float
        Minimum threshold temperature or "base" temperature (degrees Celsius)

    Tu : float
        Upper threshold temperature or "upper" temperature (degrees Celsius)

    Returns
    -------
    Daily thermal time (DTT): float or ndarray
        Daily thermal time (degrees Celsius)
    """
    Tavg = (Tmin+Tmax)/2
    if Tavg <= Tb:
        return 0
    elif (Tavg > Tb) and (Tavg < Tu):
        Tm = min(Tmax,Tu)
        Tn = max(Tm,Tb)
        Tavg_prime = (Tm+Tn)/2
        return Tavg_prime - Tb
    elif Tu < Tavg:
        return Tu - Tb

def MinQuadraticSmooth(x, y, eta=0.99):
    # Ensuring x, y, and eta can be numpy arrays for vector operations
    x = np.asarray(x)
    y = np.asarray(y)
    eta = np.asarray(eta)
    
    z = np.power(x + y, 2) - 4.0 * eta * x * y
    z = np.maximum(z, 1e-18)  # Ensure z doesn't go below 1e-18
    mins = (x + y - np.sqrt(z)) / (2.0 * eta)
    return mins
