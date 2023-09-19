"""
Helper functions for the climate module
"""

import numpy as np

def sun_declination_angle(DoY):
    """
    Solar declination angle calculation based only on the day-of-year
    
    Source: https://solarsena.com/solar-declination-angle-calculator/
    
    DoY = the number of days since January 1st (UTC 00:00:00). E.g. March 3rd (UTC 00:00:00), DoY = 31 + 28 + 2 = 61; or December 31st (UTC 00:00:00), DoY = 364
    
    These equations assume the earth orbits around the sun in a perfect circle. 
    They do not take into consideration: variations in Earth's orbit (e.g. eccentricity), atmospheric refraction, specifics of the Julian calendar, etc. 
    
    """
    Sun_Declin_deg = np.arcsin(np.sin(np.deg2rad(-23.44)) * np.cos(np.deg2rad((360/365) * (DoY+10) + 360/np.pi * 0.0167 * np.sin(np.deg2rad(360/365.24 * (DoY - 2))))))
    return Sun_Declin_deg

def hour_angle_sunrise(latitude,sun_declin_angle):
    """
    Solar hour angle calculation based on the latitude and the solar declination angle.
    
    Source: NOAA Solar Calculations (https://gml.noaa.gov/grad/solcalc/calcdetails.html)
    
    latitude = latitude (degrees), + for North
    sun_declin_angle = sun declination angle (degrees), between -23.44 to +23.44
    """
    HA_Sunrise_deg = np.rad2deg(np.arccos(np.cos(np.deg2rad(90.833))/(np.cos(np.deg2rad(latitude))*np.cos(np.deg2rad(sun_declin_angle)))-np.tan(np.deg2rad(latitude))*np.tan(np.deg2rad(sun_declin_angle))))
    return HA_Sunrise_deg
    
def sunlight_duration(latitude,DoY):
    """
    Sunlight duration (hours of daylight within the 24 hour period) for the given day-of-year.
    
    Parameters
    ----------
    latitude : scalar or ndarray
        Array containing latitude (degrees), + for North.
    DoY : scalar ndarray
        Array containing the number of days since January 1st (UTC 00:00:00). 
        E.g. March 3rd (UTC 00:00:00), DoY = 31 + 28 + 2 = 61; or December 31st (UTC 00:00:00), DoY = 364.

    Returns
    -------
    sunlight_duration_hrs : scalar or ndarray (see dtype of parameters)
        Array of sunlight duration in hours.

    """    
    sun_declin_angle = sun_declination_angle(DoY)
    ha_sunrise = hour_angle_sunrise(latitude,np.rad2deg(sun_declin_angle))
    sunlight_duration_hrs = 8*ha_sunrise/60
    return sunlight_duration_hrs