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
    cellArea = 1  ## area of the unit considered m2

    ## Unit conversion factors
    rainConv: float = 0.001  ## Conversion factor for mm/day to m/day

    def time_discretisation(self, t, dt=1):
        """
        t  = array of consecutive time steps (in days) e.g. a 2 year run with a 1-day time-step would require t=np.arange(1,2*365+1,1)
        dt = time step size (default is dt=1 day, TODO: Convert all time dimension units to seconds (t, dt))

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