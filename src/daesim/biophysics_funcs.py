"""
Biophysics helper functions used across more than one DAESim module
"""

import numpy as np

def TempCoeff(airTempC,optTemperature=20):
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