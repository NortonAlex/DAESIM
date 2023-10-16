"""
Soil model class: Includes differential equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp


@define
class SoilModuleCalculator:
    """
    Calculator of soil biophysics
    """

    # Class parameters
    labileDecompositionRate: float = field(default=0.01)  ## Question: Where does this value come from? What are the units?

    ## TODO: These are only temporary parameters for model testing
    optTemperature: float = field(default=20)  ## optimal temperature


    def calculate(
        self,
        LabileDetritus,
        _PhBioMort,
        _NPhBioMort,
        airTempC,
    ) -> Tuple[float]:
        Cin = 2.0
        Cout = 2.0

        # Call the initialisation method
        # SoilConditions = self._initialise(self.iniSoilConditions)

        _TempCoeff = np.exp(0.20 * (airTempC - self.optTemperature)) * np.abs(
            ((40 - airTempC) / (40 - self.optTemperature))
        ) ** (
            0.2 * (40 - self.optTemperature)
        )  ## TODO: This is actually a variable calculated from the plant module. Make sure we're not duplicating these calculations.

        LDDecomp = self.calculate_LDDecomp(LabileDetritus, _TempCoeff)

        LDin = _PhBioMort + _NPhBioMort
        LDout = LDDecomp

        # ODE for soil mass
        dCdt = LDin - LDout

        return dCdt

    def calculate_LDDecomp(self, LabileDetritus, _TempCoeff):
        # if Water.calPropUnsat_WatMoist>thresholdWater:
        #  LDDecomp = Labile_Detritus*labilelDecompositionRate*CropGrowth.TempCoeff*Decomposing_Microbes
        # else:
        #  LDDecomp = 0.01
        LDDecomp = LabileDetritus * self.labileDecompositionRate * _TempCoeff

        return LDDecomp