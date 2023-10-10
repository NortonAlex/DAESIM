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
    test_param: float = field(
        default=1.0
    )


    def calculate(
        self,
        SoilMass,
    ) -> Tuple[float]:
        
    	Cin = 2.0
    	Cout = 2.0

        # Call the initialisation method
        #SoilConditions = self._initialise(self.iniSoilConditions)

        # ODE for soil mass
        dCdt = Cin - Cout

        return dCdt
