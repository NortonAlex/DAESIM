"""
Soil model class: Includes differential equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
from daesim.biophysics_funcs import func_TempCoeff


@define
class SoilModuleCalculator:
    """
    Calculator of soil biophysics
    """

    # Class parameters
    labileDecompositionRate: float = field(default=0.01)  ## Question: Where does this value come from? What are the units?
    propPHLigninContent: float = field(
        default=0.025
    )  ## the proportion of the lignin content in photosynthetic biomass. Table 1 in Martin and Aber, 1997, Ecological Applications
    propNPHLigninContent: float = field(
        default=0.5
    )  ## the proportion of the lignin content in non-photosynthetic biomass. Table 1 in Martin and Aber, 1997, Ecological Applications
    propHarvPhLeft: float = field(
        default=0.1
    )  ## the percentage of photosynthetic biomass left after harvesting
    propHarvNPhLeft: float = field(
        default=0.8
    )  ## the percentage of non-photosynthetic biomass left after harvesting
    thresholdWater: float = field(
        default=0.05
    )  ## the moisture threshold above which microbes can continue to accelerate the biological decomposition process

    ## TODO: These are only temporary parameters for model testing
    optTemperature: float = field(default=20)  ## optimal temperature

    ## TODO: These are temporary parameters that really belong in a separate "Management" module
    propTillage: float = field(
        default=0.025
    )  ## Management module: propTillage=intensityTillage/10 (ErrorCheck: in the Stella code propTillage=intensityTillage/10 but in the documentation propTillage=(intensityTillage*9)/(5*10), why?)


    def calculate(
        self,
        LabileDetritus,
        _PhBioMort,
        _NPhBioMort,
        _PhBioHarvest,
        _NPhBioHarvest,
        Water_calPropUnsat_WatMoist,
        airTempC,
    ) -> Tuple[float]:
        Cin = 2.0
        Cout = 2.0
        Decomposing_Microbes = 0.03

        # Call the initialisation method
        # SoilConditions = self._initialise(self.iniSoilConditions)

        TempCoeff = func_TempCoeff(airTempC,optTemperature=self.optTemperature)

        LDin = self.calculate_LDin(_PhBioMort,_NPhBioMort,_PhBioHarvest,_NPhBioHarvest)

        LDDecomp = self.calculate_LDDecomp(LabileDetritus, Decomposing_Microbes, Water_calPropUnsat_WatMoist, TempCoeff)

        # ODE for labile detritus
        dCdt = LDin - LDDecomp #+ SDDecompLD - MicUptakeLD - OxidationLabile - LDErosion

        return dCdt

    def calculate_LDin(self,PhBio_mort,NPhBio_mort,PhBioHarvest,NPhBioHarvest):

    	propPHLigninContentFarmMethod = (1.0000001 - self.propTillage)*self.propPHLigninContent  ## TODO: note that propTillage is really from the Management module
    	propNPHLigninContentFarmMethod = (1.0000001 - self.propTillage)*self.propNPHLigninContent  ## TODO: note that propTillage is really from the Management module
    	propHarvPhLeftFarmMethod = (1.0000001 - self.propTillage)*self.propHarvPhLeft  ## TODO: note that propTillage is really from the Management module
    	propHarvNPhLeftFarmMethod = (1.0000001 - self.propTillage)*self.propHarvNPhLeft  ## TODO: note that propTillage is really from the Management module


    	_propManuIndirect = 0.2  ## the percentage of manure that directly gets decomposed into organic matter and nutrients
    	_Manure = 0.0
    	_LDManure = _Manure*_propManuIndirect  ## TODO: Placeholder for some future date when Livestock module is included

    	LDin = ((1-propPHLigninContentFarmMethod)*(PhBio_mort+PhBioHarvest*propHarvPhLeftFarmMethod+_LDManure) + 
    		(1-propNPHLigninContentFarmMethod)*(NPhBio_mort+NPhBioHarvest*propHarvNPhLeftFarmMethod))  ## Question: This equation is different between the Stella code and the Taghikhah et al. (2022) publication appendix ("manure" is included in the publication but not in code)

    	return LDin

    def calculate_LDDecomp(self, LabileDetritus, Decomposing_Microbes, Water_calPropUnsat_WatMoist, TempCoeff):
        _vfunc = np.vectorize(self.calculate_LDDecomp_conditional)
        LDDecomp = _vfunc(LabileDetritus, Decomposing_Microbes, Water_calPropUnsat_WatMoist, TempCoeff)

        return LDDecomp

    def calculate_LDDecomp_conditional(self, Labile_Detritus, Decomposing_Microbes, Water_calPropUnsat_WatMoist, TempCoeff):
    	if Water_calPropUnsat_WatMoist > self.thresholdWater:
    		return Labile_Detritus * self.labileDecompositionRate * TempCoeff * Decomposing_Microbes
    	else:
    		return 0.01

