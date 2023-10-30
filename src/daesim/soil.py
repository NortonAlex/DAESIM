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
    stableLabile_DecompositionRate: float = field(default=0.00001)  ## Question: Where does this value come from? What are the units?
    stableMineral_DecompositionRate: float = field(default=0.00007)  ## Question: What does this represent? How come it is used for both SDDecompLD and SDDecompMine? Why 
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
    iniLabileOxi: float = field(
        default=0.04
    )  ## Question: What does this represent biophysically and what are the units?
    coeffLabile: float = field(
        default=0.0002
    )  ## the microbe consumption rate of labile detritus as fuels for growth
    labileErodible: float = field(
        default=0.165
    )  ## the proportion of labile detritus that can be eroded
    
    RainfallErosivityFactor: float = field(
        default=1430
    )  ## R, rainfall erosivity factor. Energy release of rainfall / kinetic energy of rainfall. Question: Are the units MJ mm ha− 1 h− 1 y− 1 (see https://doi.org/10.1016/j.geoderma.2017.08.006)?
    ErodibilityFactor: float = field(default=0.0277) # K or K-factor, soil erodibility factor (e.g. Okorafor and Adeyemo, 2018, doi:10.5923/j.re.20180801.02; Panagos et al., 2014, doi:10.1016/j.scitotenv.2014.02.010). TODO: This is either prescribed as a fixed parameter or calculated dynamically (see Stella for equation), for now I only prescribe it. 
    empiricalC: float = field(default=0.08) # C or C-factor, cover management factor on soil erosion. TODO: This is either prescribed as a fixed parameter or calculated dynamically (see Stella for equation), for now I only prescribe it. 
    P: float = field(default = 1) # Support practices factor


    ## TODO: These are only temporary parameters for model testing
    optTemperature: float = field(default=20)  ## optimal temperature

    ## TODO: These are temporary parameters that really belong in a separate "Management" module
    propTillage: float = field(
        default=0.5
    )  ## Management module: propTillage=intensityTillage/10 (ErrorCheck: in the Stella code propTillage=intensityTillage/10 but in the documentation propTillage=(intensityTillage*9)/(5*10), why?)


    def calculate(
        self,
        LabileDetritus,
        StableDetritus,
        SoilMass,
        _PhBioMort,
        _NPhBioMort,
        _PhBioHarvest,
        _NPhBioHarvest,
        Water_calPropUnsat_WatMoist,
        Water_SurfWatOutflux,
        airTempC,
        Site,
    ) -> Tuple[float]:
        Cin = 2.0
        Cout = 2.0
        Decomposing_Microbes = 0.03

        # Call the initialisation method
        # SoilConditions = self._initialise(self.iniSoilConditions)

        TempCoeff = func_TempCoeff(airTempC,optTemperature=self.optTemperature)

        LDin = self.calculate_LDin(_PhBioMort,_NPhBioMort,_PhBioHarvest,_NPhBioHarvest)

        LDDecomp = self.calculate_LDDecomp(LabileDetritus, Decomposing_Microbes, Water_calPropUnsat_WatMoist, TempCoeff)

        OxidationLabile = self.calculate_oxidation_labile(LabileDetritus)

        MicUptakeLD = self.calculate_MicUptakeLD(LabileDetritus)

        ErosionRate = self.calculate_ErosionRate(SoilMass,Water_SurfWatOutflux,Site.degSlope,Site.slopeLength)

        LDErosion = self.calculate_LDErosion(LabileDetritus,ErosionRate)

        # ODE for labile detritus
        dLabiledt = LDin - LDDecomp - OxidationLabile - MicUptakeLD - LDErosion #+ SDDecompLD

        SDin = self.calculate_SDin(_PhBioMort,_NPhBioMort,_PhBioHarvest,_NPhBioHarvest)
        SDDecompLD = self.calculate_SDDecompLD(StableDetritus,Decomposing_Microbes,Water_calPropUnsat_WatMoist,TempCoeff)
        # ODE for stable detritus
        dStabledt = SDin - SDDecompLD #- SDDecompMine - MicUptakeSD - OxidationStable - SDErosion

        # ODE for soil mass
        dSoilMassdt = - ErosionRate

        return (dLabiledt,dStabledt,dSoilMassdt)

    def calculate_LDin(self,PhBio_mort,NPhBio_mort,PhBioHarvest,NPhBioHarvest):

        ## TODO: The four variables below are calculated in two different places, LDin and SDin. Need to consolidate these into one location.
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

    def calculate_oxidation_labile(self, Labile_Detritus):
    	"""
    	Question: What does this flux represent biophysically? How does it differ from LDDecomp?
    	"""
    	labileOxi = (1+self.iniLabileOxi-(1-self.propTillage)/10)*self.iniLabileOxi
    	return labileOxi * Labile_Detritus

    def calculate_MicUptakeLD(self, Labile_Detritus):
    	return self.coeffLabile * Labile_Detritus

    def calculate_LDErosion(self, Labile_Detritus, ErosionRate):
    	return ErosionRate * Labile_Detritus * self.labileErodible

    def calculate_ErosionRate(self,SoilMass,Water_SurfWatOutflux,degSlope,slopeLength):
    	"""
    	Question: Is the Stella code using the USLE or RUSLE or CSLE equation for estimating soil erosion rate?
    	I think its a form of the RUSLE equation as described in Zhang et al., 2017, doi:10.1016/j.geoderma.2017.08.006

    	The validity is limited to slope gradients less than 50%, due to empirical nature of the S factor

    	TODO: Update the ErosionRate calculation to be better suited to landscape modeling, following Desmet and Govers (1997).

    	This is the Revised Universal Soil Loss Equation (RUSLE) as described in:
    	 - McCool et al. (1987) Transactions of the ASAE 30.5 (1987): 1387-1396, and
    	 - Renard et al. (1997) Predicting soil erosion by water: A guide to conservation planning with the Revised Universal Soil Loss Equation (RUSLE). Agricultural Handbook No. 703, U. S. Dept. of Agr, Washington DC (1997), p. 384

    	"""
    	R = self.RainfallErosivityFactor
    	K = self.ErodibilityFactor
    	
    	perSlope = 100*np.tan(np.deg2rad(degSlope))  ## angle of the slope (%) ; ErrorCheck: Modification: I have corrected an error in the Stella code in this calculation. Errorcheck: In the Stella code perSlope=tan(degSlope), but this is missing the x100, the slope as a percentage should be 100 * tan(degSlope).
    	beta = (np.sin(np.deg2rad(degSlope))/0.086) / (3*np.sin(np.deg2rad(degSlope))**0.8 + 0.56)  ## Modification: This equation is not included in the Stella code but its needed for the calculation of "m" below. Equation from Foster et al. (1977) and McCool et al. (1989)
    	m = beta/(beta+1)   ## the exponent on the length slope, its value depends on slope or slope and rill/interrill ratio. Called "MLS" in the Stella code. Modification: I have changed this from a conditional calculation (if/elseif/else) to a smooth function, following Foster et al. (1977) and McCool et al. (1989)
    	L = (slopeLength/22.13)**m

    	## S factor
    	## Modification: This empirical calculation of L, S and LS differs from that in Stella code, the origin of the Stella S-factor equation is apparently from Goldman (1986) yet the equations produces oddly large magnitudes.
    	S_i = 10.8 * np.sin(np.deg2rad(degSlope)) + 0.03  ## if perSlope < 9%
    	S_j = 16.8 * np.sin(np.deg2rad(degSlope)) - 0.5  ## if perSlope >= 9%
    	S_k = 3.0*np.sin(np.deg2rad(degSlope))**0.8 + 0.56  ## if slopeLength < 4.57 m (15 ft)
    	S = (S_i*(perSlope < 9) + S_j*(perSlope >= 9))*(slopeLength>4.57) + S_k*(slopeLength<=4.57)  ## Modification: Differs to Stella code. See McCool et al. (1987). S = S_i if perSlope >= 9%, S = S_j if perSlope < 9%, S = S_k if slopeLength < 4.57 m (and see Renard et al., 1997)

    	LS = L * S  # slope length and steepness factor (LS), representing the effect of the topography on erosion rate

    	## TODO: The Cfactor is either prescribed fixed or calculated dynamically, I am only using the prescribed option here.
    	Cfactor = self.empiricalC  ## Cover management factor "represents the ratio of soil loss from land cropped under specific conditions to the corresponding loss from a tilled, continuous fallow condition. This factor can be estimated in various ways depending on the level of information available. It is an estimate of the combined effects of canopy cover, surface vegetation, surface roughness, prior land use, mulch cover and organic material below the soil surface (Mhangara et al., 2012)." (Teng et al., 2017, doi:10.1016/j.envsoft.2015.11.024)

    	ErosionRate = R * SoilMass * Water_SurfWatOutflux * K * LS * Cfactor * self.P / 365   # divide by 365 to convert from annual to daily rate

    	return ErosionRate

    def calculate_SDin(self,PhBio_mort,NPhBio_mort,PhBioHarvest,NPhBioHarvest):

        ## TODO: The four variables below are calculated in two different places, LDin and SDin. Need to consolidate these into one location.
        propPHLigninContentFarmMethod = (1.0000001 - self.propTillage)*self.propPHLigninContent  ## TODO: note that propTillage is really from the Management module
        propNPHLigninContentFarmMethod = (1.0000001 - self.propTillage)*self.propNPHLigninContent  ## TODO: note that propTillage is really from the Management module
        propHarvPhLeftFarmMethod = (1.0000001 - self.propTillage)*self.propHarvPhLeft  ## TODO: note that propTillage is really from the Management module
        propHarvNPhLeftFarmMethod = (1.0000001 - self.propTillage)*self.propHarvNPhLeft  ## TODO: note that propTillage is really from the Management module

        _propManuIndirect = 0.2  ## the percentage of manure that directly gets decomposed into organic matter and nutrients
        _Manure = 0.0
        _SDManure = _Manure*_propManuIndirect  ## TODO: Placeholder for some future date when Livestock module is included

        SDin = ((propPHLigninContentFarmMethod)*(PhBio_mort+PhBioHarvest*propHarvPhLeftFarmMethod+_SDManure) +
            (propNPHLigninContentFarmMethod)*(NPhBio_mort+NPhBioHarvest*propHarvNPhLeftFarmMethod))  ## Question: This equation is different between the Stella code and the Taghikhah et al. (2022) publication appendix ("manure" is included in the publication but not in code)

        return SDin

    def calculate_SDDecompLD(self,Stable_Detritus,Decomposing_Microbes,Water_calPropUnsat_WatMoist,TempCoeff):
        _vfunc = np.vectorize(self.calculate_SDDecompLD_conditional)
        SDDecompLD = _vfunc(Stable_Detritus,Decomposing_Microbes,Water_calPropUnsat_WatMoist,TempCoeff)
        return SDDecompLD

    def calculate_SDDecompLD_conditional(self,Stable_Detritus,Decomposing_Microbes,Water_calPropUnsat_WatMoist,TempCoeff):
        if Water_calPropUnsat_WatMoist > self.thresholdWater:
            return Stable_Detritus*self.stableLabile_DecompositionRate*TempCoeff*Decomposing_Microbes
        else:
            return self.stableMineral_DecompositionRate  ## Question: In this case, how does this differ from the SDDecompMine flux? Why is there a flux to the Litter_Detritus at all?
