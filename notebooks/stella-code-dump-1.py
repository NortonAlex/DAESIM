# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext nb_black

# %%
import numpy as np
import pandas as pd
from typing import Tuple, Callable
from attrs import define, field
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# %%
from daesim.climate_funcs import *

# %%
from daesim.plant import PlantModuleCalculator, PlantModelSolver

# %% [markdown]
# ### Notes for translating code
#
# "TODO": Notes on things to improve in the code, just programming things, not the actual model biophysics.
#
# "ErrorCheck": Check with Firouzeh/Justin to see if this is correct or an error.
#
# "[Question...]": Ask Firouzeh/Justin why this equation/variable/value is used, whether it makes sense.
#
# "Modification": A piece of code that HAD to be modified compared with the Stella version in order to work within this framework or in Python more generally. 

# %% [markdown]
# ## Model run variables

# %%
start_time = 1
end_time = 400
dt = 1
time = np.arange(start_time, end_time + dt, dt)

it = 0
t = time[it]

# %% [markdown]
# ## Forcing data

# %%
file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim/data/StellaOutFile.csv"

df_forcing = pd.read_csv(file)

# %%

# %% [markdown]
# ## Climate

# %%
## Green Stella variables
# TODO: Convert to parameter set
Climate_CLatDeg = (
    -33.715
)  ## latitude of site in degrees; Beltsville = 39.0; min = 64.; max = 20.
Climate_rainConv = 0.001  ## Conversion factor for mm/day to m/day
Climate_Elevation = 70.74206543  ## IF there is sediment surface BELOW MSL (e.g., tidal creeks) then use the bathimetry data (depth below MSL) to determine elevation of sediments above the base datum.; ELSE use the land elevation data above MSL (indicating distance from MSL to the soil surface) plus the distance from the datum to mean sea level; ALL VALUES ARE POSITIVE (m) above base datum.
Climate_nYear = 0  # 3   ## Number of years we have empirical data for
Climate_cellArea = 1  ## area of the unit considered m2

# julian day, 1 thru 365
Climate_DayJul = (t - dt) % (365 + 1)  ## TODO: This is really the "ordinal date", not the Julian day. Rename it.
Climate_DayJulPrev = (t - dt - dt) % (365 + 1)  ## TODO: This is really the "ordinal date", not the Julian day. Rename it.
Climate_DayJul = (
    t - dt
) % 365 + 1  # Modification: Changed this equation so that Jan 1st (UTC 00:00:00) is represented by 1 (not 0). December 31st (UTC 00:00:00) is represented by 365.
Climate_DayJulPrev = (t - 2 * dt) % 365 + 1

Climate_ampl = np.exp(7.42 + 0.045 * Climate_CLatDeg) / 3600   ## ErrorCheck: Where does this equation come from? Is it globally applicable?
Climate_dayLength = Climate_ampl * np.sin((Climate_DayJul - 79) * 0.01721) + 12  ## ErrorCheck: This formulation seems odd. It doesn't return expected behaviour of a day-length calculator. E.g. it gives a shorter day length amplitude (annual min to annual max) at higher latitudes (e.g. -60o compared to -30o), it should be the other way around! I am going to replace it with my own solar calculations
Climate_dayLengthPrev = Climate_ampl * np.sin((Climate_DayJulPrev - 79) * 0.01721) + 12

Climate_dayLength = sunlight_duration(Climate_CLatDeg, Climate_DayJul - 1)
Climate_dayLengthPrev = sunlight_duration(Climate_CLatDeg, Climate_DayJulPrev - 1)



# %%
## Climate module variables:
Climate_solRadGrd = 20.99843025  ## correction for cloudy days
Climate_airTempC = 21.43692112  ## Including the F to C conversion. May need be changed if data are in C already; So as the data are already in C we have changed this


# %% [markdown]
# ## Conditions for plant

# %%
## Variables that are external to this module but are needed to run a test
plantingDay = 30

# %%

# %%
## Green Stella variables
# TODO: Convert to parameter set
maxAboveBM = 0.6  ## Max above ground biomass kg/m2. 0.9
maxPropPhAboveBM = 0.75  ## proportion of photosynthetic biomass in the above ground biomass (0,1)(alfa*). 0.65
propPhBEverGreen = 0.3  ## Proportion of evergreen photo biomass [Question: What does this mean, in a physiological sense? What does it represent in a real-world plant? ]
iniNPhAboveBM = (
    0.04  ## initially available above ground non-photosynthetic tissue. kg/m2 
)
propAboveBelowNPhBM = (
    0.85  ## Ratio of above to below ground non-photosynthetic biomas (beta)
)
heightMaxBM = 1.2  ## Height at maximum biomass
iniRootDensity = 0.05
propNPhRoot = 0.002  ## [Question: If "NPh" means non-photosynthetic, then why isn't propNPhRoot = 1? 100% of root biomass should be non-photosynthetic]


## Climate module variables:
Climate_Elevation = Climate_Elevation

## Soil module variables:
Soil_calSoilDepth = 0.09


## Plant module variables: Blue Stella variables - calculated outside this module (e.g. state variables)
Non_photosynthetic_Biomass = 0.0871  ## "Non-photosynthetic_Biomass" :: Carbon biomass of the non-photosynthetic portion of macrophytes. Units  kg/m^2
Photosynthetic_Biomass = 0.036  ## iniPhBM :: Carbon biomass of the photosynthetic portion of plant; Units = kg/m^2
# TODO: Do we really need this variable? Firouzeh's notes suggest it is a dummy variable required in Stella
Max_Photosynthetic_Biomass = 0.6  ## A dummy, stella enforced variable that is needed for the sole purpose of tracking the maximum attained value of non-photosynthetic biomass; Finding the max of Ph in the whole period of model run; maximal biomass reached during the season


## Blue Stella variables - calculated within this module
# TODO: work out how to implement the PULSE builtin function from Stella into Python
## PULSE( 1 , plantingDay,  365)   ## Planting time May 1  jul calendar day 121. Source Phipps, 1995 pers.comm. see worksheet [corn]; IF Climate.DayJul= plantingDay THEN 1 ELSE 0
if (t <= plantingDay) and (t + dt > plantingDay):
    PlantTime = 1
else:
    PlantTime = 0
maxPhAboveBM = (
    maxAboveBM * maxPropPhAboveBM
)  ## The maximum general photosynthetic biomass above ground (not site specific). kg/m2*dimless=kg/m2
maxNPhBM = (maxAboveBM - maxPhAboveBM) * (
    1 + 1 / propAboveBelowNPhBM
)  ## maximum above ground non phtosynthetic+(max above ground non pythosynthetic/proportion of above to below ground non photosynthetic)kg/m2
maxNPhAboveBM = (
    iniNPhAboveBM * (propAboveBelowNPhBM + 1) / propAboveBelowNPhBM
)  ## initial non-photo above ground biomass. [Question: What is this really representing? Errorcheck: If this is truly the "initial non-photo above ground biomass", then it should be equivalent to iniNPhAboveBM. ]
maxBM = maxNPhBM + maxPhAboveBM  ## kg/m2
iniPhBM = (
    propPhBEverGreen * iniNPhAboveBM * maxPropPhAboveBM / (1 - maxPropPhAboveBM)
)  ## initial biomass of photosynthetic tissue  (kgC/m^2).[]. calculated based on the available above ground non-photosynthetic tissue. PH/(PH+Ab_BM)=Max_Ph_to_Ab_BM. The introduction of the PhBio_evgrn_prop in this equation eliminates all leaves from deciduous trees.  Only coniferous trees are photosynthesizing! [Question: Why use propPhBEverGreen at all? Question: In Stella this is calculated based on parameters and the initial value of "iniNPhAboveBM". Why not initialise with a value then derive ]
rootBM = Non_photosynthetic_Biomass / (
    propAboveBelowNPhBM + 1
)  ## NPHbelow=root biomass=nonphotobiomass*below/(above+below)= nonphoto/(above/below+1). kg/m2
NPhAboveBM = (
    propAboveBelowNPhBM * rootBM
)  ## Above ground non-photosynthetic biomass; (kg/m2)/(kg/m3)=m
if NPhAboveBM == 0:
    propPhAboveBM = 0
else:
    propPhAboveBM = Photosynthetic_Biomass / (Photosynthetic_Biomass + NPhAboveBM)

rootDensity = max(
    iniRootDensity, Non_photosynthetic_Biomass * propNPhRoot * 1 / Soil_calSoilDepth
)  ##

# ErrorCheck: What is (Climate_Elevation/100)-1 meant to represent??
RootDepth = max(
    (Climate_Elevation / 100) - 1, (rootBM / rootDensity)
)  ## Make sure that roots don't get longer than the elevation - needed in calculations of available DIN.; W= winter wheat (2.2 m) , spring wheat (1.1 m); B= 1.2; R= 1.8; M=12; C= 1.65; Ct= 0.3; S=1.4; Wheat roots penetrated to a maximum depth of 104 cm in crops sown in May, the optimum time of sowing for maximum yield, while delayed sowing reduced total root biomass and limited rooting depth to 73-83 cm
rootDensity = max(
    iniRootDensity, Non_photosynthetic_Biomass * propNPhRoot * 1 / Soil_calSoilDepth
)

if NPhAboveBM == 0:
    propPhAboveBM = 0
else:
    propPhAboveBM = Photosynthetic_Biomass / (Photosynthetic_Biomass + NPhAboveBM)

TotalBM = Photosynthetic_Biomass + Non_photosynthetic_Biomass

if maxBM > 0:
    CalcuHeight = heightMaxBM * TotalBM / maxBM
else:
    CalcuHeight = 0

if Photosynthetic_Biomass > Max_Photosynthetic_Biomass:
    PHinflow = Photosynthetic_Biomass / DT
else:
    PHinflow = 0

if (Photosynthetic_Biomass > Max_Photosynthetic_Biomass) or (PlantTime == 1):
    PHoutflow = Max_Photosynthetic_Biomass / DT
else:
    PHoutflow = 0

# %%
### Plant water stress

## Green Stella variables
# TODO: Convert to parameter set
calibrationMinWater = 0.29  ## When > 1 there is always a limitation if not optimal; when <1 there is almost no limitation unless conditions -> 0;  the smaller the value - the less sensitivity to water stress; water deficit tolerance coefficient
calibrationMaxWater = 0.1  ## The larger the parameter, the more sensitivity to high water stress; 0 - no sensitivity


## Water module variables:
Water_waterAvailable = 0.850826175866  ## See Stella docs
Water_unsatWatDepth = 38.74206543  ## Depth (m) of the unsaturated zone.
Water_Surface_Water = 0  ## See Stella docs

# WatStressLow
WatStressLow = max(0, np.sin(Water_waterAvailable * np.pi / 2) ** calibrationMinWater)

# WatStressHigh
if (calibrationMaxWater < 0) and (Water_Surface_Water > -calibrationMaxWater):
    WatStressHigh = max(0, 1 / (1 + np.exp(Water_Surface_Water + calibrationMaxWater)))
elif (calibrationMaxWater > 0) and (RootDepth > 0):
    WatStressHigh = min(1, Water_unsatWatDepth / RootDepth / calibrationMaxWater)
else:
    WatStressHigh = 1

# %%
### Bio time for plant

## Climate module variables:
Climate_airTempC = Climate_airTempC


# Biological time counter
if PlantTime == 1:
    cf_Air_Temp = -Bio_time / DT  ## TODO: Remove time-step dependency
elif Climate_airTempC > 5:
    cf_Air_Temp = Climate_airTempC
else:
    cf_Air_Temp = 0

### TODO: Make Bio_time a state variable, as it stores information from time-step to time-step. It is analagous to cumulative degree days
# Bio_time(t - dt) + (cf_Air_Temp) * dt
Bio_time = df_forcing["PlantGrowth.Bio time"].values[it]


# %%
ratio_above_to_below_NPh = 0.85  # = propAboveBelowNPhBM
frac_Green_in_AboveBM = 0.75  # = maxPropPhAboveBM
frac_Evergreen_in_Green = 0.3  # = propPhBEverGreen

frac_NPh_in_AboveBM = 1 - frac_Green_in_AboveBM

initial_NPhAboveBM = 0.04  # kg/m2
print("initial above-ground (non-photosynthetic) biomass =", initial_NPhAboveBM)
belowground_NPhBM = 0.04 / 0.85
print("initial below-ground (non-photosynthetic) biomass =", belowground_NPhBM)

initial_PhBM = initial_NPhAboveBM * frac_Green_in_AboveBM / (1 - frac_Green_in_AboveBM)
print(initial_PhBM)
print()

print("Verification:")
print("  ratio above- to below-ground NPh =", initial_NPhAboveBM / belowground_NPhBM)
print(
    "  frac Ph (green) in above-ground BM =",
    initial_PhBM / (initial_PhBM + initial_NPhAboveBM),
)

# %%

# %% [markdown]
# ## Plant

# %%
DMP = 3291.13599  # TODO: This is a forcing variable   ## dry matter production kgDM/ha/day
empiricalNPP = DMP * 0.45 * 0.1 / 1000  ## gC/m2/day

# %% [markdown]
# ### NPP

# %%
## Green Stella variables
# TODO: Convert to parameter set
halfSaturCoeffP = 0.01  ## half-saturation coefficient for P; 0.000037 [Question: why do the Stella docs give a value (0.000037) that is different to the actual value used (0.01)?]
halfSaturCoeffN = 0.02  ## half-saturation coefficient for Nitrogen; 0.00265 [Question: why do the Stella docs give a value (0.00265) that is different to the actual value used (0.002)?]
optTemperature = 20  ## optimal temperature
NPPCalibration = (
    0.45  ## NPP 1/day(The rate at which an ecosystem accumulates energy); 0.12; 0.25
)

## Climate module variables:
Climate_solRadGrd = Climate_solRadGrd
Climate_airTempC = Climate_airTempC

# Saturating light intensity (langleys/d) for the selected crop
# {for macrophytes-> . Source:  599 [corn](Chang, 1981 Table 1)
# 1600 (µmol irradiance m^2 *s) coniferous forests ( Gholtz et all 1994, page40)
# Unit conversion:
# 10 langley = 1kcal/m^2
# 1 langleys/d =0.48458 watt
# Irradiance =1 watt/m^2 (1 watt =1joule/sec)
# 1.43e-3 Langleys/min =1µE M/s
saturatingLightIntensity = 600

## Blue Stella variables - calculated within this module
PO4Aval = 0.2  # TODO: Change name to "PO4Avail"; ErrorCheck: What is this? Why is it set to 0 in Stella? This makes the NutrCoeff=0, NPPControlCoeff=0, then calculatedPhBioNPP=0
DINAvail = 0.2  # [Question: No documentation. Check paper or ask Firouzeh/Justin to provide.] ErrorCheck: What is this? Why is it set to 0 in Stella? This makes the NutrCoeff=0, NPPControlCoeff=0, then calculatedPhBioNPP=0
NutrCoeff = min(
    (DINAvail / (DINAvail + halfSaturCoeffN)), (PO4Aval / (PO4Aval + halfSaturCoeffP))
)

LightCoeff = (
    Climate_solRadGrd
    * 10
    / saturatingLightIntensity
    * np.exp(1 - Climate_solRadGrd / saturatingLightIntensity)
)

WaterCoeff = min(WatStressHigh, WatStressLow)

TempCoeff = np.exp(0.20 * (Climate_airTempC - optTemperature)) * np.abs(
    ((40 - Climate_airTempC) / (40 - optTemperature))
) ** (
    0.2 * (40 - optTemperature)
)  ## See Stella docs

NPPControlCoeff = (
    min(LightCoeff, TempCoeff) * WaterCoeff * NutrCoeff
)  ## Total control function for primary production,  using minimum of physical control functions and multiplicative nutrient and water controls.  Units=dimensionless.

# calculatedPhBioNPP ## Estimated net primary productivity
if Photosynthetic_Biomass < maxPhAboveBM:
    calculatedPhBioNPP = (
        NPPControlCoeff
        * NPPCalibration
        * Photosynthetic_Biomass
        * (1 - Photosynthetic_Biomass / maxPhAboveBM)
    )
else:
    calculatedPhBioNPP = 0

# %% [markdown]
# ### PhNPP
#
# Module decides whether to use NPP from data (empiricalNPP) or calculated dynamically (calculatedPhBioNPP)

# %%
if Climate_DayJul < 366 * Climate_nYear:
    PhNPP = empiricalNPP
# elif HarvestTime > 0:   ## TODO: Include HarvestTime info
#     PhNPP = 0
else:
    PhNPP = calculatedPhBioNPP

# %%

# %% [markdown]
# ### Photosynthetic mortality

# %%
## Green Stella variables
# TODO: Convert to parameter set
propPhMortality = 0.015  ## Proportion of photosynthetic biomass that dies in fall time
propPhtoNPhMortality = 0  ## A proportion that decides how much of the Ph biomass will die or go to the roots at the fall time
propPhBLeafRate = 0  ## Leaf fall rate
dayLengRequire = 13  ## [Question: Need some information on this.]

## Climate module variables:
Climate_dayLength = Climate_dayLength
Climate_dayLengthPrev = Climate_dayLengthPrev

## Blue Stella variables - calculated within this module

WaterCoeff = min(
    WatStressHigh, WatStressLow
)  ## TODO: Modify the structure here as it is used a couple of times in the Plant module

PropPhMortDrought = 0.1 * max(0, (1 - WaterCoeff))

# FallLitterCalc
if (Climate_dayLength > dayLengRequire) or (Climate_dayLength >= Climate_dayLengthPrev):
    FallLitterCalc = 0
elif Photosynthetic_Biomass < 0.01 * (1 - propPhBEverGreen):
    FallLitterCalc = (
        (1 - propPhBEverGreen) * Photosynthetic_Biomass / DT
    )  ## TODO: Remove use of DT and time-step dependency
else:
    FallLitterCalc = (1 - propPhBEverGreen) * (
        Max_Photosynthetic_Biomass * propPhBLeafRate / Photosynthetic_Biomass
    ) ** 3

Fall_litter = min(
    FallLitterCalc,
    max(
        0,
        Photosynthetic_Biomass
        - propPhBEverGreen * Max_Photosynthetic_Biomass / (1 + propPhBEverGreen),
    ),
)

## PhBioMort: mortality of photosynthetic biomass as influenced by seasonal cues plus mortality due to current
## (not historical) water stress.  Use maximum specific rate of mortality and constraints due to unweighted
## combination of seasonal litterfall and water stress feedbacks (both range 0,1). units = 1/d * kg * (dimless +dimless)  = kg/d
PhBioMort = Fall_litter * (1 - propPhtoNPhMortality) + Photosynthetic_Biomass * (
    PropPhMortDrought + propPhMortality
)

# %% [markdown]
# ### Transdown

# %%
## Green Stella variables
# TODO: Convert to parameter set

BioRepro = 1800  ## bio time when reproductive organs start to grow; 1900
propPhtoNphReproduction = 0.005  ## fraction of photo biomass that may be transferred to non-photo when reproduction occurs
## TODO: This parameter is used in calculation of both PhBioMort and Transdown, make sure its available to both calculations
propPhtoNPhMortality = 0  ## A proportion that decides how much of the Ph biomass will die or go to the roots at the fall time
## TODO: This parameter is used in calculation of both "Conditions for plant", Transdown and Transup, make sure its available to both calculations
maxPropPhAboveBM = 0.75  ## proportion of photosynthetic biomass in the above ground biomass (0,1)(alfa*); 0.65


## Blue Stella variables - calculated within this module

# TransdownRate:
# The plant attempts to obtain the optimum photobiomass to total above ground biomass ratio.  Once this is reached,
# NPP is used to grow more Nonphotosythethic biomass decreasing the optimum ratio.  This in turn allows new Photobiomass
# to compensate for this loss; IF Ph_to_Ab_BM[Habitat,Soil]>•Max_Ph_to_Ab_BM[Habitat,Soil] THEN 1; ELSE Ph_to_Ab_BM[Habitat,Soil]/•Max_Ph_to_Ab_BM[Habitat,Soil]

if Bio_time > BioRepro + 1:
    TransdownRate = 1 - 1 / (Bio_time - BioRepro) ** 0.5
elif propPhAboveBM < maxPropPhAboveBM:
    TransdownRate = 0
else:
    TransdownRate = np.cos((maxPropPhAboveBM / propPhAboveBM) * np.pi / 2) ** 0.1


## TODO: Include HarvestTime info
# if HarvestTime > 0:
#     Transdown = 0
# else:
#     Transdown = TransdownRate*(PhNPP+propPhtoNphReproduction*Photosynthetic_Biomass)+(propPhtoNPhMortality*Fall_litter)
Transdown = TransdownRate * (
    PhNPP + propPhtoNphReproduction * Photosynthetic_Biomass
) + (propPhtoNPhMortality * Fall_litter)


# %% [markdown]
# ### Transup

# %%
## Green Stella variables
# TODO: Convert to parameter set
sproutRate = 0.01  ## Sprouting rate. Rate of translocation of assimilates from non-photo to photo bimass during early growth period
bioStart = 20  ## start of sprouting. [Question: What is the physiological definition?]
bioEnd = 80  ## [Question: What is the physiological definition?]
## TODO: This parameter is used in calculation of both "Conditions for plant", Transdown and Transup, make sure its available to both calculations
maxPropPhAboveBM = 0.75


if (propPhAboveBM < maxPropPhAboveBM) and (Bio_time > bioStart) and (Bio_time < bioEnd):
    Sprouting = 1
else:
    Sprouting = 0

Transup = Sprouting * sproutRate * Non_photosynthetic_Biomass


# %% [markdown]
# ### NPhBioMort

# %%
## Green Stella variables 
# TODO: Convert to parameter set
propNPhMortality = 0.04 ## non-photo mortality rate


## Blue Stella variables - calculated within this module

## NPhBioMort: Mortality/sensescence is assumed to occur at constant specific rate.  If the macropyhte is killed at harvest then the remaining carbon is shunted to organic matter pools. (Photosynthetic components have a greater mortality burden is during time of water stress, ultimately depleting nonphotosynthetic carbon stores.); kg*1/d 
NPhBioMort = Non_photosynthetic_Biomass*propNPhMortality



# %% [markdown]
# ### Exudation

# %%
## Green Stella variables
# TODO: Convert to parameter set
rhizodepositReleaseRate = 0.025

## Conditions for plant module variables:
rootBM = rootBM

## Blue Stella variables - calculated within this module
exudation = rootBM * rhizodepositReleaseRate  ##

# %%

# %%

# %% [markdown]
# ## State: Photosynthetic Biomass

# %%
## ODE
# Photosynthetic_Biomass(t) = Photosynthetic_Biomass(t-dt) + (PhNPP + PhBioPlanting + Transup - PhBioHarvest - Transdown - PhBioMort)*dt

## Initial value
iniPhBM

# %% [markdown]
# ## State: Non-Photosynthetic Biomass

# %%
## ODE
# Non_photosynthetic_Biomass(t) = Non_photosynthetic_Biomass(t-dt) + (NPhBioPlanting + Transdown - Transup - NPhBioHarvest - NPhBioMort - exudation)*dt

## Initial value
# ErrorCheck: Why is this maxNPhAboveBM? Shouldn't it just be an initial value of total non-photosynthetic biomass, not just the above-ground portion? It seems inconsistent with the rest of hte model structure e.g. how does rootBM, propAboveBelowNPhBM, etc play into this?
maxNPhAboveBM

# %% [markdown]
# ## State: Bio time

# %%
## ODE
# Bio_time(t) = Bio_time(t-dt) + (cf_Air_Temp)*dt

## Initial value
# Bio_time = 0

## Modification: This ODE has discrete step-changes which cannot be handled with solvers like solve_ivp. 
## - the issue arises in calculating cf_Air_Temp, as it depends on plantingDay and must be reset to zero at pre-defined time-steps (i.e. a day-of-year)
## - Solution: We would have to create a special case for handling of this that did the following: stop the integration at the triggering of plantingDay, change the state vector, and restart the integration. Then join the results. 
## - Alternative solution: Convert the discrete plantingDay event into a smooth, differentiable function (similar to Knorr et al., 2010)


# %% [markdown]
# # Plant Module Calculator

# %% [markdown]
# #### - Initialise the Plant module
#
# You can initialise it simply with the default parameters or you can initialise it and assign different parameters. 
#
# This example initalises it with a mortality_constant=0.002 and the rest of the parameters are default values.

# %%
Plant1 = PlantModuleCalculator(mortality_constant=0.002, dayLengRequire=12)

# %%
Plant1._initialise(0.04)

# %%
Plant1._initialise(0.04)["iniPhBM"]

# %% [markdown]
# #### - Use the `calculate` method to compute the RHS for the state

# %%
plt.plot(df_forcing["PlantGrowth.Bio time"].values)

df_forcing["PlantGrowth.Bio time"].values[0:10]

# %%
_PhBM = 2.0
_NPhBM = 1.0
_solRadGrd = 20.99843025
_airTempC = 21.43692112
_dayLength = 11.900191330084594
_dayLengthPrev = 11.89987139219148
_Bio_time = 0.0
_propPhAboveBM = 0.473684210526

print(
    "dy/dt =",
    Plant1.calculate(
        _PhBM,
        _NPhBM,
        _solRadGrd,
        _airTempC,
        _dayLength,
        _dayLengthPrev,
        _Bio_time,
        #         _propPhAboveBM,
    ),
)

# %% [markdown]
# #### - Use one of the calculate methods to compute a flux e.g. PhBioNPP or PhBioMort

# %%
PhBioNPP = Plant1.calculate_PhBioNPP(_PhBM, _solRadGrd, _airTempC, 1, 0.99)
print("PhBioNPP =", PhBioNPP)

# %%
_dayLength = 11.900191330084594
_dayLengthPrev = 11.89987139219148

PhBioMort, Fall_litter = Plant1.calculate_PhBioMortality(
    _PhBM, _dayLength, _dayLengthPrev, 1, 0.99
)

print("PhBioMort =", PhBioMort)
print("Fall_litter =", Fall_litter)

# %%
rootBM = Plant1.calculate_rootBM(_NPhBM)
print("rootBM =", rootBM)

propPhAboveBM = Plant1.calculate_propPhAboveBM(_PhBM, rootBM)
print("propPhAboveBM =", propPhAboveBM)

# %%
Transdown = Plant1.calculate_Transdown(
    _PhBM, PhBioNPP, Fall_litter, propPhAboveBM, _Bio_time
)
print(
    "Transdown =",
    Transdown,
)

# %%
Transup = Plant1.calculate_Transup(
    _NPhBM,
    _Bio_time,
    propPhAboveBM,
)
print(
    "Transup =",
    Transup,
)

# %% [markdown]
# ## Create interpolation function for the forcing data
#
# We'll be using the solve_ivp ODE solver. It does the time-stepping by itself (rather than pre-defined time-points like in odeint).
#
# So, in order for discrete forcing data to be used with solve_ivp, the solver must be able to compute the forcing at any point over the temporal domain. To do this, we interpolate the forcing data and pass the interpolated function to the model. 

# %%
time = np.arange(1, 1001)

constant_co2_exp = interp1d(time, 280 * np.ones_like(time))

linear_plateau_co2_exp = interp1d(
    time, np.concatenate((np.linspace(280, 330, num=100), 330 * np.ones(900)))
)

## Select any time within the time domain
t1 = 100
print(constant_co2_exp(t1), linear_plateau_co2_exp(t1))

## plot the interpolated time-series
plt.plot(constant_co2_exp(time))
plt.plot(linear_plateau_co2_exp(time))

# %%
df_forcing["PlantGrowth.dayLengthPrev"]

# %%
time = df_forcing["Days"].values

Climate_airTempC_f = interp1d(time, df_forcing["Climate.airTempC"].values)
Climate_solRadGrd_f = interp1d(time, df_forcing["Climate.solRadGrd"].values)
Climate_dayLength_f = interp1d(time, df_forcing["Climate.dayLength"].values)
PlantGrowth_dayLengthPrev_f = interp1d(
    time, df_forcing["PlantGrowth.dayLengthPrev"].values
)
PlantGrowth_Bio_time_f = interp1d(time, df_forcing["PlantGrowth.Bio time"].values)

## Select any time within the time domain
t1 = 100

## plot the interpolated time-series
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes[0, 0].plot(Climate_solRadGrd_f(time), label="solRadGrd")
axes[0, 0].legend()
axes[0, 1].plot(Climate_airTempC_f(time), label="airTempC")
axes[0, 1].legend()
axes[1, 0].plot(Climate_dayLength_f(time), label="dayLength")
axes[1, 0].legend()
axes[1, 1].plot(PlantGrowth_dayLengthPrev_f(time), label="dayLengthPrev")
axes[1, 1].legend()
axes[2, 0].plot(PlantGrowth_Bio_time_f(time), label="Bio_time")
axes[2, 0].legend()

# %% [markdown]
# Initialise the calculator. Then create the Model class with that calculator, initial conditions and start time. 

# %%
PlantX = PlantModuleCalculator(mortality_constant=0.0003)

Model = PlantModelSolver(
    calculator=PlantX, state1_init=0.036, state2_init=0.0870588, time_start=1
)

# %% [markdown]
# Define a time-axis over which to execute the model. Then run the model given some forcing data.

# %%
time_axis = np.arange(1, 1001, 1)

res = Model.run(
    airTempC=Climate_airTempC_f,
    solRadGrd=Climate_solRadGrd_f,
    dayLength=Climate_dayLength_f,
    dayLengthPrev=PlantGrowth_dayLengthPrev_f,
    Bio_time=PlantGrowth_Bio_time_f,
    time_axis=time_axis,
)

# %% [markdown]
# The result that is returned from the run() method is (for now) just the evolution of the ODE over the time domain.

# %%
res.y[0]

# %%
res.y[1]

# %% [markdown]
# Now that the model ODE has been evaluated, you can compute any related "diagnostic" quantities.

# %%
PhBioNPP = PlantX.calculate_PhBioNPP(
    res.y[0],
    Climate_solRadGrd_f(time_axis),
    Climate_airTempC_f(time_axis),
    1,
    0.99,
)

PhBioMort, Fall_litter = PlantX.calculate_PhBioMortality(
    res.y[0],
    Climate_dayLength_f(time_axis),
    PlantGrowth_dayLengthPrev_f(time_axis),
    1,
    0.99,
)

rootBM = PlantX.calculate_rootBM(res.y[1])

propPhAboveBM = PlantX.calculate_propPhAboveBM(res.y[0], rootBM)

Transdown = PlantX.calculate_Transdown(
    res.y[0],
    PhBioNPP,
    Fall_litter,
    PlantGrowth_Bio_time_f(time_axis),
    propPhAboveBM,
)

Transup = PlantX.calculate_Transup(
    res.y[1],
    PlantGrowth_Bio_time_f(time_axis),
    propPhAboveBM,
)

exudation = PlantX.calculate_exudation(rootBM)


# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 10))

axes[0, 0].plot(time_axis, Climate_solRadGrd_f(time_axis), c="0.5")
axes[0, 0].set_ylabel("solRadGrd")
axes[0, 1].plot(time_axis, Climate_airTempC_f(time_axis), c="0.5")
axes[0, 1].set_ylabel("airTempC")
axes[0, 2].plot(res.t, res.y[0], c="C0")
axes[0, 2].set_ylabel("State variable 1\nPhotosynthetic_Biomass")

ax2 = axes[0, 2].twinx()
ax2.plot(res.t, res.y[1], c="C1")
ax2.set_ylabel("State variable 2\nNon_photosynthetic_Biomass")

axes[1, 0].plot(time_axis, PhBioNPP, c="C1", label="PhBioNPP")
axes[1, 0].set_ylabel("Diagnostic Flux: PhBioNPP")
axes[1, 1].plot(time_axis, PhBioMort, c="C2", label="PhBioMort")
axes[1, 1].set_ylabel("Diagnostic Flux: PhBioMort")
axes[1, 2].plot(time_axis, Transdown, c="C3", label="Transdown")
axes[1, 2].plot(time_axis, Transup, c="C4", label="Transup")
axes[1, 2].plot(time_axis, exudation, c="C5", label="exudation")
axes[1, 2].set_ylabel("Diagnostic Fluxes:\nTransdown, Transup, exudation")
axes[1, 2].legend()

plt.tight_layout()

# %%

# %%

# %%

# %%
