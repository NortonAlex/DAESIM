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
import numpy as np
import pandas as pd

# %% [markdown]
# ### Notes for translating code
#
# "TODO": Notes on things to improve in the code, just programming things, not the actual model biophysics.
#
# "ErrorCheck": Check with Firouzeh/Justin to see if this is correct or an error.
#
# "[Question...]": Ask Firouzeh/Justin why this equation/variable/value is used, whether it makes sense.

# %% [markdown]
# ## Model run variables

# %%
start_time = 1
end_time = 365
dt = 1
time = np.arange(start_time,end_time+dt,dt)

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
Climate_CLatDeg = -33.715   ## latitude of site in degrees; Beltsville = 39.0; min = 64.; max = 20.
Climate_rainConv = 0.001   ## Conversion factor for mm/day to m/day
Climate_Elevation = 70.74206543   ## IF there is sediment surface BELOW MSL (e.g., tidal creeks) then use the bathimetry data (depth below MSL) to determine elevation of sediments above the base datum.; ELSE use the land elevation data above MSL (indicating distance from MSL to the soil surface) plus the distance from the datum to mean sea level; ALL VALUES ARE POSITIVE (m) above base datum.
Climate_nYear = 0 # 3   ## Number of years we have empirical data for
Climate_cellArea = 1   ## area of the unit considered m2

# julian day, 1 thru 365
Climate_DayJul = (t - dt)% (365+1)
Climate_DayJulPrev = (t - dt - dt)% (365+1)

Climate_ampl = np.exp(7.42+0.045*Climate_CLatDeg)/3600
Climate_dayLength = Climate_ampl*np.sin((Climate_DayJul-79)*0.01721)+12

Climate_dayLengthPrev = Climate_ampl*np.sin((Climate_DayJulPrev-79)*0.01721)+12


# %%

# %% [markdown]
# ## Conditions for plant

# %%
## Variables that are external to this module but are needed to run a test
plantingDay = 30

# %%
## Green Stella variables 
# TODO: Convert to parameter set
maxAboveBM = 0.6  ## Max above ground biomass kg/m2. 0.9
maxPropPhAboveBM = 0.75   ## proportion of photosynthetic biomass in the above ground biomass (0,1)(alfa*). 0.65
propPhBEverGreen = 0.3  ## Proportion of evergreen photo biomass
iniNPhAboveBM = 0.04  ## initially available above ground non-photosynthetic tissue. kg/m2
propAboveBelowNPhBM = 0.85 ## Ratio of above to below ground non-photosynthetic biomas (beta)
heightMaxBM = 1.2   ## Height at maximum biomass
iniRootDensity = 0.05
propNPhRoot = 0.002    ## [Question: If "NPh" means non-photosynthetic, then why isn't propNPhRoot = 1? 100% of root biomass should be non-photosynthetic]


## Soil module variables:
Soil_calSoilDepth = 0.09


## Plant module variables: Blue Stella variables - calculated outside this module (e.g. state variables)
Non_photosynthetic_Biomass = 0.0871   ## "Non-photosynthetic_Biomass" :: Carbon biomass of the non-photosynthetic portion of macrophytes. Units  kg/m^2 
Photosynthetic_Biomass = 0.036   ## iniPhBM :: Carbon biomass of the photosynthetic portion of plant; Units = kg/m^2 
# TODO: Do we really need this variable? Firouzeh's notes suggest it is a dummy variable required in Stella
Max_Photosynthetic_Biomass = 0.6   ## A dummy, stella enforced variable that is needed for the sole purpose of tracking the maximum attained value of non-photosynthetic biomass; Finding the max of Ph in the whole period of model run; maximal biomass reached during the season


## Blue Stella variables - calculated within this module
# TODO: work out how to implement the PULSE builtin function from Stella into Python
## PULSE( 1 , plantingDay,  365)   ## Planting time May 1  jul calendar day 121. Source Phipps, 1995 pers.comm. see worksheet [corn]; IF Climate.DayJul= plantingDay THEN 1 ELSE 0
if (t <= plantingDay) and (t+dt > plantingDay):
    PlantTime = 1
else: 
    PlantTime = 0
maxPhAboveBM = maxAboveBM*maxPropPhAboveBM  ## The maximum general photosynthetic biomass above ground (not site specific). kg/m2*dimless=kg/m2
maxNPhBM = (maxAboveBM-maxPhAboveBM)*(1+1/propAboveBelowNPhBM)   ## maximum above ground non phtosynthetic+(max above ground non pythosynthetic/proportion of above to below ground non photosynthetic)kg/m2
maxBM = maxNPhBM+maxPhAboveBM   ## kg/m2
iniPhBM = propPhBEverGreen*iniNPhAboveBM*maxPropPhAboveBM/(1-maxPropPhAboveBM)   ## initial biomass of photosynthetic tissue  (kgC/m^2).[]. calculated based on the available above ground non-photosynthetic tissue. PH/(PH+Ab_BM)=Max_Ph_to_Ab_BM. The introduction of the PhBio_evgrn_prop in this equation eliminates all leaves from deciduous trees.  Only coniferous trees are photosynthesizing!
rootBM = Non_photosynthetic_Biomass/(propAboveBelowNPhBM+1)   ## NPHbelow=root biomass=nonphotobiomass*below/(above+below)= nonphoto/(above/below+1). kg/m2
NPhAboveBM = propAboveBelowNPhBM*rootBM   ## Above ground non-photosynthetic biomass; (kg/m2)/(kg/m3)=m
if NPhAboveBM == 0:
    propPhAboveBM = 0 
else:
    propPhAboveBM = Photosynthetic_Biomass/(Photosynthetic_Biomass+NPhAboveBM)

rootDensity = max(iniRootDensity, Non_photosynthetic_Biomass*propNPhRoot*1/Soil_calSoilDepth)   ## 

# ErrorCheck: What is (Climate_Elevation/100)-1 meant to represent??
RootDepth = max((Climate_Elevation/100)-1,(rootBM/rootDensity))  ## Make sure that roots don't get longer than the elevation - needed in calculations of available DIN.; W= winter wheat (2.2 m) , spring wheat (1.1 m); B= 1.2; R= 1.8; M=12; C= 1.65; Ct= 0.3; S=1.4; Wheat roots penetrated to a maximum depth of 104 cm in crops sown in May, the optimum time of sowing for maximum yield, while delayed sowing reduced total root biomass and limited rooting depth to 73-83 cm
rootDensity = max(iniRootDensity, Non_photosynthetic_Biomass*propNPhRoot*1/Soil_calSoilDepth)

if NPhAboveBM == 0:
    propPhAboveBM = 0
else:
    propPhAboveBM = Photosynthetic_Biomass/(Photosynthetic_Biomass+NPhAboveBM)
    
TotalBM = Photosynthetic_Biomass+Non_photosynthetic_Biomass

if maxBM>0:
    CalcuHeight = heightMaxBM * TotalBM/maxBM
else:
    CalcuHeight = 0
    
if Photosynthetic_Biomass>Max_Photosynthetic_Biomass:
    PHinflow = Photosynthetic_Biomass/DT 
else:
    PHinflow = 0
    
if (Photosynthetic_Biomass>Max_Photosynthetic_Biomass) or (PlantTime == 1):
    PHoutflow = Max_Photosynthetic_Biomass/DT 
else:
    PHoutflow = 0

# %%
### Plant water stress

## Green Stella variables 
# TODO: Convert to parameter set
calibrationMinWater = 0.29   ## When > 1 there is always a limitation if not optimal; when <1 there is almost no limitation unless conditions -> 0;  the smaller the value - the less sensitivity to water stress; water deficit tolerance coefficient
calibrationMaxWater = 0.1   ## The larger the parameter, the more sensitivity to high water stress; 0 - no sensitivity


## Water module variables:
Water_waterAvailable = 0.850826175866   ## See Stella docs
Water_unsatWatDepth = 38.74206543   ## Depth (m) of the unsaturated zone. 
Water_Surface_Water = 0   ## See Stella docs

# WatStressLow
WatStressLow = max(0, np.sin(Water_waterAvailable*np.pi/2)**calibrationMinWater)

# WatStressHigh
if (calibrationMaxWater < 0) and (Water_Surface_Water>-calibrationMaxWater):
    WatStressHigh = max(0, 1/(1+np.exp(Water_Surface_Water+calibrationMaxWater)))
elif (calibrationMaxWater > 0) and (RootDepth > 0):
    WatStressHigh = min(1, Water_unsatWatDepth/RootDepth/calibrationMaxWater)
else:
    WatStressHigh = 1

# %%
### Bio time for plant
# Biological time counter
if PlantTime == 1:
    cf_Air_Temp = -Bio_time/DT   ## TODO: Remove time-step dependency
elif Climate_airTempC > 5:
    cf_Air_Temp = Climate_airTempC
else:
    cf_Air_Temp = 0
    
### TODO: Make Bio_time a state variable, as it stores information from time-step to time-step. It is analagous to cumulative degree days
# Bio_time(t - dt) + (cf_Air_Temp) * dt
Bio_time = df_forcing["PlantGrowth.Bio time"].values[it]


# %% [markdown]
# ## Plant

# %%
DMP = 3291.13599   # TODO: This is a forcing variable   ## dry matter production kgDM/ha/day
empiricalNPP = DMP*0.45*0.1/1000   ## gC/m2/day

# %% [markdown]
# ### NPP

# %%
## Green Stella variables 
# TODO: Convert to parameter set
halfSaturCoeffP = 0.01   ## half-saturation coefficient for P; 0.000037 [Question: why do the Stella docs give a value (0.000037) that is different to the actual value used (0.01)?]
halfSaturCoeffN = 0.02  ## half-saturation coefficient for Nitrogen; 0.00265 [Question: why do the Stella docs give a value (0.00265) that is different to the actual value used (0.002)?]
optTemperature = 20   ## optimal temperature
NPPCalibration = 0.45   ## NPP 1/day(The rate at which an ecosystem accumulates energy); 0.12; 0.25

# Saturating light intensity (langleys/d) for the selected crop
# {for macrophytes-> . Source:  599 [corn](Chang, 1981 Table 1)
# 1600 (µmol irradiance m^2 *s) coniferous forests ( Gholtz et all 1994, page40)
# Unit conversion:
# 10 langley = 1kcal/m^2
# 1 langleys/d =0.48458 watt
# Irradiance =1 watt/m^2 (1 watt =1joule/sec)
# 1.43e-3 Langleys/min =1µE M/s
saturatingLightIntensity = 600 

## Climate module variables:
Climate_solRadGrd = 20.99843025   ## correction for cloudy days
Climate_airTempC = 21.43692112   ## Including the F to C conversion. May need be changed if data are in C already; So as the data are already in C we have changed this

## Blue Stella variables - calculated within this module
PO4Aval = 0.2   # TODO: Change name to "PO4Avail"; ErrorCheck: What is this? Why is it set to 0 in Stella? This makes the NutrCoeff=0, NPPControlCoeff=0, then calculatedPhBioNPP=0
DINAvail = 0.2   # [Question: No documentation. Check paper or ask Firouzeh/Justin to provide.] ErrorCheck: What is this? Why is it set to 0 in Stella? This makes the NutrCoeff=0, NPPControlCoeff=0, then calculatedPhBioNPP=0
NutrCoeff = min( (DINAvail/(DINAvail+halfSaturCoeffN)),  (PO4Aval/(PO4Aval+halfSaturCoeffP) ))

LightCoeff = Climate_solRadGrd*10/saturatingLightIntensity*np.exp(1-Climate_solRadGrd/saturatingLightIntensity)

WaterCoeff = min(WatStressHigh,WatStressLow)

TempCoeff = np.exp(0.20*(Climate_airTempC-optTemperature)) * np.abs(((40-Climate_airTempC)/(40-optTemperature)))**(0.2*(40-optTemperature))   ## See Stella docs

NPPControlCoeff = min(LightCoeff,TempCoeff)*WaterCoeff*NutrCoeff   ## Total control function for primary production,  using minimum of physical control functions and multiplicative nutrient and water controls.  Units=dimensionless.

# calculatedPhBioNPP ## Estimated net primary productivity
if (Photosynthetic_Biomass<maxPhAboveBM):
    calculatedPhBioNPP = NPPControlCoeff*NPPCalibration*Photosynthetic_Biomass*(1-Photosynthetic_Biomass/maxPhAboveBM)
else:
    calculatedPhBioNPP = 0

# %%
if Climate_DayJul < 366*Climate_nYear:
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
propPhMortality = 0.015   ## Proportion of photosynthetic biomass that dies in fall time
propPhtoNPhMortality = 0   ## A proportion that decides how much of the Ph biomass will die or go to the roots at the fall time
propPhBLeafRate = 0   ## Leaf fall rate
dayLengRequire = 13   ## [Question: Need some information on this.]

## Climate module variables:
Climate_dayLength = Climate_ampl*np.sin((Climate_DayJul-79)*0.01721)+12


## Blue Stella variables - calculated within this module

WaterCoeff = min(WatStressHigh,WatStressLow)   ## TODO: Modify the structure here as it is used a couple of times in the Plant module

PropPhMortDrought = 0.1*max(0,(1-WaterCoeff))

# FallLitterCalc
if (Climate_dayLength > dayLengRequire) or (Climate_dayLength>=Climate_dayLengthPrev):
    FallLitterCalc = 0 
elif Photosynthetic_Biomass < 0.01*(1-propPhBEverGreen):
    FallLitterCalc = (1-propPhBEverGreen)*Photosynthetic_Biomass/DT    ## TODO: Remove use of DT and time-step dependency
else:
    FallLitterCalc = (1-propPhBEverGreen)*(Max_Photosynthetic_Biomass*propPhBLeafRate/Photosynthetic_Biomass)**3

Fall_litter = min(FallLitterCalc, max(0,Photosynthetic_Biomass-propPhBEverGreen*Max_Photosynthetic_Biomass/(1+propPhBEverGreen)))

## PhBioMort: mortality of photosynthetic biomass as influenced by seasonal cues plus mortality due to current 
## (not historical) water stress.  Use maximum specific rate of mortality and constraints due to unweighted 
## combination of seasonal litterfall and water stress feedbacks (both range 0,1). units = 1/d * kg * (dimless +dimless)  = kg/d
PhBioMort = Fall_litter*(1-propPhtoNPhMortality)+Photosynthetic_Biomass*(PropPhMortDrought +propPhMortality)

# %% [markdown]
# ### Transdown

# %%
## Green Stella variables 
# TODO: Convert to parameter set

BioRepro = 1800   ## bio time when reproductive organs start to grow; 1900
propPhtoNphReproduction = 0.005   ## fraction of photo biomass that may be transferred to non-photo when reproduction occurs
## TODO: This parameter is used in calculation of both PhBioMort and Transdown, make sure its available to both calculations
propPhtoNPhMortality = 0  ## A proportion that decides how much of the Ph biomass will die or go to the roots at the fall time
## TODO: This parameter is used in calculation of both "Conditions for plant", Transdown and Transup, make sure its available to both calculations
maxPropPhAboveBM = 0.75   ## proportion of photosynthetic biomass in the above ground biomass (0,1)(alfa*); 0.65


## Blue Stella variables - calculated within this module

# TransdownRate: 
# The plant attempts to obtain the optimum photobiomass to total above ground biomass ratio.  Once this is reached, 
# NPP is used to grow more Nonphotosythethic biomass decreasing the optimum ratio.  This in turn allows new Photobiomass 
# to compensate for this loss; IF Ph_to_Ab_BM[Habitat,Soil]>•Max_Ph_to_Ab_BM[Habitat,Soil] THEN 1; ELSE Ph_to_Ab_BM[Habitat,Soil]/•Max_Ph_to_Ab_BM[Habitat,Soil]

if Bio_time>BioRepro+1:
    TransdownRate = 1-1/(Bio_time-BioRepro)**0.5 
elif propPhAboveBM<maxPropPhAboveBM:
    TransdownRate = 0
else:
    TransdownRate = np.cos((maxPropPhAboveBM/propPhAboveBM)*np.pi/2)**0.1

    
## TODO: Include HarvestTime info
# if HarvestTime > 0:
#     Transdown = 0
# else:
#     Transdown = TransdownRate*(PhNPP+propPhtoNphReproduction*Photosynthetic_Biomass)+(propPhtoNPhMortality*Fall_litter)
Transdown = TransdownRate*(PhNPP+propPhtoNphReproduction*Photosynthetic_Biomass)+(propPhtoNPhMortality*Fall_litter)


# %% [markdown]
# ### Transup

# %%
## Green Stella variables 
# TODO: Convert to parameter set
sproutRate = 0.01   ## Sprouting rate. Rate of translocation of assimilates from non-photo to photo bimass during early growth period
bioStart = 20   ## start of sprouting. [Question: What is the physiological definition?]
bioEnd = 80   ## [Question: What is the  physiological definition?]
## TODO: This parameter is used in calculation of both "Conditions for plant", Transdown and Transup, make sure its available to both calculations
maxPropPhAboveBM = 0.75


if (propPhAboveBM < maxPropPhAboveBM) and (Bio_time > bioStart) and (Bio_time < bioEnd):
    Sprouting = 1
else:
    Sprouting = 0

Transup = Sprouting * sproutRate * Non_photosynthetic_Biomass


# %%
