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
# "TODO": Notes on things to improve in the code, just programming things, not the actual model biophysics
#
# "ErrorCheck": Check with Firouzeh/Justin to see why this equation is used and whether it makes sense

# %% [markdown]
# ## Model run variables

# %%
START_TIME = 1
END_TIME = 365
DT = 1
TIME = np.arange(START_TIME,END_TIME+DT,DT)

# %% [markdown]
# ## Forcing data

# %%
file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim-stella/StellaOutFile.csv"

df_forcing = pd.read_csv(file)

# %% [markdown]
# ## Climate

# %%
## Green Stella variables 
# TODO: Convert to parameter set
CLatDeg = -33.715   ## latitude of site in degrees; Beltsville = 39.0; min = 64.; max = 20.
rainConv = 0.001   ## Conversion factor for mm/day to m/day
Elevation = 70.74206543   ## IF there is sediment surface BELOW MSL (e.g., tidal creeks) then use the bathimetry data (depth below MSL) to determine elevation of sediments above the base datum.; ELSE use the land elevation data above MSL (indicating distance from MSL to the soil surface) plus the distance from the datum to mean sea level; ALL VALUES ARE POSITIVE (m) above base datum.
nYear = 3   ## Number of years we have empirical data for
cellArea = 1   ## area of the unit considered m2


# julian day, 1 thru 365
DayJul = TIME

# %%

# %%

# %% [markdown]
# ## Conditions for plant

# %%
## Variables that are external to this module but are needed to run a test
t = 1
DT = 1
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
propNPhRoot = 0.002    ## Question: If "NPh" means non-photosynthetic, then why isn't propNPhRoot = 1? 100% of root biomass should be non-photosynthetic


## Climate module variables:
Climate_Elevation = 70.74206543   ## IF there is sediment surface BELOW MSL (e.g., tidal creeks) then use the bathimetry data (depth below MSL) to determine elevation of sediments above the base datum.; ELSE use the land elevation data above MSL (indicating distance from MSL to the soil surface) plus the distance from the datum to mean sea level; ALL VALUES ARE POSITIVE (m) above base datum.


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
if (TIME <= plantingDay) and (TIME+DT > plantingDay):
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

# %% [markdown]
# ## Plant

# %%
DMP = 3291.13599   # TODO: This is a forcing variable   ## dry matter production kgDM/ha/day
empiricalNPP = DMP*0.45*0.1/1000   ## gC/m2/day

# %%
if Climate.DayJul < 366*Climate.nYear:
    PhNPP = empiricalNPP
elif HarvestTime > 0:
    PhNPP = 0
else:
    # TODO: include function for "calculatedPhBioNPP"
    PhNPP = calculatedPhBioNPP

# %%
