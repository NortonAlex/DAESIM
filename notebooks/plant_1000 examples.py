# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext nb_black

# %%
from attrs import define, field
from typing import Tuple, Callable
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

# %%
from daesim.climate import *
from daesim.biophysics_funcs import func_TempCoeff, growing_degree_days_DTT_nonlinear, growing_degree_days_DTT_linear1, growing_degree_days_DTT_linear2, growing_degree_days_DTT_linear3
from daesim.plantgrowthphases import PlantGrowthPhases
from daesim.management import ManagementModule
from daesim.soillayers import SoilLayers
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.boundarylayer import BoundaryLayerModule
from daesim.leafgasexchange import LeafGasExchangeModule
from daesim.leafgasexchange2 import LeafGasExchangeModule2
from daesim.canopygasexchange import CanopyGasExchange
from daesim.plantcarbonwater import PlantModel as PlantCH2O
from daesim.plantallocoptimal import PlantOptimalAllocation

# %%
from daesim.plant_1000 import PlantModuleCalculator

# %%

# %% [markdown]
# # Site Level Simulation

# %% [markdown]
# ### Initialise site

# %% [markdown]
# #### - Import forcing data

# %%
# file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim/data/DAESim_forcing_Milgadara_2021.csv"
file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim/data/DAESim_forcing_Harden_2001.csv"

df_forcing = pd.read_csv(file)

# %%
df_forcing

# %% [markdown]
# #### - Generation temporally-interpolated and scaled soil moisture forcing
#
# The model requires soil moisture to be in units of volumetric soil water content (m3 m-3 or %). Soil moisture from the forcing data above is in units of mm. So we make a (pretty crude) assumption on how to convert mm to volumetrics soil water content. 

# %%
df_forcing["Soil moisture interp"] = df_forcing["Soil moisture"].interpolate('quadratic')

## Assume that the forcing data (units: mm) can be equated to relative changes in volumetric soil moisture between two arbitrary minimum and maximum values
f_soilTheta_min = 0.25
f_soilTheta_max = 0.40

f_soilTheta_min_mm = df_forcing["Soil moisture interp"].min()
f_soilTheta_max_mm = df_forcing["Soil moisture interp"].max()

f_soilTheta_norm_mm = (df_forcing["Soil moisture interp"].values - f_soilTheta_min_mm)/(f_soilTheta_max_mm - f_soilTheta_min_mm)
f_soilTheta_norm = f_soilTheta_min + f_soilTheta_norm_mm * (f_soilTheta_max - f_soilTheta_min)


fig, axes = plt.subplots(1,2,figsize=(9,3))

axes[0].scatter(df_forcing.index.values, df_forcing["Soil moisture"].values)
axes[0].plot(df_forcing.index.values, df_forcing["Soil moisture interp"].values)
axes[0].set_ylabel("Soil moisture (mm)")

axes[1].plot(df_forcing.index.values, f_soilTheta_norm)
axes[1].set_ylabel("Volumetric soil moisture")

plt.tight_layout()

# %% [markdown]
# #### - Initialise site module

# %%
## Milgadara site location-34.38904277303204, 148.46949938279096
# SiteX = ClimateModule(CLatDeg=-34.389,CLonDeg=148.469,timezone=10)
# start_doy = 1.0
# start_year = 2021
# nrundays = df_forcing.index.size

## Harden CSIRO site location-34.52194, 148.30472
SiteX = ClimateModule(CLatDeg=-34.52194,CLonDeg=148.30472,timezone=10)
start_doy = 1.0
start_year = 2001
nrundays = df_forcing.index.size

# %% [markdown]
# #### - Time discretization

# %%
## Time discretisation
time_nday, time_doy, time_year = SiteX.time_discretisation(start_doy, start_year, nrundays=nrundays)
## Adjust daily time-step to represent midday on each day
time_doy = [time_doy[i]+0.5 for i in range(len(time_doy))]

# %% [markdown]
# ### Create discrete forcing data

# %%
## Make some assumption about the fraction of diffuse radiation
diffuse_fraction = 0.2

## Shortwave radiation at surface (convert MJ m-2 d-1 to W m-2)
_Rsb_Wm2 = (1-diffuse_fraction) * df_forcing["SRAD"].values * 1e6 / (60*60*24)
_Rsd_Wm2 = diffuse_fraction * df_forcing["SRAD"].values * 1e6 / (60*60*24)

## Create synthetic data for other forcing variables
_p = 101325*np.ones(nrundays)
_es = SiteX.compute_sat_vapor_pressure_daily(df_forcing["Minimum temperature"].values,df_forcing["Maximum temperature"].values)
_RH = SiteX.compute_relative_humidity(df_forcing["VPeff"].values/10,_es/1000)
_RH[_RH > 100] = 100
_CO2 = 400*(_p/1e5)*1e-6     ## carbon dioxide partial pressure (bar)
_O2 = 209000*(_p/1e5)*1e-6   ## oxygen partial pressure (bar)
_soilTheta =  0.35*np.ones(nrundays)   ## volumetric soil moisture content (m3 m-3)
_soilTheta = f_soilTheta_norm

## Create a multi-layer soil moisture forcing dataset
## Option 1: Same soil moisture across all layers
nlevmlsoil = 2
_soilTheta_z = np.repeat(_soilTheta[:, np.newaxis], nlevmlsoil, axis=1)
## Option 2: Adjust soil moisture in each layer
_soilTheta_z0 = _soilTheta-0.04
_soilTheta_z1 = _soilTheta+0.04
_soilTheta_z = np.column_stack((_soilTheta_z0, _soilTheta_z1))

# %% [markdown]
# ### Convert discrete to continuous forcing data 

# %%
Climate_doy_f = interp_forcing(time_nday, time_doy, kind="pconst", fill_value=(time_doy[0],time_doy[-1]))
Climate_year_f = interp_forcing(time_nday, time_year, kind="pconst", fill_value=(time_year[0],time_year[-1]))
Climate_airTempCMin_f = interp1d(time_nday, df_forcing["Minimum temperature"].values)
Climate_airTempCMax_f = interp1d(time_nday, df_forcing["Maximum temperature"].values)
Climate_airTempC_f = interp1d(time_nday, (df_forcing["Minimum temperature"].values+df_forcing["Maximum temperature"].values)/2)
Climate_solRadswskyb_f = interp1d(time_nday, _Rsb_Wm2)
Climate_solRadswskyd_f = interp1d(time_nday, _Rsd_Wm2)
Climate_airPressure_f = interp1d(time_nday, _p)
Climate_airRH_f = interp1d(time_nday, _RH)
Climate_airU_f = interp1d(time_nday, df_forcing["Uavg"].values)
Climate_airCO2_f = interp1d(time_nday, _CO2)
Climate_airO2_f = interp1d(time_nday, _O2)
Climate_soilTheta_f = interp1d(time_nday, _soilTheta)
Climate_soilTheta_z_f = interp1d(time_nday, _soilTheta_z, axis=0)  # Interpolates across timesteps, handles all soil layers at once
Climate_nday_f = interp1d(time_nday, time_nday)   ## nday represents the ordinal day-of-year plus each simulation day (e.g. a model run starting on Jan 30 and going for 2 years will have nday=30+np.arange(2*365))


# %% [markdown]
# ### Test the rate of change calculate method

# %%
# _nday = 1
# _Cleaf = 100.0
# _Cstem = 50.0
# _Croot = 90.0
# _Cseed = 0.0
# _Bio_time = 0.0
# _VD_time = 0.0
# _HTT_time = 0.0
# _Cprod = 0.0
# _solRadswskyb = 800    ## incoming shortwave radiation, beam (W m-2)
# _solRadswskyd = 200    ## incoming shortwave radiation, diffuse (W m-2)
# _airTempCMin = 13.88358116
# _airTempCMax = 28.99026108
# _airTempC = (_airTempCMin + _airTempCMax)/2
# _airPressure = 101325  ## atmospheric pressure (Pa)
# _airRH = 65.0   ## relative humidity (%) 
# _airCO2 = 400*(_airPressure/1e5)*1e-6     ## carbon dioxide partial pressure (bar)
# _airO2 = 209000*(_airPressure/1e5)*1e-6   ## oxygen partial pressure (bar)
# _soilTheta = np.array([0.30,0.30])   ## volumetric soil water content (m3 m-3)
# _doy = time_doy[_nday-1]
# _year = time_year[_nday-1]

# management = ManagementModule(sowingDay=30,harvestDay=235)
# site = ClimateModule()
# soillayers = SoilLayers(nlevmlsoil=2,z_max=2.0)
# canopy = CanopyLayers()
# canopyrad = CanopyRadiation(Canopy=canopy)
# leaf = LeafGasExchangeModule2(Site=site)
# canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopyRad=canopyrad)
# plantch2o = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange)
# plantalloc = PlantOptimalAllocation(Plant=plantch2o,dWL_factor=1.01,dWR_factor=1.02)
# plant = PlantModuleCalculator(Site=site,Management=management,PlantCH2O=plantch2o,PlantAlloc=plantalloc,remob_phase="anthesis")

# dydt = plant.calculate(
#     _Cleaf,
#     _Cstem,
#     _Croot,
#     _Cseed,
#     _Bio_time,
#     _VD_time,
#     _HTT_time,
#     _Cprod,
#     _solRadswskyb,
#     _solRadswskyd,
#     _airTempCMin,
#     _airTempCMax,
#     _airPressure,
#     _airRH,
#     _airCO2,
#     _airO2,
#     _soilTheta,
#     _doy,
#     _year,
# )
# print("dy/dt =", dydt)
# print()
# print("  dydt(Cleaf) = %1.4f" % dydt[0])
# print("  dydt(Cstem) = %1.4f" % dydt[1])
# print("  dydt(Croot) = %1.4f" % dydt[2])
# print("  dydt(Cseed) = %1.4f" % dydt[3])
# print("  Bio_time = %1.4f" % dydt[4])
# print("  VRN_time = %1.4f" % dydt[5])
# print("  HTT_time = %1.4f" % dydt[6])
# print("  Cprod = %1.4f" % dydt[7])

# %% [markdown]
# ## Use Model with Numerical Solver

# %%
from daesim.utils import ODEModelSolver

# %% [markdown]
# ### Initialise aggregated model with its classes, initial values for the states, and time axis

# %%
# time_axis = np.arange(122, 332, 1)   ## Note: time_axis represents the simulation day (_nday) and must be the same x-axis upon which the forcing data was interpolated on
# sowing_date=122
# harvest_date=332

time_axis = np.arange(135, 335, 1)   ## Note: time_axis represents the simulation day (_nday) and must be the same x-axis upon which the forcing data was interpolated on
sowing_date = 135
harvest_date = 330

# %%
## PlantDev
# PlantDevX = PlantGrowthPhases(
#     phases=["germination", "vegetative", "anthesis", "grainfill", "maturity"],
#     gdd_requirements=[120,850,350,200,300],
#     vd_requirements=[0,30,0,0,0],
#     allocation_coeffs = [
#         [0.0, 0.1, 0.9, 0.0, 0.0],
#         [0.75, 0.05, 0.2, 0.0, 0.0],
#         [0.25, 0.4, 0.25, 0.1, 0.0],
#         [0.15, 0.2, 0.15, 0.5, 0.0],
#         [0.15, 0.2, 0.15, 0.5, 0.0]
#     ],
#     turnover_rates = [[5.000e-04, 5.000e-04, 5.000e-04, 0.000e+00, 0.000e+00],
#        [1.830e-02, 1.000e-03, 4.150e-03, 0.000e+00, 0.000e+00],
#        [3.165e-02, 1.000e-03, 4.150e-03, 0.000e+00, 0.000e+00],
#        [5.000e-02, 4.000e-03, 2.500e-02, 5.000e-05, 0.000e+00],
#         [0.10, 0.033, 0.10, 0.0002, 0.0]])


## PlantDev with specific spike formation phase - especially important for for wheat
PlantDevX = PlantGrowthPhases(
    phases=["germination", "vegetative", "spike", "anthesis", "grainfill", "maturity"],
    gdd_requirements=[50,500,200,110,300,100],
    vd_requirements=[0, 40, 0, 0, 0, 0],
    allocation_coeffs = [
        [0.2, 0.1, 0.7, 0.0, 0.0],
        [0.5, 0.1, 0.4, 0.0, 0.0],
        [0.20, 0.6, 0.20, 0.0, 0.0],
        [0.25, 0.5, 0.25, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.7, 0.0],
        [0.1, 0.1, 0.1, 0.7, 0.0]
    ],
    turnover_rates = [[0.001,  0.001, 0.001, 0.0, 0.0],
                      [0.01, 0.002, 0.008, 0.0, 0.0],
                      [0.01, 0.002, 0.008, 0.0, 0.0],
                      [0.01, 0.002, 0.008, 0.0, 0.0],
                      [0.033, 0.016, 0.033, 0.0002, 0.0],
                      [0.10, 0.033, 0.10, 0.0002, 0.0]])

# %%
ManagementX = ManagementModule(cropType="Wheat",sowingDay=sowing_date,harvestDay=harvest_date,propHarvestLeaf=0.75)

BoundLayerX = BoundaryLayerModule(Site=SiteX)
LeafX = LeafGasExchangeModule2(Site=SiteX)
CanopyX = CanopyLayers(nlevmlcan=3)
CanopyRadX = CanopyRadiation(Canopy=CanopyX)
CanopyGasExchangeX = CanopyGasExchange(Leaf=LeafX,Canopy=CanopyX,CanopyRad=CanopyRadX)
SoilLayersX = SoilLayers(nlevmlsoil=2,z_max=2.0)
PlantCH2OX = PlantCH2O(Site=SiteX,SoilLayers=SoilLayersX,CanopyGasExchange=CanopyGasExchangeX,BoundaryLayer=BoundLayerX,maxLAI=5.0,ksr_coeff=1000,SLA=0.040)
PlantAllocX = PlantOptimalAllocation(Plant=PlantCH2OX,dWL_factor=1.02,dWR_factor=1.02)
PlantX = PlantModuleCalculator(
    Site=SiteX,
    Management=ManagementX,
    PlantDev=PlantDevX,
    PlantCH2O=PlantCH2OX,
    PlantAlloc=PlantAllocX,
    GDD_method="linear1",
    GDD_Tbase=0.0,
    GDD_Tupp=25.0,
    hc_max_GDDindex=sum(PlantDevX.gdd_requirements[0:2])/PlantDevX.totalgdd,
    d_r_max=2.0,
    Vmaxremob=3.0,
    Kmremob=0.5,
    remob_phase="grainfill",
    specified_phase="spike",
    grainfill_phase="grainfill",
)

# %%
## Define the callable calculator that defines the right-hand-side ODE function
PlantXCalc = PlantX.calculate

Model = ODEModelSolver(calculator=PlantXCalc, states_init=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_start=time_axis[0], log_diagnostics=True)


forcing_inputs = [Climate_solRadswskyb_f,
                  Climate_solRadswskyd_f,
                  Climate_airTempCMin_f,
                  Climate_airTempCMax_f,
                  Climate_airPressure_f,
                  Climate_airRH_f,
                  Climate_airCO2_f,
                  Climate_airO2_f,
                  Climate_airU_f,
                  Climate_soilTheta_z_f,
                  Climate_doy_f,
                  Climate_year_f]

reset_days = [PlantX.Management.sowingDay, PlantX.Management.harvestDay]

res = Model.run(
    time_axis=time_axis,
    forcing_inputs=forcing_inputs,
    solver="euler",
    zero_crossing_indices=[4,5,6],
    reset_days=reset_days,
    # rtol=1e-2,
    # atol=1e-2
)

# %% [markdown]
# ### Diagnostic outputs

# %%
# Convert the defaultdict to a regular dictionary
_diagnostics = dict(Model.diagnostics)
# Convert each list in the dictionary to a NumPy array
diagnostics = {key: np.array(value) for key, value in _diagnostics.items()}

# %%
## Conversion notes: When _E units are mol m-2 s-1, multiply by molar mass H2O to get g m-2 s-1, divide by 1000 to get kg m-2 s-1, multiply by 60*60*24 to get kg m-2 d-1, and 1 kg m-2 d-1 = 1 mm d-1. 
## Noting that 1 kg of water is equivalent to 1 liter (L) of water (because the density of water is 1000 kg/m³), and 1 liter of water spread over 1 square meter results in a depth of 1 mm
diagnostics["E_mmd"] = diagnostics["E"]*18.015/1000*(60*60*24)

# Turnover rates per pool
_tr_Leaf = np.zeros(diagnostics['t'].size)
_tr_Root = np.zeros(diagnostics['t'].size)
_tr_Stem = np.zeros(diagnostics['t'].size)
_tr_Seed = np.zeros(diagnostics['t'].size)
for it, t in enumerate(diagnostics['t']):
    tr_ = PlantX.PlantDev.turnover_rates[diagnostics['idevphase'][it]]
    _tr_Leaf[it] = tr_[PlantX.PlantDev.ileaf]
    _tr_Root[it] = tr_[PlantX.PlantDev.iroot]
    _tr_Stem[it] = tr_[PlantX.PlantDev.istem]
    _tr_Seed[it] = tr_[PlantX.PlantDev.iseed]

diagnostics['tr_Leaf'] = _tr_Leaf
diagnostics['tr_Root'] = _tr_Root
diagnostics['tr_Stem'] = _tr_Stem
diagnostics['tr_Seed'] = _tr_Seed

# %% [markdown]
# ### Scalar Variables
#
# State variables and all diagnostic variables (e.g. GPP, NPP, carbon allocation fluxes, carbon allocation coefficients, stem remobilisation rate, transpiration rate, daily thermal time, vernalisation time, relative GDD, etc.)
#
# ### - Carbon and water
#  - Total dry biomass at peak biomass and harvest
#  - Leaf dry biomass at peak biomass and harvest
#  - Root dry biomass at peak biomass
#  - Stem dry biomass at peak biomass and harvest and harvest
#  - Stem dry biomass at start of spike   # to infer spike dry weight
#  - Stem dry biomass at start of anthesis   # to infer spike dry weight
#  - Total (integrated) seasonal GPP   # when to end the integration period?? At start of maturity? At harvest?
#  - Total (integrated) seasonal NPP   # when to end the integration period?? At start of maturity? At harvest?
#  - Total (integrated) seasonal Rml   # when to end the integration period?? At start of maturity? At harvest?
#  - Total (integrated) seasonal Rmr   # when to end the integration period?? At start of maturity? At harvest?
#  - Total (integrated) seasonal Rg   # when to end the integration period?? At start of maturity? At harvest?
#  - Total (integrated) seasonal turnover losses   # when to end the integration period?? At start of maturity? At harvest?
#  - Total (integrated) remobilisation to grain   # to determine fraction of grain supported by stem remobilisation
#  - Total (integrated) allocation to grain   # to determine fraction of grain supported by new assimilates
#  - Total (integrated) seasonal transpiration
#  - Leaf area index at peak biomass
#
# ### - Development
#  - Total GDD to maturity   # diagnosed from inputs
#  - Relative GDD to vegetative
#  - Relative GDD to spike
#  - Relative GDD to anthesis
#  - Relative GDD to grain filling
#  - Relative GDD to maturity
#
# ### - Grain production
#  - Spike dry biomass at anthesis (Difference between stem dry biomass between start of spike and start of anthesis)
#  - Grain yield at maturity
#  - Potential grain number
#  - Actual grain number

# %%
total_carbon_t = res["y"][PlantX.PlantDev.ileaf,:] + res["y"][PlantX.PlantDev.istem,:] + res["y"][PlantX.PlantDev.iroot,:] + res["y"][PlantX.PlantDev.iseed,:]
total_carbon_exclseed_t = res["y"][PlantX.PlantDev.ileaf,:] + res["y"][PlantX.PlantDev.istem,:] + res["y"][PlantX.PlantDev.iroot,:]

it_peakbiomass = np.argmax(total_carbon_t)
it_peakbiomass_exclseed = np.argmax(total_carbon_exclseed_t)

it_sowing = np.where(time_axis == PlantX.Management.sowingDay)[0][0]
it_harvest = np.where(time_axis == PlantX.Management.harvestDay)[0][0]

# Diagnose time indexes when developmental phase transitions occur
it_phase_transitions = np.where(np.diff(diagnostics['idevphase']) != 0)[0] + 1
# Filter out transitions that occur after the harvest day
it_phase_transitions = [t for t in it_phase_transitions if time_axis[t] <= PlantX.Management.harvestDay]

# %%
# Developmental phase indexes
igermination = PlantX.PlantDev.phases.index("germination")
ivegetative = PlantX.PlantDev.phases.index("vegetative")
if PlantX.Management.cropType == "Wheat":
    ispike = PlantX.PlantDev.phases.index("spike")
ianthesis = PlantX.PlantDev.phases.index("anthesis")
igrainfill = PlantX.PlantDev.phases.index("grainfill")
imaturity = PlantX.PlantDev.phases.index("maturity")

print("--- Carbon and Water ---")
print()
print("Total dry biomass at peak biomass =", total_carbon_t[it_peakbiomass]/PlantX.PlantCH2O.f_C)
print("Leaf dry biomass at peak biomass =", res["y"][PlantX.PlantDev.ileaf,it_peakbiomass]/PlantX.PlantCH2O.f_C)
print("Root dry biomass at peak biomass =", res["y"][PlantX.PlantDev.istem,it_peakbiomass]/PlantX.PlantCH2O.f_C)
print("Stem dry biomass at peak biomass =", res["y"][PlantX.PlantDev.iroot,it_peakbiomass]/PlantX.PlantCH2O.f_C)
if PlantX.Management.cropType == "Wheat":
    ip = np.where(diagnostics['idevphase'][it_phase_transitions] == PlantX.PlantDev.phases.index('spike'))[0][0]
    print("Stem dry biomass at start of spike =", res["y"][PlantX.PlantDev.istem,it_phase_transitions[ip]]/PlantX.PlantCH2O.f_C)
ip = np.where(diagnostics['idevphase'][it_phase_transitions] == PlantX.PlantDev.phases.index('anthesis'))[0][0]
print("Stem dry biomass at start of anthesis =", res["y"][PlantX.PlantDev.istem,it_phase_transitions[ip]]/PlantX.PlantCH2O.f_C)
print("Total (integrated) seasonal GPP =", np.sum(diagnostics['GPP'][it_sowing:it_harvest+1]))
print("Total (integrated) seasonal NPP =", np.sum(diagnostics['NPP'][it_sowing:it_harvest+1]))
print("Total (integrated) seasonal Rml =", np.sum(diagnostics['Rml'][it_sowing:it_harvest+1]))
print("Total (integrated) seasonal Rmr =", np.sum(diagnostics['Rmr'][it_sowing:it_harvest+1]))
print("Total (integrated) seasonal Rg =", np.sum(diagnostics['Rg'][it_sowing:it_harvest+1]))
print("Total (integrated) seasonal turnover losses =", np.sum(diagnostics['trflux_total'][it_sowing:it_harvest+1]))
print("Total (integrated) remobilisation to grain =", np.sum(diagnostics['F_C_stem2grain'][it_sowing:it_harvest+1]))
_Cflux_NPP2grain = diagnostics['u_Seed'] * diagnostics['NPP']
print("Total (integrated) allocation to grain =", np.sum(_Cflux_NPP2grain[it_sowing:it_harvest+1]))
print("Total (integrated) seasonal transpiration =", np.sum(diagnostics['E_mmd'][it_sowing:it_harvest+1]))
print("Leaf area index at peak biomass =", diagnostics['LAI'][it_peakbiomass])
print()
print("--- Development ---")
print()
print("Total GDD to maturity =", PlantX.PlantDev.totalgdd)
print("Relative GDD to vegetative =", (sum(PlantX.PlantDev.gdd_requirements[:ivegetative-1]) + PlantX.PlantDev.gdd_requirements[ivegetative-1])/PlantX.PlantDev.totalgdd)
if PlantX.Management.cropType == "Wheat":
    print("Relative GDD to spike =", (sum(PlantX.PlantDev.gdd_requirements[:ispike-1]) + PlantX.PlantDev.gdd_requirements[ispike-1])/PlantX.PlantDev.totalgdd)
print("Relative GDD to anthesis =", (sum(PlantX.PlantDev.gdd_requirements[:ianthesis-1]) + PlantX.PlantDev.gdd_requirements[ianthesis-1])/PlantX.PlantDev.totalgdd)
print("Relative GDD to grain filling =", (sum(PlantX.PlantDev.gdd_requirements[:igrainfill-1]) + PlantX.PlantDev.gdd_requirements[igrainfill-1])/PlantX.PlantDev.totalgdd)
print("Relative GDD to maturity =", (sum(PlantX.PlantDev.gdd_requirements[:imaturity-1]) + PlantX.PlantDev.gdd_requirements[imaturity-1])/PlantX.PlantDev.totalgdd)
print()
print("--- Grain Production ---")
print()
print("Spike dry biomass at anthesis =", res["y"][7,it_harvest]/PlantX.PlantCH2O.f_C)
print("Grain yield at harvest =", res["y"][PlantX.PlantDev.iseed,it_harvest]/PlantX.PlantCH2O.f_C)
ip = np.where(diagnostics['idevphase'][it_phase_transitions] == PlantX.PlantDev.phases.index('maturity'))[0][0]
print("Grain yield at maturity =", res["y"][PlantX.PlantDev.iseed,it_phase_transitions[ip]]/PlantX.PlantCH2O.f_C)
print("Potential seed density (grain number density) =", diagnostics['S_d_pot'][it_harvest])
print("Actual grain number =", res["y"][PlantX.PlantDev.iseed,it_harvest]/PlantX.PlantCH2O.f_C/PlantX.W_seedTKW0)

# %% [markdown]
# ### Create figures

# %%
site_year = "2001"
site_name = "Harden - Wheat"
site_filename = "Harden_2001_Wheat"

# %%
fig, axes = plt.subplots(4,1,figsize=(8,8),sharex=True)

axes[0].plot(res["t"], Climate_solRadswskyb_f(time_axis)+Climate_solRadswskyd_f(time_axis), c='0.4', label="Global")
axes[0].plot(res["t"], Climate_solRadswskyb_f(time_axis), c='goldenrod', alpha=0.5, label="Direct")
axes[0].plot(res["t"], Climate_solRadswskyd_f(time_axis), c='C0', alpha=0.5, label="Diffuse")
axes[0].set_ylabel("Solar radiation\n"+r"($\rm W \; m^{-2}$)")
axes[0].legend(loc=1,handlelength=0.75)
# axes[0].tick_params(axis='x', labelrotation=45)
axes[0].annotate("Solar radiation", (0.02,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[0].set_ylim([0,400])

axes[1].plot(res["t"], Climate_airTempCMin_f(time_axis), c="lightsteelblue", label="Min")
axes[1].plot(res["t"], Climate_airTempCMax_f(time_axis), c="indianred", label="Max")
axes[1].set_ylabel("Air Temperature\n"+r"($\rm ^{\circ}C$)")
# axes[1].tick_params(axis='x', labelrotation=45)
axes[1].legend(loc=1,handlelength=0.75)
axes[1].annotate("Air temperature", (0.02,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[1].set_ylim([-5,45])

axes[2].plot(res["t"], Climate_airRH_f(time_axis), c="0.4")
axes[2].set_ylabel("Relative humidity\n"+r"(%)")
# axes[2].tick_params(axis='x', labelrotation=45)
axes[2].annotate("Relative humidity", (0.02,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)

xcolors = np.linspace(0.9,0.1,PlantX.PlantCH2O.SoilLayers.nlevmlsoil).astype(str)
for iz in range(PlantX.PlantCH2O.SoilLayers.nlevmlsoil):
    axes[3].plot(res["t"], 100*Climate_soilTheta_z_f(time_axis)[:,iz], c=xcolors[iz])
axes[3].set_ylabel("Soil moisture\n"+r"(%)")
# axes[3].tick_params(axis='x', labelrotation=45)
axes[3].annotate("Soil moisture", (0.02,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[3].set_ylim([20,50])
axes[3].set_xlabel("Time (day of year)")

ax2 = axes[3].twinx()
i0, i1 = time_axis[0]-1, time_axis[-1]
ax2.bar(time_axis, df_forcing["Precipitation"].values[i0:i1], color="0.4")
ax2.set_ylabel("Daily Precipitation\n(mm)")
axes[3].annotate("Precipitation", (0.98,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='right', fontsize=12)

axes[0].set_xlim([PlantX.Management.sowingDay,time_axis[-1]])
# axes[0].set_xlim([0,time_axis[-1]])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESIM2_%s_climate.png" % site_filename,dpi=300,bbox_inches='tight')



# %%
fig, axes = plt.subplots(5,1,figsize=(8,10),sharex=True)

axes[0].plot(diagnostics["t"], diagnostics["LAI"])
axes[0].set_ylabel("LAI\n"+r"($\rm m^2 \; m^{-2}$)")
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].annotate("Leaf area index", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[0].set_ylim([0,5])

axes[1].plot(diagnostics["t"], diagnostics["GPP"])
axes[1].set_ylabel("GPP\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].annotate("Photosynthesis", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[1].set_ylim([0,30])

# axes[2].plot(res["t"], _E*1e3)
# axes[2].set_ylabel(r"$\rm E$"+"\n"+r"($\rm mmol \; H_2O \; m^{-2} \; s^{-1}$)")
axes[2].plot(diagnostics["t"], diagnostics["E_mmd"])
axes[2].set_ylabel(r"$\rm E$"+"\n"+r"($\rm mm \; d^{-1}$)")
axes[2].tick_params(axis='x', labelrotation=45)
axes[2].annotate("Transpiration Rate", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[2].set_ylim([0,20])

# axes[4].plot(df_forcing.index.values[364:-1], 0.5*np.cumsum(GPP[364:]))
axes[3].plot(res["t"], res["y"][4])
axes[3].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C d)")
axes[3].set_xlabel("Time (days)")
axes[3].annotate("Growing Degree Days - Developmental Phase", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
ax = axes[3]
ylimmin, ylimmax = 0, np.max(res["y"][4,:])*1.05
for itime in it_phase_transitions:
    ax.vlines(x=res["t"][itime], ymin=ylimmin, ymax=ylimmax, color='0.5',linestyle="--")
    text_x = res["t"][itime] + 1.5
    text_y = 0.5 * ylimmax
    phase = PlantX.PlantDev.phases[diagnostics['idevphase'][itime]]
    ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
            fontsize=8, alpha=0.7, rotation=90)
ax.set_ylim([ylimmin, ylimmax])

alp = 0.6
axes[4].plot(res["t"], res["y"][0]+res["y"][1]+res["y"][2]+res["y"][3],c='k',label="Plant", alpha=alp)
axes[4].plot(res["t"], res["y"][0],label="Leaf", alpha=alp)
axes[4].plot(res["t"], res["y"][1],label="Stem", alpha=alp)
axes[4].plot(res["t"], res["y"][2],label="Root", alpha=alp)
axes[4].plot(res["t"], res["y"][3],label="Seed", alpha=alp)
# axes[4].plot(res["t"], res["y"][8],label="Dead", c='0.5', alpha=alp)
axes[4].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^2$)")
axes[4].set_xlabel("Time (day of year)")
axes[4].legend(loc=3,fontsize=9,handlelength=0.8)

# Check if harvestDay is in time_axis
if PlantX.Management.harvestDay in time_axis:
    # if it is, then return the time index
    itime_HI = list(time_axis).index(PlantX.Management.harvestDay)
else:
    # if it is not, then return the last index for the time_axis
    itime_HI = len(time_axis) - 1

accumulated_carbon = res["y"][0,itime_HI]+res["y"][1,itime_HI]+res["y"][2,itime_HI]+res["y"][3,itime_HI]
eos_accumulated_carbon = accumulated_carbon    # end-of-season total carbon (at the end of the simulation period)
peak_accumulated_carbon_noseed = np.max(res["y"][0])+np.max(res["y"][1])+np.max(res["y"][2])    # peak carbon, excluding seed biomass
peak_accumulated_carbon_noseedroot = np.max(res["y"][0])+np.max(res["y"][1])    # peak carbon, excluding seed biomass
harvest_index = res["y"][3,itime_HI]/(res["y"][0,itime_HI]+res["y"][1,itime_HI]+res["y"][2,itime_HI]+res["y"][3,itime_HI])
harvest_index_peak = res["y"][3,itime_HI]/peak_accumulated_carbon_noseed
harvest_index_peak_noroot = res["y"][3,itime_HI]/peak_accumulated_carbon_noseedroot
yield_from_seed_Cpool = res["y"][3,itime_HI]/100 * (1/PlantX.PlantCH2O.f_C)   ## convert gC m-2 to t dry biomass ha-1
axes[4].annotate("Yield = %1.2f t/ha" % (yield_from_seed_Cpool), (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[4].annotate("Harvest index = %1.2f" % (harvest_index_peak), (0.01,0.81), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[4].set_ylim([0,600])

print("Harvest index (end-of-simulation seed:end-of-simulation plant) = %1.2f" % harvest_index)
print("Harvest index (end-of-simulation seed:peak plant biomass (excl seed)) = %1.2f" % harvest_index_peak)
print("Harvest index (end-of-simulation seed:peak plant biomass (excl seed, root)) = %1.2f" % harvest_index_peak_noroot)

# axes[0].set_xlim([PlantX.Management.sowingDay,292])
axes[0].set_xlim([PlantX.Management.sowingDay,time_axis[-1]])

axes[0].set_title("%s - %s" % (site_year,site_name))
# axes[0].set_title("Harden: %s" % site_year)
plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESIM2_%s_plant1000_dynamics.png" % site_filename,dpi=300,bbox_inches='tight')



# %%
fig, axes = plt.subplots(3,1,figsize=(8,6),sharex=True)

axes[0].plot(diagnostics["t"], diagnostics["GPP"])
axes[0].set_ylabel("GPP\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].annotate("Photosynthesis", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)

axes[1].plot(diagnostics["t"], diagnostics["Rm"]+diagnostics["Rg"], label=r"$\rm R_a$", c="k")
axes[1].plot(diagnostics["t"], diagnostics["Rg"], label=r"$\rm R_g$", c="C1")
axes[1].plot(diagnostics["t"], diagnostics["Rm"], label=r"$\rm R_m$", c="C0")
axes[1].plot(diagnostics["t"], diagnostics["Rml"], c="C0", linestyle="--", label=r"$\rm R_{m,L}$")
axes[1].plot(diagnostics["t"], diagnostics["Rmr"], c="C0", linestyle=":", label=r"$\rm R_{m,R}$")
axes[1].set_ylabel("Plant Respiration\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].annotate("Plant Respiration", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[1].legend(fontsize=10)

axes[2].plot(diagnostics["t"], diagnostics["NPP"]/diagnostics["GPP"])
axes[2].set_ylabel(r"CUE")
# axes[2].tick_params(axis='x', labelrotation=45)
axes[2].annotate("Carbon-Use Efficiency", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[2].set_ylim([0,1.0])

axes[0].set_xlim([PlantX.Management.sowingDay,292])
axes[0].set_xlim([PlantX.Management.sowingDay,time_axis[-1]])

axes[0].set_title("%s - %s" % (site_year,site_name))
# axes[0].set_title("Harden: %s" % site_year)
plt.tight_layout()

# %%
fig, axes = plt.subplots(3,1,figsize=(8,6),sharex=True)

axes[0].plot(diagnostics['t'], diagnostics['GPP'], c="k")
axes[0].set_ylabel("GPP\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].annotate("Photosynthesis", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[0].set_ylim([0,30])

axes[1].plot(diagnostics['t'], diagnostics['Rm'] + diagnostics['Rg'], label=r"$\rm R_a$", c='k')
axes[1].set_ylabel(r"$\rm R_a$"+"\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].annotate("Plant Respiration", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].set_ylim([0,10])

axes[2].plot(diagnostics['t'], diagnostics['trflux_total'], c="k")
# _Cflux_harvest_total = _Cflux_harvest_Leaf+_Cflux_harvest_Stem+_Cflux_harvest_Seed
# axes[2].plot(res["t"], _Cflux_harvest_total)
axes[2].set_ylabel("Turnover\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
# axes[2].tick_params(axis='x', labelrotation=45)
axes[2].annotate("Plant Turnover (loss of 'green' biomass)", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[2].set_ylim([0,50])

## Add annotations for developmental phases
for ax in axes:
    ylimmin, ylimmax = 0, ax.get_ylim()[1]
    for itime in it_phase_transitions:
        ax.vlines(x=res["t"][itime], ymin=ylimmin, ymax=ylimmax, color='0.5',linestyle="--")
        text_x = res["t"][itime] + 1.5
        text_y = 0.5 * ylimmax
        phase = PlantX.PlantDev.phases[diagnostics['idevphase'][itime]]
        ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
                fontsize=8, alpha=0.7, rotation=90)
    ax.set_ylim([ylimmin, ylimmax])


axes[0].set_xlim([PlantX.Management.sowingDay,292])
axes[0].set_xlim([PlantX.Management.sowingDay,time_axis[-1]])

axes[0].set_title("%s - %s" % (site_year,site_name))
# axes[0].set_title("Harden: %s" % site_year)
plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESIM2_%s_plant1000_carbon_balance.png" % site_filename,dpi=300,bbox_inches='tight')

# %%
fig, axes = plt.subplots(1,3,figsize=(15,3),sharex=True)

axes[0].plot(time_axis, res["y"][1,:]/PlantX.PlantCH2O.f_C, label="Stem")
axes[0].set_ylabel("Stem dry weight\n"+r"($\rm g \; d.wt \; m^{-2}$)")       
SDW_a = res["y"][7,-1]/PlantX.PlantCH2O.f_C
axes[0].text(0.07, 0.92, r"$\rm SDW_a$=%1.0f g d.wt m$\rm^{-2}$" % SDW_a, horizontalalignment='left', verticalalignment='center', transform = axes[0].transAxes)
axes[0].set_ylim([0,500])

axes[1].plot(diagnostics['t'], diagnostics['S_d_pot'], c='0.25', label="Potential seed density")
axes[1].plot(time_axis, res["y"][3,:]/PlantX.PlantCH2O.f_C/PlantX.W_seedTKW0, label="Actual seed density")
axes[1].set_ylabel(r"$\rm S_d$"+"\n"+r"($\rm thsnd \; grains \; m^{-2}$)")
axes[1].legend()
# axes[1].set_ylim([0,22])

axes[2].plot(time_axis, res["y"][3,:]/PlantX.PlantCH2O.f_C)
axes[2].set_ylabel("Grain dry weight\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[2].annotate("Yield = %1.2f t/ha" % (yield_from_seed_Cpool), (0.07,0.92), xycoords='axes fraction', verticalalignment='center', horizontalalignment='left')
axes[2].set_ylim([0,500])

## Add annotations for developmental phases
for ax in axes:
    ylimmin, ylimmax = 0, ax.get_ylim()[1]
    for itime in it_phase_transitions:
        ax.vlines(x=res["t"][itime], ymin=ylimmin, ymax=ylimmax, color='0.5',linestyle="--")
        text_x = res["t"][itime] + 1.5
        text_y = 0.5 * ylimmax
        phase = PlantX.PlantDev.phases[diagnostics['idevphase'][itime]]
        ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
                fontsize=8, alpha=0.7, rotation=90)
    ax.set_ylim([ylimmin, ylimmax])

axes[0].set_xlim([PlantX.Management.sowingDay,time_axis[-1]])
axes[0].set_xlabel("Time (day of year)")
axes[1].set_xlabel("Time (day of year)")
axes[2].set_xlabel("Time (day of year)")

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESIM2_%s_plant1000_grainprod.png" % (site_filename),dpi=300,bbox_inches='tight')

# %%
fig, axes = plt.subplots(2,3,figsize=(12,6),sharex=True)

axes[0,0].plot(res["t"], res["y"][4])
axes[0,0].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C)")
axes[0,0].set_xlabel("Time (days)")
axes[0,0].set_title("Growing Degree Days")
ax = axes[0,0]
ylimmin, ylimmax = ax.get_ylim()
for itime in it_phase_transitions:
    ax.vlines(x=res["t"][itime], ymin=ylimmin, ymax=ylimmax, color='0.5',linestyle="--")
    text_x = res["t"][itime] + 1.5
    text_y = 0.5 * ylimmax
    phase = PlantX.PlantDev.phases[diagnostics['idevphase'][itime]]
    ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
            fontsize=8, alpha=0.7, rotation=90)
ax.set_ylim([0, ylimmax])

axes[1,0].plot(diagnostics['t'], diagnostics['DTT'], c='C0', linestyle=":", label="DTT")
axes[1,0].plot(diagnostics['t'], diagnostics['DTT'] * diagnostics['fV'], c='C0', linestyle="-", label=r"$\rm DTT \times f_V$")
axes[1,0].plot(diagnostics['t'], diagnostics['DTT'] * diagnostics['fV'] * diagnostics['fGerm'], c='C0', linestyle=":", label=r"$\rm DTT \times f_V \times f_{germ}$")
axes[1,0].set_ylabel("Daily Thermal Time\n"+r"($\rm ^{\circ}$C)")
axes[1,0].set_xlabel("Time (days)")
axes[1,0].set_title("Growing Degree Days")
axes[1,0].legend()

axes[0,1].plot(res["t"], res["y"][5])
axes[0,1].set_ylabel("Vernalization Days\n"+r"(-)")
axes[0,1].set_xlabel("Time (days)")
axes[0,1].set_title("Vernalization Days")
# axes[0,1].set_ylim([0,125])
ax = axes[0,1]
ylimmin, ylimmax = ax.get_ylim()
for itime in it_phase_transitions:
    ax.vlines(x=res["t"][itime], ymin=ylimmin, ymax=ylimmax, color='0.5',linestyle="--")
    text_x = res["t"][itime] + 1.5
    text_y = 0.5 * ylimmax
    phase = PlantX.PlantDev.phases[diagnostics['idevphase'][itime]]
    ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
            fontsize=8, alpha=0.7, rotation=90)
ax.set_ylim([0, ylimmax])

axes[1,1].plot(diagnostics['t'], diagnostics['fV'])
axes[1,1].set_ylabel("Vernalization Factor")
axes[1,1].set_xlabel("Time (days)")
axes[1,1].set_title("Vernalization Factor\n(Modifier on GDD)")
# axes[1,1].set_ylim([0,1.03])
ax = axes[1,1]
ylimmin, ylimmax = ax.get_ylim()
for itime in it_phase_transitions:
    ax.vlines(x=res["t"][itime], ymin=ylimmin, ymax=ylimmax, color='0.5',linestyle="--")
    text_x = res["t"][itime] + 1.5
    text_y = 0.5 * ylimmax
    phase = PlantX.PlantDev.phases[diagnostics['idevphase'][itime]]
    ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
            fontsize=8, alpha=0.7, rotation=90)
ax.set_ylim([0, ylimmax])

axes[0,2].plot(res["t"], res["y"][6])
axes[0,2].set_ylabel("Hydrothermal Time\n"+r"($\rm MPa \; ^{\circ}$C d)")
axes[0,2].set_xlabel("Time (days)")
axes[0,2].set_title("Hydrothermal Time")

# axes[1,2].plot(res["t"], _deltaHTT)
# axes[1,2].set_ylabel("Daily Hydrothermal Time\n"+r"($\rm MPa \; ^{\circ}$C)")
# axes[1,2].set_xlabel("Time (days)")
# axes[1,2].set_title("Hydrothermal Time per Day")

plt.xlim([time_axis[0],time_axis[-1]])
plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESIM2_%s_plantdev.png" % (site_filename),dpi=300,bbox_inches='tight')



# %%
# Creating a DataFrame
df_u_W = pd.DataFrame({
    'Time': diagnostics['t'],
    'u_Leaf': diagnostics['u_Leaf'],
    'u_Root': diagnostics['u_Root'],
    'u_Stem': diagnostics['u_Stem'],
    'u_Seed': diagnostics['u_Seed'],
})

# Window size for running mean
window_size = 7  # For example, a 7-day running mean

# Calculate running means using pandas
df_u_W['RunningMean_u_Leaf'] = df_u_W['u_Leaf'].rolling(window=window_size).mean()
df_u_W['RunningMean_u_Root'] = df_u_W['u_Root'].rolling(window=window_size).mean()
df_u_W['RunningMean_u_Stem'] = df_u_W['u_Stem'].rolling(window=window_size).mean()
df_u_W['RunningMean_u_Seed'] = df_u_W['u_Seed'].rolling(window=window_size).mean()


fig, axes = plt.subplots(2,1,figsize=(8,4),sharex=True)

ax = axes[0]
ax.plot(df_u_W['Time'], df_u_W['u_Leaf'], label='Leaf')
ax.plot(df_u_W['Time'], df_u_W['u_Root'], label='Root')
ax.plot(df_u_W['Time'], df_u_W['u_Stem'], label='Stem')
ax.plot(df_u_W['Time'], df_u_W['u_Seed'], label='Seed')
ax.set_ylim([0,1.01])
ax.set_xlim([PlantX.Management.sowingDay,time_axis[-1]])
ax.set_xlabel("Time (day of year)")
ax.set_ylabel("Carbon allocation\ncoefficient")
ax.legend(handlelength=0.75)

ax = axes[1]
_tr_Leaf
ax.plot(diagnostics['t'], diagnostics['tr_Leaf'], label='Leaf')
ax.plot(diagnostics['t'], diagnostics['tr_Root'], label='Root')
ax.plot(diagnostics['t'], diagnostics['tr_Stem'], label='Stem')
ax.plot(diagnostics['t'], diagnostics['tr_Seed'], label='Seed')
# ax.set_ylim([0,1.01])
ax.set_xlim([PlantX.Management.sowingDay,time_axis[-1]])
ax.set_xlabel("Time (day of year)")
ax.set_ylabel("Turnover rate\n"+r"($\rm days^{-1}$)")
# ax.legend(handlelength=0.75)

yminlim, ymaxlim = 0, 0.105
ax.set_ylim([yminlim, ymaxlim])
ax.set_xlim([PlantX.Management.sowingDay,time_axis[-1]])

## Add annotations for developmental phases
for ax in axes:
    ylimmin, ylimmax = ax.get_ylim()
    for itime in it_phase_transitions:
        ax.vlines(x=res["t"][itime], ymin=ylimmin, ymax=ylimmax, color='0.5',linestyle="--")
        text_x = res["t"][itime] + 1.5
        text_y = 0.5 * ylimmax
        phase = PlantX.PlantDev.phases[diagnostics['idevphase'][itime]]
        ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='center',
                fontsize=8, alpha=0.7, rotation=90)
    ax.set_ylim([ylimmin, ylimmax])

# plt.grid(True)
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESIM2_%s_plant1000_alloctr.png" % (site_filename),dpi=300,bbox_inches='tight')
plt.show

# %%

# %% [markdown]
# ### Calculate diagnostic variables

# %%
# LAI = PlantX.PlantCH2O.calculate_LAI(res["y"][0])

# eqtime, houranglesunrise, theta = SiteX.solar_calcs(Climate_year_f(time_axis),Climate_doy_f(time_axis))
# airTempC = PlantX.Site.compute_mean_daily_air_temp(Climate_airTempCMin_f(time_axis),Climate_airTempCMax_f(time_axis))

# W_L = res["y"][0]/PlantX.f_C
# W_R = res["y"][2]/PlantX.f_C

# ## Calculate diagnostic variables
# _GPP_gCm2d = np.zeros(time_axis.size)
# _Rm_gCm2d = np.zeros(time_axis.size)
# _Rm_l = np.zeros(time_axis.size) # maintenance respiration of leaves
# _Rm_r = np.zeros(time_axis.size) # maintenance respiration of roots
# _Ra = np.zeros(time_axis.size) # autotrophic respiration 
# _E = np.zeros(time_axis.size)
# _DTT = np.zeros(time_axis.size)
# _deltaVD = np.zeros(time_axis.size)
# _deltaHTT = np.zeros(time_axis.size)
# _PHTT = np.zeros(time_axis.size)
# _fV = np.zeros(time_axis.size)
# _fGerm = np.zeros(time_axis.size)
# _relativeGDD = np.zeros(time_axis.size)
# _hc = np.zeros(time_axis.size)
# _airUhc = np.zeros(time_axis.size)
# _d_rpot = np.zeros(time_axis.size)
# _d_r = np.zeros(time_axis.size)
# _Psi_s = np.zeros(time_axis.size)

# _Cfluxremob = np.zeros(time_axis.size)
# _GN_pot = np.zeros(time_axis.size) # potential grain number

# for it,t in enumerate(time_axis):
#     sunrise, solarnoon, sunset = PlantX.Site.solar_day_calcs(Climate_year_f(time_axis[it]),Climate_doy_f(time_axis[it]))
    
#     # Development phase index
#     idevphase = PlantX.PlantDev.get_active_phase_index(res["y"][4,it])
#     _relativeGDD[it] = PlantX.PlantDev.calc_relative_gdd_index(res["y"][4,it])
#     _hc[it] = PlantX.calculate_canopy_height(_relativeGDD[it])
#     relative_gdd_anthesis = PlantX.PlantDev.calc_relative_gdd_to_anthesis(res["y"][4,it])
#     _d_rpot[it] = PlantX.calculate_root_depth(relative_gdd_anthesis)   # potential root depth based on developmental rate
#     _d_r[it] = PlantX.PlantCH2O.calculate_root_depth(res["y"][2,it], _d_rpot[it])   # actual root depth based on developmental rate and root biomass
#     PlantX.PlantDev.update_vd_state(res["y"][5,it],res["y"][4,it])    # Update vernalization state information to track developmental phase changes
#     VD = PlantX.PlantDev.get_phase_vd()    # Get vernalization state for current developmental phase
#     # Update vernalization days requirement for current developmental phase
#     PlantX.VD50 = 0.5 * PlantX.PlantDev.vd_requirements[idevphase]
#     _deltaVD[it] = PlantX.calculate_vernalizationtime(Climate_airTempCMin_f(time_axis[it]),Climate_airTempCMax_f(time_axis[it]),sunrise,sunset)
#     _fV[it] = PlantX.vernalization_factor(res["y"][5,it])
#     _fGerm[it] = PlantX.calculate_sowingdepth_factor(res["y"][4,it])

#     _deltaHTT[it] = PlantX.calculate_dailyhydrothermaltime(airTempC[it], Climate_soilTheta_f(time_axis[it]))
#     #_PHTT[it] = PlantX.calculate_P_HTT(res["y"][6,it], airTempC[it], Climate_soilTheta_f(time_axis[it]))

#     _DTT[it] = PlantX.calculate_dailythermaltime(Climate_airTempCMin_f(time_axis[it]),Climate_airTempCMax_f(time_axis[it]),sunrise,sunset)

#     ## GPP and Transpiration (E)
#     if (W_L[it] == 0) or (W_R[it] == 0):
#         _GPP_gCm2d[it] = 0
#         _Rm_gCm2d[it] = 0
#         _E[it] = 0
#         _Rm_l[it] = 0
#         _Rm_r[it] = 0
#         _Psi_s[it] = 0
#     else:
#         # Calculate wind speed at top-of-canopy
#         _airUhc[it] = PlantX.calculate_wind_speed_hc(Climate_airU_f(time_axis[it]),_hc[it],LAI[it]+PlantX.SAI)    ## TODO: Make sure SAI is handled consistently across modules
#         GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = PlantX.PlantCH2O.calculate(W_L[it],W_R[it],Climate_soilTheta_z_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airRH_f(time_axis)[it],Climate_airCO2_f(time_axis)[it],Climate_airO2_f(time_axis)[it],Climate_airPressure_f(time_axis)[it],_airUhc[it],Climate_solRadswskyb_f(time_axis)[it],Climate_solRadswskyd_f(time_axis)[it],theta[it],PlantX.SAI,_hc[it],_d_rpot[it])
#         _GPP_gCm2d[it] = GPP * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
#         _Rm_gCm2d[it] = (Rml+Rmr) * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
#         _E[it] = E
#         _Rm_l[it] = Rml
#         _Rm_r[it] = Rmr
#         _Psi_s[it] = Psis   ## Note: this is the soil water potential in the root zone only
    
#     _GN_pot[it] = PlantX.calculate_wheat_grain_number(res["y"][7,it]/PlantX.f_C)
#     _Cfluxremob[it] = PlantX.calculate_nsc_stem_remob(res["y"][1,it], res["y"][0,it], res["y"][3,it]/PlantX.f_C, _GN_pot[it]*PlantX.W_seedTKW0, res["y"][4,it])


# NPP_gCm2d, Rg_gCm2d = PlantX.calculate_NPP_RmRgpropto(_GPP_gCm2d, _Rm_gCm2d)
# # Rg_gCm2d = PlantX.alpha_Rg * (_GPP_gCm2d - _Rm_gCm2d)
# Ra_gCm2d = _GPP_gCm2d - NPP_gCm2d  # units of gC m-2 d-1
# Rm_l_gCm2d = _Rm_l * 12.01 * (60*60*24) / 1e6
# Rm_r_gCm2d = _Rm_r * 12.01 * (60*60*24) / 1e6

# ## Conversion notes: When _E units are mol m-2 s-1, multiply by molar mass H2O to get g m-2 s-1, divide by 1000 to get kg m-2 s-1, multiply by 60*60*24 to get kg m-2 d-1, and 1 kg m-2 d-1 = 1 mm d-1. 
# ## Noting that 1 kg of water is equivalent to 1 liter (L) of water (because the density of water is 1000 kg/m³), and 1 liter of water spread over 1 square meter results in a depth of 1 mm
# _E_mmd = _E*18.015/1000*(60*60*24)

# BioHarvestSeed = PlantX.calculate_BioHarvest(res["y"][3],Climate_doy_f(time_axis),ManagementX.harvestDay,PlantX.propHarvestSeed,ManagementX.PhHarvestTurnoverTime)

# fV = PlantX.vernalization_factor(res["y"][5])

# %%
# ## Calculate diagnostic variables for allocation coefficients and turnover rates
# _u_Leaf = np.zeros(time_axis.size)
# _u_Root = np.zeros(time_axis.size)
# _u_Stem = np.zeros(time_axis.size)
# _u_Seed = np.zeros(time_axis.size)

# _marginalgain_leaf = np.zeros(time_axis.size)
# _marginalgain_root = np.zeros(time_axis.size)
# _marginalcost_leaf = np.zeros(time_axis.size)
# _marginalcost_root = np.zeros(time_axis.size)

# _tr_Leaf = np.zeros(time_axis.size)
# _tr_Root = np.zeros(time_axis.size)
# _tr_Stem = np.zeros(time_axis.size)
# _tr_Seed = np.zeros(time_axis.size)
# _trflux_Leaf = np.zeros(time_axis.size)
# _trflux_Root = np.zeros(time_axis.size)
# _trflux_Stem = np.zeros(time_axis.size)
# _trflux_Seed = np.zeros(time_axis.size)

# _Cflux_harvest_Leaf = np.zeros(time_axis.size)
# _Cflux_harvest_Stem = np.zeros(time_axis.size)
# _Cflux_harvest_Seed = np.zeros(time_axis.size)

# for it,t in enumerate(time_axis):

#     idevphase = PlantX.PlantDev.get_active_phase_index(res["y"][4,it])
#     # Allocation fractions per pool
#     alloc_coeffs = PlantX.PlantDev.allocation_coeffs[idevphase]

#     # Calculate allocation fraction to grain
#     _u_Seed[it] = PlantX.calculate_grain_alloc_coeff(alloc_coeffs[PlantX.PlantDev.iseed], res["y"][3,it]/PlantX.f_C, _GN_pot[it]*PlantX.W_seedTKW0, res["y"][4,it])
#     _u_Stem[it] = alloc_coeffs[PlantX.PlantDev.istem]
#     # Turnover rates per pool
#     tr_ = PlantX.PlantDev.turnover_rates[idevphase]
#     _tr_Leaf[it] = tr_[PlantX.PlantDev.ileaf]
#     _tr_Root[it] = tr_[PlantX.PlantDev.iroot]
#     _tr_Stem[it] = tr_[PlantX.PlantDev.istem]
#     _tr_Seed[it] = tr_[PlantX.PlantDev.iseed]
#     # Set any constant allocation coefficients for optimal allocation
#     PlantX.PlantAlloc.u_Stem = _u_Stem[it]
#     PlantX.PlantAlloc.u_Seed = _u_Seed[it]

#     # Set pool turnover rates for optimal allocation
#     PlantX.PlantAlloc.tr_L = tr_[PlantX.PlantDev.ileaf]    #1 if tr_[self.PlantDev.ileaf] == 0 else max(1, 1/tr_[self.PlantDev.ileaf])
#     PlantX.PlantAlloc.tr_R = tr_[PlantX.PlantDev.iroot]    #1 if tr_[self.PlantDev.iroot] == 0 else max(1, 1/tr_[self.PlantDev.ileaf])

#     if (W_L[it] == 0) or (W_R[it] == 0):
#         u_L = alloc_coeffs[PlantX.PlantDev.ileaf]
#         u_R = alloc_coeffs[PlantX.PlantDev.iroot]
#         dGPPRmdWleaf, dGPPRmdWroot, dSdWleaf, dSdWroot = 0, 0, 0, 0
#     else:
#         u_L, u_R, dGPPRmdWleaf, dGPPRmdWroot, dSdWleaf, dSdWroot = PlantX.PlantAlloc.calculate(
#             W_L[it],
#             W_R[it],
#             Climate_soilTheta_z_f(time_axis[it]),
#             Climate_airTempC_f(time_axis[it]),
#             Climate_airTempC_f(time_axis[it]),
#             Climate_airRH_f(time_axis[it]),
#             Climate_airCO2_f(time_axis[it]),
#             Climate_airO2_f(time_axis[it]),
#             Climate_airPressure_f(time_axis[it]),
#             _airUhc[it],
#             Climate_solRadswskyb_f(time_axis[it]),
#             Climate_solRadswskyd_f(time_axis[it]),
#             theta[it],
#             PlantX.SAI,
#             _hc[it],
#             _d_rpot[it])

#     _u_Leaf[it] = u_L
#     _u_Root[it] = u_R

#     _marginalgain_leaf[it] = dGPPRmdWleaf
#     _marginalgain_root[it] = dGPPRmdWroot
#     _marginalcost_leaf[it] = dSdWleaf
#     _marginalcost_root[it] = dSdWroot

#     _trflux_Leaf[it] = _tr_Leaf[it]*res["y"][0,it]
#     _trflux_Stem[it] = _tr_Leaf[it]*res["y"][1,it]
#     _trflux_Root[it] = _tr_Leaf[it]*res["y"][2,it]
#     _trflux_Seed[it] = _tr_Leaf[it]*res["y"][3,it]

#     _Cflux_harvest_Leaf[it] = PlantX.calculate_BioHarvest(res["y"][0,it],Climate_doy_f(time_axis[it]),PlantX.Management.harvestDay,PlantX.propHarvestLeaf,PlantX.Management.PhHarvestTurnoverTime)
#     _Cflux_harvest_Stem[it] = PlantX.calculate_BioHarvest(res["y"][1,it],Climate_doy_f(time_axis[it]),PlantX.Management.harvestDay,PlantX.propHarvestStem,PlantX.Management.PhHarvestTurnoverTime)
#     _Cflux_harvest_Seed[it] = PlantX.calculate_BioHarvest(res["y"][3,it],Climate_doy_f(time_axis[it]),PlantX.Management.harvestDay,PlantX.propHarvestSeed,PlantX.Management.PhHarvestTurnoverTime)


# %%

# %% [markdown]
# ### Compare numerical solvers

# %% [markdown]
# ### - Scipy solve_ivp

# %%
result_ivp = Model.run(
    time_axis=time_axis,
    forcing_inputs=forcing_inputs,
    solver="ivp",
    zero_crossing_indices=[4],
    reset_days=reset_days,
    rtol=1e-5,
    atol=1e-6
)

res = result_ivp

eqtime, houranglesunrise, theta = SiteX.solar_calcs(Climate_year_f(time_axis),Climate_doy_f(time_axis))
LAI = res["y"][0] / PlantX.PlantCH2O.SLA
W_L = res["y"][0]/PlantX.f_C
W_R = res["y"][2]/PlantX.f_C

## Calculate diagnostic variables
_GPP_gCm2d = np.zeros(time_axis.size)
_E = np.zeros(time_axis.size)

for it,t in enumerate(time_axis):
    GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = PlantCH2OX.calculate(W_L[it],W_R[it],Climate_soilTheta_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airRH_f(time_axis)[it],Climate_airCO2_f(time_axis)[it],Climate_airO2_f(time_axis)[it],Climate_airPressure_f(time_axis)[it],Climate_solRadswskyb_f(time_axis)[it],Climate_solRadswskyd_f(time_axis)[it],theta[it])

    _GPP_gCm2d[it] = GPP * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
    _E[it] = E

NPP = PlantX.calculate_NPP(_GPP_gCm2d)

BioHarvestSeed = PlantX.calculate_BioHarvest(res["y"][3],Climate_doy_f(time_axis),ManagementX.harvestDay,PlantX.propHarvestSeed,ManagementX.PhHarvestTurnoverTime)

GPP_ivp = _GPP_gCm2d

# %% [markdown]
# ### - Explicit Euler method

# %%
result_euler = Model.run(
    time_axis=time_axis,
    forcing_inputs=forcing_inputs,
    solver="euler",
    zero_crossing_indices=[4],
    reset_days=reset_days,
)

res = result_euler

eqtime, houranglesunrise, theta = SiteX.solar_calcs(Climate_year_f(time_axis),Climate_doy_f(time_axis))
LAI = res["y"][0] / PlantX.PlantCH2O.SLA
W_L = res["y"][0]/PlantX.f_C
W_R = res["y"][2]/PlantX.f_C

## Calculate diagnostic variables
_GPP_gCm2d = np.zeros(time_axis.size)
_E = np.zeros(time_axis.size)

for it,t in enumerate(time_axis):
    GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = PlantCH2OX.calculate(W_L[it],W_R[it],Climate_soilTheta_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airRH_f(time_axis)[it],Climate_airCO2_f(time_axis)[it],Climate_airO2_f(time_axis)[it],Climate_airPressure_f(time_axis)[it],Climate_solRadswskyb_f(time_axis)[it],Climate_solRadswskyd_f(time_axis)[it],theta[it])

    _GPP_gCm2d[it] = GPP * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
    _E[it] = E

NPP = PlantX.calculate_NPP(_GPP_gCm2d)

BioHarvestSeed = PlantX.calculate_BioHarvest(res["y"][3],Climate_doy_f(time_axis),ManagementX.harvestDay,PlantX.propHarvestSeed,ManagementX.PhHarvestTurnoverTime)

GPP_euler = _GPP_gCm2d

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

axes[0, 0].plot(time_axis, Climate_airTempCMin_f(time_axis),c="b",alpha=0.6,label="Min")
axes[0, 0].plot(time_axis, Climate_airTempCMax_f(time_axis),c="r",alpha=0.6,label="Max")
axes[0, 0].set_ylabel("Daily Air Temperature\n"+r"($\rm ^{\circ}$C)")
axes[0, 0].legend()

axes[0, 1].plot(res["t"], GPP_ivp,c="C0")
axes[0, 1].plot(res["t"], GPP_euler,c="C0",linestyle="--")
axes[0, 1].set_ylabel("GPP\n"+r"(gC$\rm m^{-2} \; d^{-1}$)")

axes[1, 0].plot(res["t"], result_ivp["y"][0],label="Leaf")
axes[1, 0].plot(res["t"], result_ivp["y"][1],label="Stem")
axes[1, 0].plot(res["t"], result_ivp["y"][2],label="Root")
axes[1, 0].plot(res["t"], result_ivp["y"][3],label="Seed")

axes[1, 0].plot(res["t"], result_euler["y"][0],label="Leaf",c="C0",linestyle="--")
axes[1, 0].plot(res["t"], result_euler["y"][1],label="Stem",c="C1",linestyle="--")
axes[1, 0].plot(res["t"], result_euler["y"][2],label="Root",c="C2",linestyle="--")
axes[1, 0].plot(res["t"], result_euler["y"][3],label="Seed",c="C3",linestyle="--")

axes[1, 0].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^2$)")
axes[1, 0].set_xlabel("Time (days)")
axes[1, 0].legend()

axes[1, 1].plot(res["t"], result_ivp["y"][4])
axes[1, 1].plot(res["t"], result_euler["y"][4],c="C0",linestyle="--")
axes[1, 1].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C)")
axes[1, 1].set_xlabel("Time (days)")

ax = axes[1, 1]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    print("Plant dev phase:", PlantDevX.phases[iphase],"reached at t =",res["t"][itime])
    ax.vlines(x=res["t"][itime],ymin=0,ymax=res["y"][4,itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

plt.xlim([time_axis[0],time_axis[-1]])

plt.tight_layout()

# %%

# %%

# %%
