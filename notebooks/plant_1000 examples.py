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
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.leafgasexchange import LeafGasExchangeModule
from daesim.leafgasexchange2 import LeafGasExchangeModule2
from daesim.canopygasexchange import CanopyGasExchange
from daesim.plantcarbonwater import PlantModel as PlantCH2O

# %%
from daesim.plant_1000 import PlantModuleCalculator, PlantModelSolver

# %% [markdown]
# # Milgadara Simulation

# %% [markdown]
# ### Import forcing data

# %%
file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim/data/StellaOutFile.csv"

df_forcing = pd.read_csv(file)

## Milgadara site location-34.38904277303204, 148.46949938279096
SiteX = ClimateModule(CLatDeg=-34.389,CLonDeg=148.469)
start_doy = 1.0
start_year = 2018
nrundays = df_forcing.index.size

time_nday, time_doy, time_year = SiteX.time_discretisation(start_doy, start_year, nrundays=nrundays)

# %%
time_doy = [time_doy[i]+0.5 for i in range(len(time_doy))]  ## Adjust daily time-step to represent midday on each day

# %% [markdown]
# ### Test the rate of change calculate method

# %%
_nday = 1
_Cleaf = 100.0
_Cstem = 50.0
_Croot = 90.0
_Cseed = 0.0
_Bio_time = 0.0
_solRadswskyb = 800    ## incoming shortwave radiation, beam (W m-2)
_solRadswskyd = 200    ## incoming shortwave radiation, diffuse (W m-2)
_airTempCMin = 13.88358116
_airTempCMax = 28.99026108
_airTempC = (_airTempCMin + _airTempCMax)/2
_airPressure = 101325  ## atmospheric pressure (Pa)
_airRH = 65.0   ## relative humidity (%) 
_airCO2 = 400*(_airPressure/1e5)*1e-6     ## carbon dioxide partial pressure (bar)
_airO2 = 209000*(_airPressure/1e5)*1e-6   ## oxygen partial pressure (bar)
_soilTheta = 0.30   ## volumetric soil water content (m3 m-3)
_doy = time_doy[_nday-1]
_year = time_year[_nday-1]
_propPhAboveBM = 0.473684210526

management = ManagementModule(plantingDay=30,harvestDay=235)
site = ClimateModule()
canopy = CanopyLayers()
canopyrad = CanopyRadiation(Canopy=canopy)
leaf = LeafGasExchangeModule2(Site=site)
canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopySolar=canopyrad)
plantch2o = PlantCH2O(Site=site,CanopyGasExchange=canopygasexchange)
plant = PlantModuleCalculator(Site=site,Management=management,PlantCH2O=plantch2o)

dydt = plant.calculate(
    _Cleaf,
    _Cstem,
    _Croot,
    _Cseed,
    _Bio_time,
    _solRadswskyb,
    _solRadswskyd,
    _airTempCMin,
    _airTempCMax,
    _airPressure,
    _airRH,
    _airCO2,
    _airO2,
    _soilTheta,
    _doy,
    _year,
)
print("dy/dt =", dydt)
print()
print("  dydt(Cleaf) = %1.4f" % dydt[0])
print("  dydt(Cstem) = %1.4f" % dydt[1])
print("  dydt(Croot) = %1.4f" % dydt[2])
print("  dydt(Cseed) = %1.4f" % dydt[3])
print("  Bio_time = %1.4f" % dydt[4])

# %% [markdown]
# ### Initialise site

# %%
file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim/data/DAESim_forcing_Milgadara_2018.csv"

df_forcing = pd.read_csv(file)

## Milgadara site location-34.38904277303204, 148.46949938279096
SiteX = ClimateModule(CLatDeg=-34.389,CLonDeg=148.469,timezone=10)
start_doy = 1.0
start_year = 2018
nrundays = df_forcing.index.size

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
_soilTheta = 0.35*np.ones(nrundays)   ## volumetric soil moisture content (m3 m-3)

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
Climate_airCO2_f = interp1d(time_nday, _CO2)
Climate_airO2_f = interp1d(time_nday, _O2)
Climate_soilTheta_f = interp1d(time_nday, _soilTheta)
Climate_nday_f = interp1d(time_nday, time_nday)   ## nday represents the ordinal day-of-year plus each simulation day (e.g. a model run starting on Jan 30 and going for 2 years will have nday=30+np.arange(2*365))


# %% [markdown]
# ### Initialise aggregated model with its classes, initial values for the states, and time axis

# %%
time_axis = np.arange(119, 300, 1)   ## Note: time_axis represents the simulation day (_nday) and must be the same x-axis upon which the forcing data was interpolated on

ManagementX = ManagementModule(plantingDay=120,harvestDay=330)
PlantDevX = PlantGrowthPhases(gdd_requirements=[100,1000,100,100])
LeafX = LeafGasExchangeModule2()
CanopyX = CanopyLayers(nlevmlcan=1)
CanopyRadX = CanopyRadiation(Canopy=CanopyX)
CanopyGasExchangeX = CanopyGasExchange(Leaf=LeafX,Canopy=CanopyX,CanopySolar=CanopyRadX)
PlantCH2OX = PlantCH2O(Site=SiteX,CanopyGasExchange=CanopyGasExchangeX,maxLAI=2.0,SLA=0.05,ksr_coeff=5000)
PlantX = PlantModuleCalculator(
    Site=SiteX,
    Management=ManagementX,
    PlantCH2O=PlantCH2OX,
    LMA=20.0)

Model = PlantModelSolver(
    calculator=PlantX, site=SiteX, management=ManagementX, plantdev=PlantDevX, leaf=LeafX, canopy=CanopyX, canopyrad=CanopyRadX, canopygasexch=CanopyGasExchangeX, plantch2o=PlantCH2OX, state1_init=0.5, state2_init=0.1, state3_init=0.5, state4_init=0, state5_init=0, time_start=time_axis[0]
)

# %%
fig, axes = plt.subplots(2,3,figsize=(12,6))

axes[0,0].plot(time_axis,Climate_solRadswskyb_f(time_axis),label="Direct")
axes[0,0].plot(time_axis,Climate_solRadswskyd_f(time_axis),label="Diffuse")
axes[0,0].legend()
axes[0,0].set_ylabel('solRadswsky (W m-2)')

axes[0,1].plot(time_axis,Climate_airTempCMin_f(time_axis),label="Min")
axes[0,1].plot(time_axis,Climate_airTempCMax_f(time_axis),label="Max")
axes[0,1].legend()
axes[0,1].set_ylabel('airTempC (oC)')

axes[0,2].plot(time_axis,Climate_airPressure_f(time_axis))
axes[0,2].set_ylabel('airPressure (Pa)')

axes[1,0].plot(time_axis,Climate_airRH_f(time_axis))
axes[1,0].set_ylabel('RH (%)')

axes[1,1].plot(time_axis,1e6*Climate_airCO2_f(time_axis),label=r"$\rm CO_2 \; (\mu bar)$")
axes[1,1].plot(time_axis,1e3*Climate_airO2_f(time_axis),label=r"$\rm O_2$ (mbar)")
axes[1,1].set_ylabel('Gas partial pressure')
axes[1,1].legend()

axes[1,2].plot(time_axis,Climate_soilTheta_f(time_axis))
axes[1,2].set_ylabel(r'Volumetric soil moisture ($\rm m^3 \; m^{-3}$)')

plt.tight_layout()

# %% [markdown]
# ### Run model ODE solver

# %%
res = Model.run(
    solRadswskyb=Climate_solRadswskyb_f,
    solRadswskyd=Climate_solRadswskyd_f,
    airTempCMin=Climate_airTempCMin_f,
    airTempCMax=Climate_airTempCMax_f,
    airP=Climate_airPressure_f,
    airRH=Climate_airRH_f,
    airCO2=Climate_airCO2_f,
    airO2=Climate_airO2_f,
    soilTheta=Climate_soilTheta_f,
    _doy=Climate_doy_f,
    _year=Climate_year_f,
    time_axis=time_axis,
)

# %% [markdown]
# ### Calculate diagnostic variables

# %%
LAI = res.y[0] / PlantX.LMA

eqtime, houranglesunrise, theta = SiteX.solar_calcs(Climate_year_f(time_axis),Climate_doy_f(time_axis))

W_L = res.y[0]/PlantX.f_C
W_R = res.y[2]/PlantX.f_C

## Calculate diagnostic variables
_GPP_gCm2d = np.zeros(time_axis.size)
_E = np.zeros(time_axis.size)

for it,t in enumerate(time_axis):
    GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = PlantCH2OX.calculate(W_L[it],W_R[it],Climate_soilTheta_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airRH_f(time_axis)[it],Climate_airCO2_f(time_axis)[it],Climate_airO2_f(time_axis)[it],Climate_airPressure_f(time_axis)[it],Climate_solRadswskyb_f(time_axis)[it],Climate_solRadswskyd_f(time_axis)[it],SiteX,LeafX,CanopyGasExchangeX,CanopyX,CanopyRadX)

    _GPP_gCm2d[it] = GPP * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
    _E[it] = E

NPP = PlantX.calculate_NPP(_GPP_gCm2d)

BioHarvestSeed = PlantX.calculate_BioHarvest(res.y[3],Climate_doy_f(time_axis),ManagementX.harvestDay,PlantX.propHarvestSeed,ManagementX.PhHarvestTurnoverTime)

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

axes[0, 0].plot(time_axis, Climate_airTempCMin_f(time_axis),c="b",alpha=0.6,label="Min")
axes[0, 0].plot(time_axis, Climate_airTempCMax_f(time_axis),c="r",alpha=0.6,label="Max")
axes[0, 0].set_ylabel("Daily Air Temperature\n"+r"($\rm ^{\circ}$C)")
axes[0, 0].legend()

axes[0, 1].plot(res.t, _GPP_gCm2d)
axes[0, 1].set_ylabel("GPP\n"+r"(gC$\rm m^{-2} \; d^{-1}$)")

axes[1, 0].plot(res.t, res.y[0],label="Leaf")
axes[1, 0].plot(res.t, res.y[1],label="Stem")
axes[1, 0].plot(res.t, res.y[2],label="Root")
axes[1, 0].plot(res.t, res.y[3],label="Seed")
axes[1, 0].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^2$)")
axes[1, 0].set_xlabel("Time (days)")
axes[1, 0].legend()


axes[1, 1].plot(res.t, res.y[4])
axes[1, 1].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C)")
axes[1, 1].set_xlabel("Time (days)")

ax = axes[1, 1]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res.y[4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    print("Plant dev phase:", PlantDevX.phases[iphase],"reached at t =",res.t[itime])
    ax.vlines(x=res.t[itime],ymin=0,ymax=res.y[4,itime],color='0.5')
    text_x = res.t[itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

plt.xlim([time_axis[0],time_axis[-1]])

plt.tight_layout()

# %% [markdown]
# ## Testing new numerical solver

# %%
from daesim.utils import ODEModelSolver

# %%
time_axis = np.arange(119, 300, 1)   ## Note: time_axis represents the simulation day (_nday) and must be the same x-axis upon which the forcing data was interpolated on

ManagementX = ManagementModule(plantingDay=120,harvestDay=330)
PlantDevX = PlantGrowthPhases(gdd_requirements=[100,1000,100,100])
LeafX = LeafGasExchangeModule2(Site=SiteX)
CanopyX = CanopyLayers(nlevmlcan=1)
CanopyRadX = CanopyRadiation(Canopy=CanopyX)
CanopyGasExchangeX = CanopyGasExchange(Leaf=LeafX,Canopy=CanopyX,CanopySolar=CanopyRadX)
PlantCH2OX = PlantCH2O(Site=SiteX,CanopyGasExchange=CanopyGasExchangeX,maxLAI=2.0,SLA=0.05,ksr_coeff=5000)
PlantX = PlantModuleCalculator(
    Site=SiteX,
    Management=ManagementX,
    PlantCH2O=PlantCH2OX,
    LMA=20.0
)

# %%
## Define the callable calculator that defines the right-hand-side ODE function
PlantXCalc = PlantX.calculate

Model = ODEModelSolver(calculator=PlantXCalc, states_init=[0.5, 0.1, 0.5, 0.0, 0.0], time_start=time_axis[0])


forcing_inputs = [Climate_solRadswskyb_f,
                  Climate_solRadswskyd_f,
                  Climate_airTempCMin_f,
                  Climate_airTempCMax_f,
                  Climate_airPressure_f,
                  Climate_airRH_f,
                  Climate_airCO2_f,
                  Climate_airO2_f,
                  Climate_soilTheta_f,
                  Climate_doy_f,
                  Climate_year_f]
reset_days = []

# result_ivp = Model.run(
#     time_axis=time_axis,
#     forcing_inputs=forcing_inputs,
#     solver="ivp",
#     zero_crossing_indices=[4],
#     reset_days=reset_days,
#     rtol=1e-2,
#     atol=1e-2
# )

result_euler = Model.run(
    time_axis=time_axis,
    forcing_inputs=forcing_inputs,
    solver="euler",
    zero_crossing_indices=[4],
    reset_days=reset_days,
    rtol=1e-2,
    atol=1e-2
)

# %%
res = result_euler

# %% [markdown]
# ### Calculate diagnostic variables

# %%
LAI = res["y"][0] / PlantX.LMA

eqtime, houranglesunrise, theta = SiteX.solar_calcs(Climate_year_f(time_axis),Climate_doy_f(time_axis))

W_L = res["y"][0]/PlantX.f_C
W_R = res["y"][2]/PlantX.f_C

## Calculate diagnostic variables
_GPP_gCm2d = np.zeros(time_axis.size)
_E = np.zeros(time_axis.size)

for it,t in enumerate(time_axis):
    GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = PlantCH2OX.calculate(W_L[it],W_R[it],Climate_soilTheta_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airRH_f(time_axis)[it],Climate_airCO2_f(time_axis)[it],Climate_airO2_f(time_axis)[it],Climate_airPressure_f(time_axis)[it],Climate_solRadswskyb_f(time_axis)[it],Climate_solRadswskyd_f(time_axis)[it])#,SiteX,LeafX,CanopyGasExchangeX,CanopyX,CanopyRadX)

    _GPP_gCm2d[it] = GPP * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
    _E[it] = E

NPP = PlantX.calculate_NPP(_GPP_gCm2d)

BioHarvestSeed = PlantX.calculate_BioHarvest(res["y"][3],Climate_doy_f(time_axis),ManagementX.harvestDay,PlantX.propHarvestSeed,ManagementX.PhHarvestTurnoverTime)

# %%

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

axes[0, 0].plot(time_axis, Climate_airTempCMin_f(time_axis),c="b",alpha=0.6,label="Min")
axes[0, 0].plot(time_axis, Climate_airTempCMax_f(time_axis),c="r",alpha=0.6,label="Max")
axes[0, 0].set_ylabel("Daily Air Temperature\n"+r"($\rm ^{\circ}$C)")
axes[0, 0].legend()

axes[0, 1].plot(res["t"], _GPP_gCm2d)
axes[0, 1].set_ylabel("GPP\n"+r"(gC$\rm m^{-2} \; d^{-1}$)")

axes[1, 0].plot(res["t"], res["y"][0],label="Leaf")
axes[1, 0].plot(res["t"], res["y"][1],label="Stem")
axes[1, 0].plot(res["t"], res["y"][2],label="Root")
axes[1, 0].plot(res["t"], res["y"][3],label="Seed")
axes[1, 0].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^2$)")
axes[1, 0].set_xlabel("Time (days)")
axes[1, 0].legend()

axes[1, 1].plot(res["t"], res["y"][4])
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
