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
from daesim.plantallocoptimal import PlantOptimalAllocation

# %%
from daesim.plant_1000 import PlantModuleCalculator

# %% [markdown]
# # Site Level Simulation

# %% [markdown]
# ### Initialise site

# %% [markdown]
# #### - Import forcing data

# %%
file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim/data/DAESim_forcing_Milgadara_2018.csv"

df_forcing = pd.read_csv(file)

# %% [markdown]
# #### - Initialise site module

# %%
## Milgadara site location-34.38904277303204, 148.46949938279096
SiteX = ClimateModule(CLatDeg=-34.389,CLonDeg=148.469,timezone=10)
start_doy = 1.0
start_year = 2021
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
# ### Test the rate of change calculate method

# %%
_nday = 1
_Cleaf = 100.0
_Cstem = 50.0
_Croot = 90.0
_Cseed = 0.0
_Bio_time = 0.0
_VD_time = 0.0
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

management = ManagementModule(plantingDay=30,harvestDay=235)
site = ClimateModule()
canopy = CanopyLayers()
canopyrad = CanopyRadiation(Canopy=canopy)
leaf = LeafGasExchangeModule2(Site=site)
canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopyRad=canopyrad)
plantch2o = PlantCH2O(Site=site,CanopyGasExchange=canopygasexchange)
plantalloc = PlantOptimalAllocation(Plant=plantch2o,dWL_factor=1.01,dWR_factor=1.02)
plant = PlantModuleCalculator(Site=site,Management=management,PlantCH2O=plantch2o,PlantAlloc=plantalloc)

dydt = plant.calculate(
    _Cleaf,
    _Cstem,
    _Croot,
    _Cseed,
    _Bio_time,
    _VD_time,
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
print("  VRN_time = %1.4f" % dydt[5])

# %% [markdown]
# ## Use Model with Numerical Solver

# %%
from daesim.utils import ODEModelSolver

# %% [markdown]
# ### Initialise aggregated model with its classes, initial values for the states, and time axis

# %%
time_axis = np.arange(119, 332, 1)   ## Note: time_axis represents the simulation day (_nday) and must be the same x-axis upon which the forcing data was interpolated on

ManagementX = ManagementModule(plantingDay=120,harvestDay=330)
PlantDevX = PlantGrowthPhases(
    gdd_requirements=[100,1000,80,80],
    turnover_rates = [[0.001,  0.001, 0.001, 0.0, 0.0],
                      [0.0366, 0.002, 0.0083, 0.0, 0.0],
                      [0.0633, 0.002, 0.0083, 0.0, 0.0],
                      [0.1, 0.008, 0.05, 0.0001, 0.0]])
LeafX = LeafGasExchangeModule2(Site=SiteX)
CanopyX = CanopyLayers(nlevmlcan=1)
CanopyRadX = CanopyRadiation(Canopy=CanopyX)
CanopyGasExchangeX = CanopyGasExchange(Leaf=LeafX,Canopy=CanopyX,CanopyRad=CanopyRadX)
PlantCH2OX = PlantCH2O(Site=SiteX,CanopyGasExchange=CanopyGasExchangeX,maxLAI=2.0,ksr_coeff=3000)
PlantAllocX = PlantOptimalAllocation(Plant=PlantCH2OX,dWL_factor=1.02,dWR_factor=1.02)
PlantX = PlantModuleCalculator(
    Site=SiteX,
    Management=ManagementX,
    PlantDev=PlantDevX,
    PlantCH2O=PlantCH2OX,
    PlantAlloc=PlantAllocX,
    propHarvestLeaf=0.75,
)

# %%
## Define the callable calculator that defines the right-hand-side ODE function
PlantXCalc = PlantX.calculate

Model = ODEModelSolver(calculator=PlantXCalc, states_init=[0.5, 0.1, 0.5, 0.0, 0.0, 0.0], time_start=time_axis[0])


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

reset_days = [PlantX.Management.plantingDay, PlantX.Management.harvestDay]

res = Model.run(
    time_axis=time_axis,
    forcing_inputs=forcing_inputs,
    solver="euler",
    zero_crossing_indices=[4],
    reset_days=reset_days,
    # rtol=1e-2,
    # atol=1e-2
)

# %% [markdown]
# ### Calculate diagnostic variables

# %%
LAI = PlantX.PlantCH2O.calculate_LAI(res["y"][0])

eqtime, houranglesunrise, theta = SiteX.solar_calcs(Climate_year_f(time_axis),Climate_doy_f(time_axis))

W_L = res["y"][0]/PlantX.f_C
W_R = res["y"][2]/PlantX.f_C

## Calculate diagnostic variables
_GPP_gCm2d = np.zeros(time_axis.size)
_E = np.zeros(time_axis.size)
_deltaVD = np.zeros(time_axis.size)
_fV = np.zeros(time_axis.size)

for it,t in enumerate(time_axis):
    sunrise, solarnoon, sunset = PlantX.Site.solar_day_calcs(Climate_year_f(time_axis[it]),Climate_doy_f(time_axis[it]))
    
    # Development phase index
    idevphase = PlantX.PlantDev.get_active_phase_index(res["y"][4,it])
    PlantX.PlantDev.update_vd_state(res["y"][5,it],res["y"][4,it])    # Update vernalization state information to track developmental phase changes
    VD = PlantX.PlantDev.get_phase_vd()    # Get vernalization state for current developmental phase
    # Update vernalization days requirement for current developmental phase
    PlantX.VD50 = 0.5 * PlantX.PlantDev.vd_requirements[idevphase]
    _deltaVD[it] = PlantX.calculate_vernalizationtime(Climate_airTempCMin_f(time_axis[it]),Climate_airTempCMax_f(time_axis[it]),sunrise,sunset)
    _fV[it] = PlantX.vernalization_factor(res["y"][5,it])

    ## GPP and Transpiration (E)
    GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = PlantX.PlantCH2O.calculate(W_L[it],W_R[it],Climate_soilTheta_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airRH_f(time_axis)[it],Climate_airCO2_f(time_axis)[it],Climate_airO2_f(time_axis)[it],Climate_airPressure_f(time_axis)[it],Climate_solRadswskyb_f(time_axis)[it],Climate_solRadswskyd_f(time_axis)[it],theta[it])
    _GPP_gCm2d[it] = GPP * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
    _E[it] = E

NPP = PlantX.calculate_NPP(_GPP_gCm2d)

BioHarvestSeed = PlantX.calculate_BioHarvest(res["y"][3],Climate_doy_f(time_axis),ManagementX.harvestDay,PlantX.propHarvestSeed,ManagementX.PhHarvestTurnoverTime)

fV = PlantX.vernalization_factor(res["y"][5])

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

axes[0, 0].plot(res["t"], _GPP_gCm2d)
axes[0, 0].set_ylabel("GPP\n"+r"(gC$\rm m^{-2} \; d^{-1}$)")
axes[0, 0].set_ylim([0,30])

axes[0, 1].plot(res["t"], _E)
axes[0, 1].set_ylabel("E\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)")
axes[0, 1].set_ylim([0,0.015])

axes[1, 0].plot(res["t"], res["y"][0],label="Leaf")
axes[1, 0].plot(res["t"], res["y"][1],label="Stem")
axes[1, 0].plot(res["t"], res["y"][2],label="Root")
axes[1, 0].plot(res["t"], res["y"][3],label="Seed")
axes[1, 0].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^2$)")
axes[1, 0].set_xlabel("Time (days)")
axes[1, 0].legend(loc=2)
axes[1, 0].set_ylim([0,300])

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
fig, axes = plt.subplots(1,3,figsize=(12,3),sharex=True)

axes[0].plot(res["t"], res["y"][4])
axes[0].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C)")
axes[0].set_xlabel("Time (days)")
axes[0].set_title("Growing Degree Days")
axes[0].set_ylim([0,1600])
ax = axes[0]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    ax.vlines(x=res["t"][itime],ymin=0,ymax=res["y"][4,itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

axes[1].plot(res["t"], res["y"][5])
axes[1].set_ylabel("Vernalization Days\n"+r"(-)")
axes[1].set_xlabel("Time (days)")
axes[1].set_title("Vernalization Days")
axes[1].set_ylim([0,125])
ax = axes[1]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    ax.vlines(x=res["t"][itime],ymin=0,ymax=res["y"][5,itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

axes[2].plot(res["t"], _fV)
axes[2].set_ylabel("Vernalization Factor")
axes[2].set_xlabel("Time (days)")
axes[2].set_title("Vernalization Factor\n(Modifier on GDD)")
axes[2].set_ylim([0,1.03])
ax = axes[2]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    ax.vlines(x=res["t"][itime],ymin=0,ymax=_fV[itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)


plt.xlim([time_axis[0],time_axis[-1]])
plt.tight_layout()



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
