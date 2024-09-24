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
file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim/data/DAESim_forcing_Milgadara_2021.csv"

df_forcing = pd.read_csv(file)

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
_soilTheta = f_soilTheta_norm

## Create a multi-layer soil moisture forcing dataset
## Option 1: Same soil moisture across all layers
nlevmlsoil = 2
_soilTheta_z = np.repeat(_soilTheta[:, np.newaxis], nlevmlsoil, axis=1)
## Option 2: Adjust soil moisture in each layer
# _soilTheta_z0 = _soilTheta-0.04
# _soilTheta_z1 = _soilTheta+0.04
# _soilTheta_z = np.column_stack((_soilTheta_z0, _soilTheta_z1))

# %%

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
Climate_soilTheta_z_f = interp1d(time_nday, _soilTheta_z, axis=0)  # Interpolates across timesteps, handles all soil layers at once
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
_HTT_time = 0.0
_solRadswskyb = 800    ## incoming shortwave radiation, beam (W m-2)
_solRadswskyd = 200    ## incoming shortwave radiation, diffuse (W m-2)
_airTempCMin = 13.88358116
_airTempCMax = 28.99026108
_airTempC = (_airTempCMin + _airTempCMax)/2
_airPressure = 101325  ## atmospheric pressure (Pa)
_airRH = 65.0   ## relative humidity (%) 
_airCO2 = 400*(_airPressure/1e5)*1e-6     ## carbon dioxide partial pressure (bar)
_airO2 = 209000*(_airPressure/1e5)*1e-6   ## oxygen partial pressure (bar)
_soilTheta = np.array([0.30,0.30])   ## volumetric soil water content (m3 m-3)
_doy = time_doy[_nday-1]
_year = time_year[_nday-1]

management = ManagementModule(plantingDay=30,harvestDay=235)
site = ClimateModule()
soillayers = SoilLayers(nlevmlsoil=2,z_max=2.0)
canopy = CanopyLayers()
canopyrad = CanopyRadiation(Canopy=canopy)
leaf = LeafGasExchangeModule2(Site=site)
canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopyRad=canopyrad)
plantch2o = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange)
plantalloc = PlantOptimalAllocation(Plant=plantch2o,dWL_factor=1.01,dWR_factor=1.02)
plant = PlantModuleCalculator(Site=site,Management=management,PlantCH2O=plantch2o,PlantAlloc=plantalloc,remob_phase="anthesis")

dydt = plant.calculate(
    _Cleaf,
    _Cstem,
    _Croot,
    _Cseed,
    _Bio_time,
    _VD_time,
    _HTT_time,
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
print("  HTT_time = %1.4f" % dydt[6])

# %% [markdown]
# ## Use Model with Numerical Solver

# %%
from daesim.utils import ODEModelSolver

# %% [markdown]
# ### Initialise aggregated model with its classes, initial values for the states, and time axis

# %%
time_axis = np.arange(122, 332, 1)   ## Note: time_axis represents the simulation day (_nday) and must be the same x-axis upon which the forcing data was interpolated on

sowing_date=122
harvest_date=332

# %%
## PlantDev
# PlantDevX = PlantGrowthPhases(
#     gdd_requirements=[100,600,160,140],
#     allocation_coeffs = [
#         [0.0, 0.1, 0.9, 0.0, 0.0],
#         [0.5, 0.1, 0.4, 0.0, 0.0],
#         [0.25, 0.4, 0.25, 0.1, 0.0],
#         [0.15, 0.2, 0.15, 0.5, 0.0]
#     ],
#     turnover_rates = [[0.001,  0.001, 0.001, 0.0, 0.0],
#                       [0.0366, 0.002, 0.0083, 0.0, 0.0],
#                       [0.0633, 0.002, 0.0083, 0.0, 0.0],
#                       [0.1, 0.008, 0.05, 0.0001, 0.0]])


## PlantDev with specific spike formation phase - especially important for for wheat
PlantDevX = PlantGrowthPhases(
    phases=["germination", "vegetative", "spike", "anthesis", "fruiting"],
    gdd_requirements=[50,500,150,160,140],
    vd_requirements=[0, 40, 0, 0, 0],
    allocation_coeffs = [
        [0.0, 0.1, 0.9, 0.0, 0.0],
        [0.5, 0.1, 0.4, 0.0, 0.0],
        [0.20, 0.6, 0.20, 0.0, 0.0],
        [0.25, 0.4, 0.25, 0.1, 0.0],
        [0.1, 0.1, 0.1, 0.7, 0.0]
    ],
    turnover_rates = [[0.001,  0.001, 0.001, 0.0, 0.0],
                      [0.0366, 0.002, 0.0083, 0.0, 0.0],
                      [0.0366, 0.002, 0.0083, 0.0, 0.0],
                      [0.0633, 0.002, 0.0083, 0.0, 0.0],
                      [0.1, 0.008, 0.05, 0.0001, 0.0]])

# %%
ManagementX = ManagementModule(plantingDay=sowing_date,harvestDay=harvest_date)

LeafX = LeafGasExchangeModule2(Site=SiteX)
CanopyX = CanopyLayers(nlevmlcan=3)
CanopyRadX = CanopyRadiation(Canopy=CanopyX)
CanopyGasExchangeX = CanopyGasExchange(Leaf=LeafX,Canopy=CanopyX,CanopyRad=CanopyRadX)
SoilLayersX = SoilLayers(nlevmlsoil=2,z_max=2.0)
PlantCH2OX = PlantCH2O(Site=SiteX,SoilLayers=SoilLayersX,CanopyGasExchange=CanopyGasExchangeX,maxLAI=2.5,ksr_coeff=10000)
PlantAllocX = PlantOptimalAllocation(Plant=PlantCH2OX,dWL_factor=1.02,dWR_factor=1.02)
PlantX = PlantModuleCalculator(
    Site=SiteX,
    Management=ManagementX,
    PlantDev=PlantDevX,
    PlantCH2O=PlantCH2OX,
    PlantAlloc=PlantAllocX,
    propHarvestLeaf=0.75,
    hc_max_GDDindex=sum(PlantDevX.gdd_requirements[0:2])/PlantDevX.totalgdd,
    d_r_max=2.0,
    Vmaxremob=5.0,
    Kmremob=0.5,
)

# %%
## Define the callable calculator that defines the right-hand-side ODE function
PlantXCalc = PlantX.calculate

Model = ODEModelSolver(calculator=PlantXCalc, states_init=[0.5, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0], time_start=time_axis[0])


forcing_inputs = [Climate_solRadswskyb_f,
                  Climate_solRadswskyd_f,
                  Climate_airTempCMin_f,
                  Climate_airTempCMax_f,
                  Climate_airPressure_f,
                  Climate_airRH_f,
                  Climate_airCO2_f,
                  Climate_airO2_f,
                  Climate_soilTheta_z_f,
                  Climate_doy_f,
                  Climate_year_f]

reset_days = [PlantX.Management.plantingDay, PlantX.Management.harvestDay]

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
# ### Calculate diagnostic variables

# %%
LAI = PlantX.PlantCH2O.calculate_LAI(res["y"][0])

eqtime, houranglesunrise, theta = SiteX.solar_calcs(Climate_year_f(time_axis),Climate_doy_f(time_axis))
airTempC = PlantX.Site.compute_mean_daily_air_temp(Climate_airTempCMin_f(time_axis),Climate_airTempCMax_f(time_axis))

W_L = res["y"][0]/PlantX.f_C
W_R = res["y"][2]/PlantX.f_C

## Calculate diagnostic variables
_GPP_gCm2d = np.zeros(time_axis.size)
_Rm_l = np.zeros(time_axis.size) # maintenance respiration of leaves
_Rm_r = np.zeros(time_axis.size) # maintenance respiration of roots
_Ra = np.zeros(time_axis.size) # autotrophic respiration 
_E = np.zeros(time_axis.size)
_DTT = np.zeros(time_axis.size)
_deltaVD = np.zeros(time_axis.size)
_deltaHTT = np.zeros(time_axis.size)
_PHTT = np.zeros(time_axis.size)
_fV = np.zeros(time_axis.size)
_relativeGDD = np.zeros(time_axis.size)
_hc = np.zeros(time_axis.size)
_d_r = np.zeros(time_axis.size)
_Psi_s = np.zeros(time_axis.size)

_Cfluxremob = np.zeros(time_axis.size)

for it,t in enumerate(time_axis):
    sunrise, solarnoon, sunset = PlantX.Site.solar_day_calcs(Climate_year_f(time_axis[it]),Climate_doy_f(time_axis[it]))
    
    # Development phase index
    idevphase = PlantX.PlantDev.get_active_phase_index(res["y"][4,it])
    _relativeGDD[it] = PlantX.PlantDev.calc_relative_gdd_index(res["y"][4,it])
    _hc[it] = PlantX.calculate_canopy_height(_relativeGDD[it])
    relative_gdd_anthesis = PlantX.PlantDev.calc_relative_gdd_to_anthesis(res["y"][4,it])
    _d_r[it] = PlantX.calculate_root_depth(relative_gdd_anthesis)
    PlantX.PlantDev.update_vd_state(res["y"][5,it],res["y"][4,it])    # Update vernalization state information to track developmental phase changes
    VD = PlantX.PlantDev.get_phase_vd()    # Get vernalization state for current developmental phase
    # Update vernalization days requirement for current developmental phase
    PlantX.VD50 = 0.5 * PlantX.PlantDev.vd_requirements[idevphase]
    _deltaVD[it] = PlantX.calculate_vernalizationtime(Climate_airTempCMin_f(time_axis[it]),Climate_airTempCMax_f(time_axis[it]),sunrise,sunset)
    _fV[it] = PlantX.vernalization_factor(res["y"][5,it])

    _deltaHTT[it] = PlantX.calculate_dailyhydrothermaltime(airTempC[it], Climate_soilTheta_f(time_axis[it]))
    #_PHTT[it] = PlantX.calculate_P_HTT(res["y"][6,it], airTempC[it], Climate_soilTheta_f(time_axis[it]))

    _DTT[it] = PlantX.calculate_dailythermaltime(Climate_airTempCMin_f(time_axis[it]),Climate_airTempCMax_f(time_axis[it]),sunrise,sunset)

    ## GPP and Transpiration (E)
    GPP, Rml, Rmr, E, fPsil, Psil, Psir, Psis, K_s, K_sr, k_srl = PlantX.PlantCH2O.calculate(W_L[it],W_R[it],Climate_soilTheta_z_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airTempC_f(time_axis)[it],Climate_airRH_f(time_axis)[it],Climate_airCO2_f(time_axis)[it],Climate_airO2_f(time_axis)[it],Climate_airPressure_f(time_axis)[it],Climate_solRadswskyb_f(time_axis)[it],Climate_solRadswskyd_f(time_axis)[it],theta[it],_hc[it],_d_r[it])
    _GPP_gCm2d[it] = GPP * 12.01 * (60*60*24) / 1e6  ## converts umol C m-2 s-1 to g C m-2 d-1
    _E[it] = E
    _Rm_l[it] = Rml
    _Rm_r[it] = Rmr
    _Psi_s[it] = Psis   ## Note: this is the soil water potential in the root zone only
    _Cfluxremob[it] = PlantX.calculate_nsc_stem_remob(res["y"][1,it], res["y"][0,it], res["y"][4,it])
    

NPP = PlantX.calculate_NPP(_GPP_gCm2d)
Ra = _GPP_gCm2d - NPP  # units of gC m-2 d-1
Rm_l_gCm2d = _Rm_l * 12.01 * (60*60*24) / 1e6
Rm_r_gCm2d = _Rm_r * 12.01 * (60*60*24) / 1e6

BioHarvestSeed = PlantX.calculate_BioHarvest(res["y"][3],Climate_doy_f(time_axis),ManagementX.harvestDay,PlantX.propHarvestSeed,ManagementX.PhHarvestTurnoverTime)

fV = PlantX.vernalization_factor(res["y"][5])

# %%
## Calculate diagnostic variables for allocation coefficients and turnover rates
_u_Leaf = np.zeros(time_axis.size)
_u_Root = np.zeros(time_axis.size)
_u_Stem = np.zeros(time_axis.size)
_u_Seed = np.zeros(time_axis.size)

_tr_Leaf = np.zeros(time_axis.size)
_tr_Root = np.zeros(time_axis.size)
_tr_Stem = np.zeros(time_axis.size)
_tr_Seed = np.zeros(time_axis.size)

for it,t in enumerate(time_axis):

    idevphase = PlantX.PlantDev.get_active_phase_index(res["y"][4,it])
    # Allocation fractions per pool
    alloc_coeffs = PlantX.PlantDev.allocation_coeffs[idevphase]
    _u_Stem[it] = alloc_coeffs[PlantX.PlantDev.istem]
    _u_Seed[it] = alloc_coeffs[PlantX.PlantDev.iseed]
    # Turnover rates per pool
    tr_ = PlantX.PlantDev.turnover_rates[idevphase]
    _tr_Leaf[it] = tr_[PlantX.PlantDev.ileaf]
    _tr_Root[it] = tr_[PlantX.PlantDev.iroot]
    _tr_Stem[it] = tr_[PlantX.PlantDev.istem]
    _tr_Seed[it] = tr_[PlantX.PlantDev.iseed]
    # Set any constant allocation coefficients for optimal allocation
    PlantX.PlantAlloc.u_Stem = alloc_coeffs[PlantX.PlantDev.istem]
    PlantX.PlantAlloc.u_Seed = alloc_coeffs[PlantX.PlantDev.iseed]
    # Set pool turnover rates for optimal allocation
    PlantX.PlantAlloc.tr_L = tr_[PlantX.PlantDev.ileaf]    #1 if tr_[self.PlantDev.ileaf] == 0 else max(1, 1/tr_[self.PlantDev.ileaf])
    PlantX.PlantAlloc.tr_R = tr_[PlantX.PlantDev.iroot]    #1 if tr_[self.PlantDev.iroot] == 0 else max(1, 1/tr_[self.PlantDev.ileaf])

    u_L, u_R, _, _, _, _ = PlantX.PlantAlloc.calculate(
        W_L[it],
        W_R[it],
        Climate_soilTheta_z_f(time_axis[it]),
        Climate_airTempC_f(time_axis[it]),
        Climate_airTempC_f(time_axis[it]),
        Climate_airRH_f(time_axis[it]),
        Climate_airCO2_f(time_axis[it]),
        Climate_airO2_f(time_axis[it]),
        Climate_airPressure_f(time_axis[it]),
        Climate_solRadswskyb_f(time_axis[it]),
        Climate_solRadswskyd_f(time_axis[it]),
        theta[it],
        _hc[it],
        _d_r[it])

    _u_Leaf[it] = u_L
    _u_Root[it] = u_R

# %% [markdown]
# ### Create figures

# %%
site_year = "2021"
site_name = "Site - Generic Crop"
site_filename = "Site_GenericCrop_wspikephase_wstemremob_CUE0p6"

# %%
fig, axes = plt.subplots(4,1,figsize=(8,8),sharex=True)

axes[0].plot(res["t"], Climate_solRadswskyb_f(time_axis)+Climate_solRadswskyd_f(time_axis), c='0.4')
axes[0].plot(res["t"], Climate_solRadswskyb_f(time_axis), c='goldenrod', alpha=0.5)
axes[0].plot(res["t"], Climate_solRadswskyd_f(time_axis), c='C0', alpha=0.5)
axes[0].set_ylabel("Solar radiation\n"+r"($\rm W \; m^{-2}$)")
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

axes[0].set_xlim([PlantX.Management.plantingDay,time_axis[-1]])
# axes[0].set_xlim([0,time_axis[-1]])

plt.tight_layout()
plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/MilgaSite_Climate_2021_mlsoil_test2.png",dpi=300,bbox_inches='tight')



# %%
fig, axes = plt.subplots(5,1,figsize=(8,10),sharex=True)

axes[0].plot(res["t"], LAI)
axes[0].set_ylabel("LAI\n"+r"($\rm m^2 \; m^{-2}$)")
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].annotate("Leaf area index", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[0].set_ylim([0,3])

axes[1].plot(res["t"], _GPP_gCm2d)
axes[1].set_ylabel("GPP\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].annotate("Photosynthesis", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[1].set_ylim([0,20])

# axes[2].plot(res["t"], _E*1e3)
# axes[2].set_ylabel(r"$\rm E$"+"\n"+r"($\rm mmol \; H_2O \; m^{-2} \; s^{-1}$)")
## Conversion notes: When _E units are mol m-2 s-1, multiply by molar mass H2O to get g m-2 s-1, divide by 1000 to get kg m-2 s-1, multiply by 60*60*24 to get kg m-2 d-1, and 1 kg m-2 d-1 = 1 mm d-1. 
## Noting that 1 kg of water is equivalent to 1 liter (L) of water (because the density of water is 1000 kg/m³), and 1 liter of water spread over 1 square meter results in a depth of 1 mm
axes[2].plot(res["t"], _E*18.015/1000*(60*60*24))   
axes[2].set_ylabel(r"$\rm E$"+"\n"+r"($\rm mm \; d^{-1}$)")
axes[2].tick_params(axis='x', labelrotation=45)
axes[2].annotate("Transpiration Rate", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[2].set_ylim([0,6])

# axes[4].plot(df_forcing.index.values[364:-1], 0.5*np.cumsum(GPP[364:]))
axes[3].plot(res["t"], res["y"][4])
axes[3].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C d)")
axes[3].set_xlabel("Time (days)")
axes[3].annotate("Growing Degree Days - Developmental Phase", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
ax = axes[3]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    print("iphase, phase =",iphase, phase)
    print(" itime =",itime, " res[t]=",res["t"][itime])
    # print("   res[y][4] =",np.abs(res["y"][4]))
    # print("   np.cumsum(PlantDevX.gdd_requirements)[iphase] =", np.cumsum(PlantDevX.gdd_requirements)[iphase])
    ax.vlines(x=res["t"][itime],ymin=0,ymax=res["y"][4,itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

alp = 0.6
axes[4].plot(res["t"], res["y"][0]+res["y"][1]+res["y"][2]+res["y"][3],c='k',label="Plant", alpha=alp)
axes[4].plot(res["t"], res["y"][0],label="Leaf", alpha=alp)
axes[4].plot(res["t"], res["y"][1],label="Stem", alpha=alp)
axes[4].plot(res["t"], res["y"][2],label="Root", alpha=alp)
axes[4].plot(res["t"], res["y"][3],label="Seed", alpha=alp)
axes[4].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^2$)")
axes[4].set_xlabel("Time (day of year)")
axes[4].legend(loc=3,fontsize=9,handlelength=0.8)
axes[4].set_ylim([-7,180])

accumulated_carbon = res["y"][0]+res["y"][1]+res["y"][2]+res["y"][3]
eos_accumulated_carbon = accumulated_carbon[-1]    # end-of-season total carbon (at the end of the simulation period)
peak_accumulated_carbon_noseed = np.max(res["y"][0])+np.max(res["y"][1])+np.max(res["y"][2])    # peak carbon, excluding seed biomass
peak_accumulated_carbon_noseedroot = np.max(res["y"][0])+np.max(res["y"][1])    # peak carbon, excluding seed biomass
harvest_index = res["y"][3][-1]/(res["y"][0][-1]+res["y"][1][-1]+res["y"][2][-1]+res["y"][3][-1])
harvest_index_peak = res["y"][3][-1]/peak_accumulated_carbon_noseed
harvest_index_peak_noroot = res["y"][3][-1]/peak_accumulated_carbon_noseedroot
yield_from_seed_Cpool = res["y"][3][-1]/100 * (1/PlantX.f_C)   ## convert gC m-2 to t dry biomass ha-1
axes[4].annotate("Yield = %1.2f t/ha" % (yield_from_seed_Cpool), (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[4].annotate("Harvest index = %1.2f" % (harvest_index_peak), (0.01,0.81), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[4].set_ylim([0,200])

print("Harvest index (end-of-simulation seed:end-of-simulation plant) = %1.2f" % harvest_index)
print("Harvest index (end-of-simulation seed:peak plant biomass (excl seed)) = %1.2f" % harvest_index_peak)
print("Harvest index (end-of-simulation seed:peak plant biomass (excl seed, root)) = %1.2f" % harvest_index_peak_noroot)

axes[0].set_xlim([PlantX.Management.plantingDay,time_axis[-1]])

axes[0].set_title("%s - %s" % (site_year,site_name))
plt.tight_layout()
plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/MilgaSite_DAESim_%s_%s.png" % (site_year,site_filename),dpi=300,bbox_inches='tight')



# %%

# %%
_Cseed_add_remob = np.zeros(time_axis.size) #res["t"][4]
t_fruiting_start = 295
t_fruiting_end = 308

for it,t in enumerate(time_axis):
    # print(it,t)
    if (t >= t_fruiting_start) & (t <= t_fruiting_end):
        _Cseed_add_remob[it] = res["y"][3,it] + _Cfluxremob[it]
    else:
        _Cseed_add_remob[it] = res["y"][3,it]

# %%
fig, axes = plt.subplots(5,1,figsize=(8,10),sharex=True)

axes[0].plot(res["t"], _Cfluxremob)
axes[0].set_ylabel("Remobilization flux\n"+r"($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].tick_params(axis='x', labelrotation=45)
# axes[0].annotate("Photosynthesis", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
# axes[0].set_ylim([0,20])

axes[1].plot(res["t"], res["y"][3], label=r"$\rm C_{seed}$")
axes[1].plot(res["t"], _Cseed_add_remob, label=r"$\rm C_{seed} + Remob.$")

axes[2].plot(res["t"], res["y"][1]/res["y"][0],label=r"$\rm R_{S,L}$")
axes[2].plot(res["t"], res["y"][1]/(res["y"][0]+res["y"][2]),label=r"$\rm R_{S,L+R}$")
axes[2].plot(res["t"], res["y"][1]/(res["y"][0]+res["y"][1]+res["y"][2]),label=r"$\rm R_{S,L+R+S}$")
axes[2].set_ylabel("Plant pool ratios\n"+r"(-)")
axes[2].set_xlabel("Time (days)")
axes[2].legend(fontsize=9,handlelength=0.8)
axes[2].set_ylim([0,1])

axes[3].plot(res["t"], res["y"][4])
axes[3].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C d)")
axes[3].set_xlabel("Time (days)")
axes[3].annotate("Growing Degree Days - Developmental Phase", (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
ax = axes[3]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    # print("iphase, phase =",iphase, phase)
    # print(" itime =",itime)
    # print("   res[y][4] =",np.abs(res["y"][4]))
    # print("   np.cumsum(PlantDevX.gdd_requirements)[iphase] =", np.cumsum(PlantDevX.gdd_requirements)[iphase])
    ax.vlines(x=res["t"][itime],ymin=0,ymax=res["y"][4,itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

alp = 0.6
axes[4].plot(res["t"], res["y"][0]+res["y"][1]+res["y"][2]+res["y"][3],c='k',label="Plant", alpha=alp)
axes[4].plot(res["t"], res["y"][0],label="Leaf", alpha=alp)
axes[4].plot(res["t"], res["y"][1],label="Stem", alpha=alp)
axes[4].plot(res["t"], res["y"][2],label="Root", alpha=alp)
axes[4].plot(res["t"], res["y"][3],label="Seed", alpha=alp)
axes[4].set_ylabel("Carbon Pool Size\n"+r"(g C $\rm m^2$)")
axes[4].set_xlabel("Time (day of year)")
axes[4].legend(loc=3,fontsize=9,handlelength=0.8)
axes[4].set_ylim([-7,180])

accumulated_carbon = res["y"][0]+res["y"][1]+res["y"][2]+res["y"][3]
eos_accumulated_carbon = accumulated_carbon[-1]    # end-of-season total carbon (at the end of the simulation period)
peak_accumulated_carbon_noseed = np.max(res["y"][0])+np.max(res["y"][1])+np.max(res["y"][2])    # peak carbon, excluding seed biomass
peak_accumulated_carbon_noseedroot = np.max(res["y"][0])+np.max(res["y"][1])    # peak carbon, excluding seed biomass
harvest_index = res["y"][3][-1]/(res["y"][0][-1]+res["y"][1][-1]+res["y"][2][-1]+res["y"][3][-1])
harvest_index_peak = res["y"][3][-1]/peak_accumulated_carbon_noseed
harvest_index_peak_noroot = res["y"][3][-1]/peak_accumulated_carbon_noseedroot
yield_from_seed_Cpool = res["y"][3][-1]/100 * (1/PlantX.f_C)   ## convert gC m-2 to t dry biomass ha-1
axes[4].annotate("Yield = %1.2f t/ha" % (yield_from_seed_Cpool), (0.01,0.93), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)
axes[4].annotate("Harvest index = %1.2f" % (harvest_index_peak), (0.01,0.81), xycoords='axes fraction', verticalalignment='top', horizontalalignment='left', fontsize=12)

print("Harvest index (end-of-simulation seed:end-of-simulation plant) = %1.2f" % harvest_index)
print("Harvest index (end-of-simulation seed:peak plant biomass (excl seed)) = %1.2f" % harvest_index_peak)
print("Harvest index (end-of-simulation seed:peak plant biomass (excl seed, root)) = %1.2f" % harvest_index_peak_noroot)

axes[0].set_xlim([PlantX.Management.plantingDay,time_axis[-1]])

axes[0].set_title("%s - %s" % (site_year,site_name))
plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/MilgaSite_DAESim_%s_%s_RH2.png" % (site_year,site_filename),dpi=300,bbox_inches='tight')


# %%
fig, axes = plt.subplots(2,3,figsize=(12,6),sharex=True)

axes[0,0].plot(res["t"], res["y"][4])
axes[0,0].set_ylabel("Thermal Time\n"+r"($\rm ^{\circ}$C)")
axes[0,0].set_xlabel("Time (days)")
axes[0,0].set_title("Growing Degree Days")
# axes[0,0].set_ylim([0,1600])
# axes[0,0].vlines(x=PlantX.Management.plantingDay,ymin=0,ymax=res["y"][4,itime],color='0.5')
ax = axes[0,0]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    ax.vlines(x=res["t"][itime],ymin=0,ymax=res["y"][4,itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

axes[1,0].plot(res["t"], _DTT, c='C0', linestyle=":", label="DTT")
axes[1,0].plot(res["t"], _DTT*_fV, c='C0', linestyle="-", label=r"$\rm DTT \times f_V$")
axes[1,0].set_ylabel("Daily Thermal Time\n"+r"($\rm ^{\circ}$C)")
axes[1,0].set_xlabel("Time (days)")
axes[1,0].set_title("Growing Degree Days")
axes[1,0].legend()
# axes[1,0].vlines(x=PlantX.Management.plantingDay,ymin=0,ymax=res["y"][4,itime],color='0.5')


axes[0,1].plot(res["t"], res["y"][5])
axes[0,1].set_ylabel("Vernalization Days\n"+r"(-)")
axes[0,1].set_xlabel("Time (days)")
axes[0,1].set_title("Vernalization Days")
# axes[0,1].set_ylim([0,125])
ax = axes[0,1]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    ax.vlines(x=res["t"][itime],ymin=0,ymax=res["y"][5,itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

axes[1,1].plot(res["t"], _fV)
axes[1,1].set_ylabel("Vernalization Factor")
axes[1,1].set_xlabel("Time (days)")
axes[1,1].set_title("Vernalization Factor\n(Modifier on GDD)")
# axes[1,1].set_ylim([0,1.03])
ax = axes[1,1]
for iphase,phase in enumerate(PlantDevX.phases):
    itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
    ax.vlines(x=res["t"][itime],ymin=0,ymax=_fV[itime],color='0.5')
    text_x = res["t"][itime]
    text_y = 0.04
    ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)


axes[0,2].plot(res["t"], res["y"][6])
axes[0,2].set_ylabel("Hydrothermal Time\n"+r"($\rm MPa \; ^{\circ}$C d)")
axes[0,2].set_xlabel("Time (days)")
axes[0,2].set_title("Hydrothermal Time")
# axes[1].set_ylim([0,125])
# ax = axes[1]
# for iphase,phase in enumerate(PlantDevX.phases):
#     itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
#     ax.vlines(x=res["t"][itime],ymin=0,ymax=res["y"][5,itime],color='0.5')
#     text_x = res["t"][itime]
#     text_y = 0.04
#     ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

axes[1,2].plot(res["t"], _deltaHTT)
axes[1,2].set_ylabel("Daily Hydrothermal Time\n"+r"($\rm MPa \; ^{\circ}$C)")
axes[1,2].set_xlabel("Time (days)")
axes[1,2].set_title("Hydrothermal Time per Day")
# axes[1,2].set_ylim([0,1.03])
# ax = axes[2]
# for iphase,phase in enumerate(PlantDevX.phases):
#     itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase]))
#     ax.vlines(x=res["t"][itime],ymin=0,ymax=_fV[itime],color='0.5')
#     text_x = res["t"][itime]
#     text_y = 0.04
#     ax.text(text_x, text_y, phase, horizontalalignment='right', verticalalignment='bottom', fontsize=8, alpha=0.7, rotation=90)

plt.xlim([time_axis[0],time_axis[-1]])
plt.tight_layout()
plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/MilgaSite_DAESim_%s_%s_plantdev.png" % (site_year,site_filename),dpi=300,bbox_inches='tight')



# %%
# Creating a DataFrame
df_u_W = pd.DataFrame({
    'Time': time_axis,
    'u_Leaf': _u_Leaf,
    'u_Root': _u_Root,
    'u_Stem': _u_Stem,
    'u_Seed': _u_Seed
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
ax.plot(df_u_W['Time'], df_u_W['RunningMean_u_Leaf'], label='Leaf')
ax.plot(df_u_W['Time'], df_u_W['RunningMean_u_Root'], label='Root')
ax.plot(df_u_W['Time'], df_u_W['u_Stem'], label='Stem')
ax.plot(df_u_W['Time'], df_u_W['u_Seed'], label='Seed')
ax.set_ylim([0,1.01])
ax.set_xlim([PlantX.Management.plantingDay,time_axis[-1]])
ax.set_xlabel("Time (day of year)")
ax.set_ylabel("Carbon allocation\ncoefficient")
ax.legend(handlelength=0.75)

for iphase, phase in enumerate(PlantDevX.phases):
    if iphase == 0:
        # Special case for the first phase
        itime = np.argmin(np.abs(res["y"][4] - 0))
    else:
        # For subsequent phases, calculate itime based on cumulative GDD requirements
        itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase - 1]))
    # Plot vertical line at the determined time point
    ax.vlines(x=res["t"][itime], ymin=0, ymax=1, color='0.5')
    # Set text position and plot text
    text_x = res["t"][itime] + 1.5
    text_y = 1
    ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='top',
            fontsize=8, alpha=0.7, rotation=90)


ax = axes[1]
ax.plot(time_axis, _tr_Leaf, label='Leaf')
ax.plot(time_axis, _tr_Root, label='Root')
ax.plot(time_axis, _tr_Stem, label='Stem')
ax.plot(time_axis, _tr_Seed, label='Seed')
# ax.set_ylim([0,1.01])
ax.set_xlim([PlantX.Management.plantingDay,time_axis[-1]])
ax.set_xlabel("Time (day of year)")
ax.set_ylabel("Turnover rate\n"+r"($\rm days^{-1}$)")
# ax.legend(handlelength=0.75)

xminlim, xmaxlim = 0, 0.105

for iphase, phase in enumerate(PlantDevX.phases):
    if iphase == 0:
        # Special case for the first phase
        itime = np.argmin(np.abs(res["y"][4] - 0))
    else:
        # For subsequent phases, calculate itime based on cumulative GDD requirements
        itime = np.argmin(np.abs(res["y"][4] - np.cumsum(PlantDevX.gdd_requirements)[iphase - 1]))
    # Plot vertical line at the determined time point
    ax.vlines(x=res["t"][itime], ymin=0, ymax=1, color='0.5')
    # Set text position and plot text
    text_x = res["t"][itime] + 1.5
    text_y = xmaxlim
    ax.text(text_x, text_y, phase, horizontalalignment='left', verticalalignment='top',
            fontsize=8, alpha=0.7, rotation=90)

ax.set_ylim([xminlim, xmaxlim])

# plt.grid(True)
plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/MilgaSite_DAESim_%s_%s_alloctr_mlsoil.png" % (site_year,site_filename),dpi=300,bbox_inches='tight')
plt.show()


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
