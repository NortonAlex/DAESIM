# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext nb_black

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
from daesim.climate import ClimateModule
from daesim.water import WaterModule
from daesim.boundarylayer import BoundaryLayerModule

# %% [markdown]
# ## Potential Evapotranspiration and Net Radiation Calculations

# %% [markdown]
# #### Define a site using the ClimateModule and initialise the WaterModule. 

# %%
site = ClimateModule(CLatDeg=-35.0,CLonDeg=0.0,timezone=0)
water = WaterModule()
boundarylayer = BoundaryLayerModule()

# %% [markdown]
# #### Define a set of inputs including time, radiation and meteorological variables. 

# %%
year = 2020
doy = 1.5
Tmin = 10.0
Tmax = 25.0
RH = 65
albedo = 0.20
fsunhrs = 1.0

## Example to calculate net radiation given above inputs
site.calculate_radiation_net(year,doy,Tmin,Tmax,RH,albedo,fsunhrs)

# %% [markdown]
# #### Define an array of inputs including time, radiation and meteorological variables. 

# %%
doy = np.arange(1,366)
n = doy.size
year = 2020*np.ones(n)
Tmin = 10.0*np.ones(n)   ## daily minimum air temperature (degrees Celsius)
Tmax = 25.0*np.ones(n)   ## daily minimum air temperature (degrees Celsius)
RH = 60.0*np.ones(n)     ## daily relative humidity (%)
P = 101325 * np.ones(n)  ## atmospheric pressure (Pa)
albedo = 0.20            ## surface albedo (-)
fsunhrs = np.ones(n)     ## fraction of full (direct) sunshine hours for each day (hrs)
u_2 = 1.5*np.ones(n)     ## wind speed at 2 m height (m s-1)
G = 0.0*np.ones(n)       ## ground heat flux (MJ m-2 d-1)

T = (Tmin+Tmax)/2        ## calculate daily average air temperature (degrees Celsius)

# %% [markdown]
# #### Run the site methods to determine radiation variables

# %%
_Ra = site.calculate_solarradiation_Ra(year,doy)
_Rs = site.calculate_solarradiation_Rs(year,doy,_Ra,fsunhrs)
_Rso = site.calculate_solarradiation_clearsky(_Ra)
_Rns = site.calculate_radiation_netshortwave(_Rs,albedo)
_Rnl = site.calculate_radiation_netlongwave(_Rs,_Rso,Tmin,Tmax,RH)
_Rnet = site.calculate_radiation_net(year,doy,Tmin,Tmax,RH,albedo,fsunhrs)

# %% [markdown]
# #### Run water module calculations of potential evapotranspiration

# %%
_ET0 = water.calculate_PenmanMonteith_ET0_FAO56(year,doy,T,Tmin,Tmax,RH,u_2,P,G,albedo,fsunhrs,site)
_ET0_hargreaves = water.calculate_Hargreaves_ET0(year,doy,T,Tmin,Tmax,site)

# %% [markdown]
# #### For actual evapotranspiration using the full Penman-Monteith equation, we require surface and aerodynamic resistances

# %%
## A simple empirical approach to determine the aerodynamic resistance is via wind speed and canopy height
canopy_height = 0.12
r_a = boundarylayer.calculate_aerodynamic_resistance(u_2,2.0,canopy_height)

## A simple empirical approach to determine the surfacee resistance is via bulk canopy conductance (via stomata) and "active" leaf area index
r_1 = 100.0
LAI_active = 0.5*(24*canopy_height)
r_s = boundarylayer.calculate_surface_resistance(r_1, LAI_active)

## Calculate evapotranspiration
_ETPM = water.calculate_PenmanMonteith_ET(year,doy,T,Tmin,Tmax,RH,u_2,P,G,albedo,fsunhrs,r_a,r_s,site)

# %% [markdown]
# ### Create Plots

# %%
fig, axes = plt.subplots(1,4,figsize=(14,2.5))

axes[0].plot(doy,_Ra,c='k',label="TOA")
axes[0].plot(doy,_Rso,linestyle="--",label="Surface (clear-sky)")
axes[0].plot(doy,_Rs,label="Surface")
axes[0].legend(handlelength=0.7)
axes[0].set_ylabel("Incoming Shortwave\nRadiation\n"+r"($\rm MJ \; m^{-2} \; d^{-1}$)")

axes[1].plot(doy,_Rns)
axes[1].set_ylabel("Net Shortwave\n"+r"($\rm MJ \; m^{-2} \; d^{-1}$)")
axes[1].plot(doy,_Rnl)
axes[1].set_ylabel("Net Longwave\n"+r"($\rm MJ \; m^{-2} \; d^{-1}$)")
axes[2].plot(doy,_Rnet)
axes[2].set_ylabel("Net Radiation\n"+r"($\rm MJ \; m^{-2} \; d^{-1}$)")
axes[3].plot(doy,_ET0,label="Allen 1998")
axes[3].plot(doy,_ET0_hargreaves,label="Hargreaves 1985")
axes[3].plot(doy,_ETPM,label="PM")
axes[3].set_ylabel("Potential ET\n"+r"($\rm mm \; d^{-1}$)")
axes[3].legend(handlelength=0.7)

axes[0].set_xlabel("Day of Year")
axes[1].set_xlabel("Day of Year")
axes[2].set_xlabel("Day of Year")
axes[3].set_xlabel("Day of Year")

plt.tight_layout()

# %%

# %%
