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
import numpy as np
import matplotlib.pyplot as plt
from attrs import define, field
from typing import Tuple, Callable
from scipy.stats import beta

# %%
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation

# %% [markdown]
# ## Class CanopyLayers 

# %%
## Instance of class
Canopy = CanopyLayers(nlevmlcan=8)

## input variables
LAI = 3.0
SAI = 0.2
canopy_height = 1.0

# %%
print("Number of canopy layers:",Canopy.nlevmlcan)

# %% [markdown]
# #### Distribute leaf area index over canopy layers

# %%
fig, ax = plt.subplots(1,1,figsize=(4,3))

dlai = Canopy.cast_parameter_over_layers_betacdf(LAI,2,3.5)
ax.plot(dlai,np.arange(0,Canopy.nlevmlcan),label="Leaf (LAI)")
dsai = Canopy.cast_parameter_over_layers_betacdf(SAI,4,1)
ax.plot(dsai,np.arange(0,Canopy.nlevmlcan),label="Stem (SAI)")
ax.plot(dlai+dsai,np.arange(0,Canopy.nlevmlcan),label="Plant (Leaf+Stem)")
ax.legend()
ax.set_xlabel("Layer LAI, SAI, PAI\n"+r"($\rm m^2 \; m^{-2}$)")
ax.set_ylabel("Canopy layer")


# %% [markdown]
# #### Distribute other parameters over canopy layers (e.g. Vcmax)

# %%
Vcmax25_ml = Canopy.cast_parameter_over_layers_exp(80,0.15,LAI)

fig, ax = plt.subplots(1,1,figsize=(4,3))
ax.plot(Vcmax25_ml,np.arange(Canopy.nlevmlcan))
ax.set_xlabel("Layer Vcmax25")
ax.set_ylabel("Canopy layer")
plt.show()

# %% [markdown]
# ## Class CanopyRadiation

# %%
LAI = 3.0
PAI = 0.0
CI = 0.5
canopy_height = 1.0
sza = 10.0

Canopy = CanopyLayers(nlevmlcan=8)
ntop, nbot = Canopy.index_canopy()
print("ntop, nbot =",ntop,",", nbot)

CanopySolar = CanopyRadiation(Canopy=Canopy)

# %%
dpai = 0.5
kb = 10.0
omega = 0.2   # Layer leaf/stem scattering coefficient
avmu = 1/0.7   # average inverse diffuse optical depth per unit leaf area
betad = 1.0
betab = 1.0
albb_below = 0.1
albd_below = 0.1
tbi = 0.1

iabsb_sun, iabsb_sha, iupwb0, idwnb, iabsd_sun, iabsd_sha, iupwd0, idwnd, albb_below, albd_below = CanopySolar.calculate_radiative_flux_layer(dpai, kb, CI, omega, avmu, betad, betab, tbi, albb_below, albd_below)

print("iabsb_sun:",iabsb_sun)
print("iabsb_sha:",iabsb_sha)
print("iupwb0:",iupwb0)
print("idwnb:",idwnb)
print("iabsd_sun:",iabsd_sun)
print("iabsd_sha:",iabsd_sha)
print("iupwd0:",iupwd0)
print("idwnd:",idwnd)
print("albb_below:",albb_below)
print("albd_below:",albd_below)



# %%
(fracsun, kb, omega, avmu, betab, betad, tbi) = CanopySolar.calculateRTProperties(LAI,SAI,CI,canopy_height,sza)

fig, axes = plt.subplots(2,3,figsize=(8,5),sharey=True)

axes[0,0].plot(fracsun,np.arange(Canopy.nlevmlcan),label="fracsun")
axes[0,0].set_xlabel("Sunlit fraction")
axes[0,0].legend()
axes[0,0].set_ylabel("Canopy layer")
axes[0,1].plot(kb,np.arange(Canopy.nlevmlcan),label="kb")
axes[0,1].set_xlabel("Extinction coefficient")
axes[0,1].legend()
axes[0,2].plot(omega,np.arange(Canopy.nlevmlcan),label="omega")
axes[0,2].set_xlabel("Leaf/stem scattering\ncoefficient")
axes[0,2].legend()
axes[1,0].plot(fracsun,np.arange(Canopy.nlevmlcan),label="avmu")
axes[1,0].set_xlabel("Average inverse diffuse\noptical depth\nper unit leaf area")
axes[1,0].legend()
axes[1,0].set_ylabel("Canopy layer")
axes[1,1].plot(betab,np.arange(Canopy.nlevmlcan),label="betab (beam)")
axes[1,1].plot(betad,np.arange(Canopy.nlevmlcan),label="betad (diffuse)")
axes[1,1].set_xlabel("Upscatter parameter")
axes[1,1].legend()
axes[1,2].plot(fracsun,np.arange(Canopy.nlevmlcan),label="tbi")
axes[1,2].set_xlabel("Cumulative transmittance of\ndirect beam onto\ncanopy layer")
axes[1,2].legend()

plt.tight_layout()


# %%

# %%
swskyb = 400.0 # Atmospheric direct beam solar radiation (W/m2)
swskyd = 100.0 # Atmospheric diffuse solar radiation (W/m2)
albsoib = 0.2
albsoid = 0.2

## Calculate RT properties
(fracsun, kb, omega, avmu, betab, betad, tbi) = CanopySolar.calculateRTProperties(LAI,SAI,CI,canopy_height,sza)

dlai = Canopy.cast_parameter_over_layers_betacdf(LAI,Canopy.beta_lai_a,Canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)
dsai = Canopy.cast_parameter_over_layers_betacdf(SAI,Canopy.beta_sai_a,Canopy.beta_sai_b)  # Canopy layer stem area index (m2/m2)
dpai = dlai+dsai  # Canopy layer plant area index (m2/m2)

clump_fac = np.full(Canopy.nlevmlcan, CI)

swleaf = CanopySolar.calculateTwoStream(swskyb,swskyd,dpai,fracsun,kb,clump_fac,omega,avmu,betab,betad,tbi,albsoib,albsoid)

# %%
(fracsun, kb, omega, avmu, betab, betad, tbi) = CanopySolar.calculateRTProperties(LAI,SAI,CI,canopy_height,sza)
fracsun.size

# %%

# %%
fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(swleaf[:,Canopy.isun],np.arange(Canopy.nlevmlcan),label="beam")
axes[0].plot(swleaf[:,Canopy.isha],np.arange(Canopy.nlevmlcan),label="diffuse")
axes[0].set_xlabel("APAR per LAI\n"+r"($\rm W \; m^{-2}$)")
axes[0].set_ylabel("Canopy layer")


axes[1].plot(swleaf[:,Canopy.isun]*dlai,np.arange(Canopy.nlevmlcan),label="beam")
axes[1].plot(swleaf[:,Canopy.isha]*dlai,np.arange(Canopy.nlevmlcan),label="diffuse")
axes[1].set_xlabel("APAR x LAI\n"+r"($\rm W \; m^{-2}$)")
axes[1].legend()

plt.tight_layout()


# %%

# %% [markdown]
# ### Testing scalar vs array_like inputs

# %%
## Input arguments are scalars
_LAI = 1.5
_SAI = 0.0
_CI = 0.5
_z = 1.0
_sza = 30.0
_swskyb = 400.0  # Atmospheric direct beam solar radiation (W/m2)
_swskyd = 100.0  # Atmospheric diffuse solar radiation (W/m2)

swleaf = CanopySolar.calculate(_LAI,_SAI,_CI,_z,_sza,_swskyb,_swskyd)

# %%
swleaf


# %%
## Input arguments are array_like
_LAI = np.array([1.5, 1.5])
_SAI = np.array([0.0, 0.0])
_CI = np.array([0.5, 0.5])
_z = np.array([1.0, 1.0])
_sza = np.array([30.0, 30.0])
_swskyb = np.array([400.0,400.0]) # Atmospheric direct beam solar radiation (W/m2)
_swskyd = np.array([100.0,100.0]) # Atmospheric diffuse solar radiation (W/m2)

swleaf = CanopySolar.calculate(_LAI,_SAI,_CI,_z,_sza,_swskyb,_swskyd)

# %%
from daesim.utils import array_like_wrapper

canopyrad_arraylikewrap = array_like_wrapper(CanopySolar.calculate, ["LAI","SAI","CI","z","sza","swskyb","swskyd"])
swleaf = canopyrad_arraylikewrap(_LAI,_SAI,_CI,_z,_sza,_swskyb,_swskyd)


# %%

# %%

# %%
