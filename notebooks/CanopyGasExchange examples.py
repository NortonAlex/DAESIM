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
import numpy as np
import matplotlib.pyplot as plt

# %%
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.leafgasexchange import LeafGasExchangeModule
from daesim.leafgasexchange2 import LeafGasExchangeModule2
from daesim.canopygasexchange import CanopyGasExchange
from daesim.climate import *
from daesim.biophysics_funcs import fT_Q10, fT_arrhenius, fT_arrheniuspeaked
from daesim.boundarylayer import BoundaryLayerModule
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.climate import *
from daesim.biophysics_funcs import fT_Q10, fT_arrhenius, fT_arrheniuspeaked
from daesim.boundarylayer import BoundaryLayerModule

# %% [markdown]
# # Canopy Gas Exchange
#
# Below is an example of how to calculate canopy photosynthesis and stomatal conductance.
#
# To calculate canopy gas exchange we first require a description of:
#
#  - The canopy itself. This is discretized into layers and requires inputs of leaf area index (LAI), stem area index (SAI) and their vertical distributions.
#  - The canopy radiation scheme. This includes a description of the absorption of radiation by canopy foliage, a key driver of canopy gas exchange.
#  - The leaf gas exchange scheme. This includes equations for photosynthesis and stomatal conductance. 

# %% [markdown]
# ## First, create an instance of a site

# %%
Site = ClimateModule()

# %% [markdown]
# ## Define input parameters

# %%
## input variables for canopy layers and canopy radiation
LAI = 1.5    ## leaf area index (m2 m-2)
SAI = 0.2    ## stem area index (m2 m-2)
clumping_factor = 0.5   ## foliage clumping index (-)
canopy_height = 1.0     ## canopy height (m)
sza = 30.0       ## solar zenith angle (degrees)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)
albsoib = 0.2    ## Soil background albedo for beam radiation
albsoid = 0.2    ## Soil background albedo for diffuse radiation
leafreflectance = 0.10  ## leaf reflectance (-)

## input variables for leaf gas exchange model
p = 101325   ## air pressure (Pa)
T = 25.0     ## leaf temperature (degrees Celsius)
Cs = 400*(p/1e5)*1e-6     ## carbon dioxide partial pressure (bar)
O = 209000*(p/1e5)*1e-6   ## oxygen partial pressure (bar)
RH = 65.0    ## relative humidity (%)

# %% [markdown]
# ### Create instance of LeafGasExchange class

# %%
Leaf = LeafGasExchangeModule2()

# %% [markdown]
# ## Big-Leaf Approach
#
# This approach treats the canopy as a single "big leaf". 

# %%
from daesim.canopyradiation_bigleaf import CanopyRadiation as CanopyRadiationBigLeaf

# %%
if clumping_factor == 0.5:
    LAD_type = "spherical"

canopyrad_bigleaf = CanopyRadiationBigLeaf(LAD_type=LAD_type,rho=leafreflectance)

# %%
P = canopyrad_bigleaf.canopy_gap_fraction(np.deg2rad(sza),LAI,LAD_type=LAD_type)
print("Gap fraction, P(theta) = %1.3f" % P)
fAPAR = canopyrad_bigleaf.canopy_fraction_absorbed_irradiance(np.deg2rad(sza),LAI,LAD_type=LAD_type)
print("Fraction of absorbed shortwave irradiance = %1.3f" % fAPAR)

swdown = swskyb + swskyd
print("Absorbed shortwave irradiance = %1.3f" % (fAPAR*swdown),"W m-2")

# %%
Q = 1e-6 * (swdown*fAPAR) * CanopyRadiation().J_to_umol  # absorbed PPFD, umol PAR m-2 s-1
A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH,1.0)

# %%
Anet_bigleaf = A*1e6
gs_bigleaf = gs

print("Canopy An = %1.1f" % (A*1e6),"umol m-2 s-1")
print("Canopy GPP = %1.1f" % ((A+Rd)*1e6),"umol m-2 s-1")

# %% [markdown]
# ## Multi-Layer Canopy Approach
#
# This follows the canopy radiation scheme of Bonan et al. (2021, doi:10.1016/j.agrformet.2021.108435) which is implemented in the Community Land Model multi-layer model v1 (CLM-ml v1). 
#
# Currently, we only implement the multi-layer canopy radiative transfer scheme. We do not incorporate corresponding formulations for latent heat flux, sensible heat flux, friction velocity or vertical profiles of within-canopy temperature or specific humidity. 

# %%
## Instance of CanopyLayers class
canopy = CanopyLayers(nlevmlcan=8)
canopy.set_index()

## Instance of CanopyRadiation class with upstream module dependencies
canopyrad = CanopyRadiation(Canopy=canopy,rhol=leafreflectance)

## Instance of CanopyGasExchange class with upstream module dependencies
canopygasexchange = CanopyGasExchange(Leaf=Leaf,Canopy=canopy,CanopyRad=canopyrad)

# %% [markdown]
# ### Testing scalar vs array_like inputs

# %%
(fracsun, kb, omega, avmu, betab, betad, tbi) = canopyrad.calculateRTProperties(LAI,SAI,clumping_factor,canopy_height,sza)

fracsun

# %%
n = 3
_T = np.array([20,22,24])
_Cs = Cs*np.ones(n)
_O = O*np.ones(n)
_RH = RH*np.ones(n)
_LAI = np.array([1.5,1.6,1.7])
_SAI = SAI*np.ones(n)
_CI = clumping_factor*np.ones(n)
_z = canopy_height*np.ones(n)
_sza = sza*np.ones(n)
_swskyb = np.array([180,200,220])
_swskyd = np.array([80,80,80])

An_ml, gs_ml, Rd_ml = canopygasexchange.calculate(_T,_Cs,_O,_RH,1.0,LAI,SAI,clumping_factor,canopy_height,_sza,_swskyb,_swskyd)


# %%
# swleaf, swveg, swvegsun, swvegsha = canopyrad.calculate(np.array([LAI,LAI]),SAI,clumping_factor,canopy_height,sza,swskyb,swskyd,Canopy=canopy)

# swleaf

# _vfunc = np.vectorize(canopyrad.calculateRTProperties)
# (np.array([LAI,LAI]),SAI,clumping_factor,canopy_height,sza,Canopy=canopy)


# _vfunc = np.vectorize(canopy.cast_parameter_over_layers_betacdf)
# dlai = _vfunc(_LAI,canopy.beta_lai_a,canopy.beta_lai_b)


_vfunc = np.vectorize(canopy.cast_parameter_over_layers_uniform)
dlai = _vfunc(_LAI)
dlai

# %%

# %%
An_ml

# %% [markdown]
# ## Run radiative transfer calculations

# %%
An_ml, gs_ml, Rd_ml = canopygasexchange.calculate(T,Cs,O,RH,1.0,LAI,SAI,clumping_factor,canopy_height,sza,swskyb,swskyd)

An_total = np.sum(1e6*An_ml)
gs_total = np.sum(gs_ml)
Rd_total = np.sum(1e6*Rd_ml)

fig, axes = plt.subplots(1,3,figsize=(12,3))

axes[0].plot(1e6*An_ml,np.arange(1,canopy.nlevmlcan+1),c='k',alpha=0.5)
axes[0].set_xlabel("Net photosynthetic rate (umol m-2 s-1)\non ground area basis")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(gs_ml,np.arange(1,canopy.nlevmlcan+1),c='k',alpha=0.5)
axes[1].set_xlabel("Stomatal conductance (mol m-2 s-1)\non ground area basis")
axes[1].set_ylabel("Canopy layer")

axes[2].plot(1e6*Rd_ml,np.arange(1,canopy.nlevmlcan+1),c='k',alpha=0.5)
axes[2].set_xlabel("Mitochondrial respiration (umol m-2 s-1)\non ground area basis")
axes[2].set_ylabel("Canopy layer")

plt.show()

print("Sum of An_ml over canopy layers = %1.2f umol m-2 s-1" % (An_ml.sum()*1e6))
print("Sum of gs_ml over canopy layers = %1.2f mol m-2 s-1" % (gs_ml.sum()))
print("Sum of Rd_ml over canopy layers = %1.2f umol m-2 s-1" % (Rd_ml.sum()*1e6))


# %%
## Option: Prescribe selected forcing data per canopy element (by layer and sunlit/shaded leaves)
## - vertical gradient in leaf water potential effect on stomatal conductance
fgsw_ml = np.zeros((canopy.nlevmlcan, canopy.nleaf))
fgsw_ml[:,0] = canopy.cast_parameter_over_layers_exp(1,0.3,1)[::-1]
fgsw_ml[:,1] = canopy.cast_parameter_over_layers_exp(1,0.3,1)[::-1]

An_ml1, gs_ml1, Rd_ml1 = canopygasexchange.calculate(T,Cs,O,RH,fgsw_ml,LAI,SAI,clumping_factor,canopy_height,sza,swskyb,swskyd)


fig, axes = plt.subplots(1,3,figsize=(12,3))

axes[0].plot(1e6*An_ml,np.arange(1,canopy.nlevmlcan+1),c='k',alpha=0.5)
axes[0].plot(1e6*An_ml1,np.arange(1,canopy.nlevmlcan+1),c='crimson',alpha=0.5)
axes[0].set_xlabel("Net photosynthetic rate (umol m-2 s-1)\non ground area basis")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(gs_ml,np.arange(1,canopy.nlevmlcan+1),c='k',alpha=0.5)
axes[1].plot(gs_ml1,np.arange(1,canopy.nlevmlcan+1),c='crimson',alpha=0.5)
axes[1].set_xlabel("Stomatal conductance (mol m-2 s-1)\non ground area basis")
axes[1].set_ylabel("Canopy layer")

axes[2].plot(1e6*Rd_ml,np.arange(1,canopy.nlevmlcan+1),c='k',alpha=0.5)
axes[2].plot(1e6*Rd_ml1,np.arange(1,canopy.nlevmlcan+1),c='crimson',alpha=0.5)
axes[2].set_xlabel("Mitochondrial respiration (umol m-2 s-1)\non ground area basis")
axes[2].set_ylabel("Canopy layer")

plt.show()

print("Sum of An_ml over canopy layers = %1.2f umol m-2 s-1" % (An_ml1.sum()*1e6))
print("Sum of gs_ml over canopy layers = %1.2f mol m-2 s-1" % (gs_ml1.sum()))
print("Sum of Rd_ml over canopy layers = %1.2f umol m-2 s-1" % (Rd_ml1.sum()*1e6))


# %% [markdown]
# ### Show steps involved in canopy gas exchange calculations

# %%
## Calculate radiative transfer properties of the canopy
(fracsun, kb, omega, avmu, betab, betad, tbi) = canopyrad.calculateRTProperties(LAI,SAI,clumping_factor,canopy_height,sza)

# %%
## Determine vertical distribution of canopy properties such as LAI, SAI and clumping index
dlai = canopy.cast_parameter_over_layers_betacdf(LAI,canopy.beta_lai_a,canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)
dsai = canopy.cast_parameter_over_layers_betacdf(SAI,canopy.beta_sai_a,canopy.beta_sai_b)  # Canopy layer stem area index (m2/m2)
dpai = dlai+dsai  # Canopy layer plant area index (m2/m2)
clump_fac = canopy.cast_parameter_over_layers_uniform(clumping_factor)

# %%
## Calculate two stream approximation calculations
## Note: swleaf is the absorption per leaf area index (W/m2 per leaf)
swleaf = canopyrad.calculateTwoStream(swskyb,swskyd,dpai,fracsun,kb,clump_fac,omega,avmu,betab,betad,tbi,albsoib,albsoid)

# %%
## Calculate total absorbed shortwave radiation by sunlit and shaded vegetation
swveg = 0
swvegsun = 0
swvegsha = 0
for ic in range(canopy.nbot, canopy.ntop + 1):
    sun = swleaf[ic,canopy.isun] * fracsun[ic] * dlai[ic]
    sha = swleaf[ic,canopy.isha] * (1.0 - fracsun[ic]) * dlai[ic]
    swveg += (sun + sha)
    swvegsun += sun
    swvegsha += sha

print("Absorbed shortwave radiation = %1.3f W m-2" % swveg)
print("Absorbed shortwave radiation by sunlit leaves = %1.3f W m-2" % swvegsun)
print("Absorbed shortwave radiation by sunlit leaves = %1.3f W m-2" % swvegsha)


# %%
## Run leaf gas exchange calculations over each canopy element
_An = np.zeros((canopy.nlevmlcan,canopy.nleaf))
_gs = np.zeros((canopy.nlevmlcan,canopy.nleaf))
_Rd = np.zeros((canopy.nlevmlcan,canopy.nleaf))

for ileaf in range(canopy.nleaf):
    for ic in range(canopy.nbot, canopy.ntop+1):
        Q = 1e-6 * swleaf[ic,ileaf] * canopyrad.J_to_umol  # absorbed PPFD, mol PAR m-2 s-1
        An, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH,1.0)
        _An[ic,ileaf] = An
        _gs[ic,ileaf] = gs
        _Rd[ic,ileaf] = Rd

# %% [markdown]
# ### Plot results

# %%
fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(fracsun,np.arange(canopy.nlevmlcan))
axes[0].set_xlabel("Sunlit fraction\n"+r"(-)")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(dpai*fracsun,np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(dpai*(1-fracsun),np.arange(canopy.nlevmlcan),label="shaded")
axes[1].plot(dpai,np.arange(canopy.nlevmlcan),label="total",c='k')
axes[1].set_xlabel("LAI\n"+r"($\rm m^2 \; m^{-2}$)")
axes[1].legend()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(swleaf[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[0].plot(swleaf[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[0].set_xlabel("APAR per leaf\n"+r"($\rm W \; m^{-2} \; leaf^{-1}$)")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(swleaf[:,canopy.isun]*fracsun*dpai,np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(swleaf[:,canopy.isha]*(1-fracsun)*dpai,np.arange(canopy.nlevmlcan),label="shaded")
axes[1].set_xlabel("APAR per layer\n"+r"($\rm W \; m^{-2}$)")
axes[1].legend()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(1e6*_An[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[0].plot(1e6*_An[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[0].set_xlabel(r"$\rm A_{net}$ per leaf"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(_gs[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(_gs[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[1].set_xlabel(r"$\rm g_s$ per leaf"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[1].legend()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(dlai*fracsun*1e6*_An[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
# axes[0].plot(dlai*fracsun*1e6*(_An[:,canopy.isun]+_Rd[:,canopy.isun]),np.arange(canopy.nlevmlcan),label="sunlit",c="C0",linestyle=":")
axes[0].plot(dlai*(1-fracsun)*1e6*_An[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
# axes[0].plot(dlai*(1-fracsun)*1e6*(_An[:,canopy.isha]+_Rd[:,canopy.isun]),np.arange(canopy.nlevmlcan),label="shaded",c="C1",linestyle=":")
axes[0].set_xlabel(r"$\rm A_{net}$ per layer"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[0].set_ylabel("Canopy layer")
axes[0].set_xlim([0,4])

axes[1].plot(dlai*fracsun*_gs[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(dlai*(1-fracsun)*_gs[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[1].set_xlabel(r"$\rm g_s$ per layer"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[1].legend()
axes[1].set_xlim([0,0.07])

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(fracsun,np.arange(canopy.nlevmlcan))
axes[0].set_xlabel("Sunlit fraction\n"+r"(-)")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(dpai*fracsun,np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(dpai*(1-fracsun),np.arange(canopy.nlevmlcan),label="shaded")
axes[1].plot(dpai,np.arange(canopy.nlevmlcan),label="total",c='k')
axes[1].set_xlabel("LAI\n"+r"($\rm m^2 \; m^{-2}$)")
axes[1].legend()


plt.tight_layout()
plt.show()




fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(swleaf[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[0].plot(swleaf[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[0].set_xlabel("APAR per leaf\n"+r"($\rm W \; m^{-2} \; leaf^{-1}$)")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(swleaf[:,canopy.isun]*fracsun*dpai,np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(swleaf[:,canopy.isha]*(1-fracsun)*dpai,np.arange(canopy.nlevmlcan),label="shaded")
axes[1].set_xlabel("APAR per layer\n"+r"($\rm W \; m^{-2}$)")
axes[1].legend()

plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(1e6*_An[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[0].plot(1e6*_An[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[0].set_xlabel(r"$\rm A_{net}$ per leaf"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(_gs[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(_gs[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[1].set_xlabel(r"$\rm g_s$ per leaf"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[1].legend()

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(dlai*fracsun*1e6*_An[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
# axes[0].plot(dlai*fracsun*1e6*(_An[:,canopy.isun]+_Rd[:,canopy.isun]),np.arange(canopy.nlevmlcan),label="sunlit",c="C0",linestyle=":")
axes[0].plot(dlai*(1-fracsun)*1e6*_An[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
# axes[0].plot(dlai*(1-fracsun)*1e6*(_An[:,canopy.isha]+_Rd[:,canopy.isun]),np.arange(canopy.nlevmlcan),label="shaded",c="C1",linestyle=":")
axes[0].set_xlabel(r"$\rm A_{net}$ per layer"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(dlai*fracsun*_gs[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(dlai*(1-fracsun)*_gs[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[1].set_xlabel(r"$\rm g_s$ per layer"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[1].legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Summary of canopy totals


# %%
print("Canopy radiation:")
print("  Canopy sunlit APAR = %1.1f" % swvegsun, "W m-2")
print("  Canopy shaded APAR = %1.1f" % swvegsha, "W m-2")
print("  Canopy total APAR = %1.1f" % swveg, "W m-2")
print()
print("Photosynthesis:")
print("  Canopy sunlit Anet = %1.1f" % (np.sum(dlai*fracsun*1e6*_An[:,canopy.isun])),"umol m-2 s-1")
print("  Canopy shaded Anet = %1.1f" % (np.sum(dlai*(1-fracsun)*1e6*_An[:,canopy.isha])),"umol m-2 s-1")
print("  Canopy total Anet = %1.1f" % (np.sum(dlai*fracsun*1e6*_An[:,canopy.isun])+np.sum(dlai*(1-fracsun)*1e6*_An[:,canopy.isha])),"umol m-2 s-1")
print("  Canopy total GPP = %1.1f" % (np.sum(dlai*fracsun*1e6*(_An[:,canopy.isun]+_Rd[:,canopy.isun]))+np.sum(dlai*(1-fracsun)*1e6*(_An[:,canopy.isha]+_Rd[:,canopy.isha]))),"umol m-2 s-1")
print()
print("Stomatal conductance to water vapor:")
gs_multilayer = np.sum(dlai*fracsun*_gs[:,canopy.isun])+np.sum(dlai*(1-fracsun)*_gs[:,canopy.isha])
print("  Canopy sunlit gs = %1.2f" % (np.sum(dlai*fracsun*_gs[:,canopy.isun])),"mol m-2 s-1")
print("  Canopy shaded gs = %1.2f" % (np.sum(dlai*(1-fracsun)*_gs[:,canopy.isha])),"mol m-2 s-1")
print("  Sum of gs over canopy layers = %1.3f" % gs_multilayer,"mol m-2 s-1")
print("  Sum of rs over canopy layers = %1.3f" % (1/gs_multilayer),"m2 s mol-1")
conversion_factor = p/(Site.R_w_mol*(T+273.15))  ## this is the molar density of water vapor at the given temperature and pressure
print("  Sum of rs over canopy layers = %1.4f" % (conversion_factor/gs_multilayer),"s m-1")
print()


# %% [markdown]
# #### Compare big-leaf to multi-layer canopy approaches

# %%
fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(dlai*fracsun*1e6*_An[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[0].plot(dlai*(1-fracsun)*1e6*_An[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[0].scatter(Anet_bigleaf/canopy.nlevmlcan,canopy.nlevmlcan-1,c='k',s=25)
axes[0].set_xlabel(r"$\rm A$ per layer"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[0].set_ylabel("Canopy layer")


axes[1].plot(dlai*fracsun*_gs[:,canopy.isun],np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(dlai*(1-fracsun)*_gs[:,canopy.isha],np.arange(canopy.nlevmlcan),label="shaded")
axes[1].scatter(gs_bigleaf/canopy.nlevmlcan,canopy.nlevmlcan-1,c='k',s=25,label="big-leaf/nlayers")
axes[1].set_xlabel(r"$\rm g_s$ per layer"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[1].legend()

plt.tight_layout()


# %% [markdown]
# ### Optional: Account for vertical variation in leaf gas exchange parameters (e.g. Vcmax)

# %%
def calculate_leaf_variables(Q, T, Cs, O, RH, Leaf):
    return Leaf.calculate(Q, T, Cs, O, RH, 1.0)

_An_mlp = np.zeros((canopy.nlevmlcan,canopy.nleaf))
_gs_mlp = np.zeros((canopy.nlevmlcan,canopy.nleaf))
_Rd_mlp = np.zeros((canopy.nlevmlcan,canopy.nleaf))

## Here we define the leaf property scaling factor for each layer of the canopy
## Relative LAI from top-of-canopy down
relative_LAI = np.cumsum(dlai[::-1]) / LAI
## Ensure we store the leaf property attribute before it is modified. I assume this represents the top-of-canopy value. 
Vcmax_opt_toc = Leaf.Vcmax_opt
Vcmax_opt_ml_scalefactor = canopy.cast_scalefactor_to_layer_exp(0.5,LAI,relative_LAI)[::-1]

## Loop over canopy layers, set leaf parameter and calculate gas exchange
for ileaf in range(canopy.nleaf):
    for ic in range(canopy.nbot, canopy.ntop+1):
        Q = 1e-6 * swleaf[ic,ileaf] * canopyrad.J_to_umol  # absorbed PPFD, mol PAR m-2 s-1
        An, gs, Ci, Vc, Ve, Vs, Rd = calculate_leaf_variables(Q, T, Cs, O, RH, Leaf)
        Leaf.set_Vcmax_for_layer(Vcmax_opt_toc, Vcmax_opt_ml_scalefactor[ic])  # Adjust Vcmax_opt for the layer
        _An_mlp[ic,ileaf] = An
        _gs_mlp[ic,ileaf] = gs
        _Rd_mlp[ic,ileaf] = Rd

Leaf.set_Vcmax_for_layer(Vcmax_opt_toc, 1)  # Reset leaf Vcmax_opt to its top-of-canopy value


# %%
fig, axes = plt.subplots(1,2,figsize=(6,2.5))

ax = axes[0]
relative_LAI = np.cumsum(dlai[::-1]) / LAI

scalefactor = canopy.cast_scalefactor_to_layer_exp(0.5,LAI,relative_LAI)[::-1]
ax.plot(scalefactor,relative_LAI,label='k=0.5')

scalefactor = canopy.cast_scalefactor_to_layer_exp(0.7,LAI,relative_LAI)[::-1]
ax.plot(scalefactor,relative_LAI,label='k=0.7')

scalefactor = canopy.cast_scalefactor_to_layer_exp(0.9,LAI,relative_LAI)[::-1]
ax.plot(scalefactor,relative_LAI,label='k=0.9')
ax.legend(title="Extinction\ncoefficient",fontsize=9,handlelength=0.7,loc=3)

ax.hlines(y=1,xmin=0,xmax=1,color='0.25',lw=1)
ax.text(0.5, 1.0, 'top of canopy', horizontalalignment='center',verticalalignment='bottom')#, transform=ax.transAxes)
ax.set_xlim([0,1])
ax.set_ylim([0,1.1])
ax.set_xlabel("Scaling factor for leaf parameter")
ax.set_ylabel("Cumulative relative LAI")

ax = axes[1]
ax.plot(Vcmax_opt_ml_scalefactor,np.arange(canopy.nlevmlcan))
ax.set_xlabel("Scaling factor for leaf parameter")
ax.set_ylabel("Canopy layer")
ax.set_xlim([0,1])
ax.set_ylim([0,1.1*(canopy.nlevmlcan-1)])
ax.hlines(y=canopy.nlevmlcan-1,xmin=0,xmax=1,color='0.25',lw=1)
ax.text(0.5, canopy.nlevmlcan-1, 'top of canopy', horizontalalignment='center',verticalalignment='bottom')#, transform=ax.transAxes)

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(2,2,figsize=(5.33,5))

ax = axes[0,0]
ax.plot(1e6*_An[:,0],np.arange(canopy.nlevmlcan),label=r"Uniform $\rm V_{cmax,opt}$")
ax.plot(1e6*_An_mlp[:,0],np.arange(canopy.nlevmlcan),label=r"Exp $\rm V_{cmax,opt}$",c='crimson')
ax.set_xlabel(r"$\rm A_{net}$ per leaf"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
ax.set_ylabel("Canopy layer")
ax.legend(handlelength=0.7,fontsize=9)

ax = axes[0,1]
ax.plot(_gs[:,0],np.arange(canopy.nlevmlcan),label="Uniform Vcmax")
ax.plot(_gs_mlp[:,0],np.arange(canopy.nlevmlcan),label="Exp Vcmax",c='crimson')
ax.set_xlabel(r"$\rm g_{sw}$ per leaf"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)")
ax.set_ylabel("Canopy layer")

ax = axes[1,0]
ax.plot(1e6*_An[:,0]*dlai*fracsun,np.arange(canopy.nlevmlcan),label="Uniform Vcmax")
ax.plot(1e6*_An_mlp[:,0]*dlai*fracsun,np.arange(canopy.nlevmlcan),label="Exp Vcmax",c='crimson')
ax.set_xlabel(r"$\rm A_{net}$ per layer"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
ax.set_ylabel("Canopy layer")

ax = axes[1,1]
ax.plot(_gs[:,0]*dlai*fracsun,np.arange(canopy.nlevmlcan),label="Uniform Vcmax")
ax.plot(_gs_mlp[:,0]*dlai*fracsun,np.arange(canopy.nlevmlcan),label="Exp Vcmax",c='crimson')
ax.set_xlabel(r"$\rm g_{sw}$ per layer"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)")
ax.set_ylabel("Canopy layer")

plt.tight_layout()
plt.show()

# %%
