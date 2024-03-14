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
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.leafgasexchange import LeafGasExchangeModule
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
Leaf = LeafGasExchangeModule()

# %% [markdown]
# ## Big-Leaf Approach
#
# This approach treats the canopy as a single "big leaf". 

# %%
from daesim.canopyradiation_bigleaf import CanopyRadiation as CanopyRadiationBigLeaf

# %%
if clumping_factor == 0.5:
    LAD_type = "spherical"

canopysolar_bigleaf = CanopyRadiationBigLeaf(LAD_type=LAD_type,rho=leafreflectance)

# %%
P = canopysolar_bigleaf.canopy_gap_fraction(np.deg2rad(sza),LAI,LAD_type=LAD_type)
print("Gap fraction, P(theta) = %1.3f" % P)
fAPAR = canopysolar_bigleaf.canopy_fraction_absorbed_irradiance(np.deg2rad(sza),LAI,LAD_type=LAD_type)
print("Fraction of absorbed shortwave irradiance = %1.3f" % fAPAR)

swdown = swskyb + swskyd
print("Absorbed shortwave irradiance = %1.3f" % (fAPAR*swdown),"W m-2")

# %%
Q = 1e-6 * (swdown*fAPAR) * CanopyRadiation().J_to_umol  # absorbed PPFD, umol PAR m-2 s-1
A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH)

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
canopy = CanopyLayers(nlevmlcan_enforce=8)#beta_lai_a=1,beta_lai_b=1,beta_sai_a=1,beta_sai_b=1)
canopy.set_nlayers(LAI,canopy_height)
canopy.set_index()

## Instance of CanopyRadiation class
canopysolar = CanopyRadiation(rhol=leafreflectance)

# %% [markdown]
# ### Run radiative transfer calculations

# %%
(fracsun, kb, omega, avmu, betab, betad, tbi) = canopysolar.calculateRTProperties(LAI,SAI,clumping_factor,canopy_height,sza,Canopy=canopy)

# %%
dlai = canopy.cast_parameter_over_layers_betacdf(LAI,canopy.beta_lai_a,canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)
dsai = canopy.cast_parameter_over_layers_betacdf(SAI,canopy.beta_sai_a,canopy.beta_sai_b)  # Canopy layer stem area index (m2/m2)
dpai = dlai+dsai  # Canopy layer plant area index (m2/m2)
clump_fac = canopy.cast_parameter_over_layers_uniform(clumping_factor)

## Note: swleaf is the absorption per leaf area index (W/m2 per leaf)
swleaf = canopysolar.calculateTwoStream(swskyb,swskyd,dpai,fracsun,kb,clump_fac,omega,avmu,betab,betad,tbi,albsoib,albsoid,Canopy=canopy)

# %%
swveg = 0
swvegsun = 0
swvegsha = 0
for ic in range(canopy.nbot, canopy.ntop + 1):
    sun = swleaf[ic,canopy.isun] * fracsun[ic] * dpai[ic]
    sha = swleaf[ic,canopy.isha] * (1.0 - fracsun[ic]) * dpai[ic]
    swveg += (sun + sha)
    swvegsun += sun
    swvegsha += sha

print("Absorbed shortwave radiation = %1.3f" % swveg)
print("Absorbed shortwave radiation by sunlit leaves = %1.3f" % swvegsun)
print("Absorbed shortwave radiation by sunlit leaves = %1.3f" % swvegsha)


# %% [markdown]
# ### Run gas exchange calculations given absorbed radiation per layer from radiative transfer calculations

# %%
_An = np.zeros((canopy.nlevmlcan,canopy.nleaf))
_gs = np.zeros((canopy.nlevmlcan,canopy.nleaf))
_Rd = np.zeros((canopy.nlevmlcan,canopy.nleaf))

for ileaf in range(canopy.nleaf):
    for ic in range(canopy.nbot, canopy.ntop+1):
        Q = 1e-6 * swleaf[ic,ileaf] * canopysolar.J_to_umol  # absorbed PPFD, mol PAR m-2 s-1
        An, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH)
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
conversion_factor = p/(Site.R_w_mol*T)  ## this is the molar density of water vapor at the given temperature and pressure
print("  Sum of rs over canopy layers = %1.4f" % (conversion_factor/gs_multilayer),"s m-1")
print()


# %%

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

# %%
