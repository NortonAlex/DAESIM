# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
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
from attrs import define, field
from typing import Tuple, Callable
from functools import partial
from scipy.optimize import bisect

# %%
from daesim.climate import ClimateModule
from daesim.leafgasexchange import LeafGasExchangeModule
from daesim.leafgasexchange2 import LeafGasExchangeModule2
from daesim.canopygasexchange import CanopyGasExchange
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.boundarylayer import BoundaryLayerModule
from daesim.plantcarbonwater import PlantModel as PlantCH2O
from daesim.soillayers import SoilLayers

# %% [markdown]
# ### Create instances of each module

# %%
site = ClimateModule()
leaf = LeafGasExchangeModule2(g0=0.0)
canopy = CanopyLayers(nlevmlcan=6)
soillayers = SoilLayers(nlevmlsoil=10,z_max=2.0,Psi_e=-0.1)
canopyrad = CanopyRadiation(Canopy=canopy)
canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopyRad=canopyrad)
boundarylayer = BoundaryLayerModule(Site=site,k_wl=0.006)

## Module with upstream module dependencies
plant = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,BoundaryLayer=boundarylayer,maxLAI=1.5,ksr_coeff=100,sf=1.5)

# %% [markdown]
# ### Example of soil-root profile and rooting depth functions

# %%
z_soil, d_soil = soillayers.discretise_layers()

print(z_soil, d_soil)

d_r = 2.0  # Rooting depth
fc_r_z = plant.calculate_root_distribution(d_r, d_soil)
## Calculate actual root fraction per layer (by difference, no loops needed)
f_r_z = np.diff(np.insert(fc_r_z, 0, 0, axis=0), axis=0)  # Fractional root density per layer

d_soil_midpoints = []
for i in range(soillayers.nlevmlsoil):
    d_soil_midpoints.append(d_soil[i] - 0.5*z_soil[i])

## figure
fig, axes = plt.subplots(1,2,figsize=(8,3))

xlimmin, xlimmax = 0.5, 1.0
axes[0].plot(fc_r_z, -np.array(d_soil_midpoints), marker='s')
axes[0].hlines(y=-np.array(d_soil), xmin=xlimmin, xmax=xlimmax, color='0.5', alpha=0.5)
axes[0].set_ylim([np.min(-np.array(d_soil)), 0])
axes[0].set_xlim([xlimmin, xlimmax])
axes[0].set_xlabel("Cumulative fraction of roots per layer")
axes[0].set_ylabel("Soil depth (m)")

xlimmin, xlimmax = 0, 1.0
axes[1].plot(f_r_z, -np.array(d_soil_midpoints), marker='s')
axes[1].hlines(y=-np.array(d_soil), xmin=xlimmin, xmax=xlimmax, color='0.5', alpha=0.5)
axes[1].set_ylim([np.min(-np.array(d_soil)), 0])
axes[1].set_xlim([xlimmin, xlimmax])
axes[1].set_xlabel("Fraction of roots per layer")
axes[1].set_ylabel("Soil depth (m)")

plt.tight_layout()

# %%
fig, axes = plt.subplots(1,2,figsize=(8,3))

## Effect of potential rooting depth
n = 100
_W_R = np.linspace(5,400,n)
_d_r_dynamic0 = np.zeros(n)
_d_r_dynamic1 = np.zeros(n)
_d_r_dynamic2 = np.zeros(n)
_d_r_dynamic3 = np.zeros(n)

plant0 = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,maxLAI=1.5,ksr_coeff=100,sf=1.5,SRD=0.010)
 
for i,x in enumerate(_W_R):
    _d_r_dynamic0[i] = plant0.calculate_root_depth(x, 0.5)
    _d_r_dynamic1[i] = plant0.calculate_root_depth(x, 1.0)
    _d_r_dynamic2[i] = plant0.calculate_root_depth(x, 1.5)
    _d_r_dynamic3[i] = plant0.calculate_root_depth(x, 2.0)

axes[0].plot(_W_R, _d_r_dynamic0, label=r"$\rm d_{r,pot}=0.5$")
axes[0].plot(_W_R, _d_r_dynamic1, label=r"$\rm d_{r,pot}=1.0$")
axes[0].plot(_W_R, _d_r_dynamic2, label=r"$\rm d_{r,pot}=1.5$")
axes[0].plot(_W_R, _d_r_dynamic3, label=r"$\rm d_{r,pot}=2.0$")
axes[0].legend(handlelength=0.75)
axes[0].set_title("Potential Root Depth Sensitivity")


## Effect of specific root depth
n = 100
_W_R = np.linspace(5,400,n)
_d_r_dynamic0 = np.zeros(n)
_d_r_dynamic1 = np.zeros(n)
_d_r_dynamic2 = np.zeros(n)
_d_r_dynamic3 = np.zeros(n)

d_rpot = 2.0

plant0 = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,maxLAI=1.5,ksr_coeff=100,sf=1.5,SRD=0.005)
plant1 = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,maxLAI=1.5,ksr_coeff=100,sf=1.5,SRD=0.010)
plant2 = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,maxLAI=1.5,ksr_coeff=100,sf=1.5,SRD=0.020)
plant3 = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,maxLAI=1.5,ksr_coeff=100,sf=1.5,SRD=0.040)

for i,x in enumerate(_W_R):
    _d_r_dynamic0[i] = plant0.calculate_root_depth(x, d_rpot)
    _d_r_dynamic1[i] = plant1.calculate_root_depth(x, d_rpot)
    _d_r_dynamic2[i] = plant2.calculate_root_depth(x, d_rpot)
    _d_r_dynamic3[i] = plant3.calculate_root_depth(x, d_rpot)

axes[1].plot(_W_R, _d_r_dynamic0, label="SRD=%1.3f" % plant0.SRD)
axes[1].plot(_W_R, _d_r_dynamic1, label="SRD=%1.3f" % plant1.SRD)
axes[1].plot(_W_R, _d_r_dynamic2, label="SRD=%1.3f" % plant2.SRD)
axes[1].plot(_W_R, _d_r_dynamic3, label="SRD=%1.3f" % plant3.SRD)
axes[1].legend(handlelength=0.75)
axes[1].set_title("Specific Root Depth Sensitivity")

# %%

# %%

# %% [markdown]
# ## Considering the multi-layer soil when determining supply-side transpiration rate
#
# The DAESIM2 model assumes that plant water uptake through the roots is balanced by the plant water loss from canopy transpiration. The plant water uptake through the roots is considered the water supply and is calculated as follows:
#
# $E = k_{tot} (\Psi_s - \Psi_l)$
#
# Where $E$ is the transpiration rate (leaf-area specific; mol m-2 s-1 MPa-1), $k_{tot}$ is the total leaf-area specific soil-to-leaf hydraulic conductance (mol m-2 s-1 MPa-1), $\Psi_s$ is the soil water potential (MPa) and $\Psi_l$ is the leaf water potential (MPa). Note: the variable $k_{tot}$ is determined by the soil water potential, soil properties, root density and root properties. The function above works fine when assuming a single soil layer with a single $\Psi_s$ value and therefore a single $k_{tot}$ value. However, it is more complicated when considering a multi-layer soil where $\Psi_s$ can vary, and subsequently $k_{tot}$ can vary depending on what layer or layers we consider in the soil-to-root hydraulic conductance. 
#
# For a given soil layer, we can consider the supply-based, layer-specific ($z$) potential transpiration rate as follows:
#
# $E(z) = k_{tot}(z) (\Psi_s(z) - \Psi_l)$
#
# Note: we only consider a bulk canopy average leaf water potential i.e. $\Psi_l$ is not discretised by canopy layer. The question is: How should $k_{tot}(z)$ and $\Psi_s(z)$ be determined to give a single value for the supply-based transpiration rate ($E$). Should we consider weighting each layer somehow? How would this occur? 
#
# First, let's show how $k_{tot}$ is determined:
#
# $k_{tot} = \frac{(k_{srl} k_{rl})}{(k_{rl} + k_{srl})}$
#
# This assumes a one-dimensional pathway (in series) and Ohm's law for the hydraulic conductances i.e. the relationship $1/k_tot = 1/k_srl + 1/k_rl$. In DAESIM2, $k_{rl}$ is assumed to be a plant-type specific parameter which is constant in time. The variable $k_{srl}$ is the soil-to-root hydraulic conductance (leaf-area specific) and is determined by:
#
# $k_{srl} = \frac{K_{sr}}{LAI}$
#
# and:
#
# $K_{sr} = K_s \frac{f_r W_R}{d_{soil} k_{sr,coeff}} = K_s \frac{L_v}{k_{sr,coeff}}$
#
# Noting that $f_r \times W_R$ is the root biomass in the given soil layer and $(f_r W_R)/d_{soil}$ is the root biomass density in the given soil layer, $L_v$. Finally, $K_s$ is calculated as follows:
#
# $K_s = K_{sat} \left( \frac{\Psi_e}{Psi_s} \right)^{2+3/b_{soil}}$
#
# Thus, $K_s$ is dependent upon the soil water potential. The above equations account for variations over the soil profile in soil water potential and the root density, which determine $k_{srl}$. To determine $k_{tot}$ and then $E$, we need to determine how to weight or average $k_{srl}$ and $\Psi_s$ in each layer. 
#

# %% [markdown]
# ### Input variables for canopy layers, canopy radiation and canopy gas exchange

# %%
LAI = 1.5    ## leaf area index (m2 m-2)
SAI = 0.1    ## stem area index (m2 m-2)
CI = 0.8     ## foliage clumping index (-)
hc = 1.0     ## canopy height (m)
d_r = 2.0    ## rooting depth (m)
sza = 30.0       ## solar zenith angle (degrees)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)

## input variables for leaf gas exchange model
leafTempC = 20.0
airTempC = 20.0
airRH = 70.0
airP = 101325    ## air pressure, Pa
soilTheta = np.linspace(0.24, 0.38, soillayers.nlevmlsoil)  #np.array([0.26, 0.30, 0.34, 0.38])  # np.array([[0.26],[0.30],[0.34],[0.38]])  ## volumetric soil moisture (m3 m-3), now defined on a per layer basis (first dimension of array represent the layers)
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)
airUhc = 2.0   ## wind speed at top-of-canopy (m s-1)

## model state variables
W_R = 40
W_L = 70

# %%

# %% [markdown]
# ### Example run of plant methods

# %%
LAI = plant.calculate_LAI(W_L)

## Calculate wind speed profile within canopy, given canopy properties
dlai = plant.CanopyGasExchange.Canopy.cast_parameter_over_layers_betacdf(LAI, plant.CanopyGasExchange.Canopy.beta_lai_a, plant.CanopyGasExchange.Canopy.beta_lai_b)   # Canopy layer leaf area index (m2/m2)
dsai = plant.CanopyGasExchange.Canopy.cast_parameter_over_layers_betacdf(SAI, plant.CanopyGasExchange.Canopy.beta_sai_a, plant.CanopyGasExchange.Canopy.beta_sai_b)   # Canopy layer stem area index (m2/m2)
dpai = dlai+dsai    # Canopy layer plant area index (m2/m2)
ntop, nbot = plant.CanopyGasExchange.Canopy.index_canopy()
airUz = plant.BoundaryLayer.calculate_wind_profile_exp(airUhc,dpai,ntop)   # Wind speed at mid-point of each canopy layer

print("LAI =",LAI)
print("airUz =",airUz)
print()
GPP, E, Rd, E_l = plant.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, airUz, 1.0, LAI, SAI, CI, hc, sza, swskyb, swskyd)

print("GPP =", GPP)
print("E =", E)
print("Rd =", Rd)
print("E_l =", E_l)

# %%
LAI = plant.calculate_LAI(W_L)
print("LAI =",LAI)
print()
GPP, E, Rd, E_l = plant.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, airUhc, 0.08302861252972676, LAI, SAI, CI, hc, sza, swskyb, swskyd)

print("GPP =", GPP)
print("E =", E)
print("Rd =", Rd)
print("E_l =", E_l)

# %%
GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)

print("GPP =", GPP_0)
print("E =", E_0)
print("f_Psi_l =",fPsil_0)
print("Psi_l =",Psil_0)
print("Psi_r =",Psir_0)
print("Psi_s =",Psis_0)



# %% [markdown]
# ## Model Sensitivity Tests

# %% [markdown]
# ### Forcing and State Variable Sensitivity Tests

# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = np.array([20,50,100,200])
_W_L = np.linspace(5,400,n)

for ix, xWR in enumerate(_W_R):
    W_R = xWR
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xWL in enumerate(_W_L):
        W_L = xWL
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_W_L,GPP_0_,label=r"$\rm W_R=%d$" % W_R)
    axes[1].plot(_W_L,E_0_,label=r"$\rm W_R=%d$" % W_R)
    axes[2].plot(_W_L,fPsil_0_,label=r"$\rm W_R=%d$" % W_R)
    axes[3].plot(_W_L,Psil_0_,label=r"$\rm W_R=%d$" % W_R)
    axes[4].plot(_W_L,Psir_0_,label=r"$\rm W_R=%d$" % W_R)
    axes[5].plot(_W_L,Psis_0_,label=r"$\rm W_R=%d$" % W_R)
    axes[6].plot(_W_L,K_s_0_,label=r"$\rm W_R=%d$" % W_R)
    axes[7].plot(_W_L,k_srl_0_,label=r"$\rm W_R=%d$" % W_R)


axes[0].set_xlabel("Leaf biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Leaf biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Leaf biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Leaf biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Leaf biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Leaf biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Leaf biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Leaf biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[7].set_ylabel(r"$k_{srl}$")

axes[0].set_ylim([0,40])
# axes[1].set_ylim([0.00295,0.003])
# axes[3].set_ylim([-2,0])
# axes[4].set_ylim([-2,0])
# axes[5].set_ylim([-2,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_sensitivity_test_WL_plantsoilhydraulics_by_WR_3.png",dpi=300,bbox_inches='tight')

# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = np.linspace(20,400,n)
_W_L = np.array([20,50,100,200])

for ix, xWL in enumerate(_W_L):
    W_L = xWL
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xWR in enumerate(_W_R):
        W_R = xWR
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_W_R,GPP_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[1].plot(_W_R,E_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[2].plot(_W_R,fPsil_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[3].plot(_W_R,Psil_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[4].plot(_W_R,Psir_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[5].plot(_W_R,Psis_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[6].plot(_W_R,K_s_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[7].plot(_W_R,k_srl_0_,label=r"$\rm W_L=%d$" % W_L)

axes[0].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[7].set_ylabel(r"$k_{srl}$")

axes[0].set_ylim([0,40])
# axes[1].set_ylim([0.00295,0.003])
# axes[3].set_ylim([-2,0])
# axes[4].set_ylim([-2,0])
# axes[5].set_ylim([-2,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_sensitivity_test_WR_plantsoilhydraulics_by_WL_2.png",dpi=300,bbox_inches='tight')

# %% [markdown]
# ### Soil Moisture Sensitivity - uniform soil moisture profile

# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = np.array([20,50,100,200])
_W_L = _W_R*1.
_soilTheta = np.linspace(0.20,plant.SoilLayers.soilThetaMax,n)

for ix, xWL in enumerate(_W_L):
    W_L = xWL
    W_R = _W_R[ix]
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xsoilTheta in enumerate(_soilTheta):
        soilTheta = xsoilTheta*np.ones(plant.SoilLayers.nlevmlsoil)
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_soilTheta,GPP_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[1].plot(_soilTheta,E_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[2].plot(_soilTheta,fPsil_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[3].plot(_soilTheta,Psil_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[4].plot(_soilTheta,Psir_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[5].plot(_soilTheta,Psis_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[6].plot(_soilTheta,K_s_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[7].plot(_soilTheta,k_srl_0_,label=r"$\rm W_L=%d$" % W_L)

axes[0].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[7].set_ylabel(r"$k_{srl}$")

# axes[0].set_ylim([0,15])
# axes[1].set_ylim([0,0.00085])
axes[3].set_ylim([-4,0])
axes[4].set_ylim([-4,0])
axes[5].set_ylim([-4,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_sensitivity_test_soilmoisture_plantsoilhydraulics_by_WL_2.png",dpi=300,bbox_inches='tight')


# %% [markdown]
# ### Soil Moisture Sensitivity - linear decline in soil moisture from surface to bottom layer saturation

# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = np.array([20,50,100,200])
_W_L = _W_R*1.
_soilTheta0 = np.linspace(0.20,plant.SoilLayers.soilThetaMax,n)

for ix, xWL in enumerate(_W_L):
    W_L = xWL
    W_R = _W_R[ix]
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xsoilTheta in enumerate(_soilTheta0):
        soilTheta = np.linspace(xsoilTheta, plant.SoilLayers.soilThetaMax, plant.SoilLayers.nlevmlsoil)
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_soilTheta,GPP_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[1].plot(_soilTheta,E_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[2].plot(_soilTheta,fPsil_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[3].plot(_soilTheta,Psil_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[4].plot(_soilTheta,Psir_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[5].plot(_soilTheta,Psis_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[6].plot(_soilTheta,K_s_0_,label=r"$\rm W_L=%d$" % W_L)
    axes[7].plot(_soilTheta,k_srl_0_,label=r"$\rm W_L=%d$" % W_L)

axes[0].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[7].set_ylabel(r"$k_{srl}$")

# axes[0].set_ylim([0,15])
# axes[1].set_ylim([0,0.00085])
axes[3].set_ylim([-4,0])
axes[4].set_ylim([-4,0])
axes[5].set_ylim([-4,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_sensitivity_test_soilmoisture_plantsoilhydraulics_by_WL_2.png",dpi=300,bbox_inches='tight')

# %% [markdown]
# ### - Parameter sensitivity tests

# %%
n = 11

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = np.linspace(20,400,n)
_W_L = 80*np.ones(4)
_ksr_coeff = np.array([100,1000,5000,15000])

soilTheta = np.linspace(0.24, 0.38, soillayers.nlevmlsoil)

for ix, xWL in enumerate(_W_L):
    W_L = xWL
    ksr_coeff = _ksr_coeff[ix]
    plant.ksr_coeff = ksr_coeff
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xWR in enumerate(_W_R):
        W_R = xWR
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_W_R,GPP_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[1].plot(_W_R,E_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[2].plot(_W_R,fPsil_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[3].plot(_W_R,Psil_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[4].plot(_W_R,Psir_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[5].plot(_W_R,Psis_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[6].plot(_W_R,K_s_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[7].plot(_W_R,k_srl_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)

axes[0].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Root biomass\n"+r"($\rm g \; d.wt \; m^{-2}$)")
axes[7].set_ylabel(r"$k_{srl}$")

# axes[3].set_ylim([-2,0])
# axes[4].set_ylim([-2,0])
# axes[5].set_ylim([-2,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_psensitivity_test_WR_plantsoilhydraulics_by_ksrcoeff.png",dpi=300,bbox_inches='tight')


# %%

# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = 80 * np.ones(4)
_W_L = _W_R*1.
_ksr_coeff = np.array([100,1000,5000,15000])

_soilTheta = np.linspace(0.20,plant.SoilLayers.soilThetaMax,n)

for ix, xWL in enumerate(_W_L):
    W_L = xWL
    W_R = _W_R[ix]
    ksr_coeff = _ksr_coeff[ix]
    plant.ksr_coeff = ksr_coeff
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xsoilTheta in enumerate(_soilTheta):
        # soilTheta = xsoilTheta
        soilTheta = np.linspace(xsoilTheta, plant.SoilLayers.soilThetaMax, plant.SoilLayers.nlevmlsoil)
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_soilTheta,GPP_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[1].plot(_soilTheta,E_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[2].plot(_soilTheta,fPsil_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[3].plot(_soilTheta,Psil_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[4].plot(_soilTheta,Psir_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[5].plot(_soilTheta,Psis_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[6].plot(_soilTheta,K_s_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)
    axes[7].plot(_soilTheta,k_srl_0_,label=r"$\rm k_{sr,coeff}=%d$" % ksr_coeff)

axes[0].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[7].set_ylabel(r"$k_{srl}$")

# axes[0].set_ylim([0,15])
# axes[1].set_ylim([0,0.00085])
axes[3].set_ylim([-4,0])
axes[4].set_ylim([-4,0])
axes[5].set_ylim([-4,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_psensitivity_test_soilmoisture_plantsoilhydraulics_by_ksrcoeff.png",dpi=300,bbox_inches='tight')


# %%

# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = 80 * np.ones(4)
_W_L = _W_R*1.
_Psi_f = np.array([-3.0, -2.0, -1.0, -0.5])

_soilTheta = np.linspace(0.20,plant.SoilLayers.soilThetaMax,n)

for ix, xWL in enumerate(_W_L):
    W_L = xWL
    W_R = _W_R[ix]
    Psi_f = _Psi_f[ix]
    plant.Psi_f = Psi_f
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xsoilTheta in enumerate(_soilTheta):
        soilTheta = xsoilTheta
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_soilTheta,GPP_0_,label=r"$\rm \Psi_f=%1.1f$" % Psi_f)
    axes[1].plot(_soilTheta,E_0_,label=r"$\rm \Psi_f=%1.1f$" % Psi_f)
    axes[2].plot(_soilTheta,fPsil_0_,label=r"$\rm \Psi_f=%1.1f$" % Psi_f)
    axes[3].plot(_soilTheta,Psil_0_,label=r"$\rm \Psi_f=%1.1f$" % Psi_f)
    axes[4].plot(_soilTheta,Psir_0_,label=r"$\rm \Psi_f=%1.1f$" % Psi_f)
    axes[5].plot(_soilTheta,Psis_0_,label=r"$\rm \Psi_f=%1.1f$" % Psi_f)
    axes[6].plot(_soilTheta,K_s_0_,label=r"$\rm \Psi_f=%1.1f$" % Psi_f)
    axes[7].plot(_soilTheta,k_srl_0_,label=r"$\rm \Psi_f=%1.1f$" % Psi_f)

axes[0].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[7].set_ylabel(r"$k_{srl}$")

# axes[0].set_ylim([0,15])
# axes[1].set_ylim([0,0.00085])
axes[3].set_ylim([-4,0])
axes[4].set_ylim([-4,0])
axes[5].set_ylim([-4,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_psensitivity_test_soilmoisture_plantsoilhydraulics_by_Psif.png",dpi=300,bbox_inches='tight')


# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = 80 * np.ones(4)
_W_L = _W_R*1.
_sf = np.array([1,2,4,8])

_soilTheta = np.linspace(0.20,plant.SoilLayers.soilThetaMax,n)

for ix, xWL in enumerate(_W_L):
    W_L = xWL
    W_R = _W_R[ix]
    sf = _sf[ix]
    plant.sf = sf
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xsoilTheta in enumerate(_soilTheta):
        soilTheta = xsoilTheta
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_soilTheta,GPP_0_,label=r"$\rm s_f=%d$" % sf)
    axes[1].plot(_soilTheta,E_0_,label=r"$\rm s_f=%d$" % sf)
    axes[2].plot(_soilTheta,fPsil_0_,label=r"$\rm s_f=%d$" % sf)
    axes[3].plot(_soilTheta,Psil_0_,label=r"$\rm s_f=%d$" % sf)
    axes[4].plot(_soilTheta,Psir_0_,label=r"$\rm s_f=%d$" % sf)
    axes[5].plot(_soilTheta,Psis_0_,label=r"$\rm s_f=%d$" % sf)
    axes[6].plot(_soilTheta,K_s_0_,label=r"$\rm s_f=%d$" % sf)
    axes[7].plot(_soilTheta,k_srl_0_,label=r"$\rm s_f=%d$" % sf)

axes[0].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[7].set_ylabel(r"$k_{srl}$")

# axes[0].set_ylim([0,15])
# axes[1].set_ylim([0,0.00085])
axes[3].set_ylim([-4,0])
axes[4].set_ylim([-4,0])
axes[5].set_ylim([-4,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_psensitivity_test_soilmoisture_plantsoilhydraulics_by_sf.png",dpi=300,bbox_inches='tight')


# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = 80 
_W_L = _W_R*2.
_soilThetaMax = np.array([0.2,0.3,0.4,0.5])

plant.sf = 1.0
plant.Psi_f = -5.0
plant.ksr_coeff = 500

for ix, xparam in enumerate(_soilThetaMax):
    W_L = _W_L
    W_R = _W_R
    plant.SoilLayers.soilThetaMax = xparam
    _soilTheta = np.linspace(0.02,plant.SoilLayers.soilThetaMax,n)
    soilTheta_zbot = plant.SoilLayers.soilThetaMax   ## Bottom soil layer soil moisture is at saturation
    _soilTheta_z0 = np.linspace(0.02,soilTheta_zbot,n)  ## Surface soil moisture is modified
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xsoilTheta in enumerate(_soilTheta_z0):
        # soilTheta = xsoilTheta
        soilTheta_1d = np.linspace(xsoilTheta, soilTheta_zbot, plant.SoilLayers.nlevmlsoil)
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta_1d,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_soilTheta,GPP_0_,label=r"$\rm \theta_{max}=%1.1f$" % plant.SoilLayers.soilThetaMax)
    axes[1].plot(_soilTheta,E_0_,label=r"$\rm \theta_{max}=%1.1f$" % plant.SoilLayers.soilThetaMax)
    axes[2].plot(_soilTheta,fPsil_0_,label=r"$\rm \theta_{max}=%1.1f$" % plant.SoilLayers.soilThetaMax)
    axes[3].plot(_soilTheta,Psil_0_,label=r"$\rm \theta_{max}=%1.1f$" % plant.SoilLayers.soilThetaMax)
    axes[4].plot(_soilTheta,Psir_0_,label=r"$\rm \theta_{max}=%1.1f$" % plant.SoilLayers.soilThetaMax)
    axes[5].plot(_soilTheta,Psis_0_,label=r"$\rm \theta_{max}=%1.1f$" % plant.SoilLayers.soilThetaMax)
    axes[6].plot(_soilTheta,K_s_0_,label=r"$\rm \theta_{max}=%1.1f$" % plant.SoilLayers.soilThetaMax)
    axes[7].plot(_soilTheta,k_srl_0_,label=r"$\rm \theta_{max}=%1.1f$" % plant.SoilLayers.soilThetaMax)

axes[0].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[7].set_ylabel(r"$k_{srl}$")

# axes[0].set_ylim([0,15])
# axes[1].set_ylim([0,0.00085])
axes[3].set_ylim([-4,0])
axes[4].set_ylim([-4,0])
axes[5].set_ylim([-4,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_psensitivity_test_soilmoisture_plantsoilhydraulics_by_soilThetaMax.png",dpi=300,bbox_inches='tight')

# %%

# %% [markdown]
# ### - Rooting depth effect in the multi-layer soil model

# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = 200 * np.ones(4)
_W_L = _W_R*1.
_rooting_depth = np.array([0.1, 0.6, 1.2, 2.0])

soilTheta_zbot = 0.40   ## Bottom soil layer soil moisture is kept constant
_soilTheta_z0 = np.linspace(0.20,soilTheta_zbot,n)  ## Surface soil moisture is modified

plant.ksr_coeff = 5000

for ix, xWL in enumerate(_W_L):
    W_L = xWL
    W_R = _W_R[ix]
    rooting_depth = _rooting_depth[ix]
    _d_r = rooting_depth
    GPP_0_ = np.zeros(n)
    Rml_0_ = np.zeros(n)
    Rmr_0_ = np.zeros(n)
    E_0_ = np.zeros(n)
    fPsil_0_ = np.zeros(n)
    Psil_0_ = np.zeros(n)
    Psir_0_ = np.zeros(n)
    Psis_0_ = np.zeros(n)
    K_s_0_ = np.zeros(n)
    K_sr_0_ = np.zeros(n)
    k_srl_0_ = np.zeros(n)
    GPP_0_nofPsil_ = np.zeros(n)
    
    for ix,xsoilTheta in enumerate(_soilTheta_z0):
        # soilTheta = xsoilTheta
        soilTheta_1d = np.linspace(xsoilTheta, soilTheta_zbot, plant.SoilLayers.nlevmlsoil)
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta_1d,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,_d_r)
        GPP_0_[ix] = GPP_0
        Rml_0_[ix] = Rml_0
        Rmr_0_[ix] = Rmr_0
        E_0_[ix] = E_0
        fPsil_0_[ix] = fPsil_0
        Psil_0_[ix] = Psil_0
        Psir_0_[ix] = Psir_0
        Psis_0_[ix] = Psis_0
        K_s_0_[ix] = K_s_0
        K_sr_0_[ix] = K_sr_0
        k_srl_0_[ix] = k_srl_0

    
    axes[0].plot(_soilTheta_z0,GPP_0_,label=r"$\rm d_r=%1.1f$" % rooting_depth)
    axes[1].plot(_soilTheta_z0,E_0_,label=r"$\rm d_r=%1.1f$" % rooting_depth)
    axes[2].plot(_soilTheta_z0,fPsil_0_,label=r"$\rm d_r=%1.1f$" % rooting_depth)
    axes[3].plot(_soilTheta_z0,Psil_0_,label=r"$\rm d_r=%1.1f$" % rooting_depth)
    axes[4].plot(_soilTheta_z0,Psir_0_,label=r"$\rm d_r=%1.1f$" % rooting_depth)
    axes[5].plot(_soilTheta_z0,Psis_0_,label=r"$\rm d_r=%1.1f$" % rooting_depth)
    axes[6].plot(_soilTheta_z0,K_s_0_,label=r"$\rm d_r=%1.1f$" % rooting_depth)
    axes[7].plot(_soilTheta_z0,k_srl_0_,label=r"$\rm d_r=%1.1f$" % rooting_depth)

axes[0].set_xlabel("Top layer soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[0].set_ylabel(r"GPP ($\rm g C \; m^{-2} \; d^{-1}$)")
axes[0].legend(handlelength=0.7,fontsize=9)

axes[1].set_xlabel("Top layer soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[1].set_ylabel(r"E ($\rm mol \; H_2O \; m^{-2} \; s^{-1}$)")

axes[2].set_xlabel("Top layer soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[2].set_ylabel(r"$\rm f_{\Psi_l}$ (-)")
axes[2].set_ylim([0,1])

axes[3].set_xlabel("Top layer soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[3].set_ylabel(r"$\Psi_L$ (MPa)")

axes[4].set_xlabel("Top layer soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[4].set_ylabel(r"$\Psi_R$ (MPa)")

axes[5].set_xlabel("Top layer soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[5].set_ylabel(r"$\Psi_S$ (MPa)")

axes[6].set_xlabel("Top layer soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[6].set_ylabel(r"$K_s$")

axes[7].set_xlabel("Top layer soil Moisture\n"+r"($\rm m^3 \; m^{-3}$)")
axes[7].set_ylabel(r"$k_{srl}$")

# axes[0].set_ylim([0,15])
# axes[1].set_ylim([0,0.00085])
# axes[3].set_ylim([-4,0])
# axes[4].set_ylim([-4,0])
# axes[5].set_ylim([-4,0])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESim_psensitivity_test_soilmoisture_plantsoilhydraulics_by_rootdepth.png",dpi=300,bbox_inches='tight')

# %%

# %% [markdown]
# ## Optimal Trajectory Allocation
#
# This approach to carbon allocation is modified from Potkay et al. "Coupled whole-tree optimality and xylem hydraulics explain dynamic biomass partitioning" (2021, doi: 10.1111/nph.17242). 
#
# The instantaneous allocation fraction to pool $i$ is proportional to the ratio of the marginal gain divided by the marginal cost:
#
# $u_i \propto \frac{marginal gain_i}{marginal cost_i}$
#
# where $u_i$ is the instantaneous allocation fraction to pool $i$. The marginal gain per pool is equal to 
#
# $marginal \; gain_i = \frac{d}{dC_k} [ a_L(A_n + R_d) - R_m] = \frac{d}{dC_k} [ GPP - R_m]$
#
# where $a_L$ is the leaf area index, $A_n$ is net photosynthetic rate, $R_d$ is leaf mitochondrial respiration rate, and $R_m$ is the maintenance respiration rate. 
#
# The marginal cost takes into account the mean residence time of the pool. As discussed in Potkay et al. (2021) this means that carbon allocation "considers how long any investment of carbon will last and potentially benefit a tree [plant]. Investments with short-lived payoffs (i.e. small $\tau_i$) benefit the tree only briefly and thus reflect poor investments over the duration of a tree’s [plant's] life." To account for this, one may consider the instantaneous senescence rate of a given pool ($S_i$; g C m-2 d-1):
#
# $S_i = \frac{C_i}{\tau_i} = k_{C}^i \; C_i$
#
# where $C_i$ is the pool size (g C m-2) and $\tau_i$ is the mean life span of the pool (days), and $k_{C}^i$ is the turnover rate (days-1). The marginal cost is calculated as the change in senescence rate divided by the change in pool size. 
#
# $marginal \; cost_k = \frac{dS_k}{dC_k} = \frac{1}{\tau_i} = k_{C}^i $
#
# I note here that Potkay et al. (2021) seems to have written out the above equation incorrectly in their supplementary material (Equation S.8.3), incorrectly writing $dS_k/dC_k = tau_i$. 
#
# In practice, when the economic gain is negative, the allocation is set to zero. Furthermore, to ensure all allocation fraction to all pools sum to unity, each marginal gain-cost ratio is normalised to the sum of marginal gain-cost ratios for all pools:
#
# $u_k = \frac{max(0,\frac{marginal gain_k}{marginal cost_k})}{\sum_j max(0,\frac{marginal gain_j}{marginal cost_j})}$
#
# where $j$ is the vector of $k$ carbon pools.
#
# _____________
#
# ### Accounting for fixed allocation fractions to other pools
#
# We can modify this to include constant allocation fractions for some pools. For example, we can assume that the carbon allocation to leaves and roots follows the optimal trajectory principle outlined above, but that allocation to stems ($u_S$) and grains ($u_G$) is fixed. To account for this we can do the following. First, we note that the sum of all four allocation fractions must sum to unity:
#
# $u_L + u_R + u_S + u_G = 1$
#
# The allocation fractions to $u_L$ and $u_R$ are calculated using the optimal trajectory equations further above, which we denote $u_L^{\prime}$ and $u_R^{\prime}$, respectively, remembering that $u_L^{\prime} + u_R^{\prime} = 1$. To account for constant, non-zero terms for $u_S$ or $u_G$, we must scale $u_L^{\prime}$ and $u_R^{\prime}$ by a factor $\alpha$:
#
# $u_L = \alpha u_L^{\prime}$
#
# $u_R = \alpha u_R^{\prime}$
#
# Therefore, we now have: 
#
# $\alpha u_L^{\prime} + \alpha u_R^{\prime} + u_S + u_G = \alpha (u_L^{\prime} + u_R^{\prime}) + u_S + u_G = 1$
#
# We know that $u_L^{\prime} + u_R^{\prime} = 1$, therefore:
#
# $\alpha (1) + u_S + u_G = 1$
#
# $\alpha = 1 - (u_S + u_G)$
#
# So, the actual allocation fractions to the leaf and root pools are: 
#
# $u_L = (1 - (u_S + u_G)) u_L^{\prime}$
#
# $u_R = (1 - (u_S + u_G)) u_R^{\prime}$
#
# This constrains the optimal trajectory coefficients equally and in a way that maintains the total sum of allocation coefficients equal to 1. 
#
# **TODO: Develop more biophysical constraints**
#
# These can be used to define different species types. 
#
# There are biophysical constraints on the amount of biomass in each pool. First, leaf area index is defined by the multiplication of leaf dry structural biomass ($W_L$; g d.wt m-2) and the specific leaf area ($SLA$; m2 g d.wt-1), there is a parameter for the maximum potential LAI, which constrains the maximum of $W_L$. 
#
# It may be necessary or favorable to incorporate a similar biophysical constraints on the amount of biomass in the root pool. Can we define a parameter for the maximum potential root to shoot ratio, which defines the maximum amount of root biomass per unit of shoot biomass ($r_{r,s}$, unitless). 
#
# <!-- Finally, I want a biophysical constraint that defines the amount of stem biomass. This includes a constraint of stem growth, defined by a parameter for the increment of stem structural dry weight per unit growth of leaves ($Phi_L$; dimensionless; typical range [0.1-1]), and a constraint on maximum amount of biomass in the stem pool -->

# %%
from daesim.plantallocoptimal import PlantOptimalAllocation

# %%
site = ClimateModule()
leaf = LeafGasExchangeModule2(g0=0.0,Vcmax_opt=80e-6,Jmax_opt_rVcmax=0.89,Jmax_opt_rVcmax_method="log",Rds=0.005,g1=3.0,VPDmin=0.1)
canopy = CanopyLayers(nlevmlcan=6)
soillayers = SoilLayers(nlevmlsoil=10,z_max=2.0)
canopyrad = CanopyRadiation(Canopy=canopy)
canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopyRad=canopyrad)
boundarylayer = BoundaryLayerModule(Site=site,k_wl=0.006)

## Module with upstream module dependencies
plant = PlantCH2O(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,BoundaryLayer=boundarylayer,maxLAI=5.0,ksr_coeff=1000,SLA=0.020,Psi_f=-3.0,sf=2.5,m_r_r_opt=0.006)

# %%
## initialise model
plantalloc = PlantOptimalAllocation(PlantCH2O=plant, tr_L=0.01, tr_R=0.01, gradient_method="fd_forward", min_step_rel_WL=0.02, min_step_rel_WR=0.02, gradient_threshold_WL=1e-6, gradient_threshold_WR=1e-6)


# %%
## input variables for canopy layers, canopy radiation and canopy gas exchange
leafTempC = 24.0
airTempC = 20.0
airRH = 65.0
airP = 101325    ## air pressure, Pa
soilTheta = np.linspace(0.28,0.38,plant.SoilLayers.nlevmlsoil)   #np.array([0.30,0.30,0.30,0.30])
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)
airUhc = 2.0     ## wind speed at top-of-canopy (m s-1)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)
sza = 20.0    ## Solar zenith angle (degrees)
SAI = 0.0   ## stem area index (m2 m-2)
CI = 0.8    ## foliage clumping index (-)
hc = 1.0    ## canopy height (m)
d_rpot = 2.0   ## potential rooting depth (m)

# %%
## Leaf biomass
n = 50
_W_L = np.linspace(20,400,n)

## Define model inputs
# W_L = 80
W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
dSdWleaf = np.zeros(n)
dSdWroot = np.zeros(n)
GPP_0_, GPP_soilThetaMax_ = np.zeros(n), np.zeros(n)
Rml_0_, Rml_soilThetaMax_ = np.zeros(n), np.zeros(n)
Rmr_0_, Rmr_soilThetaMax_ = np.zeros(n), np.zeros(n)
E_0_, E_soilThetaMax_ = np.zeros(n), np.zeros(n)
f_Psil_0_, f_Psil_soilThetaMax_ = np.zeros(n), np.zeros(n)
Psil_0_, Psil_soilThetaMax_ = np.zeros(n), np.zeros(n)
Psir_0_, Psir_soilThetaMax_ = np.zeros(n), np.zeros(n)
Psis_0_, Psis_soilThetaMax_ = np.zeros(n), np.zeros(n)
K_s_0_, K_s_soilThetaMax_ = np.zeros(n), np.zeros(n)
K_sr_0_, K_sr_soilThetaMax_ = np.zeros(n), np.zeros(n)
k_srl_0_, k_srl_soilThetaMax_ = np.zeros(n), np.zeros(n)

for ix,xW_L in enumerate(_W_L):
    
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0_[ix], Psil_0_[ix], Psir_0_[ix], Psis_0_[ix], K_s_0_[ix], K_sr_0_[ix], k_srl_0_[ix] = plant.calculate(xW_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(xW_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)

    GPP_soilThetaMax_[ix], Rml_soilThetaMax_[ix], Rmr_soilThetaMax_[ix], E_soilThetaMax_[ix], f_Psil_soilThetaMax_[ix], Psil_soilThetaMax_[ix], Psir_soilThetaMax_[ix], Psis_soilThetaMax_[ix], K_s_soilThetaMax_[ix], K_sr_soilThetaMax_[ix], k_srl_soilThetaMax_[ix] = plant.calculate(xW_L,W_R,plant.SoilLayers.soilThetaMax*np.ones(plant.SoilLayers.nlevmlsoil),leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)


fig, axes = plt.subplots(1,4,figsize=(15,3))

ax = axes[0]
ax.plot(_W_L,GPP_0_,label="Moisture stress")
ax.plot(_W_L,GPP_soilThetaMax_,label="No moisture stress",c="C0",linestyle="--")
ax.plot(_W_L,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.plot(_W_L,E_soilThetaMax_*10000,label="No moisture stress",c="C1",linestyle="--")
# ax.plot(_W_L,Rml_0_+Rmr_0_,label=r"$\rm R_{m,L}+R_{m,R}$",c='0.25')
# ax.plot(_W_L,Rml_0_,label=r"$\rm R_{m,L}$",c='0.5',linestyle="--")
# ax.plot(_W_L,Rmr_0_,label=r"$\rm R_{m,R}$",c='0.5',linestyle=":")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend()

ax = axes[1]
ax.plot(_W_L,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_W_L,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend()
ax.set_ylim([-0.05,0.30])

ax = axes[2]
ax.plot(_W_L,dSdWleaf,label=r"$\rm W_L$")
ax.plot(_W_L,dSdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal cost ($\rm g C \; g C^{-1}$)")
ax.legend()

ax = axes[3]
ax.plot(_W_L,u_L,label=r"$\rm u_L$")
ax.plot(_W_L,u_R,label=r"$\rm u_R$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Allocation coefficient")
ax.legend(handlelength=0.7)
ax.set_ylim([0,1])
# ax.set_xlim([40,120])
ax.grid(True)

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/DAESIM2Plant_optimalalloc_xWL_Vcmax80.png",dpi=300,bbox_inches="tight")
plt.show()

# %%
## Root biomass
n = 50
_W_R = np.linspace(20,400,n)

## Define model inputs
W_L = 80
# W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
dSdWleaf = np.zeros(n)
dSdWroot = np.zeros(n)
GPP_0_ = np.zeros(n)
Rml_0_ = np.zeros(n)
Rmr_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xW_R in enumerate(_W_R):
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,xW_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(W_L,xW_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)
    
fig, axes = plt.subplots(1,4,figsize=(15,3))

ax = axes[0]
ax.plot(_W_R,GPP_0_,label=r"$\rm GPP$")
ax.plot(_W_R,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.plot(_W_R,Rml_0_+Rmr_0_,label=r"$\rm R_{m,L}+R_{m,R}$",c='0.25')
ax.plot(_W_R,Rml_0_,label=r"$\rm R_{m,L}$",c='0.5',linestyle="--")
ax.plot(_W_R,Rmr_0_,label=r"$\rm R_{m,R}$",c='0.5',linestyle=":")
ax.set_xlabel(r"Root biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_W_R,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_W_R,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Root biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[2]
ax.plot(_W_R,dSdWleaf,label=r"$\rm W_L$")
ax.plot(_W_R,dSdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Root biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal cost ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[3]
ax.plot(_W_R,u_L,label=r"$\rm u_L$")
ax.plot(_W_R,u_R,label=r"$\rm u_R$")
ax.set_xlabel(r"Root biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Allocation coefficient")
ax.legend(handlelength=0.7)
ax.set_ylim([0,1])

plt.tight_layout()


# %%
## Soil moisture
n = 50
_soilTheta = np.linspace(0.25,plant.SoilLayers.soilThetaMax,n)

## Define model inputs
W_L = 80
W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
dSdWleaf = np.zeros(n)
dSdWroot = np.zeros(n)
GPP_0_ = np.zeros(n)
Rml_0_ = np.zeros(n)
Rmr_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xsoilTheta in enumerate(_soilTheta):
    xsoilTheta_1d = xsoilTheta*np.ones(plant.SoilLayers.nlevmlsoil)
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,xsoilTheta_1d,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(W_L,W_R,xsoilTheta_1d,leafTempC,airTempC,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)

    
fig, axes = plt.subplots(1,4,figsize=(15,3))

ax = axes[0]
ax.plot(_soilTheta,GPP_0_,label=r"$\rm GPP$")
ax.plot(_soilTheta,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.plot(_soilTheta,Rml_0_+Rmr_0_,label=r"$\rm R_{m,L}+R_{m,R}$",c='0.25')
ax.plot(_soilTheta,Rml_0_,label=r"$\rm R_{m,L}$",c='0.5',linestyle="--")
ax.plot(_soilTheta,Rmr_0_,label=r"$\rm R_{m,R}$",c='0.5',linestyle=":")
ax.set_xlabel(r"Soil water content ($\rm m^3 \; m^{-3}$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_soilTheta,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_soilTheta,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Soil water content ($\rm m^3 \; m^{-3}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[2]
ax.plot(_soilTheta,dSdWleaf,label=r"$\rm W_L$")
ax.plot(_soilTheta,dSdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Soil water content ($\rm m^3 \; m^{-3}$)")
ax.set_ylabel(r"Marginal cost ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[3]
ax.plot(_soilTheta,u_L,label=r"$\rm u_L$")
ax.plot(_soilTheta,u_R,label=r"$\rm u_R$")
ax.set_xlabel(r"Soil water content ($\rm m^3 \; m^{-3}$)")
ax.set_ylabel(r"Allocation coefficient")
ax.legend(handlelength=0.7)
ax.set_ylim([0,1])

plt.tight_layout()



# %%
## Temperature
n = 50
_temperature = np.linspace(10,40,n)

## Define model inputs
W_L = 80
W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
dSdWleaf = np.zeros(n)
dSdWroot = np.zeros(n)
GPP_0_ = np.zeros(n)
Rml_0_ = np.zeros(n)
Rmr_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xTemp in enumerate(_temperature):
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,xTemp,xTemp,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(W_L,W_R,soilTheta,xTemp,xTemp,airRH,airCO2,airO2,airP,airUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)


fig, axes = plt.subplots(1,4,figsize=(15,3))

ax = axes[0]
ax.plot(_temperature,GPP_0_,label=r"$\rm GPP$")
ax.plot(_temperature,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.plot(_temperature,Rml_0_+Rmr_0_,label=r"$\rm R_{m,L}+R_{m,R}$",c='0.25')
ax.plot(_temperature,Rml_0_,label=r"$\rm R_{m,L}$",c='0.5',linestyle="--")
ax.plot(_temperature,Rmr_0_,label=r"$\rm R_{m,R}$",c='0.5',linestyle=":")
ax.set_xlabel(r"Temperature ($\rm ^{\circ}C$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_temperature,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_temperature,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Temperature ($\rm ^{\circ}C$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[2]
ax.plot(_temperature,dSdWleaf,label=r"$\rm W_L$")
ax.plot(_temperature,dSdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Temperature ($\rm ^{\circ}C$)")
ax.set_ylabel(r"Marginal cost ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[3]
ax.plot(_temperature,u_L,label=r"$\rm u_L$")
ax.plot(_temperature,u_R,label=r"$\rm u_R$")
ax.set_xlabel(r"Temperature ($\rm ^{\circ}C$)")
ax.set_ylabel(r"Allocation coefficient")
ax.legend(handlelength=0.7)
ax.set_ylim([0,1])

plt.tight_layout()

# %%
## Wind speed
n = 50
_Uhc = np.linspace(0,12,n)

## Define model inputs
W_L = 200
W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
dSdWleaf = np.zeros(n)
dSdWroot = np.zeros(n)
GPP_0_ = np.zeros(n)
Rml_0_ = np.zeros(n)
Rmr_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xUhc in enumerate(_Uhc):
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,xUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,xUhc,swskyb,swskyd,sza,SAI,CI,hc,d_rpot)


fig, axes = plt.subplots(1,4,figsize=(15,3))

ax = axes[0]
ax.plot(_Uhc,GPP_0_,label=r"$\rm GPP$")
ax.plot(_Uhc,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.plot(_Uhc,Rml_0_+Rmr_0_,label=r"$\rm R_{m,L}+R_{m,R}$",c='0.25')
ax.plot(_Uhc,Rml_0_,label=r"$\rm R_{m,L}$",c='0.5',linestyle="--")
ax.plot(_Uhc,Rmr_0_,label=r"$\rm R_{m,R}$",c='0.5',linestyle=":")
ax.set_xlabel(r"Wind speed ($\rm m \; s^{-1}$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_Uhc,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_Uhc,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Wind speed ($\rm m \; s^{-1}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[2]
ax.plot(_Uhc,dSdWleaf,label=r"$\rm W_L$")
ax.plot(_Uhc,dSdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Wind speed ($\rm m \; s^{-1}$)")
ax.set_ylabel(r"Marginal cost ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[3]
ax.plot(_Uhc,u_L,label=r"$\rm u_L$")
ax.plot(_Uhc,u_R,label=r"$\rm u_R$")
ax.set_xlabel(r"Wind speed ($\rm m \; s^{-1}$)")
ax.set_ylabel(r"Allocation coefficient")
ax.legend(handlelength=0.7)
ax.set_ylim([0,1])

plt.tight_layout()

# %%
