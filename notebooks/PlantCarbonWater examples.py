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
from daesim.plantcarbonwater import PlantModel
from daesim.soillayers import SoilLayers

# %% [markdown]
# ### Create instances of each module

# %%
site = ClimateModule()
leaf = LeafGasExchangeModule2(g0=0.0)
canopy = CanopyLayers(nlevmlcan=3)
soillayers = SoilLayers(nlevmlsoil=4,z_max=2.0)
canopyrad = CanopyRadiation(Canopy=canopy)
canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopyRad=canopyrad)

## Module with upstream module dependencies
plant = PlantModel(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,maxLAI=1.5,SAI=0.2,CI=0.5,ksr_coeff=100,Psi_e=-0.1,sf=1.5)

# %%
# soillayers = SoilLayers(nlevmlsoil=3,z_max=1.0)

z_soil, d_soil = soillayers.discretise_layers()

print("z_soil =",z_soil)
print("d_soil =",d_soil)

d_r = 2.0  # Rooting depth
plant.calculate_root_distribution(d_r, d_soil)

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
soilTheta = np.array([0.26, 0.30, 0.34, 0.38])  # np.array([[0.26],[0.30],[0.34],[0.38]])  ## volumetric soil moisture (m3 m-3), now defined on a per layer basis (first dimension of array represent the layers)
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)

## model state variables
W_R = 40
W_L = 70

# %%
z_soil, d_soil = plant.SoilLayers.discretise_layers()
d_soil

# %% [markdown]
# ### Example run of plant methods

# %%
LAI = plant.calculate_LAI(W_L)

GPP, E, Rd = plant.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, 1.0, LAI, hc, sza, swskyb, swskyd)

print("GPP =", GPP)
print("E =", E)
print("Rd =", Rd)

# %%
GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)

print("GPP =", GPP_0)
print("E =", E_0)
print("f_Psi_l =",fPsil_0)
print("Psi_l =",Psil_0)



# %%

# %% [markdown]
# ## Model Sensitivity Tests

# %% [markdown]
# ### Forcing and State Variable Sensitivity Tests

# %%

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
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


# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = np.array([20,50,100,200])
_W_L = _W_R*1.
_soilTheta = np.linspace(0.20,plant.soilThetaMax,n)

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
        soilTheta = xsoilTheta
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
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
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = np.linspace(20,400,n)
_W_L = 80*np.ones(4)
_ksr_coeff = np.array([100,1000,5000,15000])

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
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
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = 80 * np.ones(4)
_W_L = _W_R*1.
_ksr_coeff = np.array([100,1000,5000,15000])

_soilTheta = np.linspace(0.20,plant.soilThetaMax,n)

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
        soilTheta = xsoilTheta
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
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
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = 80 * np.ones(4)
_W_L = _W_R*1.
_Psi_f = np.array([-3.0, -2.0, -1.0, -0.5])

_soilTheta = np.linspace(0.20,plant.soilThetaMax,n)

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
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

_soilTheta = np.linspace(0.20,plant.soilThetaMax,n)

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
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


# %% [markdown]
# ### - Rooting depth effect in the multi-layer soil model

# %%
n = 100

fig, axes = plt.subplots(1,8,figsize=(20,2.5))

_W_R = 80 * np.ones(4)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta_1d,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,_d_r)
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
# I want a similar biophysical constraint on the amount of biomass in the root pool. Can we define a parameter for the maximum potential root to shoot ratio, which defines the maximum amount of root biomass per unit of shoot biomass ($r_{r,s}$, unitless). 
#
# <!-- Finally, I want a biophysical constraint that defines the amount of stem biomass. This includes a constraint of stem growth, defined by a parameter for the increment of stem structural dry weight per unit growth of leaves ($Phi_L$; dimensionless; typical range [0.1-1]), and a constraint on maximum amount of biomass in the stem pool -->

# %%
from daesim.plantallocoptimal import PlantOptimalAllocation

# %%
site = ClimateModule()
leaf = LeafGasExchangeModule2(g0=0.0)
canopy = CanopyLayers(nlevmlcan=3)
soillayers = SoilLayers(nlevmlsoil=4,z_max=2.0)
canopyrad = CanopyRadiation(Canopy=canopy)
canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopyRad=canopyrad)

## Module with upstream module dependencies
# plant = PlantModel(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,maxLAI=1.5,SAI=0.2,CI=0.5,ksr_coeff=3000,Psi_e=-0.1,sf=1.5)
plant = PlantModel(Site=site,SoilLayers=soillayers,CanopyGasExchange=canopygasexchange,maxLAI=5.0,ksr_coeff=1000,SLA=0.050)

# %%
## initialise model
plantalloc = PlantOptimalAllocation(Plant=plant,dWL_factor=1.01,dWR_factor=1.01,tr_L=0.01,tr_R=0.008)

# %%
## input variables for canopy layers, canopy radiation and canopy gas exchange
leafTempC = 20.0
airTempC = 20.0
airRH = 70.0
airP = 101325    ## air pressure, Pa
soilTheta = np.array([0.30,0.30,0.30,0.30])
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)
sza = 20.0    ## Solar zenith angle (degrees)
hc = 1.0    ## canopy height (m)
d_r = 2.0   ## rooting depth (m)

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
GPP_0_ = np.zeros(n)
Rml_0_ = np.zeros(n)
Rmr_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xW_L in enumerate(_W_L):
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(xW_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(xW_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
    
    
fig, axes = plt.subplots(1,4,figsize=(16,3))

ax = axes[0]
ax.plot(_W_L,GPP_0_,label=r"$\rm GPP$")
ax.plot(_W_L,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.plot(_W_L,Rml_0_+Rmr_0_,label=r"$\rm R_{m,L}+R_{m,R}$",c='0.25')
ax.plot(_W_L,Rml_0_,label=r"$\rm R_{m,L}$",c='0.5',linestyle="--")
ax.plot(_W_L,Rmr_0_,label=r"$\rm R_{m,R}$",c='0.5',linestyle=":")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_W_L,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_W_L,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[2]
ax.plot(_W_L,dSdWleaf,label=r"$\rm W_L$")
ax.plot(_W_L,dSdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal cost ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

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



# %%

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
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,xW_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(W_L,xW_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
    
fig, axes = plt.subplots(1,4,figsize=(16,3))

ax = axes[0]
ax.plot(_W_R,GPP_0_,label=r"$\rm GPP$")
ax.plot(_W_R,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.plot(_W_L,Rml_0_+Rmr_0_,label=r"$\rm R_{m,L}+R_{m,R}$",c='0.25')
ax.plot(_W_L,Rml_0_,label=r"$\rm R_{m,L}$",c='0.5',linestyle="--")
ax.plot(_W_L,Rmr_0_,label=r"$\rm R_{m,R}$",c='0.5',linestyle=":")
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
_soilTheta = np.linspace(0.25,plant.soilThetaMax,n)

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
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,xsoilTheta_1d,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(W_L,W_R,xsoilTheta_1d,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)

    
fig, axes = plt.subplots(1,4,figsize=(16,3))

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
    GPP_0_[ix], Rml_0_[ix], Rmr_0_[ix], E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,xTemp,xTemp,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)
    u_L[ix], u_R[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix], dSdWleaf[ix], dSdWroot[ix] = plantalloc.calculate(W_L,W_R,soilTheta,xTemp,xTemp,airRH,airCO2,airO2,airP,swskyb,swskyd,sza,hc,d_r)


fig, axes = plt.subplots(1,4,figsize=(16,3))

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
