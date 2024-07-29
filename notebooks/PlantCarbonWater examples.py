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

# %% [markdown]
# ### Create instances of each module

# %%
site = ClimateModule()
leaf = LeafGasExchangeModule2(g0=0.0)
canopy = CanopyLayers(nlevmlcan=3)
canopysolar = CanopyRadiation(Canopy=canopy)
canopygasexchange = CanopyGasExchange(Leaf=leaf,Canopy=canopy,CanopySolar=canopysolar)

## Module with upstream module dependencies
plant = PlantModel(Site=site,CanopyGasExchange=canopygasexchange,maxLAI=1.5,ksr_coeff=100,Psi_e=-0.1,sf=1.5)

# %% [markdown]
# ### Input variables for canopy layers, canopy radiation and canopy gas exchange

# %%
LAI = 1.5    ## leaf area index (m2 m-2)
SAI = 0.2    ## stem area index (m2 m-2)
clumping_factor = 0.5   ## foliage clumping index (-)
canopy_height = 1.0     ## canopy height (m)
sza = 30.0       ## solar zenith angle (degrees)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)

## input variables for leaf gas exchange model
leafTempC = 20.0
airTempC = 20.0
airRH = 70.0
airP = 101325    ## air pressure, Pa
soilTheta = 0.26
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)

## model state variables
W_R = 40
W_L = 70

# %% [markdown]
# ### Example run of plant methods

# %%
LAI = plant.calculate_LAI(W_L)

GPP, E = plant.calculate_canopygasexchange(airTempC, leafTempC, airCO2, airO2, airRH, airP, 1.0, LAI, SAI, clumping_factor, canopy_height, sza, swskyb, swskyd)#, leaf, canopy, canopysolar, canopygasexchange, site)

print("GPP =", GPP)
print("E =", E)

# %%
GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd)#,site,leaf,canopygasexchange,canopy,canopysolar)

print("GPP =", GPP_0)
print("E =", E_0)
print("f_Psi_l =",fPsil_0)
print("Psi_l =",Psil_0)

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd)
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

# %% [markdown]
# ## Optimal Allocation
#
# This approach to carbon allocation is modified from Potkay et al. (2021) "Coupled whole-tree optimality and xylem hydraulics explain dynamic biomass partitioning". 
#
# The instantaneous allocation fraction to pool $k$ is proportional to the ratio of the marginal gain divided by the marginal cost:
#
# $u_k \propto \frac{marginal gain_k}{marginal cost_k}$
#
# where $u_k$ is the instantaneous allocation fraction to pool $k$. The marginal gain per pool is equal to 
#
# $marginal \; gain_k = \frac{d}{dC_k} [ a_L(A_n + R_d) - R_m] = \frac{d}{dC_k} [ GPP - R_m]$
#
# while the marginal cost per pool is equal to
#
# $marginal \; cost_k = \frac{dS_k}{dC_k} = \tau_i$
#
# where $\tau$ is the carbon pool mean lifespan. 
#
# In practice, when the economic gain is negative, the allocation is set to zero. Furthermore, to ensure all allocation fraction to all pools sum to unity, each marginal gain-cost ratio is normalised to the sum of marginal gain-cost ratios for all pools:
#
# $u_k = \frac{max(0,\frac{marginal gain_k}{marginal cost_k})}{\sum_j max(0,\frac{marginal gain_j}{marginal cost_j})}$
#
# where $j$ is the vector of $k$ carbon pools.
#
# **Next: Develop more biophysical constraints**
#
# These can be used to define different species types. 
#
# There are biophysical constraints on the amount of biomass in each pool. First, leaf area index is defined by the multiplication of leaf dry structural biomass ($W_L$; g d.wt m-2) and the specific leaf area ($SLA$; m2 g d.wt-1), there is a parameter for the maximum potential LAI, which constrains the maximum of $W_L$. 
#
# I want a similar biophysical constraint on the amount of biomass in the root pool. Can we define a parameter for the maximum potential root to shoot ratio, which defines the maximum amount of root biomass per unit of shoot biomass ($r_{r,s}$, unitless). 
#
# <!-- Finally, I want a biophysical constraint that defines the amount of stem biomass. This includes a constraint of stem growth, defined by a parameter for the increment of stem structural dry weight per unit growth of leaves ($Phi_L$; dimensionless; typical range [0.1-1]), and a constraint on maximum amount of biomass in the stem pool -->

# %%
@define 
class PlantOptimalAllocation:
    """
    Calculator of plant allocation based on optimal trajectory principle
    """

    ## Class parameters
    
    ## Biomass pool step size (defined as a factor or 'multiplier') for evaluating marginal gain and marginal cost with finite difference method
    dWL_factor: float = field(default=1.01)    ## Step size for leaf biomass pool
    dWR_factor: float = field(default=1.01)    ## Step size for leaf biomass pool


    def calculate(
        self,
        W_L,         ## leaf structural dry biomass (g d.wt m-2)
        W_R,         ## root structural dry biomass (g d.wt m-2)
        soilTheta,   ## volumetric soil water content (m3 m-3)
        leafTempC,   ## leaf temperature (deg C)
        airTempC,    ## air temperature (deg C), outside leaf boundary layer 
        airRH,      ## relative humidity of air (%), outside leaf boundary layer
        airCO2,  ## leaf surface CO2 partial pressure, bar, (corrected for boundary layer effects)
        airO2,   ## leaf surface O2 partial pressure, bar, (corrected for boundary layer effects)
        airP,    ## air pressure, Pa, (in leaf boundary layer)
        swskyb, ## Atmospheric direct beam solar radiation, W/m2
        swskyd, ## Atmospheric diffuse solar radiation, W/m2
        Site=ClimateModule(),   ## It is optional to define Site for this method. If no argument is passed in here, then default setting for Site is the default ClimateModule(). Note that this may be important as it defines many site-specific variables used in the calculations.
        Leaf=LeafGasExchangeModule2(),    ## It is optional to define Leaf for this method. If no argument is passed in here, then default setting for Leaf is the default LeafGasExchangeModule().
        CanopyGasExchange=CanopyGasExchange(),    ## It is optional to define CanopyGasExchange for this method. If no argument is passed in here, then default setting for CanopyGasExchange is the default CanopyGasExchange().
        Canopy=CanopyLayers(),    ## It is optional to define Canopy for this method. If no argument is passed in here, then default setting for Canopy is the default CanopyLayers().
        CanopySolar=CanopyRadiation(),    ## It is optional to define CanopySolar for this method. If no argument is passed in here, then default setting for CanopySolar is the default CanopyRadiation().
        Plant=PlantModel(),    ## It is optional to define Plant for this method. If no argument is passed in here, then default setting for Plant is the default PlantModel().
    ) -> Tuple[float]:

        ## Calculate control run
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,Site,Leaf,CanopyGasExchange,Canopy,CanopySolar)
        
        ## Calculate sensitivity run for leaf biomass
        GPP_L, Rml_L, Rmr_L, E_L, f_Psil_L, Psil_L, Psir_L, Psis_L, K_s_L, K_sr_L, k_srl_L = plant.calculate(W_L*self.dWL_factor,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,Site,Leaf,CanopyGasExchange,Canopy,CanopySolar)
        
        ## Calculate sensitivity run for root biomass
        GPP_R, Rml_R, Rmr_R, E_R, f_Psil_R, Psil_R, Psir_R, Psis_R, K_s_R, K_sr_R, k_srl_R = plant.calculate(W_L,W_R*self.dWR_factor,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,Site,Leaf,CanopyGasExchange,Canopy,CanopySolar)
        
        
        ## Calculate change in GPP per unit change in biomass pool
        dGPPdWleaf = (GPP_L-GPP_0)/(W_L*self.dWL_factor - W_L)
        dGPPdWroot = (GPP_R-GPP_0)/(W_R*self.dWR_factor - W_R)
        
        ## Calculate change in GPP-Rm per unit change in biomass pool
        dGPPRmdWleaf = ((GPP_L - Rml_L)-(GPP_0 - Rml_0))/(W_L*self.dWL_factor - W_L)
        dGPPRmdWroot = ((GPP_R - Rmr_R)-(GPP_0 - Rmr_0))/(W_R*self.dWR_factor - W_R)
        
        ## Calculate allocation coefficients
        u_L = np.maximum(0,dGPPdWleaf)/(np.maximum(0,dGPPdWleaf)+np.maximum(0,dGPPdWroot))
        u_R = np.maximum(0,dGPPdWroot)/(np.maximum(0,dGPPdWleaf)+np.maximum(0,dGPPdWroot))

        return u_L, u_R, dGPPdWleaf, dGPPdWroot, dGPPRmdWleaf, dGPPRmdWroot

# %%

# %%
## Leaf biomass
n = 50
_W_L = np.linspace(20,400,n)

## initialise model
plant = PlantModel(maxLAI=1.5,ksr_coeff=3000,Psi_e=-0.1,sf=1.5,Psi_f=-1.0)
plantalloc = PlantOptimalAllocation(dWL_factor=1.03,dWR_factor=1.03)

## input variables for canopy layers, canopy radiation and canopy gas exchange
leafTempC = 20.0
airTempC = 20.0
airRH = 70.0
airP = 101325    ## air pressure, Pa
soilTheta = 0.30
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)

## Define model inputs
# W_L = 80
W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPdWleaf = np.zeros(n)
dGPPdWroot = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
GPP_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xW_L in enumerate(_W_L):
    GPP_0_[ix], Rml_0, Rmr_0, E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(xW_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,site,leaf,canopygasexchange,canopy,canopysolar)
    u_L[ix], u_R[ix], dGPPdWleaf[ix], dGPPdWroot[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix] = plantalloc.calculate(xW_L,W_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,site,leaf,canopygasexchange,canopy,canopysolar,plant)
    
fig, axes = plt.subplots(1,4,figsize=(16,3))

ax = axes[0]
ax.plot(_W_L,GPP_0_,label=r"$\rm GPP$")
ax.plot(_W_L,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_W_L,dGPPdWleaf,label=r"$\rm W_L$")
ax.plot(_W_L,dGPPdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[2]
ax.plot(_W_L,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_W_L,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[3]
ax.plot(_W_L,u_L,label=r"$\rm u_L$")
ax.plot(_W_L,u_R,label=r"$\rm u_R$")
ax.set_xlabel(r"Leaf biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Allocation coefficient")
ax.legend(handlelength=0.7)
ax.set_ylim([0,1])

plt.tight_layout()



# %%
## Root biomass
n = 50
_W_R = np.linspace(20,400,n)

## initialise model
plant = PlantModel(maxLAI=1.5,ksr_coeff=3000,Psi_e=-0.1,sf=1.5,Psi_f=-1.0)
plantalloc = PlantOptimalAllocation(dWL_factor=1.03,dWR_factor=1.03)

## input variables for canopy layers, canopy radiation and canopy gas exchange
leafTempC = 20.0
airTempC = 20.0
airRH = 70.0
airP = 101325    ## air pressure, Pa
soilTheta = 0.30
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)

## Define model inputs
W_L = 80
# W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPdWleaf = np.zeros(n)
dGPPdWroot = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
GPP_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xW_R in enumerate(_W_R):
    GPP_0_[ix], Rml_0, Rmr_0, E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,xW_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,site,leaf,canopygasexchange,canopy,canopysolar)
    u_L[ix], u_R[ix], dGPPdWleaf[ix], dGPPdWroot[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix] = plantalloc.calculate(W_L,xW_R,soilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,site,leaf,canopygasexchange,canopy,canopysolar,plant)
    
fig, axes = plt.subplots(1,4,figsize=(16,3))

ax = axes[0]
ax.plot(_W_R,GPP_0_,label=r"$\rm GPP$")
ax.plot(_W_R,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.set_xlabel(r"Root biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_W_R,dGPPdWleaf,label=r"$\rm W_L$")
ax.plot(_W_R,dGPPdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Root biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[2]
ax.plot(_W_R,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_W_R,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Root biomass ($\rm g \; d.wt \; m^{-2}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
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

## initialise model
plant = PlantModel(maxLAI=1.5,ksr_coeff=3000,Psi_e=-0.1,sf=1.5,Psi_f=-1.0)
plantalloc = PlantOptimalAllocation(dWL_factor=1.03,dWR_factor=1.03)

## input variables for canopy layers, canopy radiation and canopy gas exchange
leafTempC = 20.0
airTempC = 20.0
airRH = 70.0
airP = 101325    ## air pressure, Pa
soilTheta = 0.30
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)

## Define model inputs
W_L = 80
W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPdWleaf = np.zeros(n)
dGPPdWroot = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
GPP_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xsoilTheta in enumerate(_soilTheta):
    GPP_0_[ix], Rml_0, Rmr_0, E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,xsoilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,site,leaf,canopygasexchange,canopy,canopysolar)
    u_L[ix], u_R[ix], dGPPdWleaf[ix], dGPPdWroot[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix] = plantalloc.calculate(W_L,W_R,xsoilTheta,leafTempC,airTempC,airRH,airCO2,airO2,airP,swskyb,swskyd,site,leaf,canopygasexchange,canopy,canopysolar,plant)

    
fig, axes = plt.subplots(1,4,figsize=(16,3))

ax = axes[0]
ax.plot(_soilTheta,GPP_0_,label=r"$\rm GPP$")
ax.plot(_soilTheta,E_0_*10000,label=r"$\rm E(*1e4)$")
ax.set_xlabel(r"Soil water content ($\rm m^3 \; m^{-3}$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_soilTheta,dGPPdWleaf,label=r"$\rm W_L$")
ax.plot(_soilTheta,dGPPdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Soil water content ($\rm m^3 \; m^{-3}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[2]
ax.plot(_soilTheta,dGPPRmdWleaf,label=r"$\rm W_L$")
ax.plot(_soilTheta,dGPPRmdWroot,label=r"$\rm W_R$")
ax.set_xlabel(r"Soil water content ($\rm m^3 \; m^{-3}$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
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

## initialise model
plant = PlantModel(maxLAI=1.5,ksr_coeff=3000,Psi_e=-0.1,sf=1.5,Psi_f=-1.0)
plantalloc = PlantOptimalAllocation(dWL_factor=1.03,dWR_factor=1.03)

## input variables for canopy layers, canopy radiation and canopy gas exchange
leafTempC = 20.0
airTempC = 20.0
airRH = 70.0
airP = 101325    ## air pressure, Pa
soilTheta = 0.30
airCO2 = 400*(airP/1e5)*1e-6 ## carbon dioxide partial pressure (bar)
airO2 = 209000*(airP/1e5)*1e-6   ## oxygen partial pressure (bar)
swskyb = 200.0   ## Atmospheric direct beam solar radiation (W/m2)
swskyd = 80.0    ## Atmospheric diffuse solar radiation (W/m2)

## Define model inputs
W_L = 80
W_R = 80

u_L = np.zeros(n)
u_R = np.zeros(n)
dGPPdWleaf = np.zeros(n)
dGPPdWroot = np.zeros(n)
dGPPRmdWleaf = np.zeros(n)
dGPPRmdWroot = np.zeros(n)
GPP_0_ = np.zeros(n)
E_0_ = np.zeros(n)

for ix,xTemp in enumerate(_temperature):
    GPP_0_[ix], Rml_0, Rmr_0, E_0_[ix], f_Psil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,xTemp,xTemp,airRH,airCO2,airO2,airP,swskyb,swskyd,site,leaf,canopygasexchange,canopy,canopysolar)
    u_L[ix], u_R[ix], dGPPdWleaf[ix], dGPPdWroot[ix], dGPPRmdWleaf[ix], dGPPRmdWroot[ix] = plantalloc.calculate(W_L,W_R,soilTheta,xTemp,xTemp,airRH,airCO2,airO2,airP,swskyb,swskyd,site,leaf,canopygasexchange,canopy,canopysolar,plant)

    
fig, axes = plt.subplots(1,3,figsize=(12,3))

ax = axes[0]
ax.plot(_temperature,GPP_0_,label=r"$\rm GPP$",c='k')
ax.plot(_temperature,E_0_*10000,label=r"$\rm E(*1e4)$",c='k',linestyle=":")
ax.set_xlabel(r"Temperature ($\rm ^{\circ}C$)")
ax.set_ylabel(r"GPP ($\rm g C \; m^{-2} \; s^{-1}$)")
ax.legend(handlelength=0.7)

ax = axes[1]
ax.plot(_temperature,dGPPRmdWleaf,label='Leaf')#label=r"$\rm W_L$")
ax.plot(_temperature,dGPPRmdWroot,label='Root')#label=r"$\rm W_R$")
ax.set_xlabel(r"Temperature ($\rm ^{\circ}C$)")
ax.set_ylabel(r"Marginal gain ($\rm g C \; g C^{-1}$)")
ax.legend(handlelength=0.7,title="Plant pool")

ax = axes[2]
ax.plot(_temperature,u_L,label=r"$\rm u_L$")
ax.plot(_temperature,u_R,label=r"$\rm u_R$")
ax.set_xlabel(r"Temperature ($\rm ^{\circ}C$)")
ax.set_ylabel(r"Allocation coefficient")
ax.legend(handlelength=0.7)
ax.set_ylim([0,1])

plt.tight_layout()
# plt.savefig("/Users/alexandernorton/ANU/Projects/DAESim/DAESIM/results/PlantCarbonWater_optimaltrajectory_soilmoisture.png", dpi=300, bbox_inches='tight')


# %%
