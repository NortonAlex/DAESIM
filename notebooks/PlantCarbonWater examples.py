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
from daesim.plantcarbonwater import PlantModel

# %%
site = ClimateModule()
leaf = LeafGasExchangeModule2(g0=0.01)
plant = PlantModel(maxLAI=1.5,ksr_coeff=100,Psi_e=-0.1,sf=1.5)

# %%
## model forcing variables
leafTempC = 20.0
airTempC = 25.0
airRH = 70.0
airCO2 = 400e-6  ## bar TODO: properly account for pressure effect on CO2 concentration
airO2 = 209e-3   ## bar TODO: properly account for pressure effect on O2 concentration
airP = 101325    ## air pressure, Pa
Q = 800e-6
soilTheta = 0.26

## model state variables
W_R = 40
W_L = 70


_GPP, _Rml, _Rmr, _E, _fPsil, _Psil, _Psir, _Psis, _K_s, _K_sr, _k_srl = plant.calculate(W_L,W_R,soilTheta,leafTempC,airTempC,airRH,Q,airCO2,airO2,airP)

_GPP, _E, _Psil, _Psir, _Psis

# %% [markdown]
# ## Model Sensitivity Tests

# %% [markdown]
# ### - Forcing and State Variable Sensitivity Tests

# %%
## Fixed inputs
soilTheta = 0.26
Tleaf = 20.0
Tair = 21.0
airCO2 = 400e-6
airO2 = 209e-3
airP = 101325
RH = 70.0
Q = 800e-6

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,Tleaf,Tair,RH,Q,airCO2,airO2,airP,site)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,Tleaf,Tair,RH,Q,airCO2,airO2,airP,site)
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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,Tleaf,Tair,RH,Q,airCO2,airO2,airP,site)
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
## Fixed inputs
soilTheta = 0.26
Tleaf = 20.0
Tair = 20.0
airCO2 = 400e-6
airO2 = 209e-3
airP = 101325
RH = 70.0
Q = 800e-6

# %%
## initialise plant model
plant = PlantModel(maxLAI=1.5,ksr_coeff=100,Psi_e=-0.1,sf=1.5)

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,Tleaf,Tair,RH,Q,airCO2,airO2,airP,site)
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
## initialise plant model
plant = PlantModel(maxLAI=1.5,ksr_coeff=100,Psi_e=-0.1,sf=1.5)

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,Tleaf,Tair,RH,Q,airCO2,airO2,airP,site)
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
## initialise plant model
plant = PlantModel(maxLAI=1.5,ksr_coeff=100,Psi_e=-0.1,sf=1.5)

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,Tleaf,Tair,RH,Q,airCO2,airO2,airP,site)
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
## initialise plant model
plant = PlantModel(maxLAI=1.5,ksr_coeff=100,Psi_e=-0.1,sf=1.5)

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
        GPP_0, Rml_0, Rmr_0, E_0, fPsil_0, Psil_0, Psir_0, Psis_0, K_s_0, K_sr_0, k_srl_0 = plant.calculate(W_L,W_R,soilTheta,Tleaf,Tair,RH,Q,airCO2,airO2,airP,site)
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
