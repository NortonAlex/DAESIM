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
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# %%
from daesim.leafgasexchange import LeafGasExchangeModule
from daesim.climate import *
from daesim.biophysics_funcs import fT_Q10, fT_arrhenius, fT_arrheniuspeaked

# %%
Site = ClimateModule()

# %% [markdown]
# ## Photosynthesis Model - Farquhar et al. (1981) and Johnson and Berry (2021)
#
# We use the mechanistic C3 photosynthesis model described by Johnson and Berry (2021, doi: 10.1007/s11120-021-00840-4). This model extends the Farquhar et al. (1980, 149:78–90) model of C3 photosynthesis by incorporating a mechanistic description of steady-state electron transport rate. In the new model, the dark reactions of photosynthesis (carbon metabolism) are described by the traditional Farquhar et al. (1980, 149:78–90) model while light reactions of photosynthesis (electron transport) are described by the new Johnson and Berry (2021, doi: 10.1007/s11120-021-00840-4) model. This new model also derives fluorescence parameters from the mechanistic gas-exchange and electron transport expressions which can be compared to pulse-amplitude-modulated fluorescence measurements. 
#
# The potential rate of net CO2 assimilation under electron transport limitation (light-limited), $A_j$, is given by:
#
# $A_j = \frac{J_{P680}'}{4+8 \Gamma*/C} (1 - \Gamma*/C) - R_d$
#
# Where $R_d$ is dark respiration (mitochondrial respiration), $C$ is the CO2 partial pressure in the chloroplasts, $\Gamma*$ is the CO2 compensation point in the absence of dark respiration, and $J_{P680}'$ is the rate of electron transport through photosystem II. The model provides a conceptual and quantitative description of $J_{P680}'$. 
#
# The potential rate net CO2 assimilation under Rubisco limitation (light-saturated), $A_c$, is given by:
#
# $A_c = \frac{V_{cmax} C}{K_c (1 + O/K_o) + C} (1 - \Gamma*/C) - R_d$
#
# Where $V_{cmax}$ is the maximum carboxylase activity of Rubisco, $K_c$ and $K_o$ are the Michaelis-Menten constants for CO2 and O2, respectively. 
#
# ### Stomatal Conductance Model
#
# To represent stomatal conductance we implement the model of Medlyn et al. (2011, see Corrigendum doi:10.1111/j.1365-2486.2012.02790.x). This model unified the optimal and empirical approaches to representing 
# The model presented a unified approach to modeling stomatal conductance and developed an empirical expression that was consistent with the optimal stomatal behaviour as first described by Cowan and Farquhar (1977). This optimal stomatal behaviour theory postulates that stomata should act to maximize carbon gain (photosynthesis, $A$) while at the same time minimizing water lost ($E$, transpiration).
#
# The theoretical model of Equation 11 in Medlyn et al. (2011, see Corrigendum doi:10.1111/j.1365-2486.2012.02790.x) expresses the stomatal conductance of H2O, $g_{sw}$, as:
#
# $g_{sw} \approx g_0 + 1.6 (1 + \frac{g_1}{\sqrt{D}}) \frac{A}{C_a}$
#
# when D is the leaf-to-air vapor pressure deficit and is expressed in kPa, $C_a$ is the atmospheric CO2 concentration, $A$ is the rate of net CO2 assimilation, $g_0$ and $g_1$ are fitting parameters, where $g_1$ has units of kPa$^{0.5}$ and is proportional to the marginal water cost of carbon, $\lambda$, and with the CO2 compensation point, $\Gamma*$. Importantly, the Medlyn et al. (2011) model uses the light-limited rate of CO2 assimilation for $A$
#
# Equation 13 in Medlyn et al. (2011, see Corrigendum doi:10.1111/j.1365-2486.2012.02790.x):
#
# $\frac{C_i}{C_a} \approx \frac{g_1}{g_1 + \sqrt{D}}$
#
# Equation 14 in Medlyn et al. (2011, see Corrigendum doi:10.1111/j.1365-2486.2012.02790.x), the instantaneous transpiration use efficiency (ITE), is the ratio of leaf photosynthesis to transpiration which can be expressed as:
#
# $\frac{A}{E} \approx \frac{C_a P}{1.6(D + g_1 \sqrt{D}}$ 
#
# where $P$ is atmospheric pressure in kPa.
#
# ### Implementation
#
# To implement the combined photosynthesis and stomatal conducance model we take a numerical approach to estimating the optimal $C_i$. In this approach, there are five steps: Step 1 sets an initial estimate of $C_i$. Step 2 calculates A with the estimate of $C_i$. Step 3 updates $g_{sw}$ using the Medlyn et al. (2011) formulation. Step 4 updates $C_i$ using an expression following Fick's law of diffusion which relates the gradient of $C_a$ and $C_i$ to the photosynthetic rate and stomatal conductance. Step 5 evaluates the proximity of the updated $C_i$ to the previous estimate (using an error tolerance limit). If it is outside the tolerance limit, replace $C_i$ in step 1 with the updated estimate and repeat until convergence whereby the optimal $C_i$ is determined. 
#
# Once the optimal $C_i$ has been determined, it is passed back into the full photosynthetic rate equations (where both light- and rubisco-limited rates are considered) to determine that actual photosynthetic rate. 

# %% [markdown]
# ### Initialise the Leaf Gas Exchange model

# %%
Leaf = LeafGasExchangeModule(Site=Site,gm_opt=0.5)

# %% [markdown]
# ### Run the model given input data

# %%
p = 101325 # air pressure, Pa

Q = 1200e-6  # absorbed PPFD, umol PAR m-2 s-1
T = 25.0  # leaf temperature, degrees Celcius
Cs = 400*(p/1e5)*1e-6 # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6 # oxygen partial pressure, bar
RH = 65.0  # relative humidity, %

fgsw = 1.0

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

# %% [markdown]
# #### - Checking optimal vs actual A, gs and Ci

# %%
p = 101325 # air pressure, Pa

Q = 2000e-6  # absorbed PPFD, umol PAR m-2 s-1
T = 25.0  # leaf temperature, degrees Celcius
Cs = 400*(p/1e5)*1e-6 # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6 # oxygen partial pressure, bar
RH = 65.0  # relative humidity, %
fgsw = 1.0

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

print("Final calculated values:")
print(" A =",A*1e6)
print(" gs =",gs)
print(" Ci =", Ci)
print()
VPD = Site.compute_VPD(T,RH)*1e-3
print("Using actual A in the formulation")
print("Medlyn et al. (2011) equation:")
print("gs = 1.6 (1 + g1/sqrt(D)) A/Ca")
print("  -> gs =",gs)
print("  -> 1.6 (1 + g1/sqrt(D)) A/Ca=",1.6* (1 + Leaf.g1/np.sqrt(VPD))*A/Cs)
print()
print("Fick's law of diffusion:")
print("Ci = Cs - 1.6*A/gs")
print("  -> Ci =",Ci*1e6)
print("  -> Cs - 1.6*A/gs =",(Cs - 1.6*A/gs)*1e6)

# %%
Vqmax = fT_arrheniuspeaked(Leaf.Vqmax_opt,T,E_a=Leaf.Vqmax_Ea,H_d=Leaf.Vqmax_Hd,DeltaS=Leaf.Vqmax_DeltaS)       # Maximum Cyt b6f activity, mol e-1 m-2 s-1
Vcmax = fT_arrheniuspeaked(Leaf.Vcmax_opt,T,E_a=Leaf.Vcmax_Ea,H_d=Leaf.Vcmax_Hd,DeltaS=Leaf.Vcmax_DeltaS)       # Maximum Rubisco activity, mol CO2 m-2 s-1
TPU = fT_Q10(Leaf.TPU_opt_rVcmax*Leaf.Vcmax_opt,T,Q10=Leaf.TPU_Q10) 
Rd = fT_Q10(Vcmax*Leaf.Rds,T,Q10=Leaf.Rd_Q10)
S = fT_arrhenius(Leaf.spfy_opt,T,E_a=Leaf.spfy_Ea)
Kc = fT_arrhenius(Leaf.Kc_opt,T,E_a=Leaf.Kc_Ea)
Ko = fT_arrhenius(Leaf.Ko_opt,T,E_a=Leaf.Ko_Ea)
phi1P_max = Leaf.Kp1/(Leaf.Kp1+Leaf.Kd+Leaf.Kf)  # Maximum photochemical yield PS I
Gamma_star   = 0.5 / S * O      # compensation point in absence of Rd

## Establish PSII and PS I cross-sections, mol PPFD abs PS2/PS1 mol-1 PPFD
## TODO: consider moving this to a separate function
if Leaf.alpha_opt == 'static':
    a2 = Leaf.Abs*Leaf.beta
    a1 = Leaf.Abs - a2

VPD = Site.compute_VPD(T,RH)*1e-3

# Compute stomatal conductance and Ci based on optimal stomatal theory (Medlyn et al., 2011)
A, gs, Ci = Leaf.solve_Ci(Cs,Q,O,VPD,Vqmax,a1,phi1P_max,S,fgsw)    ## TODO: Check the units of A, gs, and Ci here. Is it in ppm (umol mol-1?)? or bar? 
print("Optimal calculated values where A is light-limited i.e. A=Aj, not A=min{Aj,Ac}:")
print(" A =",A*1e6)
print(" gs =",gs)
print(" Ci =", Ci)
print()
print("Using light-limited A in the formulation")
print("Medlyn et al. (2011) equation:")
print("gs = 1.6 (1 + g1/sqrt(D)) A/Ca")
print("  -> gs =",gs)
print("  -> 1.6 (1 + g1/sqrt(D)) A/Ca=",1.6* (1 + Leaf.g1/np.sqrt(VPD))*A/Cs)
print()
print("Fick's law of diffusion:")
print("Ci = Cs - 1.6*A/gs")
print("  -> Ci =",Ci*1e6)
print("  -> Cs - 1.6*A/gs =",(Cs - 1.6*A/gs)*1e6)

# %%

# %% [markdown]
# ### Simulate a light-response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = np.linspace(50,2400,n)*1e-6  #1200*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)   # relative humidity, %
fgsw = 1.0*np.ones(n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))
axes[0].plot(Q*1e6,A*1e6,label=r"$\rm A_{n}$",c="0.5")
# axes[0].plot(Q*1e6,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(Q*1e6,Vc*1e6,label="Vc",linestyle=":")
axes[0].plot(Q*1e6,Ve*1e6,label="Ve",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm Q_{abs}$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].grid(True)

axes[1].plot(Q*1e6,gs,c="0.5")
axes[1].set_xlabel(r"$\rm Q_{abs}$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("Light-response curve")#: Photosynthetic rate")
axes[1].set_title("Light-response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Simulate a CO2-response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = np.linspace(60,800,n)*(p/1e5)*1e-6   #   400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)   # relative humidity, %
fgsw = 1.0*np.ones(n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))

axes[0].plot(Cs*1e6,A*1e6,label=r"$\rm A_{n}$",c="0.5")
# axes[0].plot(Cs*1e6,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(Cs*1e6,Vc*1e6,label="Vc",linestyle=":")
axes[0].plot(Cs*1e6,Ve*1e6,label="Ve",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm C_{s}$"+"\n"+r"($\rm \mu mol \; mol^{-1}$)");
axes[0].grid(True)

axes[1].plot(Cs*1e6,gs,c="0.5")
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm C_{s}$"+"\n"+r"($\rm \mu mol \; mol^{-1}$)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("CO2-response curve")#: Photosynthetic rate")
axes[1].set_title("CO2-response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Simulate a temperature response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = np.linspace(0,50,n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)   # relative humidity, %
fgsw = 1.0*np.ones(n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))

axes[0].plot(T,A*1e6,label=r"$\rm A_n$",c="0.5")
# axes[0].plot(T,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(T,Vc*1e6,label="Vc",linestyle=":")
axes[0].plot(T,Ve*1e6,label="Ve",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm T_{leaf}$"+"\n"+r"($\rm ^{\circ}C$)");
axes[0].grid(True)

axes[1].plot(T,gs,c="0.5")
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm T_{leaf}$"+"\n"+r"($\rm ^{\circ}C$)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("Temperature-response curve")#: Photosynthetic rate")
axes[1].set_title("Temperature-response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Simulate a leaf water potential response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)   # relative humidity, %
fgsw = np.linspace(0,1,n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))

axes[0].plot(fgsw,A*1e6,label=r"$\rm A_n$",c="0.5")
# axes[0].plot(T,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(fgsw,Vc*1e6,label="Vc",linestyle=":")
axes[0].plot(fgsw,Ve*1e6,label="Ve",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm Tuzet \; f_{sv}$"+"\n"+r"(unitless)");
axes[0].grid(True)

axes[1].plot(fgsw,gs,c="0.5")
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm Tuzet \; f_{sv}$"+"\n"+r"(unitless)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("Leaf water potential response curve")#: Photosynthetic rate")
axes[1].set_title("Leaf water potential response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Photosynthesis Model - Farquhar et al. (1981)

# %%
from daesim.leafgasexchange2 import LeafGasExchangeModule2

# %%
Leaf = LeafGasExchangeModule2(Site=Site,Jmax_opt=250e-6)

# %%
p = 101325 # air pressure, Pa

Q = 1200e-6  # absorbed PPFD, umol PAR m-2 s-1
T = 25.0  # leaf temperature, degrees Celcius
Cs = 400*(p/1e5)*1e-6 # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6 # oxygen partial pressure, bar
RH = 65.0  # relative humidity, %

fgsw = 1.0

A, gs, Ci, Ac, Aj, Ap, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

# %% [markdown]
# ### Simulate a light-response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = np.linspace(50,2400,n)*1e-6  #1200*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)   # relative humidity, %
fgsw = 1.0*np.ones(n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Ac, Aj, Ap, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))
axes[0].plot(Q*1e6,A*1e6,label=r"$\rm A_{n}$",c="0.5")
# axes[0].plot(Q*1e6,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(Q*1e6,Ac*1e6,label="Ac",linestyle=":")
axes[0].plot(Q*1e6,Aj*1e6,label="Aj",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm Q_{abs}$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].grid(True)

axes[1].plot(Q*1e6,gs,c="0.5")
axes[1].set_xlabel(r"$\rm Q_{abs}$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("Light-response curve")#: Photosynthetic rate")
axes[1].set_title("Light-response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Simulate a CO2-response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = np.linspace(60,800,n)*(p/1e5)*1e-6   #   400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)   # relative humidity, %
fgsw = 1.0*np.ones(n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Ac, Aj, Ap, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))

axes[0].plot(Cs*1e6,A*1e6,label=r"$\rm A_{n}$",c="0.5")
# axes[0].plot(Cs*1e6,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(Cs*1e6,Ac*1e6,label="Ac",linestyle=":")
axes[0].plot(Cs*1e6,Aj*1e6,label="Ae",linestyle=":")
axes[0].plot(Cs*1e6,Ap*1e6,label="Ap",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm C_{s}$"+"\n"+r"($\rm \mu mol \; mol^{-1}$)");
axes[0].grid(True)

axes[1].plot(Cs*1e6,gs,c="0.5")
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm C_{s}$"+"\n"+r"($\rm \mu mol \; mol^{-1}$)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("CO2-response curve")#: Photosynthetic rate")
axes[1].set_title("CO2-response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Simulate a temperature response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = np.linspace(0,50,n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)   # relative humidity, %
fgsw = 1.0*np.ones(n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Ac, Aj, Ap, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))

axes[0].plot(T,A*1e6,label=r"$\rm A_n$",c="0.5")
# axes[0].plot(T,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(T,Ac*1e6,label="Ac",linestyle=":")
axes[0].plot(T,Aj*1e6,label="Ae",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm T_{leaf}$"+"\n"+r"($\rm ^{\circ}C$)");
axes[0].grid(True)

axes[1].plot(T,gs,c="0.5")
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm T_{leaf}$"+"\n"+r"($\rm ^{\circ}C$)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("Temperature-response curve")#: Photosynthetic rate")
axes[1].set_title("Temperature-response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Simulate a leaf water potential response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)   # relative humidity, %
fgsw = np.linspace(0,1,n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Ac, Aj, Ap, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))

axes[0].plot(fgsw,A*1e6,label=r"$\rm A_n$",c="0.5")
# axes[0].plot(T,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(fgsw,Ac*1e6,label="Ac",linestyle=":")
axes[0].plot(fgsw,Aj*1e6,label="Ae",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm Tuzet \; f_{sv}$"+"\n"+r"(unitless)");
axes[0].grid(True)

axes[1].plot(fgsw,gs,c="0.5")
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm Tuzet \; f_{sv}$"+"\n"+r"(unitless)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("Leaf water potential response curve")#: Photosynthetic rate")
axes[1].set_title("Leaf water potential response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Simulate a relative humidity response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = np.linspace(40.0,100,n)   # relative humidity, %
fgsw = 1.0*np.ones(n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Ac, Aj, Ap, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

fig, axes = plt.subplots(1,2,figsize=(8,3))

axes[0].plot(RH,A*1e6,label=r"$\rm A_n$",c="0.5")
# axes[0].plot(RH,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(RH,Ac*1e6,label="Ac",linestyle=":")
axes[0].plot(RH,Aj*1e6,label="Ae",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,45])
axes[0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm RH$"+"\n"+r"(%)");
axes[0].grid(True)

axes[1].plot(RH,gs,c="0.5")
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm RH$"+"\n"+r"(%)");
axes[1].set_ylim([0,0.7])
axes[1].grid(True)

axes[0].set_title("Relative humidity-response curve")#: Photosynthetic rate")
axes[1].set_title("Relative humidity-response curve")#: Stomatal conductance")

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Comparing responses for air temperature, leaf temperature, g1, VPDmin, RH and VPD

# %%
def esat(TdegC, Pa):
        """
        Calculate the saturation vapor pressure (esat) given temperature and pressure.

        Parameters:
        TdegC (float): Temperature in degrees Celsius.
        Pa (float): Atmospheric pressure in kilopascals.

        Returns:
        float: Saturation vapor pressure in Pascals.

        References:
        Duursma, R.A., 2015. Plantecophys - An R Package for Analysing and Modelling Leaf Gas Exchange Data. PLoS ONE 10, e0143346. doi:10.1371/journal.pone.0143346.
        """
        # Constants
        a = 611.21  # Pascals
        b = 17.502
        c = 240.97  # °C
        f = 1.0007 + 3.46e-8 * Pa * 1000  # Correction factor
        esatval = f * a * np.exp(b * TdegC / (c + TdegC))
        return esatval

def RHtoVPD(RH, TdegC, Pa=101):
    """
    Convert relative humidity (RH) to vapor pressure deficit (VPD).

    Parameters:
    RH (float): Relative humidity in percentage (0-100).
    TdegC (float): Temperature in degrees Celsius.
    Pa (float): Atmospheric pressure in kilopascals (default = 101).

    Returns:
    float: Vapor pressure deficit in kilopascals.

    References:
    Duursma, R.A., 2015. Plantecophys - An R Package for Analysing and Modelling Leaf Gas Exchange Data. PLoS ONE 10, e0143346. doi:10.1371/journal.pone.0143346.
    """
    esatval = esat(TdegC, Pa)
    e = (RH / 100) * esatval
    VPD = (esatval - e) / 1000  # Convert Pa to kPa
    return VPD


def VPDleafToAir(VPD, Tleaf, Tair, Pa=101):
    """
    Convert vapor pressure deficit (VPD) from a leaf temperature to an air-temperature basis.

    Parameters:
    VPD (float): Vapor pressure deficit at the leaf temperature (kPa).
    Tleaf (float): Leaf temperature in degrees Celsius.
    Tair (float): Air temperature in degrees Celsius.
    Pa (float): Atmospheric pressure in kilopascals (default = 101).

    Returns:
    float: Vapor pressure deficit at the air temperature (kPa).

    References:
    Duursma, R.A., 2015. Plantecophys - An R Package for Analysing and Modelling Leaf Gas Exchange Data. PLoS ONE 10, e0143346. doi:10.1371/journal.pone.0143346.
    """
    e = esat(Tleaf, Pa) - VPD * 1000
    vpd = esat(Tair, Pa) - e
    return vpd / 1000  # Convert Pa to kPa


def VPDairToLeaf(VPD, Tair, Tleaf, Pa=101):
    """
    Convert vapor pressure deficit (VPD) from an air temperature to a leaf-temperature basis.

    Parameters:
    VPD (float): Vapor pressure deficit at the air temperature (kPa).
    Tair (float): Air temperature in degrees Celsius.
    Tleaf (float): Leaf temperature in degrees Celsius.
    Pa (float): Atmospheric pressure in kilopascals (default = 101).

    Returns:
    float: Vapor pressure deficit at the leaf temperature (kPa).

    References:
    Duursma, R.A., 2015. Plantecophys - An R Package for Analysing and Modelling Leaf Gas Exchange Data. PLoS ONE 10, e0143346. doi:10.1371/journal.pone.0143346.
    """    
    e = esat(Tair, Pa) - VPD * 1000
    vpd = esat(Tleaf, Pa) - e
    return vpd / 1000  # Convert Pa to kPa


def RHleafToAir(RH, Tleaf, Tair, Pa=101):
    """
    Convert relative humidity (RH) from a leaf temperature to an air-temperature basis.

    Parameters:
    RH (float): Relative humidity at the leaf temperature (percentage, 0-100).
    Tleaf (float): Leaf temperature in degrees Celsius.
    Tair (float): Air temperature in degrees Celsius.
    Pa (float): Atmospheric pressure in kilopascals (default = 101).

    Returns:
    float: Relative humidity at the air temperature (percentage, 0-100).

    References:
    Duursma, R.A., 2015. Plantecophys - An R Package for Analysing and Modelling Leaf Gas Exchange Data. PLoS ONE 10, e0143346. doi:10.1371/journal.pone.0143346.
    """    
    e = (RH / 100) * esat(Tleaf, Pa)  # Vapor pressure at leaf temperature
    rh = e / esat(Tair, Pa)  # Relative humidity at air temperature
    return rh * 100  # Convert to percentage


def RHairToLeaf(RH, Tair, Tleaf, Pa=101):
    """
    Convert relative humidity (RH) from an air temperature to a leaf-temperature basis.

    Parameters:
    RH (float): Relative humidity at the air temperature (percentage, 0-100).
    Tair (float): Air temperature in degrees Celsius.
    Tleaf (float): Leaf temperature in degrees Celsius.
    Pa (float): Atmospheric pressure in kilopascals (default = 101).

    Returns:
    float: Relative humidity at the leaf temperature (percentage, 0-100).

    References:
    Duursma, R.A., 2015. Plantecophys - An R Package for Analysing and Modelling Leaf Gas Exchange Data. PLoS ONE 10, e0143346. doi:10.1371/journal.pone.0143346.
    """
    e = (RH / 100) * esat(Tair, Pa)  # Vapor pressure at air temperature
    rh = e / esat(Tleaf, Pa)  # Relative humidity at leaf temperature
    return rh * 100  # Convert to percentage


# %%
Leaf.g1 = 3.0
Leaf.VPDmin = 0.10

n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = 10.0*np.ones(n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = np.linspace(40.0,100,n)   # relative humidity, %
fgsw = 1.0*np.ones(n)  # stomatal conductance scaling factor based on leaf water potential, unitless

A, gs, Ci, Ac, Aj, Ap, Rd = Leaf.calculate(Q,T,Cs,O,RH,fgsw)

VPD = Leaf.Site.compute_VPD(T,RH)
VPD_plantecophys = RHtoVPD(RH,T)
Tleaf = T+3.0
VPD_air2leaf = VPDairToLeaf(VPD_plantecophys,T,Tleaf)

RH_air2leaf = RHairToLeaf(RH,T,Tleaf)
_A, _gs, _Ci, _Ac, _Aj, _Ap, _Rd = Leaf.calculate(Q,T,Cs,O,RH_air2leaf,fgsw)

fig, axes = plt.subplots(2,3,figsize=(12,6))


axes[0,0].plot(RH,A*1e6,label=r"$\rm A_n$",c="0.5")
axes[0,0].plot(RH_air2leaf,_A*1e6,label=r"$\rm A_n$",c="k",linestyle="--")
# axes[0,0].plot(RH,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0,0].legend()
# axes[0,0].set_ylim([0,10])
axes[0,0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0,0].set_xlabel(r"$\rm RH$"+"\n"+r"(%)");
axes[0,0].grid(True)

axes[0,1].plot(RH,gs,c="0.5")
axes[0,1].plot(RH_air2leaf,_gs,c="k",linestyle="--")
axes[0,1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[0,1].set_xlabel(r"$\rm RH$"+"\n"+r"(%)");
axes[0,1].set_ylim([0,0.7])
axes[0,1].grid(True)

axes[0,2].plot(RH,VPD*1e-3,c="0.5",label=r"VPD($T_{air}$)")
axes[0,2].plot(RH,VPD_air2leaf,c="k",linestyle="--",label=r"VPD($T_{leaf}$)")
axes[0,2].set_ylabel(r"$\rm VPD$"+"\n"+r"($\rm kPa$)");
axes[0,2].set_xlabel(r"$\rm RH$"+"\n"+r"(%)");
# axes[0,2].set_ylim([0,0.7])
axes[0,2].grid(True)
axes[0,2].legend(handlelength=0.75)

axes[0,0].set_title("Relative humidity-response curve")#: Photosynthetic rate")
axes[0,1].set_title("Relative humidity-response curve")#: Stomatal conductance")
axes[0,2].set_title("Relative humidity-response curve")


axes[1,0].plot(VPD*1e-3,A*1e6,label=r"$\rm A_n$",c="0.5")
axes[1,0].plot(VPD_air2leaf,_A*1e6,label=r"$\rm A_n$",c="k",linestyle="--")
# axes[1,0].plot(RH,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[1,0].legend()
# axes[1,0].set_ylim([0,10])
axes[1,0].set_ylabel(r"$\rm A_n$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[1,0].set_xlabel(r"$\rm VPD$"+"\n"+r"(kPa)");
axes[1,0].grid(True)

axes[1,1].plot(VPD*1e-3,gs,c="0.5")
axes[1,1].plot(VPD_air2leaf,_gs,c="k",linestyle="--")
axes[1,1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1,1].set_xlabel(r"$\rm VPD$"+"\n"+r"(kPa)");
axes[1,1].set_ylim([0,0.7])
axes[1,1].grid(True)

axes[1,2].plot(VPD*1e-3,RH,c="0.5",label=r"RH($T_{air}$)")
axes[1,2].plot(VPD_air2leaf,RH_air2leaf,c="k",linestyle="--",label=r"RH($T_{leaf}$)")
axes[1,2].set_ylabel(r"$\rm RH$"+"\n"+r"(%)");
axes[1,2].set_xlabel(r"$\rm VPD$"+"\n"+r"(kPa)");
# axes[1,2].set_ylim([0,0.7])
axes[1,2].grid(True)
axes[1,2].legend(handlelength=0.75)

axes[1,0].set_title("VPD-response curve")#: Photosynthetic rate")
axes[1,1].set_title("VPD-response curve")#: Stomatal conductance")
axes[1,2].set_title("VPD-response curve")

plt.suptitle("Air temperature = %1.1f deg C, Leaf temperature = %1.1f deg C" % (T[0],Tleaf[0]))
plt.tight_layout()
plt.show()

# %%
