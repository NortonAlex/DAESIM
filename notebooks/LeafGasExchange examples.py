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
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# %%
from daesim.leafgasexchange import LeafGasExchangeModule
from daesim.climate import *
from daesim.biophysics_funcs import fT_Q10, fT_arrhenius, fT_arrheniuspeaked

# %% [markdown]
# ### Photosynthesis Model
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
Leaf = LeafGasExchangeModule()

# %% [markdown]
# ### Run the model given input data

# %%
p = 101325 # air pressure, Pa

Q = 1200e-6  # absorbed PPFD, umol PAR m-2 s-1
T = 25.0  # leaf temperature, degrees Celcius
Cs = 400*(p/1e5)*1e-6 # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6 # oxygen partial pressure, bar
RH = 65.0  # relative humidity, %

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH)

# %% [markdown]
# ### Simulate a light-response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = np.linspace(50,2400,n)*1e-6  #1200*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH)

fig, axes = plt.subplots(1,2,figsize=(10,4))
axes[0].plot(Q*1e6,A*1e6,label="Anet",c="0.5")
axes[0].plot(Q*1e6,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(Q*1e6,Vc*1e6,label="Vc",linestyle=":")
axes[0].plot(Q*1e6,Ve*1e6,label="Ve",linestyle=":")
axes[0].legend()
# axes[0].set_ylim([-5,70])
axes[0].set_ylabel(r"$\rm A$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm Q_{abs}$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");

axes[1].plot(Q*1e6,gs)
axes[1].set_xlabel(r"$\rm Q_{abs}$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");

axes[0].set_title("Light-response curve: Photosynthetic rate")
axes[1].set_title("Light-response curve: Stomatal conductance")

plt.tight_layout()

# %% [markdown]
# ### Simulate a CO2-response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = 25.0*np.ones(n)  # degrees Celcius
Cs = np.linspace(60,800,n)*(p/1e5)*1e-6   #   400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH)

fig, axes = plt.subplots(1,2,figsize=(10,4))

axes[0].plot(Cs*1e6,A*1e6,label="Anet",c="0.5")
axes[0].plot(Cs*1e6,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(Cs*1e6,Vc*1e6,label="Vc",linestyle=":")
axes[0].plot(Cs*1e6,Ve*1e6,label="Ve",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,50])
axes[0].set_ylabel(r"$\rm A$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm C_{s}$"+"\n"+r"($\rm \mu mol \; mol^{-1}$)");
axes[0].grid(True)

axes[1].plot(Cs*1e6,gs)
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm C_{s}$"+"\n"+r"($\rm \mu mol \; mol^{-1}$)");
axes[1].grid(True)

axes[0].set_title("CO2-response curve: Photosynthetic rate")
axes[1].set_title("CO2-response curve: Stomatal conductance")

plt.tight_layout()

# %% [markdown]
# ### Simulate a temperature response curve

# %%
n = 50

p = 101325*np.ones(n) # air pressure, Pa
Q = 800e-6*np.ones(n)  # umol PAR m-2 s-1
T = np.linspace(0,50,n)  # degrees Celcius
Cs = 400*(p/1e5)*1e-6*np.ones(n) # carbon dioxide partial pressure, bar
O = 209000*(p/1e5)*1e-6*np.ones(n) # oxygen partial pressure, bar
RH = 65.0*np.ones(n)

A, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q,T,Cs,O,RH)

fig, axes = plt.subplots(1,2,figsize=(10,4))

axes[0].plot(T,A*1e6,label="Anet",c="0.5")
axes[0].plot(T,A*1e6+Rd*1e6,label="Anet+Rd",c="k")
axes[0].plot(T,Vc*1e6,label="Vc",linestyle=":")
axes[0].plot(T,Ve*1e6,label="Ve",linestyle=":")
axes[0].legend()
axes[0].set_ylim([-5,50])
axes[0].set_ylabel(r"$\rm A$"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)");
axes[0].set_xlabel(r"$\rm T_{leaf}$"+"\n"+r"($\rm ^{\circ}C$)");
axes[0].grid(True)

axes[1].plot(T,gs)
axes[1].set_ylabel(r"$\rm g_{sw}$"+"\n"+r"($\rm mol \; m^{-2} \; s^{-1}$)");
axes[1].set_xlabel(r"$\rm T_{leaf}$"+"\n"+r"($\rm ^{\circ}C$)");
axes[1].grid(True)

axes[0].set_title("Temperature-response curve: Photosynthetic rate")
axes[1].set_title("Temperature-response curve: Stomatal conductance")

plt.tight_layout()

# %%

# %%
