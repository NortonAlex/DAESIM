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
from daesim.climate import *
from daesim.boundarylayer import BoundaryLayerModule

# %% [markdown]
# # Boundary Layer
#
# Below is a set of example calculations and figures showing how the boundary layer is defined and applied in the DAESim model. 

# %% [markdown]
# ### Define input parameters

# %%
## input variables for canopy layers and canopy radiation
LAI = 1.5    ## leaf area index (m2 m-2)
SAI = 0.2    ## stem area index (m2 m-2)
clumping_factor = 0.5   ## foliage clumping index (-)
canopy_height = 1.5     ## canopy height (m)

## input variables for leaf gas exchange model
p = 101325   ## air pressure (Pa)
T = 25.0     ## leaf temperature (degrees Celsius)

# %% [markdown]
# ## Create instance of site and canopy layers modules

# %%
Site = ClimateModule()

# %%
## Instance of CanopyLayers class
canopy = CanopyLayers(nlevmlcan_enforce=8)#beta_lai_a=1,beta_lai_b=1,beta_sai_a=1,beta_sai_b=1)
canopy.set_nlayers(LAI,canopy_height)
canopy.set_index()

# %% [markdown]
# ## Boundary Layer

# %%
boundarylayer = BoundaryLayerModule()

# %% [markdown]
# ### Relationship between wind speed and leaf boundary layer resistance

# %%
n = 50
_u = np.linspace(0.01,10,n)
_r_bl = boundarylayer.calculate_leaf_boundary_resistance(T,p,_u,0.015,Site)

fig, axes = plt.subplots(1,2,figsize=(7,3))
ax = axes[0]
ax.plot(_u,_r_bl)
ax.set_xlabel("Wind speed\n"+r"(m $\rm s^{-1}$)")
ax.set_ylabel("Leaf boundary layer\nresistance "+r"(s $\rm m^{-1}$)")

ax = axes[1]
ax.plot(_u,1000*1/_r_bl)
ax.set_xlabel("Wind speed\n"+r"(m $\rm s^{-1}$)")
ax.set_ylabel("Leaf boundary layer\nconductance "+r"(m $\rm s^{-1}$)")

plt.tight_layout()


# %% [markdown]
# ### Replicating results of Figure 1 in Raupach (1994) for vegetation roughness length and zero-plane displc

# %%
# Example usage
h = canopy_height   # Canopy height (m)
PAI = LAI+SAI  # plant area index (m2 m-2)

z0, d = boundarylayer.calculate_z0_and_d(h, PAI)
print(f"Roughness length (z0): {z0} m")
print(f"Zero-plane displacement height (d): {d} m")


n = 100
_PAI = np.linspace(0.001,10.0,n)
_z0 = np.zeros(n)
_d = np.zeros(n)
_R_ustar_Uh = np.zeros(n)
for i,xPAI in enumerate(_PAI):
    z0, d = boundarylayer.calculate_z0_and_d(h, xPAI)
    _z0[i] = z0
    _d[i] = d
    _R_ustar_Uh[i] = boundarylayer.calculate_R_ustar_Uh(xPAI)


fig, axes = plt.subplots(1,3,figsize=(10,3))
axes[0].plot(_PAI,_R_ustar_Uh)
axes[0].set_ylabel(r"$\rm u_*/U_h$")
axes[0].set_xscale('log')
axes[1].plot(_PAI,_d/h)
axes[1].set_ylabel("d/h")
axes[1].set_xscale('log')
axes[1].set_ylim([0,1])
axes[2].plot(_PAI,_z0/h)
axes[2].set_ylabel(r"$\rm z_0/h$")
axes[2].set_xscale('log')
plt.tight_layout()

# %% [markdown]
# ### Example vertical wind speed profiles within the canopy

# %%
## First, determine which index of plant area index array is the top of the canopy, then order 
## the array accordingly so that we can determine the cumulative amount from the top-of-canopy
ntop, nbot = canopy.index_canopy()
Uh = 2.0  ## wind speed at top-of-canopy

## uniform vertical profile of LAI and SAI
dlai = canopy.cast_parameter_over_layers_betacdf(LAI,1,1)  # Canopy layer leaf area index (m2/m2)
dsai = canopy.cast_parameter_over_layers_betacdf(SAI,1,1)  # Canopy layer stem area index (m2/m2)
dpai_0 = dlai+dsai  # Canopy layer plant area index (m2/m2)
u_z_0 = boundarylayer.calculate_wind_profile_exp(Uh,dpai_0,ntop)

## non-uniform vertical profile of LAI and SAI
dlai = canopy.cast_parameter_over_layers_betacdf(LAI,canopy.beta_lai_a,canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)
dsai = canopy.cast_parameter_over_layers_betacdf(SAI,canopy.beta_sai_a,canopy.beta_sai_b)  # Canopy layer stem area index (m2/m2)
dpai_1 = dlai+dsai  # Canopy layer plant area index (m2/m2)
u_z_1 = boundarylayer.calculate_wind_profile_exp(Uh,dpai_1,ntop)

## non-uniform vertical profile of LAI and SAI with modified empirical coefficient
boundarylayer2 = BoundaryLayerModule(beta=1.0)
dlai = canopy.cast_parameter_over_layers_betacdf(LAI,canopy.beta_lai_a,canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)
dsai = canopy.cast_parameter_over_layers_betacdf(SAI,canopy.beta_sai_a,canopy.beta_sai_b)  # Canopy layer stem area index (m2/m2)
dpai_2 = dlai+dsai  # Canopy layer plant area index (m2/m2)
u_z_2 = boundarylayer2.calculate_wind_profile_exp(Uh,dpai_2,ntop)


## Create plots
fig, axes = plt.subplots(1,2,figsize=(7,3))
axes[0].plot(dpai_0,np.arange(canopy.nlevmlcan),label="Uniform")
axes[0].plot(dpai_1,np.arange(canopy.nlevmlcan),label="Beta distribution")
axes[0].plot(dpai_2,np.arange(canopy.nlevmlcan),label="Beta distribution",linestyle=":")
axes[0].set_xlabel("Plant area index (m2 m-2)")
axes[0].set_ylabel("Canopy layer")
axes[0].set_xlim([0,dpai_1.max()*1.1])
axes[0].legend(title="Canopy PAI",handlelength=0.7)
axes[1].plot(u_z_0,np.arange(canopy.nlevmlcan),label="beta=0.5")
axes[1].plot(u_z_1,np.arange(canopy.nlevmlcan),label="beta=0.5")
axes[1].plot(u_z_2,np.arange(canopy.nlevmlcan),linestyle=":",label="beta=1.0")
axes[1].set_xlabel("Wind speed (m s-1)")
axes[1].set_ylabel("Canopy layer")
axes[1].set_xlim([0,Uh*1.1])
axes[1].legend(title="Exp decay",handlelength=0.7)
plt.tight_layout()

# %% [markdown]
# ### Show assumptions about wind speed profiles

# %%
PAI = LAI+SAI   ## total canopy plant area index
h = canopy_height
z_meas = 10.0   ## measurement height for wind speed (m)
u_z_meas = 2.0     ## wind speed at measurement height (m s-1)

print("Canopy height, h =",h,"m and canopy PAI =",PAI,"m2 m-2")
d = h*boundarylayer.calculate_R_d_h(PAI)
print("then zero-plane displacement height, d = %1.2f" % d,"m")
R_ustar_Uh = boundarylayer.calculate_R_ustar_Uh(PAI)
z0 = h * boundarylayer.calculate_R_z0_h(PAI,R_ustar_Uh)
print("and roughness length, z0 = %1.2f" % z0,"m")

n = 100
_z = np.linspace(d,z_meas,n)
_u_z = boundarylayer.estimate_wind_profile_log(u_z_meas,z_meas,_z,d,z0)
plt.plot(_u_z,_z,c='0.5',linestyle="-",label="Wind profile above canopy")

## First, determine which index of plant area index array is the top of the canopy, then order 
## the array accordingly so that we can determine the cumulative amount from the top-of-canopy
ntop, nbot = canopy.index_canopy()
ih = np.argmin(np.abs(_z - h))
print("ih, _u_z[ih] =",ih,_u_z[ih])
Uh = _u_z[ih]

## non-uniform vertical profile of LAI and SAI
dlai = canopy.cast_parameter_over_layers_betacdf(LAI,canopy.beta_lai_a,canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)
dsai = canopy.cast_parameter_over_layers_betacdf(SAI,canopy.beta_sai_a,canopy.beta_sai_b)  # Canopy layer stem area index (m2/m2)
dpai = dlai+dsai  # Canopy layer plant area index (m2/m2)
u_z = boundarylayer.calculate_wind_profile_exp(Uh,dpai,ntop)    

u_z = np.append(u_z,Uh)
dz = h/canopy.nlevmlcan
_z = np.append(np.arange(0+0.5*dz,h,dz),h)
plt.plot(u_z,_z,c='k',label="Wind profile within canopy")

plt.scatter(u_z_meas,z_meas,marker='s',label="Wind speed measurement",c='k',zorder=4)
plt.hlines(h,xmin=0,xmax=u_z_meas*1.1,color='0.5',label="Top of canopy (h)",linestyle="--")
plt.hlines(d,xmin=0,xmax=u_z_meas*1.1,color='0.75',label="Displacement height (d)",linestyle="--")
plt.legend()
plt.xlim([0,u_z_meas*1.1])
plt.ylim([0,z_meas*1.1])
plt.show()

# %% [markdown]
# ### Multi-layer canopy boundary layer resistances

# %%
## non-uniform vertical profile of LAI and SAI
dlai = canopy.cast_parameter_over_layers_betacdf(LAI,canopy.beta_lai_a,canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)
dsai = canopy.cast_parameter_over_layers_betacdf(SAI,canopy.beta_sai_a,canopy.beta_sai_b)  # Canopy layer stem area index (m2/m2)
dpai = dlai+dsai  # Canopy layer plant area index (m2/m2)
ntop,nbot = canopy.index_canopy()
u_z = boundarylayer.calculate_wind_profile_exp(Uh,dpai,ntop)
_r_bl = boundarylayer.calculate_leaf_boundary_resistance(20.0,101325,u_z,0.015,Site)

fig, axes = plt.subplots(1,3,figsize=(10,3))

ax = axes[0]
ax.plot(u_z,np.arange(canopy.nlevmlcan))
ax.set_ylabel("Canopy layer")
ax.set_xlabel("Wind speed\n"+r"(m $\rm s^{-1}$)")
ax = axes[1]
ax.plot(_r_bl,np.arange(canopy.nlevmlcan))
ax.set_ylabel("Canopy layer")
ax.set_xlabel("Leaf boundary layer\nresistance "+r"(s $\rm m^{-1}$)")
ax = axes[2]
ax.plot(1000*1/_r_bl,np.arange(canopy.nlevmlcan),label="Beta")
ax.set_ylabel("Canopy layer")
ax.set_xlabel("Leaf boundary layer\nconductance "+r"(m $\rm s^{-1}$)")

## uniform vertical profile of LAI and SAI
dlai = canopy.cast_parameter_over_layers_betacdf(LAI,1,1)  # Canopy layer leaf area index (m2/m2)
dsai = canopy.cast_parameter_over_layers_betacdf(SAI,1,1)  # Canopy layer stem area index (m2/m2)
dpai = dlai+dsai  # Canopy layer plant area index (m2/m2)
u_z = boundarylayer.calculate_wind_profile_exp(Uh,dpai,ntop)
_r_bl = boundarylayer.calculate_leaf_boundary_resistance(20.0,101325,u_z,0.015,Site)

ax = axes[0]
ax.plot(u_z,np.arange(canopy.nlevmlcan))
ax.set_ylabel("Canopy layer")
ax.set_xlabel("Wind speed\n"+r"(m $\rm s^{-1}$)")
ax = axes[1]
ax.plot(_r_bl,np.arange(canopy.nlevmlcan))
ax.set_ylabel("Canopy layer")
ax.set_xlabel("Leaf boundary layer\nresistance "+r"(s $\rm m^{-1}$)")
ax = axes[2]
ax.plot(1000*1/_r_bl,np.arange(canopy.nlevmlcan),label="Uniform")
ax.set_ylabel("Canopy layer")
ax.set_xlabel("Leaf boundary layer\nconductance "+r"(m $\rm s^{-1}$)")

axes[2].legend(title="PAI Profile",handlelength=0.7)

plt.tight_layout()

# %% [markdown]
# ### Converting stomatal conductances to resistances

# %%
## Example of stomatal conductances over layers of the canopy
## - these are stomatal conductances per leaf area
_gs_ml = np.array([0.36450726, 0.35735872, 0.34899538, 0.34641221, 0.34731174, 0.35041727, 0.35477625, 0.35834702])

## Example of sunlit and shaded gs assuming shaded is 50% of sunlit 
_gs_sunlit = _gs_ml
_gs_shaded = 0.5*_gs_ml

# %%
R = 8.314    ## gas constant (J K-1 mol-1)
conversion_factor = p/(R*(T+273.15))
print("Molar density of water vapor: p/(R*T) = %1.1f" % (conversion_factor),"mol m-3")

# %%
fig, axes = plt.subplots(1,2,figsize=(5.33,2.5),sharey=True)

axes[0].plot(_gs_sunlit,np.arange(canopy.nlevmlcan),label="sunlit")
axes[0].plot(_gs_shaded,np.arange(canopy.nlevmlcan),label="shaded")
axes[0].set_xlabel(r"$\rm g_s$ per leaf"+"\n"+r"($\rm \mu mol \; m^{-2} \; s^{-1}$)")
axes[0].set_ylabel("Canopy layer")

axes[1].plot(conversion_factor/_gs_sunlit,np.arange(canopy.nlevmlcan),label="sunlit")
axes[1].plot(conversion_factor/_gs_shaded,np.arange(canopy.nlevmlcan),label="shaded")
axes[1].set_xlabel(r"$\rm r_s$ per leaf"+"\n"+r"($\rm s \; m^{-1}$)")
axes[1].legend()

plt.tight_layout()

# %%

# %%
