# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
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
from typing import Tuple, Callable
from attrs import define, field
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# %%
from daesim.climate_funcs import *
from daesim.plant import PlantModuleCalculator, PlantModelSolver
from daesim.climate import ClimateModule

# %% [markdown]
# ### Notes for translating code
#
# "TODO": Notes on things to improve in the code, just programming things, not the actual model biophysics.
#
# "ErrorCheck": Check with Firouzeh/Justin to see if this is correct or an error.
#
# "[Question...]": Ask Firouzeh/Justin why this equation/variable/value is used, whether it makes sense.
#
# "Modification": A piece of code that HAD to be modified compared with the Stella version in order to work within this framework or in Python more generally. 

# %% [markdown]
# ## Model run variables

# %%
start_time = 1
end_time = 400
dt = 1
time = np.arange(start_time, end_time + dt, dt)

it = 0
t = time[it]

# %% [markdown]
# ## Forcing data

# %%
file = "/Users/alexandernorton/ANU/Projects/DAESIM/daesim/data/StellaOutFile.csv"

df_forcing = pd.read_csv(file)

# %% [markdown]
# ## Site and Climate

# %% [markdown]
# #### - Initialise the Climate module (details about meteorology, solar, location)
#
# You can initialise it simply with the default parameters or you can initialise it and assign different parameters. 

# %%
SiteX = ClimateModule()

DayJul_X, DayJulPrev_X, dayLength_X, dayLengthPrev_X = SiteX.time_discretisation(time)

# %% [markdown]
# To initialise with a different site, you can specify a different latitude and/or elevation

# %%
SiteY = ClimateModule(CLatDeg=45.0, Elevation=100)

DayJul_Y, DayJulPrev_Y, dayLength_Y, dayLengthPrev_Y = SiteY.time_discretisation(time)

# %% [markdown]
# Compare the two sites

# %%
plt.plot(DayJul_X[0:365], dayLength_X[0:365], label="Site lat = %1.1f" % SiteX.CLatDeg)
plt.plot(DayJul_Y[0:365], dayLength_Y[0:365], label="Site lat = %1.1f" % SiteY.CLatDeg)
plt.legend()
plt.xlabel("Day of Year")
plt.ylabel("Day length (sunlight hours)")

# %% [markdown]
# There are methods (functions) in the climate module that help calculate various meteorological variables like absolute humidity

# %%
AH = SiteX.compute_absolute_humidity(
    df_forcing["Climate.airTempC"].values, df_forcing["Climate.relativeHumidity"].values
)

plt.plot(DayJul_X[0:365], AH[0:365])
plt.legend()
plt.xlabel("Day of Year")
plt.ylabel("Absolute Humidity (g m-3)")

# %% [markdown]
# ## Plant Module Calculator

# %% [markdown]
# #### - Initialise the Plant module
#
# You can initialise it simply with the default parameters or you can initialise it and assign different parameters. 
#
# This example initalises it with a mortality_constant=0.002 and the rest of the parameters are default values.

# %%
Plant1 = PlantModuleCalculator(mortality_constant=0.002, dayLengRequire=12)

# %%
Plant1._initialise(0.04)

# %%
Plant1._initialise(0.04)["iniPhBM"]

# %%
plt.plot(df_forcing["PlantGrowth.Bio time"].values)

df_forcing["PlantGrowth.Bio time"].values[0:10]

# %% [markdown]
# #### - Use the `calculate` method to compute the RHS for the state

# %%
_PhBM = 2.0
_NPhBM = 1.0
_solRadGrd = 20.99843025
_airTempC = 21.43692112
_dayLength = 11.900191330084594
_dayLengthPrev = 11.89987139219148
_Bio_time = 0.0
_propPhAboveBM = 0.473684210526


dydt = Plant1.calculate(
    _PhBM,
    _NPhBM,
    _solRadGrd,
    _airTempC,
    _dayLength,
    _dayLengthPrev,
    _Bio_time,
)
print("dy/dt =", dydt)
print()
print("  PhBM = %1.4f" % dydt[0])
print("  NPhBM = %1.4f" % dydt[1])

# %% [markdown]
# #### - Use one of the calculate methods to compute a flux e.g. PhBioNPP or PhBioMort

# %%
PhBioNPP = Plant1.calculate_PhBioNPP(_PhBM, _solRadGrd, _airTempC, 1, 0.99)
print("PhBioNPP =", PhBioNPP)

# %%
_dayLength = 11.900191330084594
_dayLengthPrev = 11.89987139219148

PhBioMort, Fall_litter = Plant1.calculate_PhBioMortality(
    _PhBM, _dayLength, _dayLengthPrev, 1, 0.99
)

print("PhBioMort =", PhBioMort)
print("Fall_litter =", Fall_litter)

# %%
rootBM = Plant1.calculate_rootBM(_NPhBM)
print("rootBM =", rootBM)

propPhAboveBM = Plant1.calculate_propPhAboveBM(_PhBM, rootBM)
print("propPhAboveBM =", propPhAboveBM)

# %%
Transdown = Plant1.calculate_Transdown(
    _PhBM, PhBioNPP, Fall_litter, propPhAboveBM, _Bio_time
)
print(
    "Transdown =",
    Transdown,
)

# %%
Transup = Plant1.calculate_Transup(
    _NPhBM,
    _Bio_time,
    propPhAboveBM,
)
print(
    "Transup =",
    Transup,
)

# %% [markdown]
# ## Interpolation function for the forcing data
#
# We'll be using the solve_ivp ODE solver. It does the time-stepping by itself (rather than pre-defined time-points like in odeint).
#
# So, in order for discrete forcing data to be used with solve_ivp, the solver must be able to compute the forcing __at any point over the temporal domain__. To do this, we interpolate the forcing data and pass this function to the model. 

# %%
time = df_forcing["Days"].values

Climate_airTempC_f = interp1d(time, df_forcing["Climate.airTempC"].values)
Climate_solRadGrd_f = interp1d(time, df_forcing["Climate.solRadGrd"].values)
Climate_dayLength_f = interp1d(time, df_forcing["Climate.dayLength"].values)
PlantGrowth_dayLengthPrev_f = interp1d(
    time, df_forcing["PlantGrowth.dayLengthPrev"].values
)
PlantGrowth_Bio_time_f = interp1d(time, df_forcing["PlantGrowth.Bio time"].values)

## Select any time within the time domain
t1 = 100

## plot the interpolated time-series
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes[0, 0].plot(Climate_solRadGrd_f(time), label="solRadGrd")
axes[0, 0].legend()
axes[0, 1].plot(Climate_airTempC_f(time), label="airTempC")
axes[0, 1].legend()
axes[1, 0].plot(Climate_dayLength_f(time), label="dayLength")
axes[1, 0].legend()
axes[1, 1].plot(PlantGrowth_dayLengthPrev_f(time), label="dayLengthPrev")
axes[1, 1].legend()
axes[2, 0].plot(PlantGrowth_Bio_time_f(time), label="Bio_time")
axes[2, 0].legend()

# %% [markdown]
# ## Model Initialisation

# %% [markdown]
# Initialise the calculator. Then create the Model class with that calculator, initial conditions and start time. 

# %%
PlantX = PlantModuleCalculator(mortality_constant=0.0003)

Model = PlantModelSolver(
    calculator=PlantX, state1_init=0.036, state2_init=0.0870588, time_start=1
)

# %% [markdown]
# Define a time-axis over which to execute the model. Then run the model given some forcing data.

# %%
time_axis = np.arange(1, 1001, 1)

res = Model.run(
    airTempC=Climate_airTempC_f,
    solRadGrd=Climate_solRadGrd_f,
    dayLength=Climate_dayLength_f,
    dayLengthPrev=PlantGrowth_dayLengthPrev_f,
    Bio_time=PlantGrowth_Bio_time_f,
    time_axis=time_axis,
)

# %% [markdown]
# The result that is returned from the run() method is (for now) just the evolution of the ODE over the time domain.

# %%
res.y[0]

# %%
res.y[1]

# %% [markdown]
# Now that the model ODE has been evaluated, you can compute any related "diagnostic" quantities.

# %%
PhBioNPP = PlantX.calculate_PhBioNPP(
    res.y[0],
    Climate_solRadGrd_f(time_axis),
    Climate_airTempC_f(time_axis),
    1,
    0.99,
)

PhBioMort, Fall_litter = PlantX.calculate_PhBioMortality(
    res.y[0],
    Climate_dayLength_f(time_axis),
    PlantGrowth_dayLengthPrev_f(time_axis),
    1,
    0.99,
)

rootBM = PlantX.calculate_rootBM(res.y[1])

propPhAboveBM = PlantX.calculate_propPhAboveBM(res.y[0], rootBM)

Transdown = PlantX.calculate_Transdown(
    res.y[0],
    PhBioNPP,
    Fall_litter,
    PlantGrowth_Bio_time_f(time_axis),
    propPhAboveBM,
)

Transup = PlantX.calculate_Transup(
    res.y[1],
    PlantGrowth_Bio_time_f(time_axis),
    propPhAboveBM,
)

exudation = PlantX.calculate_exudation(rootBM)


# %%
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

axes[0, 0].plot(res.t, res.y[0], c="C0")
axes[0, 0].set_ylabel("State variable 1\nPhotosynthetic_Biomass")
axes[0, 1].plot(res.t, res.y[1], c="C1")
axes[0, 1].set_ylabel("State variable 2\nNon_photosynthetic_Biomass")

axes[1, 0].plot(time_axis, PhBioNPP, c="C1", label="PhBioNPP")
axes[1, 0].set_ylabel("Diagnostic Flux: PhBioNPP")
axes[1, 1].plot(time_axis, PhBioMort, c="C2", label="PhBioMort")
axes[1, 1].set_ylabel("Diagnostic Flux: PhBioMort")
axes[2, 0].plot(time_axis, Transdown, c="C3", alpha=0.5, label="Transdown")
axes[2, 0].plot(time_axis, Transup, c="C4", alpha=0.5, label="Transup")
axes[2, 0].set_ylabel("Diagnostic Flux:\nTransdown, Transup")
axes[2, 0].legend()
axes[2, 1].plot(time_axis, exudation, c="C5", label="exudation")
axes[2, 1].set_ylabel("Diagnostic Flux: exudation")
axes[2, 1].legend()

plt.tight_layout()

# %%

# %%

# %%

# %%
