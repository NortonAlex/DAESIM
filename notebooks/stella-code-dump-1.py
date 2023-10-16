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
from daesim.biophysics_funcs import func_TempCoeff
from daesim.plant import PlantModuleCalculator, PlantModelSolver
from daesim.climate import ClimateModule
from daesim.soil import SoilModuleCalculator

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

# %%
_airTempC = df_forcing["Climate.airTempC"].values
_relativeHumidity = df_forcing["Climate.relativeHumidity"].values

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(_airTempC)
axes[0].set_ylabel("airTempC (oC)")
axes[1].plot(_relativeHumidity)
axes[1].set_ylabel("relativeHumidity (%)")

## Question: The relativeHumidity forcing data is shockingly low. It is always < 23%!! Where does this forcing data come from?

# %% [markdown]
# There are methods (functions) in the climate module that help calculate various meteorological variables like absolute humidity

# %%
## Original Stella equation for absolute humidity
_AH_Stella = (
    6.112
    * _relativeHumidity
    * 2.1674
    * np.exp((17.67 * _airTempC) / (_airTempC + 243.5))
    / (273.15 + _airTempC)
)

## New method in Python code for absolute humidity
AH = SiteX.compute_absolute_humidity(
    df_forcing["Climate.airTempC"].values, df_forcing["Climate.relativeHumidity"].values
)

plt.plot(DayJul_X[0:365], AH[0:365])
plt.plot(DayJul_X[0:365], _AH_Stella[0:365])
plt.legend()
plt.xlabel("Day of Year")
plt.ylabel("Absolute Humidity (g m-3)")

# %%
## Original Stella equation for vapor pressure (I assume "actual vapor pressure", not "saturation vapor pressure")
## Stella notes: Daily vapor pressure (Pa) (Nikolov and Zeller) ## Question: What is the reference? The only thing I can find online that matches it is a climate denialism website...
_vapPress_Stella = (
    _AH_Stella * 6.1078 * np.exp(17.269 * _airTempC / (_airTempC + 237.3))
)  ## Question: Where does this equation come from? What are the units supposed to be? It does not align with how I would calculate actual vapor pressure.

## New method in Python code for actual vapor pressure
_vapPress_py = SiteX.compute_actual_vapor_pressure(
    _airTempC,
    _relativeHumidity,
)  # Units: Pa

# plt.plot(
#     DayJul_X[0:365],
#     df_forcing["Climate.vapPress"].values[0:365],
#     c="r",
#     linestyle=":",
#     label=r"$E_a (output)$ (units?)",
# )
plt.plot(
    DayJul_X[0:365], 1e-2 * _vapPress_Stella[0:365], label=r"$E_a (Stella)$ (unit?)"
)
plt.plot(DayJul_X[0:365], 1e-3 * _vapPress_py[0:365], c="r", label=r"$E_a (new)$ (kPa)")

_satvapPress_py = SiteX.compute_sat_vapor_pressure(_airTempC)
plt.plot(
    DayJul_X[0:365], 1e-3 * _satvapPress_py[0:365], c="0.5", label=r"$E_s (new)$ (kPa)"
)

plt.legend()
plt.xlabel("Day of Year")
plt.ylabel("Vapor Pressure (units?)")
plt.show()

plt.scatter(_airTempC, 1e-3 * _satvapPress_py, s=2, c="0.5")
plt.scatter(_airTempC, 1e-3 * _vapPress_py, s=5)
plt.xlabel("Air Temperature (oC)")
plt.ylabel("Vapor Pressure (kPa)")
plt.show()


# %%
_VPD = SiteX.compute_VPD(_airTempC, _relativeHumidity)

plt.plot(DayJul_X[0:365], 1e-3 * _VPD[0:365])
plt.legend()
plt.xlabel("Day of Year")
plt.ylabel("VPD (kPa)")
plt.show()

# %%
_precipM = (
    SiteX.rainConv * df_forcing["Climate.Precipitation"].values
)  ## TODO: change units to mm day-1 i.e. remove use of "precipM" everywhere
_vapPress = SiteX.compute_actual_vapor_pressure(
    _airTempC,
    _relativeHumidity,
)

# _Cloudy_test = [compute_Cloudy(p, v) for p, v in zip(_precipM, _vapPress)]
_Cloudy = SiteX.compute_Cloudy(_precipM, _vapPress)

plt.plot(DayJul_X[0:365], _Cloudy[0:365])
plt.ylabel("Cloudy")

## Question: What does this "Cloudy" variable represent??

# %% [markdown]
# Below outlines what Stella takes as actual forcing data and what climate/meteorology/radiation variables are calculated within the model.
#

# %%
## Forcing data variables
_solRadAtm = df_forcing[
    "Climate.solRadAtm"
].values  ## solar radiation in the atmosphere  ## Question: What does this really represent? How is it calculated?
_solRadAtm = df_forcing[
    "Climate.solRadGrd"
].values  ## correction for cloudy days ## Question: What does this really represent (e.g. downward shortwave radiation at the surface, PAR, something else)? How is it calculated?
_Precipitation = df_forcing[
    "Climate.Precipitation"
].values  ## rainfall from Beltsville, MD 1991. (in/d). Based on GIS data it is now mm/day
_Humidity = df_forcing[
    "Climate.Humidity"
].values  ## relative humidity data used Baltimore Airport, 1991 - Beltsville has no humidity measurements  ## Question: Is the relative humidity or humidity? What are the units??
_relativeHumidity = df_forcing[
    "Climate.relativeHumidity"
].values  ## Question: No documentation here. What are the units?
_airTempMax = df_forcing[
    "Climate.airTempMax"
].values  ## Beltsville, 1991 daily maximum values (deg. F). New data in C
_airTempMin = df_forcing[
    "Climate.airTempMax"
].values  ## Beltsville, 1991 daily minimum values (deg. F). New data in C
_windSpeed = df_forcing[
    "Climate.windSpeed"
].values  ## used Baltimore Airport, 1991 - Beltsville has no wind measurements. units = nautical miles over one day. ## Errorcheck: The Stella docs say the units are "nautical miles over one day" but then the next variable "Wind" is in km/hr but it is the same??

## Calculated forcing data variables
_Wind = _windSpeed  ## Errorcheck: The Stella docs say the units for windSpeed are "nautical miles over one day" but then the next variable "Wind" is in km/hr but it is the same??
_airTempC = SiteX.compute_mean_daily_air_temp(_airTempMin, _airTempMax)
_absoluteHumidity = SiteX.compute_absolute_humidity(
    _airTempC, _relativeHumidity
)  ## Modification: Slightly different formula than that used in Stella code, same result to within <0.1%
_vapPress = SiteX.compute_actual_vapor_pressure(
    _airTempC, _relativeHumidity
)  ## Modification: This way of calculating e_a is correct but it differs to the formula used in Stella code
_precipM = SiteX.rainConv * _Precipitation
_Cloudy = SiteX.compute_Cloudy(_precipM, _vapPress)


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
# #### - Test the TempCoeff function
#
# Errorcheck: This function seems okay for temperatures below 40 degC but it goes whacky above 40 degC. This is a problem that we'll have to correct.
#
# TODO: Correct the whacky values from the calculate_TempCoeff functiono when airTempC > 40 degC.

# %%
Plant1.optTemperature

# %%
n = 100
_airTempC = np.linspace(-25, 60, n)

# Plant1
plt.plot(
    _airTempC,
    func_TempCoeff(_airTempC, optTemperature=Plant1.optTemperature),
    label="optTemperature=%d" % Plant1.optTemperature,
)
# Plant2
Plant2 = PlantModuleCalculator(optTemperature=25)
plt.plot(
    _airTempC,
    func_TempCoeff(_airTempC, optTemperature=Plant2.optTemperature),
    label="optTemperature=%d" % Plant2.optTemperature,
)
# Plant3
Plant3 = PlantModuleCalculator(optTemperature=30)
plt.plot(
    _airTempC,
    func_TempCoeff(_airTempC, optTemperature=Plant3.optTemperature),
    label="optTemperature=%d" % Plant3.optTemperature,
)
plt.ylim([0, 3])
plt.xlabel("airTempC\n(oC)")
plt.ylabel("Temperature Coefficient")
plt.legend()

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

# %% [markdown]
# ## Soil Module Calculator

# %%
# ## ODE
# Decomposing_Microbes(t) = Decomposing_Microbes(t - dt) + (MicUptakeLD + MicUptakeSD - MicDeath) * dt
# # Initial condition: iniMicrobe  ## kg/m2

# ## ODE
# Labile_Detritus(t) = Labile_Detritus(t - dt) + (LDin + SDDecompLD - MicUptakeLD - OxidationLabile - LDDecomp - LDErosion) * dt
# # Initial condition: iniLabileDetritus ## kg/m2

# ## ODE
# Stable_Detritus(t) = Stable_Detritus(t - dt) + (SDin - SDDecompLD - SDDecompMine - MicUptakeSD - OxidationStable - SDErosion) * dt
# # Initial condition: iniStableDetritus  ## Aggregates carbon. kg/m2

# ## ODE
# Mineral(t) = Mineral(t - dt) + (SDDecompMine - MineralDecomp) * dt
# # Initial condition: iniMineralDetritus  ## 53

## Diagnostic ODE
# Soil_Loss(t) = Soil_Loss(t - dt) + (ErosionRate) * dt

# ## Diagnostic ODE
# Soil_Mass(t) = Soil_Mass(t - dt) + ( - ErosionRate) * dt

# ## Diagnostic ODE
# carbonLoss(t) = carbonLoss(t - dt) + (carbonOut) * dt

# ## Diagnostic ODE
# carbonStored(t) = carbonStored(t - dt) + (ResidueIn) * dt

# %%
Soil1 = SoilModuleCalculator()

Soil1

# %% [markdown]
# #### - Use the `calculate` method to compute the RHS for the state

# %%
_LabileDetritus = 0.3956
_PhBM = 0.036
_NPhBM = 0.0870588235294
_PhBioHarvest = 0.0
_NPhBioHarvest = 0.0
_airTempC = 21.43692112
_Water_calPropUnsat_WatMoist = 0.26


dydt = Soil1.calculate(
    _LabileDetritus,
    _PhBM,
    _NPhBM,
    _PhBioHarvest,
    _NPhBioHarvest,
    _Water_calPropUnsat_WatMoist,
    _airTempC,
)
print("dy/dt =", dydt)
print()
print("  dCdt = %1.4f" % dydt)

# %% [markdown]
# #### - Use one of the calculate methods to compute a flux e.g. PhBioNPP or PhBioMort

# %%
LDin = Soil1.calculate_LDin(_PhBM, _NPhBM, _PhBioHarvest, _NPhBioHarvest)
print("LDin =", LDin)

# %%
_Decomposing_Microbes = 0.03

TempCoeff = func_TempCoeff(_airTempC, optTemperature=Soil1.optTemperature)
LDDecomp = Soil1.calculate_LDDecomp(
    _LabileDetritus, _Decomposing_Microbes, _Water_calPropUnsat_WatMoist, TempCoeff
)
print("LDDecomp =", LDDecomp)

# %%
OxidationLabile = Soil1.calculate_oxidation_labile(_LabileDetritus)
print("OxidationLabile =", OxidationLabile)


# %%

# %% [markdown]
# ## Coupled Plant-Soil Model

# %%
@define
class PlantSoilModel:
    plant_calculator: PlantModuleCalculator
    """Calculator of plant model"""

    soil_calculator: SoilModuleCalculator
    """Calculator of plant model"""

    def calculate(
        self,
        Photosynthetic_Biomass,
        Non_Photosynthetic_Biomass,
        LabileDetritus,
        solRadGrd,
        airTempC,
        dayLength,
        dayLengthPrev,
        Bio_time,
    ) -> Tuple[float]:
        dPhBMdt, dNPhBMdt = self.plant_calculator.calculate(
            Photosynthetic_Biomass,
            Non_Photosynthetic_Biomass,
            solRadGrd,
            airTempC,
            dayLength,
            dayLengthPrev,
            Bio_time,
        )

        _PhBioMort = self.plant_calculator.calculate_PhBioMort(Photosynthetic_Biomass)
        _NPhBioMort = self.plant_calculator.calculate_PhBioMort(
            Non_Photosynthetic_Biomass
        )

        dCdt = self.soil_calculator.calculate(
            LabileDetritus, _PhBioMort, _NPhBioMort, airTempC
        )

        return (dPhBMdt, dNPhBMdt, dCdt)
    
"""
Differential equation solver implementation for plant model
"""

@define
class PlantSoilModelSolver:

    """
    Plant and Soil model solver implementation
    """

    calculator: PlantSoilModel
    """Calculator of plant-soil model"""
    
    state1_init: float
    """
    Initial value for state 1
    """

    state2_init: float
    """
    Initial value for state 2
    """
    
    state3_init: float
    """
    Initial value for state 3
    """

    time_start: float
    """
    Time at which the initialisation values apply.
    """

    def run(
        self,
        airTempC: Callable[[float], float],
        solRadGrd: Callable[[float], float],
        dayLength: Callable[[float], float],
        dayLengthPrev: Callable[[float], float],
        Bio_time: Callable[
            [float], float
        ],  ## TODO: Temporary driver (calculate internally at some point)
        time_axis: float,
    ) -> Tuple[float]:
        func_to_solve = self._get_func_to_solve(
            airTempC,
            solRadGrd,
            dayLength,
            dayLengthPrev,
            Bio_time,
        )

        t_eval = time_axis
        t_span = (self.time_start, t_eval[-1])
        start_state = (
            self.state1_init,
            self.state2_init,
            self.state3_init,
        )

        solve_kwargs = {
            "t_span": t_span,
            "t_eval": t_eval,
            "y0": start_state,
        }

        res_raw = self._solve_ivp(
            func_to_solve,
            **solve_kwargs,
        )

        return res_raw

    def _get_func_to_solve(
        self,
        solRadGrd,
        airTempC,
        dayLength,
        dayLengthPrev,
        Bio_time: Callable[float, float],
    ) -> Callable[float, float]:
        def func_to_solve(t: float, y: np.ndarray) -> np.ndarray:
            """
            Function to solve i.e. f(t, y) that goes on the RHS of dy/dt = f(t, y)

            Parameters
            ----------
            t
                time

            y
                State vector

            Returns
            -------
                dy / dt (also as a vector)
            """
            airTempCh = airTempC(t).squeeze()
            solRadGrdh = solRadGrd(t).squeeze()
            dayLengthh = dayLength(t).squeeze()
            dayLengthPrevh = dayLengthPrev(t).squeeze()
            Bio_timeh = Bio_time(t).squeeze()

            dydt = self.calculator.calculate(
                Photosynthetic_Biomass=y[0],
                Non_Photosynthetic_Biomass=y[1],
                LabileDetritus=y[2],
                solRadGrd=solRadGrdh,
                airTempC=airTempCh,
                dayLength=dayLengthh,
                dayLengthPrev=dayLengthPrevh,
                Bio_time=Bio_timeh,
            )

            # TODO: Use this python magic when we have more than one state variable in dydt
            # dydt = [v for v in dydt]

            return dydt

        return func_to_solve

    def _solve_ivp(
        self, func_to_solve, t_span, t_eval, y0, rtol=1e-6, atol=1e-6, **kwargs
    ) -> OptimizeResult:
        raw = solve_ivp(
            func_to_solve,
            t_span=t_span,
            t_eval=t_eval,
            y0=y0,
            atol=atol,
            rtol=rtol,
            **kwargs,
        )
        if not raw.success:
            info = "Your model failed to solve, perhaps there was a runaway feedback?"
            error_msg = f"{info}\n{raw}"
            raise SolveError(error_msg)

        return raw



# %%
Plant1 = PlantModuleCalculator(mortality_constant=0.002, dayLengRequire=12)
Soil1 = SoilModuleCalculator()

PSModel = PlantSoilModel(plant_calculator=Plant1, soil_calculator=Soil1)

Model = PlantSoilModelSolver(
    calculator=PSModel,
    state1_init=0.036,
    state2_init=0.0870588,
    state3_init=0.3956,
    time_start=1,
)

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

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(res.t, res.y[0], c="C0")
axes[0].set_ylabel("State variable 1\nPhotosynthetic_Biomass")
axes[1].plot(res.t, res.y[1], c="C1")
axes[1].set_ylabel("State variable 2\nNon_photosynthetic_Biomass")
axes[2].plot(res.t, res.y[2], c="C1")
axes[2].set_ylabel("State variable 3\nLabileDetritus")

plt.tight_layout()

# %%

# %%
