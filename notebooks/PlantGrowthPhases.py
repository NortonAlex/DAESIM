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
from attrs import define, field

# %%
from daesim.plantgrowthphases import PlantGrowthPhases

# %%
PlantGrowthPhases()

# %%
# Example usage
plant = PlantGrowthPhases()
print("Number of development phases:", plant.ndevphases)
print("Total GDD to maturity:", plant.totalgdd)
print("Relative GDD index for GDD=500:", plant.calc_relative_gdd_index(500))
print()
print("Development Phases")
for iphase in range(plant.ndevphases):
    print("Phase:",plant.phases[iphase])
    print("  GDD requirement =",plant.gdd_requirements[iphase])
    print("  Pool allocation coefficients: Leaf=%1.1f, Stem=%1.1f, Root=%1.1f, Seed=%1.1f, Exudates=%1.1f" % (plant.allocation_coeffs[iphase][0],plant.allocation_coeffs[iphase][1],plant.allocation_coeffs[iphase][2],plant.allocation_coeffs[iphase][3],plant.allocation_coeffs[iphase][4]),", units=(-)")
    print("  Pool turnover rates         : Leaf=%1.1f, Stem=%1.1f, Root=%1.1f, Seed=%1.1f, Exudates=%1.1f" % (plant.turnover_rates[iphase][0],plant.turnover_rates[iphase][1],plant.turnover_rates[iphase][2],plant.turnover_rates[iphase][3],plant.turnover_rates[iphase][4]),", units=days-1")
    

# %% [markdown]
# ### Example of developmental phases over a season

# %%
n = 1000
_GDDcumulative = np.linspace(0,plant.totalgdd,n)

alloc_coeffs = np.zeros((n,plant.nCpools))
turno_rates = np.zeros((n,plant.nCpools))

for i,cumul_gdd in enumerate(_GDDcumulative):
    iphase = plant.get_active_phase_index(cumul_gdd)
    alloc_coeffs[i,:] = plant.allocation_coeffs[iphase]
    turno_rates[i,:] = plant.turnover_rates[iphase]

# Calculate cumulative GDD for phase transitions
phase_transitions = np.cumsum(plant.gdd_requirements)
# Add a starting point for the initial phase
phase_transitions = np.insert(phase_transitions, 0, 0)

fig, axes = plt.subplots(1,2,figsize=(9,3))

axes[0].plot(_GDDcumulative,alloc_coeffs[:,plant.ileaf],alpha=0.65)
axes[0].plot(_GDDcumulative,alloc_coeffs[:,plant.istem],alpha=0.65)
axes[0].plot(_GDDcumulative,alloc_coeffs[:,plant.iroot],alpha=0.65)
axes[0].plot(_GDDcumulative,alloc_coeffs[:,plant.iseed],alpha=0.65)
axes[0].plot(_GDDcumulative,alloc_coeffs[:,plant.iexud],alpha=0.65)
axes[0].set_ylabel("Allocation coefficient (-)")
axes[0].set_xlabel("Cumulative Growing Degree Days "+r"($\rm ^{\circ}$C)")

axes[1].plot(_GDDcumulative,turno_rates[:,plant.ileaf],alpha=0.65,label="Leaf")
axes[1].plot(_GDDcumulative,turno_rates[:,plant.istem],alpha=0.65,label="Stem")
axes[1].plot(_GDDcumulative,turno_rates[:,plant.iroot],alpha=0.65,label="Root")
axes[1].plot(_GDDcumulative,turno_rates[:,plant.iseed],alpha=0.65,label="Seed")
axes[1].plot(_GDDcumulative,turno_rates[:,plant.iexud],alpha=0.65,label="Exud")
axes[1].set_ylabel(r"Turnover rate ($\rm days^{-1}$)")
axes[1].set_xlabel("Cumulative Growing Degree Days "+r"($\rm ^{\circ}$C)")
axes[1].legend()

# Add phase names to the plots
ax = axes[0]
for i, transition in enumerate(phase_transitions[:-1]):
    # Place text halfway between current and next transition point, but ensure it does not go out of bounds
    next_transition = phase_transitions[i + 1] if i + 1 < len(phase_transitions) else _GDDcumulative[-1]
    text_x = (transition + next_transition) / 2
    text_y = ax.get_ylim()[1] * 0.95  # Near the top of the plot
    ax.text(text_x, text_y, plant.phases[i], horizontalalignment='center', verticalalignment='top', fontsize=8, alpha=0.7, rotation=90)
        
plt.tight_layout()

# %%
