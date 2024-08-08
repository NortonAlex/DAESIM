"""
Plant pools, development and growth phases model class
"""

import numpy as np
from attrs import define, field

@define
class PlantGrowthPhases:
    # Carbon pool indices
    nCpools: int = field(default=5)  ## number of plant carbon pools, must match number of pool indexes below
    ileaf: int = field(default=0)    ## leaf carbon pool index
    istem: int = field(default=1)    ## stem carbon pool index
    iroot: int = field(default=2)    ## root carbon pool index
    iseed: int = field(default=3)    ## seed carbon pool index
    iexud: int = field(default=4)    ## exudation carbon pool index (not really a "pool" but treated as one for carbon allocation)
    
    # Default values for class attributes
    default_phases = ["germination", "vegetative", "anthesis", "fruiting"]
    default_gdd_requirements = [50, 2000, 100, 200]    # Growing degree days requirement per developmental phase
    default_vd_requirements = [0, 45, 0, 0]    # Vernalization days requirement per developmental phase
    default_allocation_coeffs = [
        [0.0, 0.1, 0.9, 0.0, 0.0],  # Phase 1
        [0.48, 0.1, 0.4, 0.0, 0.02],  # Phase 2
        [0.25, 0.0, 0.25, 0.5, 0.0],  # Phase 3
        [0.0, 0.0, 0.0, 1.0, 0.0]  # Phase 4
    ]
    default_turnover_rates = [
        [0.001, 0.001, 0.001, 0.0, 0.0],  # Phase 1
        [0.01,  0.002, 0.01,  0.0, 0.0],  # Phase 2
        [0.02,  0.002, 0.04,  0.0, 0.0],  # Phase 3
        [0.10,  0.008, 0.10,  0.0, 0.0]  # Phase 4
    ]

    # Class attributes with default values or attrs field definitions
    phases: list = field(default=default_phases)
    gdd_requirements: list = field(default=default_gdd_requirements)
    vd_requirements: list = field(default=default_vd_requirements)
    allocation_coeffs: list = field(default=default_allocation_coeffs)
    turnover_rates: list = field(default=default_turnover_rates)
    ndevphases: int = field(init=False)  #field(default=4)
    totalgdd: int = field(init=False)  # This will be set based on gdd_requirements, hence no default in field definition
    vd_t: float = field(default=0)    # Current vernalization state
    vd_0: float = field(default=0)    # Vernalization state at the beginning of the current phase
    previous_phase: int = field(default=None)    # Previous developmental phase index for detecting phase changes

    @totalgdd.default
    def _totalgdd_default(self):
        return sum(self.gdd_requirements)

    @ndevphases.default
    def _ndevphases_default(self):
        """Sets the number of development phases based on the GDD requirements array."""
        return len(self.phases)
    
    def __attrs_post_init__(self):
        # Validate the length of related attributes to ensure consistency
        if not (self.ndevphases == len(self.phases) == len(self.gdd_requirements) == np.shape(self.allocation_coeffs)[0] == np.shape(self.turnover_rates)[0]):
            raise ValueError("The number of developmental phases does not match across phases, GDD requirements, allocation coefficients, and turnover rates. Please correct the attributes to ensure they all match the number of developmental phases.")
        if not (np.shape(self.allocation_coeffs)[0] == np.shape(self.turnover_rates)[0]):
            raise ValueError("The number of developmental phases does not match across phases, GDD requirements, allocation coefficients, and turnover rates. Please correct the attributes to ensure they all match the number of developmental phases.")
        for i, coeffs in enumerate(self.allocation_coeffs):
            if not np.isclose(sum(coeffs), 1):
                raise ValueError(f"The allocation coefficients for phase {i+1}, {self.phases[i]}, do not sum to 1. Please correct this.")
            
    def set_ndevphases(self):
        """Sets the number of development phases based on the GDD requirements array."""
        self.ndevphases = len(self.gdd_requirements)

    def set_totalgdd(self):
        """Sets the total GDD to maturity based on the sum of the GDD requirements array."""
        self.totalgdd = sum(self.gdd_requirements)

    def calc_relative_gdd_index(self, current_gdd):
        """
        Calculates the relative GDD index between 0 and 1, indicating the
        relative development growth phase from germination to the end of seed/fruit filling period.
        """
        if current_gdd <= 0:
            return 0
        elif current_gdd >= self.totalgdd:
            return 1
        else:
            return current_gdd / self.totalgdd

    def get_active_phase_index(self, current_cumul_gdd):
        """
        Determines the active growth phase based on the given cumulative GDD.
        Returns the index of the active growth phase.
        """
        cumulative_gdd = 0
        for i, gdd_req in enumerate(self.gdd_requirements):
            cumulative_gdd += gdd_req
            if current_cumul_gdd <= cumulative_gdd:
                return i
        return len(self.gdd_requirements) - 1  # Return the last phase if current_cumul_gdd exceeds total requirements

    def update_vd_state(self, VRN_time, current_cumul_gdd):
        """
        Updates the vernalization state at the point of phase change.
        """
        current_phase = self.get_active_phase_index(current_cumul_gdd)
        if self.previous_phase is not None and current_phase != self.previous_phase:
            self.vd_0 = VRN_time
        self.vd_t = VRN_time
        self.previous_phase = current_phase

    def get_phase_vd(self):
        """
        Returns the accumulated VD within the current developmental phase.
        """
        return self.vd_t - self.vd_0

