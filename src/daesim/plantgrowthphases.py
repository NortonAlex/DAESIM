"""
Plant development/growth model class: Includes definition of plant structure, development and growth phases
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
    ]    ## Turnover rates per pool and developmental phase (days-1)

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
        if not (self.ndevphases == len(self.phases) == len(self.gdd_requirements) == len(self.vd_requirements) == np.shape(self.allocation_coeffs)[0] == np.shape(self.turnover_rates)[0]):
            raise ValueError("The number of developmental phases does not match across phases, GDD requirements, VD requirements, allocation coefficients, and turnover rates. Please correct the attributes to ensure they all match the number of developmental phases.")
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

    # TODO: change name of current_cumul_gdd and current_gdd to be consistent throughout this module
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
        return None  # Return None if that if current_cumul_gdd exceeds total gdd requirements

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

    def calc_relative_gdd_to_phase(self, current_gdd, phase_name):
        """
        Calculates the relative GDD index between 0 and 1, indicating the
        relative development growth phase from germination to the start of
        the specified phase.
        """
        # Identify the index corresponding to the given phase
        try:
            phase_index = self.phases.index(phase_name)
        except ValueError:
            raise ValueError("%s phase not found in the growth phases. Please ensure the phase exists to determine developmental progression." % phase_name)

        phase_gdd = sum(self.gdd_requirements[:phase_index])  # GDD required to reach given phase

        if current_gdd <= 0:
            return 0
        elif current_gdd >= phase_gdd:
            return 1
        else:
            return current_gdd / phase_gdd

    def is_in_phase(self, current_gdd, phase_name):
        """
        Returns True if the current GDD falls within the given phase (phase_name), otherwise False.
        The phase is identified dynamically based on the phase name.
        """
        # Ensure the phase_name exists in the phases list
        if phase_name not in self.phases:
            raise ValueError(f"Phase '{phase_name}' not found in the growth phases. Available phases: {self.phases}")
        
        # Identify the index of the phase
        phase_index = self.phases.index(phase_name)
        
        # Calculate the cumulative GDD range for the given phase
        start_gdd = sum(self.gdd_requirements[:phase_index])  # Start GDD for the phase
        end_gdd = start_gdd + self.gdd_requirements[phase_index]  # End GDD for the phase
        
        # Return True if current_gdd is within the phase range, otherwise False
        return start_gdd <= current_gdd < end_gdd

    def index_progress_through_phase(self, current_gdd, phase_name):
        """
        Calculates the relative progress through a selected phase and returns an
        index of that progress which ranges between 0-1. E.g. an index of 0 means
        there has been no progress through that phase, an index of 0.5 means the
        plant is halfway through that phase and an index of 1 means the plant is
        at the end of that phase. If the plant is not within the define phase
        the function returns None.
        """
        if phase_name is not None:
            if self.is_in_phase(current_gdd, phase_name):
                # Identify the index of the phase
                phase_index = self.phases.index(phase_name)
                # Calculate the cumulative GDD range for the given phase
                start_gdd = sum(self.gdd_requirements[:phase_index])  # Start GDD for the phase
                end_gdd = start_gdd + self.gdd_requirements[phase_index]  # End GDD for the phase
                return min(1, (current_gdd - start_gdd)/self.gdd_requirements[phase_index])
            else:
                return None

