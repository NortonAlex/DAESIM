"""
Soil layers class: Includes parameters and functions to define and discretise the soil profile into layers.
"""

import numpy as np
from attrs import define, field

@define
class SoilLayers:
    """
    Soil discretisation into layers components, including indexing and vertical distribution of parameters.

    Discretises the soil into multiple layers. Three options for how to discretise the soil: (i) Fixed layer 
    thickness, (ii) exponentially increasing layer thickness with depth or (iii) user-defined layer thickness. 
    """
    nlevmlsoil: int = field(default=5)  ## Number of layers in multilayer soil model. Note: Initialised as None, so it must be assigned after instance is created by set_nlayers method.

    ntop: int = field(default=None)  ## Index for uppermost soil layer (nearest to surface). Note: Initialised as None, so it must be assigned after instance is created by set_index method.
    nbot: int = field(default=None)  ## Index for lowest soil layer. Note: Initialised as None, so it must be assigned after instance is created by set_index method.

    nlevmlsoil_max: int = field(default=3)  ## Maximum number of soil layers
    nlevmlsoil_min: int = field(default=3)    ## Minimum number of soil layers
    nlevmlsoil_enforce: int = field(default=None)    ## Optional: Enforce the number of soil layers, regardless of other parameters
    dz_param: float = field(default=0.2)  ## Depth increment in soil layers (m) when assuming equal layer thickness over soil profile

    z_max: float = field(default=1.0)  ## Depth of the soil profile (m), generally this would indicate the depth to bedrock but it may also just indicate the maximum depth the model simulates
    discretise_method: str = field(default="fixed")  ## 
    z_top: float = field(default=None)  ## Depth of top (uppermost) soil layer (m), only used in specific methods of discretising the soil layers
    # Add a class attribute to store horizon-based soil layers
    z_horizon: list = field(default=None)  ## List of user-defined layer thicknesses (m) for horizon method
    
    def index_soil(self):
        if self.nlevmlsoil is None:
            raise AttributeError("nlevmlsoil has not been properly initialized in SoilLayers class. Please set it by running set_nlayers method before calling index_soil.")
        ntop = 0    ## Index for top (uppermost) soil layer
        nbot = self.nlevmlsoil - 1    ## Index for bottom soil layer 
        return ntop, nbot

    def set_index(self):
        """
        Class method to set the number of layers in multilayer soil model class.
        This updates the class attribute called "nlevmlsoil" with a value assigned by 
        the nlayers method. 

        Returns
        -------
        N/A
        """
        ntop, nbot = self.index_soil()
        self.ntop = ntop
        self.nbot = nbot

    def discretise_layers(self):

        # Initialize an empty list to store layer thicknesses and layer depths
        z_soil = []
        d_soil = []

        if self.discretise_method == "fixed":
            # Fixed layer thickness - divide the soil profile into even increments
            dz = self.z_max / self.nlevmlsoil
            z_soil = [self.z_max / self.nlevmlsoil] * self.nlevmlsoil  #dz*np.ones(self.nlevmlsoil)   # soil layer thickness (m)

        elif self.discretise_method == "scaled_exp":
            if self.z_top is None:
                raise ValueError("For scaled method, the top soil layer thickness, z_top, must be defined.")
            remaining_depth = self.z_max - self.z_top
            if self.nlevmlsoil is None or self.nlevmlsoil <= 1:
                raise ValueError("For scaled method, nlevmlsoil must be greater than 1.")
            
            z_soil.append(self.z_top)
            scale_factors = np.exp(np.linspace(0, 1, self.nlevmlsoil - 1))  # Exponential scaling factors
            scale_factors /= np.sum(scale_factors)  # Normalize scale factors to sum to 1
            for factor in scale_factors:
                z_soil.append(factor * remaining_depth)

        elif self.discretise_method == "horizon":
            # Horizon-based method: user provides the layer thicknesses
            if self.z_horizon is None:
                raise ValueError("For horizon method, z_horizon (layer thicknesses) must be defined.")
            if sum(self.z_horizon) > self.z_max:
                raise ValueError("The sum of the horizon layers cannot exceed the total soil profile depth.")
            
            z_soil = self.z_horizon  # Use the user-defined layer thicknesses

        else:
            raise ValueError("Unknown soil layer discretisation method. Choose one of 'fixed' or 'scaled_exp'.")

        d_soil = list(np.cumsum(z_soil))      # depth from soil surface to bottom of each soil layer (m)

        return z_soil, d_soil

