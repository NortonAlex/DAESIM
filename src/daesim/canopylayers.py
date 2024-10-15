"""
Canopy layers class: Includes parameters and functions to define and discretise the canopy profile into layers.
"""

import numpy as np
from attrs import define, field
from scipy.stats import beta
from typing import Union

@define
class CanopyLayers:
    """
    Canopy discretisation into layers components, including indexing and vertical distribution of parameters.
    """
    nlevmlcan: int = field(default=5)  ## Number of layers in multilayer canopy model. Note: Initialised as None, so it must be assigned after instance is created by set_nlayers method.
    nleaf: int = field(default=2)      ## Number of leaf types (sunlit and shaded)
    isun: int = field(default=0)        ## Sunlit leaf index
    isha: int = field(default=1)        ## Shaded leaf index
    ntop: int = field(default=None)  ## Index for top of canopy layer. Note: Initialised as None, so it must be assigned after instance is created by set_index method.
    nbot: int = field(default=None)  ## Index for bottom of canopy layer. Note: Initialised as None, so it must be assigned after instance is created by set_index method.

    nlevmlcan_max: int = field(default=10)  ## Maximum number of canopy layers (recommend to use 10 as this is produces similar results to canopies with > 10 layers, see Section 4.3 Bonan et al., 2021, doi:10.1016/j.agrformet.2021.108435)
    nlevmlcan_min: int = field(default=3)    ## Minimum number of canopy layers
    nlevmlcan_enforce: int = field(default=None)    ## Optional: Enforce the number of canopy layers, regardless of other parameters
    dL_param: float = field(default=0.4)  ## LAI increment within canopy (m)
    dz_param: float = field(default=0.2)  ## Height increment within canopy (m)
    
    # Vertical distribution paramaters for specific canopy variables
    beta_lai_a: float = field(default=2.0)  ## Beta distribution parameter 'alpha' for vertical distribution of leaf area index
    beta_lai_b: float = field(default=3.5)  ## Beta distribution parameter 'beta' for vertical distribution of leaf area index
    beta_sai_a: float = field(default=4.0)  ## Beta distribution parameter 'alpha' for vertical distribution of stem area index
    beta_sai_b: float = field(default=1.0)  ## Beta distribution parameter 'beta' for vertical distribution of stem area index
    
    def index_canopy(self):
        if self.nlevmlcan is None:
            raise AttributeError("nlevmlcan has not been properly initialized in CanopyLayers class. Please set it by running set_nlayers method before calling index_canopy.")
        ntop = self.nlevmlcan - 1    ## Index for top leaf layer
        nbot = 0    ## Index for bottom leaf layer 
        return ntop, nbot

    def set_index(self):
        """
        Class method to set the number of layers in multilayer canopy model class.
        This updates the class attribute called "nlevmlcan" with a value assigned by 
        the nlayers method. 

        Parameters
        ----------
        L : float
            Canopy leaf area index, m2 m-2
        z : float
            Canopy height, m

        Returns
        -------
        N/A
        """
        ntop, nbot = self.index_canopy()
        self.ntop = ntop
        self.nbot = nbot

    # def nlayers(self, L, z):
    #     """
    #     Determines the number of discrete canopy layers.

    #     Parameters
    #     ----------
    #     L : float
    #         Canopy leaf area index, m2 m-2
    #     z : float
    #         Canopy height, m

    #     Returns
    #     -------
    #     nlayers : int
    #         Number of canopy layers
    #     """
    #     if self.nlevmlcan_enforce != None:
    #         nlayers = self.nlevmlcan_enforce
    #     elif (z <= 0.3) or (L <=0.5):
    #         nlayers = self.nlevmlcan_min
    #     else:
    #         nlayers_height = min(np.ceil(z/self.dz_param),self.nlevmlcan_max)
    #         nlayers_lai = min(np.ceil(L/self.dL_param),self.nlevmlcan_max)
    #         nlayers = max(nlayers_height,nlayers_lai)
    
    #     return int(nlayers)

    # def set_nlayers(self, L, z):
    #     """
    #     Class method to set the number of layers in multilayer canopy model class.
    #     This updates the class attribute called "nlevmlcan" with a value assigned by 
    #     the nlayers method. 

    #     Parameters
    #     ----------
    #     L : float
    #         Canopy leaf area index, m2 m-2
    #     z : float
    #         Canopy height, m

    #     Returns
    #     -------
    #     N/A
    #     """
    #     self.nlevmlcan = self.nlayers(L,z)


    def cast_parameter_over_layers_uniform(self,p: Union[float, np.ndarray]) -> np.ndarray:
        """
        Assigns vertically resolved parameter assuming constant value over discrete canopy layers.

        Parameters
        ----------
        p : float or array_like
            Parameter value or array of parameter values

        Returns
        -------
        layer_p : np.ndarray
            Parameter value at each canopy height, or an array of parameter values at each canopy height for each input parameter

        Notes
        -----
        - The function can take float or array_like inputs for `p` to provide flexibility in specifying
          either single values or multiple values for the parameter.
        - The function uses broadcasting to handle array_like inputs efficiently, allowing operations on 
          multi-dimensional arrays without explicit loops.
        """
        p = np.asarray(p)
        # Assign the parameter value over the levels of the multilayer canopy
        layer_p = np.broadcast_to(p[:, np.newaxis] if p.ndim == 1 else p, (p.size, self.nlevmlcan) if p.ndim == 1 else (self.nlevmlcan,))
        return layer_p

    def cast_parameter_over_layers_exp(
        self,
        ptop: Union[float, np.ndarray],
        k: float,
        L_C: Union[float, np.ndarray]
        ) -> np.ndarray:
        """
        Calculate parameter at each canopy height using an exponential decay
        function. The vertical distribution of the parameter follows a 
        decreasing exponential with cumulative relative leaf area index from 
        the top of the canopy. 
        
        Parameters
        ----------
        ptop : float or array_like
            Parameter value at top-of-canopy
        k : float
            Exponential function shape parameter (exponential decay rate)
        L_C : float or array_like
            Canopy leaf area index (total LAI over whole canopy), m2 m-2

        Returns
        -------
        layer_p : array_like
            Parameter value at each canopy height
            
        Notes
        -----
        Certain canopy traits (e.g. leaf nitrogen content) are known to decrease exponentially from 
        the top to the bottom of the canopy. This can be expressed as: $P(L) = N_0 exp^{-k L} 
        where $L$ is the canopy layer leaf area index. 

        References
        ----------
        De Pury and Farquhar, 1997, doi:10.1111/j.1365-3040.1997.00094.x
        """
        # Ensure each input is a numpy array to allow broadcasting
        ptop = np.asarray(ptop)
        k = np.asarray(k)
        L_C = np.asarray(L_C)

        # Retrieve canopy indices
        ntop, nbot = self.index_canopy()
        
        dlai = self.cast_parameter_over_layers_betacdf(L_C,self.beta_lai_a,self.beta_lai_b)
        ## Determine cumulative LAI from top of the canopy
        ## First, determine which index of dlai array is the top of the canopy
        itop = self.index_canopy()[0]
        if itop != 0:
            dlai = np.flip(dlai, axis=-1 if dlai.ndim == 2 else 0)
        
        # Compute cumulative sum
        L = np.cumsum(dlai, axis=-1 if dlai.ndim == 2 else 0)
        
        # Ensure ptop and L_C are broadcasted correctly
        ptop = np.broadcast_to(ptop[:, np.newaxis] if ptop.ndim == 1 else ptop, (ptop.size, 1) if ptop.ndim == 1 else (1,))
        L_C_broadcasted = np.broadcast_to(L_C[:, np.newaxis] if L_C.ndim == 1 else L_C, (L_C.size, self.nlevmlcan) if L_C.ndim == 1 else (self.nlevmlcan,))
        
        ## Cumulative relative LAI from top of the canopy
        relative_LAI = L/L_C_broadcasted
        
        ## Determine the parameter value over the levels of the multilayer using an exponentially declining function
        layer_p = ptop * np.exp(-k*relative_LAI)
        
        # # Assign LAI to each layer
        if layer_p.ndim == 1:
            if ntop > nbot:
                # Reverse order of array so that top-of-canopy is the last index
                layer_p = layer_p[::-1]
        else:
            if ntop > nbot:
                # Reverse order of array so that top-of-canopy is the last index
                layer_p = np.flip(layer_p, axis=-1)

        return layer_p

    def cast_scalefactor_to_layer_exp(self,k,L_C,cumulative_LAI):
        """
        Calculate the scaling factor across canopy layers based on an exponential decay function 
        and relative cumulative LAI from the top of the canopy.

        Parameters
        ----------
        k : float
            Exponential function shape parameter (exponential decay rate).
        L_C : float
            Canopy leaf area index (total LAI over the whole canopy), m2 m-2.
        cumulative_LAI : array_like
            Relative cumulative leaf area index from the top of the canopy to the bottom of the canopy, m2 m-2, where 
            the first value in the array corresponds to the top-of-canopy. 


        Returns
        -------
        scaling factor : float
            Scaling factor for the given canopy layer.
        """
        # Calculate the relative cumulative LAI from the top of the canopy
        relative_LAI = cumulative_LAI / L_C
        # Determine the adjusted parameter value for the current layer using an exponentially declining function
        layer_scalefactor = np.exp(-k * relative_LAI)
        return layer_scalefactor

    def cast_parameter_over_layers_betacdf(self,
        p: Union[float, np.ndarray],
        alpha_param: Union[float, np.ndarray],
        beta_param: Union[float, np.ndarray],
        method: str = "total"
        ) -> np.ndarray:
        """
        Calculate parameter at each canopy height using the beta distribution. Use the cumulative
        distribution function evaluated at the the bottom and top heights for the layer. The shape 
        of the beta distribution is determined by the parameters alpha_param and beta_param. Note 
        that alpha_param = beta_param = 1 gives a uniform distribution.

        Parameters
        ----------
        p : float or array_like
            Canopy level parameter
        alpha_param : float or array_like
            Beta distribution parameter 'alpha'
        beta_param : float or array_like
            Beta distribution parameter 'beta'
        method : str
            "total" = the parameter p represents the canopy total, such that: canopy p = integral of multi-layer p generated from this function
            "average" = the parameter p represents the canopy average, such that: canopy p = average of multi-layer p generated from this function

        Returns
        -------
        layer_p : np.ndarray
            Parameter value at each canopy height, or an array of parameter values at each canopy height for each input parameter

        Notes
        -----
        - The function can take float or array_like inputs for `p`, `alpha_param`, and `beta_param` to provide flexibility in specifying
          either single values or multiple values for these parameters.
        - The function uses broadcasting to handle array_like inputs efficiently, allowing operations on multi-dimensional arrays without
          explicit loops.
        - Depending on the method selected, `p` is either used directly ("total") or multiplied by `Canopy.nlevmlcan` ("average") to determine
          `pvar`, which represents the parameter to be distributed across the canopy layers.
        - The layer boundaries are calculated using `np.linspace`, and the cumulative distribution function (CDF) of the beta distribution is
          used to determine the fractions of LAI assigned to each layer.
        - If `ntop > nbot`, the order of the layers is reversed to ensure that the top of the canopy is the last index in the output array.

        References
        ----------
        Bonan et al., 2021, doi:10.1016/j.agrformet.2021.108435
        """
        # Ensure each input is a numpy array to allow broadcasting
        alpha_param = np.asarray(alpha_param)
        beta_param = np.asarray(beta_param)
        p = np.asarray(p)

        # Determine pvar based on the method
        if method == "total":
            pvar = p
        elif method == "average":
            pvar = p * self.nlevmlcan
        else:
            raise ValueError(f"Error: Chosen method, {method}, for casting parameter over canopy layers not available")

        # Generate the boundaries of the layers in the canopy
        layer_bounds = np.linspace(0, 1, self.nlevmlcan+1)

        # Ensure alpha_param and beta_param are broadcasted correctly
        alpha_param = np.broadcast_to(alpha_param[:, np.newaxis] if alpha_param.ndim == 1 else alpha_param, (alpha_param.size, self.nlevmlcan+1) if alpha_param.ndim == 1 else (self.nlevmlcan+1,))
        beta_param = np.broadcast_to(beta_param[:, np.newaxis] if beta_param.ndim == 1 else beta_param, (beta_param.size, self.nlevmlcan+1) if beta_param.ndim == 1 else (self.nlevmlcan+1,))

        # Calculate the CDF values at these boundaries
        cdf_values = beta.cdf(layer_bounds, alpha_param, beta_param)
        
        # Calculate the fraction of variable for each layer by finding the difference between successive CDF values
        fractions = np.diff(cdf_values, axis=-1)
        
        # Retrieve canopy indices
        ntop, nbot = self.index_canopy()

        # Assign LAI to each layer
        if pvar.size == 1:
            layer_p = fractions * pvar
            if ntop > nbot:
                # Reverse order of array so that top-of-canopy is the last index
                layer_p = layer_p[::-1]
        else:
            layer_p = (fractions * pvar[:, np.newaxis])
            if ntop > nbot:
                # Reverse order of array along canopy dimension so that top-of-canopy is the last index
                layer_p = layer_p[:,::-1]

        return layer_p
            
