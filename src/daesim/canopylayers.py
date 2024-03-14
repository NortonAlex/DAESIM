import numpy as np
from attrs import define, field
from scipy.stats import beta

@define
class CanopyLayers:
    """
    Canopy discretisation into layers components, including indexing and vertical distribution of parameters.
    """
    nlevmlcan: int = field(default=None)  ## Number of layers in multilayer canopy model. Note: Initialised as None, so it must be assigned after instance is created by set_nlayers method.
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

    def nlayers(self, L, z):
        """
        Determines the number of discrete canopy layers.

        Parameters
        ----------
        L : float
            Canopy leaf area index, m2 m-2
        z : float
            Canopy height, m

        Returns
        -------
        nlayers : int
            Number of canopy layers
        """
        if self.nlevmlcan_enforce != None:
            nlayers = self.nlevmlcan_enforce
        elif (z <= 0.3) or (L <=0.5):
            nlayers = self.nlevmlcan_min
        else:
            nlayers_height = min(np.ceil(z/self.dz_param),self.nlevmlcan_max)
            nlayers_lai = min(np.ceil(L/self.dL_param),self.nlevmlcan_max)
            nlayers = max(nlayers_height,nlayers_lai)
    
        return int(nlayers)

    def set_nlayers(self, L, z):
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
        self.nlevmlcan = self.nlayers(L,z)


    def cast_parameter_over_layers_uniform(self,p):
        """
        Assigns vertically resolved parameter assuming constant value over discrete canopy layers.

        Parameters
        ----------
        p : float
            Parameter value

        Returns
        -------
        layer_p : array_like
            Parameter value at each canopy height
        """
        layer_p = p*np.ones(self.nlevmlcan)  ## assign the parameter value over the levels of the multilayer canopy
        return layer_p

    def cast_parameter_over_layers_exp(self,ptop,k,L_C):
        """
        Calculate parameter at each canopy height using an exponential decay
        function. The vertical distribution of the parameter follows a 
        decreasing exponential with cumulative relative leaf area index from 
        the top of the canopy. 
        
        Parameters
        ----------
        ptop : float
            Parameter value at top-of-canopy
        k : float
            Exponential function shape parameter (exponential decay rate)
        L_C : float
            Canopy leaf area index (total LAI over whole canopy), m2 m-2

        Returns
        -------
        layer_p : array_like
            Parameter value at each canopy height
            
        Notes
        -----
        Certain canopy traits (e.g. leaf nitrogne content) are known to decrease exponentially from 
        the top to the bottom of the canopy. This can be expressed as: $P(L) = N_0 exp^{-k L} 
        where $L$ is the canopy layer leaf area index. 

        References
        ----------
        De Pury and Farquhar, 1997, doi:10.1111/j.1365-3040.1997.00094.x
        """
        dlai = self.cast_parameter_over_layers_betacdf(L_C,self.beta_lai_a,self.beta_lai_b)
        ## Determine cumulative LAI from top of the canopy
        ## First, determine which index of dlai array is the top of the canopy
        itop = self.index_canopy()[0]
        if itop == 0:
            L = np.cumsum(dlai)
        else:
            L = np.cumsum(dlai[::-1])

        ## Cumulative relative LAI from top of the canopy
        relative_LAI = L/L_C
        ## Determine the parameter value over the levels of the multilayer using an exponentially declining function
        layer_p = ptop * np.exp(-k*relative_LAI)

        ## Return multilayer in same index order as rest of canopy
        if itop == 0:
            return layer_p
        else:
            return layer_p[::-1]

    def cast_parameter_over_layers_betacdf(self,p,alpha_param,beta_param,method="total"):
        """
        Calculate parameter at each canopy height using the beta distribution. Use the cumulative
        distribution function evaluated at the the bottom and top heights for the layer. The shape 
        of the beta distribution is determined by the parameters alpha_param and beta_param. Note 
        that alpha_param = beta_param = 1 gives a uniform distribution.

        Parameters
        ----------
        p : float
            Canopy level parameter
        alpha_param : float
            Beta distribution parameter 'alpha'
        beta_param : float
            Beta distribution parameter 'beta'
        method : str
            "total" = the parameter p represents the canopy total, such that: canopy p = integral of multi-layer p generated from this function
            "average" = the parameter p represents the canopy average, such that: canopy p = average of multi-layer p generated from this function

        Returns
        -------
        layer_p : array_like
            Parameter value at each canopy height

        References
        ----------
        Bonan et al., 2021, doi:10.1016/j.agrformet.2021.108435
        """

        if method == "total":
            pvar = p
        elif method == "average":
            pvar = p * self.nlevmlcan
        else:
            print("Error: Chosen method, %s, for casting parameter over canopy layers not available" % method)
        
        # Generate the boundaries of the layers in the canopy
        layer_bounds = np.linspace(0, 1, self.nlevmlcan+1)
        
        # Calculate the CDF values at these boundaries
        cdf_values = beta.cdf(layer_bounds, alpha_param, beta_param)
        
        # Calculate the fraction of variable for each layer by finding the difference between successive CDF values
        fractions = np.diff(cdf_values)
        
        # Assign LAI to each layer
        layer_p = fractions * pvar

        ntop, nbot = self.index_canopy()
        if ntop > nbot:
            return layer_p[::-1]
        else:
            return layer_p
            
