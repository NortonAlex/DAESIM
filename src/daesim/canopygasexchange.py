"""
Leaf gas exchange model class: Includes equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple
from attrs import define, field
from daesim.canopylayers import CanopyLayers
from daesim.canopyradiation import CanopyRadiation
from daesim.leafgasexchange2 import LeafGasExchangeModule2

@define
class CanopyGasExchange:

    def calculate(
        self,
        leafTempC,    ## Leaf temperature, degrees Celsius
        Cs,   ## Leaf surface CO2 partial pressure, bar, (corrected for boundary layer effects)
        O,    ## Leaf surface O2 partial pressure, bar, (corrected for boundary layer effects)
        airRH,   ## Relative humidity, %
        fgsw, ## Leaf water potential limitation factor on stomatal conductance, unitless
        LAI,    ## Leaf area index, m2/m2
        SAI,    ## Stem area index, m2/m2
        clumping_factor,  ## Foliage clumping index (-)
        z,      ## Canopy height, m
        sza,    ## Solar zenith angle, degrees
        swskyb, ## Atmospheric direct beam solar radiation, W/m2
        swskyd, ## Atmospheric diffuse solar radiation, W/m2
        Leaf=LeafGasExchangeModule2(),    ## It is optional to define Leaf for this method. If no argument is passed in here, then default setting for Leaf is the default LeafGasExchangeModule2().
        Canopy=CanopyLayers(),  ## It is optional to define Canopy for this method. If no argument is passed in here, then default setting for Canopy is the default CanopyLayers(). Note that this may be important as it defines many canopy structure specific variables used in the calculations.
        CanopySolar=CanopyRadiation(),  ## It is optional to define CanopySolar for this method. If no argument is passed in here, then default setting for CanopySolar is the default CanopyRadiation(). Note that this may be important as it defines many canopy structural and optical parameters used in the calculations.
    ) -> Tuple[float]:

        swleaf, swveg, swvegsun, swvegsha = CanopySolar.calculate(LAI,SAI,clumping_factor,z,sza,swskyb,swskyd,Canopy=Canopy)

        # Calculate gas exchange per canopy multi-layer element (per leaf area basis)
        An_mle = np.zeros((Canopy.nlevmlcan, Canopy.nleaf))
        gs_mle = np.zeros((Canopy.nlevmlcan, Canopy.nleaf))
        Rd_mle = np.zeros((Canopy.nlevmlcan, Canopy.nleaf))
        
        # Vectorized calculation of Q (absorbed PPFD, mol PAR m-2 s-1) for all layers and leaves
        Q = 1e-6 * swleaf[Canopy.nbot:Canopy.ntop+1, :] * CanopySolar.J_to_umol  # absorbed PPFD, mol PAR m-2 s-1
        
        # Vectorized calculation of An, gs, Rd for all layers and leaves
        An, gs, Ci, Vc, Ve, Vs, Rd = Leaf.calculate(Q, leafTempC, Cs, O, airRH, fgsw)
        
        # Assign results to the respective arrays
        An_mle[Canopy.nbot:Canopy.ntop+1, :] = An
        gs_mle[Canopy.nbot:Canopy.ntop+1, :] = gs
        Rd_mle[Canopy.nbot:Canopy.ntop+1, :] = Rd

        # Determine leaf area index (LAI) per canopy layer based on prescribed distribution
        dlai = Canopy.cast_parameter_over_layers_betacdf(LAI,Canopy.beta_lai_a,Canopy.beta_lai_b)  # Canopy layer leaf area index (m2/m2)

        # Determine sunlit and shaded fractions for canopy layers
        (fracsun, kb, omega, avmu, betab, betad, tbi) = CanopySolar.calculateRTProperties(LAI,SAI,clumping_factor,z,sza,Canopy=Canopy)

        ## Calculate gas exchange per canopy layer (per ground area basis)
        ## - multiply the sunlit rates (per leaf area) by the sunlit lai, and the shaded rates (per leaf area) by the shaded lai
        An_ml = dlai*fracsun*An_mle[:,Canopy.isun] + dlai*(1-fracsun)*An_mle[:,Canopy.isha]
        gs_ml = dlai*fracsun*gs_mle[:,Canopy.isun] + dlai*(1-fracsun)*gs_mle[:,Canopy.isha]
        Rd_ml = dlai*fracsun*Rd_mle[:,Canopy.isun] + dlai*(1-fracsun)*Rd_mle[:,Canopy.isha]

        return An_ml, gs_ml, Rd_ml