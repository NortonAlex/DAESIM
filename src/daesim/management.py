"""
Management model class: Includes differential equations, calculators, and parameters
"""

import numpy as np
from typing import Tuple, Callable
from attrs import define, field

@define
class ManagementModule:
    """
    Calculator of management
    """

    # Class parameters

    ## TODO: Shift the parameters below to a separate Management module
    plantingDay: float = field(default=30)  ## day that planting (sowing) occurs, in units of ordinal day of year (DOY)
    harvestDay: float = field(default=235)  ## day that harvest occurs, in units of ordinal day of year (DOY)
    frequPlanting: float = field(default=0)  ## frequency of planting (days-1)
    propPhPlanting: float = field(default=0)  ## fraction of planted biomass that is photosynthetic (almost always equal to 0, as it is seeds that are planted, which have no photosynthetic biomass. Although, this parameter allows planting of seedlings which have some photosynthetic biomass as soon as they're planted). Modification: This variable was previously defined using "frequPlanting", which didn't match with the units or its definition.
    maxDensity: float = field(default=40)  ## number of individual plants per m2
    plantingRate: float = field(default=40)  ## number of individual plants per m2 per day (# plants m-2 d-1)
    plantWeight: float = field(default=0.0009)  ## mass of individual plant at sowing (Question: units? kg? g?)
    propPhHarvesting: float = field(default=0.3)  ## proportion of Photosynthetic_Biomass harvested
    propNPhHarvest: float = field(default=0.4)  ## proportion of Non_Photosynthetic_Biomass harvested
    PhHarvestTurnoverTime: float = field(default=1)  ## Turnover time (days). Modification: This is a new parameter required to run in this framework. It does not exist in the Stella code, but it is needed as a replacement for "DT"
    NPhHarvestTurnoverTime: float = field(default=1)  ## Turnover time (days). Modification: This is a new parameter required to run in this framework. It does not exist in the Stella code, but it is needed as a replacement for "DT"

    ## TODO: Shift the parameters below to a separate Management module
    x_plantingDay: float = field(default=30)  ## day that planting (sowing) occurs, in units of ordinal day of year (DOY)
    x_harvestDay: float = field(default=235)  ## day that harvest occurs, in units of ordinal day of year (DOY)
    x_frequPlanting: float = field(default=0)  ## frequency of planting (days-1)
    x_propPhPlanting: float = field(default=0)  ## fraction of planted biomass that is photosynthetic (almost always equal to 0, as it is seeds that are planted, which have no photosynthetic biomass. Although, this parameter allows planting of seedlings which have some photosynthetic biomass as soon as they're planted). Modification: This variable was previously defined using "frequPlanting", which didn't match with the units or its definition.
    x_maxDensity: float = field(default=40)  ## number of individual plants per m2
    x_plantingRate: float = field(default=40)  ## number of individual plants per m2 per day (# plants m-2 d-1)
    x_plantWeight: float = field(default=0.0009)  ## mass of individual plant at sowing (Question: units? kg? g?)
    x_propPhHarvesting: float = field(default=0.3)  ## proportion of Photosynthetic_Biomass harvested
    x_propNPhHarvest: float = field(default=0.4)  ## proportion of Non_Photosynthetic_Biomass harvested
    x_PhHarvestTurnoverTime: float = field(default=1)  ## Turnover time (days). Modification: This is a new parameter required to run in this framework. It does not exist in the Stella code, but it is needed as a replacement for "DT"
    x_NPhHarvestTurnoverTime: float = field(default=1)  ## Turnover time (days). Modification: This is a new parameter required to run in this framework. It does not exist in the Stella code, but it is needed as a replacement for "DT"

    def calculate(
        self,
    ) -> Tuple[float]:
        test = 0.0

        return test