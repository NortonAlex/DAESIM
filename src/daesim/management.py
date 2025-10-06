"""
Management model class: Includes parameters to define management
"""

from dataclasses import dataclass, field
from typing import Union, List

@dataclass
class ManagementModule:
    """
    Management class
    """

    # Class parameters
    cropType: str = field(default="Wheat")  ## crop type
    sowingDays: Union[int, List[int]] = field(default_factory=list)  ## day that sowing occurs, in units of ordinal day of year (DOY), as a list. If there is no sowing, set sowingDay=None
    sowingYears: Union[int, List[int]] = field(default_factory=list)  ## day that harvest occurs, in units of ordinal day of year (DOY), as a list. If there is no harvest, set harvestDay=None
    harvestDays: Union[int, List[int]] = field(default_factory=list)  ## day that harvest occurs, in units of ordinal day of year (DOY), as a list. If there is no harvest, set harvestDay=None
    harvestYears: Union[int, List[int]] = field(default_factory=list)  ## year that harvest occurs, as a list whose Values must correspond to those set in harvestDays. If there is no harvest, set harvestYears=None
    sowingRate: float = field(default=80)    ## seed sowing rate at start of season (kg ha-1)
    sowingDepth: float = field(default=0.03)  ## seed sowing depth (m)
    propHarvestSeed: float = field(default=1.0)  ## proportion of seed (grain) carbon pool removed at harvest
    propHarvestLeaf: float = field(default=0.9)  ## proportion of seed (grain) carbon pool removed at harvest
    propHarvestStem: float = field(default=0.7)  ## proportion of seed (grain) carbon pool removed at harvest
    propPhHarvesting: float = field(default=0.3)  ## proportion of Photosynthetic_Biomass harvested
    propNPhHarvest: float = field(default=0.4)  ## proportion of Non_Photosynthetic_Biomass harvested
    PhHarvestTurnoverTime: float = field(default=1)  ## Turnover time (days). Modification: This is a new parameter required to run in this framework. It does not exist in the Stella code, but it is needed as a replacement for "DT"
    NPhHarvestTurnoverTime: float = field(default=1)  ## Turnover time (days). Modification: This is a new parameter required to run in this framework. It does not exist in the Stella code, but it is needed as a replacement for "DT"
    propTillage: float = field(default=0.5)  ## Management module: propTillage=intensityTillage/10 (ErrorCheck: in the Stella code propTillage=intensityTillage/10 but in the documentation propTillage=(intensityTillage*9)/(5*10), why?)
    propHarvPhLeft: float = field(default=0.1)  ## the percentage of photosynthetic biomass left after harvesting
    propHarvNPhLeft: float = field(default=0.8)  ## the percentage of non-photosynthetic biomass left after harvesting

    def __post_init__(self):
        """Ensure key attributes are always lists."""
        for attr in ["sowingDays", "sowingYears", "harvestDays", "harvestYears"]:
            val = getattr(self, attr)
            # If it's an int (or a single scalar), wrap it in a list
            if isinstance(val, int):
                setattr(self, attr, [val])
            # If it's None, replace with an empty list
            elif val is None:
                setattr(self, attr, [])
            # If it’s already a list, leave it alone
            elif not isinstance(val, list):
                # Optionally, catch unexpected types
                setattr(self, attr, list(val))