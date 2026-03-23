############### Time Course Info ##############
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################


from dataclasses import dataclass, field
from typing import Optional, Literal, Callable

from ..crn import (
    Component,
    Reaction
)
from ..utils.constants import (
    ALLOWED_INTERRUPTIONS
)
from ..utils.base import (
    DynamicFunctionStoring,
    
    _modifyBySubstitute,
    _modifyByAdd,
    _modifyBySubtract
)
from ..utils.base import (
    _getValueName,
    _getParameterName
)

###############################################

# At time breaks, the reaction conditions of a CRN may change. Interruption objects represent these changes.
class Interruption(DynamicFunctionStoring):
    def __init__(
        self,
        interruptionObject: Component | Reaction,
        valueObtain: int | float | Callable[[dict, dict], float | int],
        valueName: Optional[Literal["ForwardRate", "BackwardRate", "Concentration"]] = None,
        interruptionType: Literal["SUBSTITUTE", "ADD", "SUBTRACT"] = "SUBSTITUTE"
    ):
        self.ValueName = _getValueName(interruptionObject, valueName)
        self.Name = _getParameterName(interruptionObject)
        self.InterruptionType = interruptionType

        if self.InterruptionType.upper() not in ALLOWED_INTERRUPTIONS:
            raise ValueError(f"InterruptionType must be one of {ALLOWED_INTERRUPTIONS}")
        
        if callable(valueObtain):
            self.getValue = valueObtain
        else:
            self.DefaultValue = valueObtain
    
        if interruptionType == "SUBSTITUTE":
            self.getModifiedList = _modifyBySubstitute
        elif interruptionType == "ADD":
            self.getModifiedList = _modifyByAdd
        elif interruptionType == "SUBTRACT":
            self.getModifiedList = _modifyBySubtract

    def getValue(self, _, __):
        return self.DefaultValue


# Represents a period of time where a CRN runs independent of reaction condition changes or sudden concentration
# jumps from external manipulation. A list of TimeSpan objects represents a full time course for a simulation
# run by the CRNOptimizer class. TimeSpan objects can have a list of Interruption objects that represent
# the changes made to the conditions of the experiment in between TimeSpan intervals.
@dataclass
class TimeSpan:
    Time: float
    Interruptions: list[Interruption] = field(default_factory=list)
