################ CRN Parameters ###############
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from typing import Literal, Optional, Callable

from ..crn import (
    Reaction,
    Component,
    _getRxnStr
)
from ..utils.base import (
    _getValueName,
    _getParameterName,
    DynamicFunctionStoring
)

###############################################


class FreeParameter:
    def __init__(
        self,
        parameterObject: Component | Reaction,
        valueName: Optional[Literal["ForwardRate", "BackwardRate", "Concentration"]] = None,
        lowerBound: float = None,
        upperBound: float = None
    ):
        self.ValueName = _getValueName(parameterObject, valueName)
        self.ParameterObject = parameterObject

        if isinstance(parameterObject, Component):
            self.Name = parameterObject.Name
            self.DefaultValue = parameterObject.Concentration
            self.StrRep = self.Name
        else:
            self.Name = parameterObject.RxnStr
            self.DefaultValue = parameterObject.ForwardRate if self.ValueName == "ForwardRate" else parameterObject.BackwardRate
            if self.ValueName == "ForwardRate": # Orient the direction of the string output to match the rate of the free parameter in question.
                subs, prods = parameterObject.Substrates, parameterObject.Products
            else:
                subs, prods = parameterObject.Products, parameterObject.Substrates
            self.StrRep = _getRxnStr(subs, prods, backwardRate = False)

        self.LowerBound = lowerBound
        self.UpperBound = upperBound
        self._Modifiable = True
    
    def tryModifying(self):
        if not self._Modifiable:
            raise ValueError("This free parameter is no longer modifiable. Modifications must be made before creating dependence on this object.")

    def changeDefaultValue(self, newVal):
        self.tryModifying()
        if (self.LowerBound and self.LowerBound > newVal) or (self.UpperBound and self.UpperBound < newVal):
            raise ValueError(f"Attempted to set default value of {self.Name} to {self.DefaultValue:.4g}, outside of provided bound(s).")
        else:
            self.DefaultValue = newVal
    
    def disableModifying(self):
        self._Modifiable = False

    def __str__(self):
        return self.StrRep


# Assign a relationship between a parameter and the rest of the reaction network's initial conditions (by default, it simply clones its reference's relevant value of interest).
class LinkedParameter(DynamicFunctionStoring):
    def __init__(
        self,
        parameterObject: Component | Reaction,
        valueName: Optional[Literal["ForwardRate", "BackwardRate", "Concentration"]] = None,
        refName: Optional[str] = None,
        conversionFn: Optional[Callable[[dict, dict], float | int]] = None
    ):
        self.ValueName = _getValueName(parameterObject, valueName)
        self.Name = _getParameterName(parameterObject)

        if refName != None and conversionFn is None:
            self.ReferenceName = refName
            if self.ValueName in ["ForwardRate", "BackwardRate"]:
                self.applyConversion = self._convertRates
            else:
                self.applyConversion = self._convertConcs
        elif conversionFn != None:
            self.applyConversion = conversionFn
        else:
            raise ValueError(f"Must provide either a reference name for the linked parameter '{self.Name}' or a custom conversion function, cannot be both or neither.")
        
    def _convertRates(self, rates, _):
        return rates[(self.ReferenceName, self.ValueName)]
    
    def _convertConcs(self, _, concs):
        return concs[self.ReferenceName]
