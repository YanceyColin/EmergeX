######### Base Helper Functions/Classes #######
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 03/03/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from typing import Literal
import time

import jax
jax.config.update("jax_enable_x64", True)  # Maintain 64-bit precision.
import jax.numpy as jnp

import inspect
import re

from pathlib import Path
import dill

from ..crn import (
    Component,
    Reaction
)

###############################################

# The default property that is chosen when creating a parameter for a given class.
DEFAULT_VALUE_NAMES = {
    Component.__name__: "Concentration",
    Reaction.__name__: "ForwardRate"
}


def getComponentHosts(compName: str, compCandidates: list[str], delimiter: str = ":"):
    delimEscape = re.escape(delimiter)
    # Start of string/delimiter -> compName <- End of string/a delimiter
    pattern = f"(^|{delimEscape}){re.escape(compName)}($|{delimEscape})"
    
    return [s for s in compCandidates if re.search(pattern, s)]


def _getMinSecStr(timeVal: int | float):
    return f"{int(timeVal // 60)} min {(timeVal % 60):.3f} sec"


def _getValueName(parameterObject: Component | Reaction, valueName: Literal["ForwardRate", "BackwardRate", "Concentration", None]):
    paramType = type(parameterObject).__name__
    if not valueName:
        return DEFAULT_VALUE_NAMES[paramType]
    if isinstance(parameterObject, Component):
        if not valueName == "Concentration":
            print(f"Warning: Provided '{valueName}' as value name for a Component object. Reverted to '{DEFAULT_VALUE_NAMES[Component]}'.")
        return DEFAULT_VALUE_NAMES[Component]
    elif not valueName in ["ForwardRate", "BackwardRate"]:
        print(f"Warning: Provided '{valueName}' as a value name for a Reaction object. Reverted to '{DEFAULT_VALUE_NAMES[Reaction]}'.")
        return DEFAULT_VALUE_NAMES[Reaction]
    return valueName


def _getParameterName(parameterObject):
    if isinstance(parameterObject, Component):
        return parameterObject.Name
    else:
        return parameterObject.RxnStr


def _modifyBySubstitute(listOfInterest, newInd, newVal, modValMin):
    return listOfInterest.at[newInd].set(jnp.maximum(modValMin, newVal))

def _modifyByAdd(listOfInterest, newInd, newVal, modValMin):
    return listOfInterest.at[newInd].set(jnp.maximum(modValMin, listOfInterest[newInd] + newVal))

def _modifyBySubtract(listOfInterest, newInd, newVal, modValMin):
    return listOfInterest.at[newInd].set(jnp.maximum(modValMin, listOfInterest[newInd] - newVal))


class DynamicFunctionStoring:
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sourceFuncData"] = []
        for name, val in inspect.getmembers(self):
            if callable(val):
                if isinstance(val, type(self.__dict__.get(name))):
                    # Exception for classes, since these are handled by dill properly.
                    if not (inspect.isfunction(val) or inspect.ismethod(val)):
                        continue
                    valSourceCode = inspect.getsource(val)
                    freeVarData = {}
                    if val.__closure__ is not None:
                        freeVarData = {
                            freeVarName: val.__closure__[i].cell_contents
                            for i, freeVarName in enumerate(val.__code__.co_freevars)
                        }
                    state["_sourceFuncData"].append((valSourceCode.strip(), freeVarData))

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for funcData in self._sourceFuncData:
            funcSource, freeVarData = funcData
            for freeVarName, freeVarVal in freeVarData.items():
                self.__dict__[freeVarName] = freeVarVal

            exec(funcSource, self.__dict__)



class ComponentCompositesHandler:
    """
    Base class for handling component composites in concentration lookups.
    
    Subclasses or instances should define `normalizeValue(normVal, compDict)` 
    either by:
    - Assigning a function to `self.normalizeValue` in `__init__`
    - Overriding the method in a subclass
    """
    def __init__(self, compNames: str | list[str], compositesAllowed: bool):
        if isinstance(compNames, str):
            self.ComponentNames = [compNames]
        else:
            self.ComponentNames = compNames
        self.CompositesAllowed = compositesAllowed

        self.RelevantComponentIndices = {}

    
    def getRelevantComponentIndices(self, allCompNames):
        allCompNameKey = id(allCompNames)
        if not allCompNameKey in self.RelevantComponentIndices:
            compIndices = []
            if self.CompositesAllowed:
                for baseCompName in self.ComponentNames:
                    for compName in getComponentHosts(baseCompName, allCompNames):
                        compIndices.append(allCompNames.index(compName))
            else:
                for compName in self.ComponentNames:
                    compIndices.append(allCompNames.index(compName))
            
            self.RelevantComponentIndices[allCompNameKey] = compIndices
            
        return self.RelevantComponentIndices[allCompNameKey]


    def getNormalizeVMAP(self, compDict):
        return jax.vmap(lambda normVal: self.normalizeValue(normVal, compDict))



class DataStore:
    def save(self, fileLocation: Path, fileName: str):
        self.SavedTimestamp = time.time()
        fullResName = fileName + ".pkl"
        with open(fileLocation / fullResName, 'wb') as f:
            dill.dump(
                self,
                f
            )
    
    @classmethod
    def load(cls, filePath: Path):
        with open(filePath, 'rb') as f:
            data = dill.load(f)
        if not isinstance(data, cls):
            print(f"Expected instance of {cls.__name__} or subclass; got {type(data).__name__}.")
        return data
