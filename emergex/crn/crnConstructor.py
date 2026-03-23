#### Chemical Reaction Network Constructor ####
#                                             #
#               By: Colin Yancey              #
#             Created February 2025           #
#            Last Edited: 03/05/2026          #
#                 Schulman Lab                #
#           Johns Hopkins University          #
#                                             #
###############################################

from typing import Optional
import re
import numpy as np
import collections
import scipy.integrate as spi
import warnings

# Strings forbidden in component names due to their use in reaction string parsing or their potential inconsistent omission.
PREVENTED_STRINGS = [" ", ",", "<-", "->", "+"]

###############################################

def _getRxnStr(substrates: list, products: list, backwardRate: bool = False):
    """
    [Internal] Generate a standardized reaction string from substrates and products.
    
    Args:
        substrates: List of substrate names.
        products: List of product names.
        backwardRate: Determines arrow used ("<->" or "->").
    
    Returns:
        A formatted reaction string (e.g., "A + B -> C" or "A + B <-> C").
    """
    substrates.sort()
    products.sort()
    return " + ".join(substrates) + (" <-> " if backwardRate else " -> ") + " + ".join(products)


def _getSubstratesAndProducts(rxnStr: str, backwardRate: bool = False):
    """
    [Internal] Parse a reaction string to extract substrates and products.
    
    Args:
        rxnStr: A reaction string (e.g., "A + B -> C" or "A + B <-> C").
        backwardRate: Determines arrow used ("<->" or "->").
    
    Returns:
        Two lists, one for substrates and one for products.
    """
    if backwardRate:
        eqnParse = re.match(r"(.+)<->(.+)", rxnStr)
    else:
        eqnParse = re.match(r"(.+)->(.*)", rxnStr)
    
    if eqnParse:
        return eqnParse.group(1).split("+"), eqnParse.group(2).split("+")
    else:
        raise ValueError("The chemical equation is not in the correct format.")


class Component:
    """
    Creates a Component object representing an item in a solution. Has a name and a concentration.

    -- Functions --
    __init__ (Initializing function)
        Initializes a Component. Must provide a name and concentration.


    -- Properties --
    Name (IMMUTABLE): Name of the Component.
    Concentration: Concentration of the Component.
    """
    def __init__(self, compName, compConc):
        if any(char in compName for char in PREVENTED_STRINGS): # Checks for forbidden characters
            raise ValueError(f"Avoid using spaces, commas, +, <-, and -> characters in component names. Violating name: '{compName}'")

        self._Name = compName
        self.Concentration = compConc
    
    @property
    def Name(self):
        return self._Name

    def __str__(self):
        return self.Name


class Reaction:
    """
    Creates a Reaction object representing a chemical reaction. Has a forward rate and a backward rate, as well as a list of substrates and products.

    -- Functions --
    __init__ (Initializing function)
        Initializes a Reaction. Must provide a forward rate and either a reaction string (i.e., "A + B -> C") or substrate and product lists (i.e., ["A", "B"], ["C"]).


    -- Properties --
    ForwardRate: The forward rate of the reaction (i.e., rate at which C is produced in A + B -> C).
    BackwardRate: The optional backward rate of the reaction (i.e., the rate at which C is consumed in A + B <-> C, independent of the ForwardRate).
    Substrates (IMMUTABLE): The substrates in the reaction.
    Products (IMMUTABLE): The products in the reaction.
    RxnStr (IMMUTABLE): The reaction string corresponding to the defined reaction. Substrates and products may be reordered from how they were provided.


    -- NOTES --
    Stoichiometry must be given without the use of coefficients, as the coefficients will be interpretted as part of the unique string. To implement
    stoichiometry (i.e., 4A + 2B -> C), add multiple of the same substrate or product together in the provided string (i.e., A + A + A + A + B + B -> C),
    or the provided substrate or product table (i.e., ["A", "A", "A", "A", "B", "B"])
    """
    def __init__(self, forwardRate: float, rxnStr: Optional[str] = None, backwardRate: float = 0.0, substrates: Optional[list] = [], products: Optional[list] = []):
        if forwardRate < 0 or backwardRate < 0:
            raise ValueError("Rates must not be negative.")
        
        if rxnStr:
            substrates, products = _getSubstratesAndProducts(rxnStr, backwardRate != 0)
        
        if len(substrates)==0:
            raise ValueError("Missing substrates; each reaction must have a substrate.")
        
        substrates = [x.strip() for x in substrates]
        if len(products) > 0:
            products = [x.strip() for x in products]
        
        if collections.Counter(substrates) == collections.Counter(products):
            raise ValueError("This reaction appears to do nothing. Check your substrates and products.")
        
        self.ForwardRate = forwardRate
        self.BackwardRate = backwardRate
        self._RxnStr = _getRxnStr(substrates, products, backwardRate > 0.0)
        self._Substrates = substrates
        self._Products = products

    @property
    def RxnStr(self):
        return self._RxnStr
    
    @property
    def Substrates(self):
        return self._Substrates
    
    @property
    def Products(self):
        return self._Products
    
    # Returns the reaction string
    def __str__(self):
        return self.RxnStr


def _compareReactions(rxn1: Reaction, rxn2: Reaction):
    """
    Compare two Reaction objects to determine their relationship.
    
    Args:
        rxn1: The first Reaction object.
        rxn2: The second Reaction object.
    
    Returns:
        "SAME" if the reactions have identical substrates and products.
        "INVERTED" if one reaction is the reverse of the other.
        "DISTINCT" if the reactions are different.
    """
    rxn1Subs = collections.Counter(rxn1.Substrates)
    rxn1Prods = collections.Counter(rxn1.Products)
    rxn2Subs = collections.Counter(rxn2.Substrates)
    rxn2Prods = collections.Counter(rxn2.Products)

    if rxn1Subs == rxn2Subs and rxn1Prods == rxn2Prods:
        return "SAME"
    elif rxn1Subs == rxn2Prods and rxn1Prods == rxn2Subs:
        return "INVERTED"
    else:
        return "DISTINCT"



# Creates a chemical reaction network capable of storing reactions and components. Can run simulations once the network is properly assembled.
class ReactionNetwork:
    """
    Creates a ReactionNetwork object that contains all the reactions and their constituent components in a given chemical network. Can run 
    simulations of the network with optional network state persistence, and accepts modifications to reaction conditions in-between simulation runs.
    
    -- Functions --
    __init__ (Initializing function)
        No parameters required. Initializes an empty ReactionNetwork.

    addRxn:
        Adds a unique Reaction object to the ReactionNetwork.
    
    addRxns:
        Adds a list of Reaction objects to the ReactionNetwork.
    
    addComponent:
        Adds or overrides a Component object in the ReactionNetwork.
    
    addComponents:
        Adds a list of Component objects to the ReactionNetwork.
    
    disableModifying:
        Prevents modifications of the network. Useful when maintaining reaction network data structures.
    
    simulateReactionFn(simTime, continueFromLast=True, method="LSODA", simDataResolution=0, evalPts=[]):
        Simulates the reaction network over the specified time. Results are stored in SimResults.
    
    getCompiledCRNMatricies:
        Returns compiled matrices for ODE calculations.
    
    getCompiledReactionODEFunction:
        Returns the component name list and a derivative function for ODE integration.
    
    clearSimResults:
        Clears all simulation results.


    -- Properties --
    Reactions: Dictionary of Reaction objects in the network, keyed by their reaction strings.
    Components: Dictionary of Component objects in the network, keyed by their names.
    SimResults: List of simulation result dictionaries containing time-series data for each component.
    SimLastPoint: ComponentGroup containing the final concentrations from the last simulation.
    Modifiable: Indicates whether the network structure can still be modified.
    CompNameList: List of component names in the order used for ODE calculations.
    
    
    """
    def __init__(self):
        self._Reactions = {}
        self._Components = {}
        self._SimResults = []
        self._SimLastPoint = None
        self._Modifiable = True
        self._CompNameList = None


    @property
    def Reactions(self):
        return self._Reactions
    
    @property
    def Components(self):
        return self._Components
    
    @property
    def SimResults(self):
        return self._SimResults
    
    @property
    def SimLastPoint(self):
        return self._SimLastPoint
    
    @property
    def Modifiable(self):
        return self._Modifiable
    
    @property
    def CompNameList(self):
        return self._CompNameList


    # Checks to see if modifying the network is allowed, throws an error if not
    def _tryModifying(self):
        if not self._Modifiable:
            raise ValueError(
                "The reaction network structure has been locked, either from compiled " \
                "simulation data reliant on the current structure or due to external dependencies."
            )


    # Prevent modifications of the network. This is useful when using the CRN library to maintain reaction network data structures, as turning this on prevents the user from accidentally modifying the network and causing unintended behavior.
    def disableModifying(self):
        self._Modifiable = False

 
    #  Adds a unique Reaction object to ReactionNetwork.
    def addRxn(self, rxnObj: Reaction):
        self._tryModifying()
        for curRxn in self._Reactions.values():
            rxnRelation = _compareReactions(curRxn, rxnObj)
            if rxnRelation == "SAME":
                raise ValueError("This reaction already exists in the network.")
            elif rxnRelation == "INVERTED":
                raise ValueError("This reaction's inverse already exists in the network. Add a backward rate instead.")
        self._Reactions[rxnObj.RxnStr] = rxnObj
        for compName in set(rxnObj.Substrates+rxnObj.Products):
            if not compName in self._Components.keys():
                self.addComponent(Component(compName, 0.0), directFromRxn = True)


    # Adds a list of Reaction objects to ReactionNetwork.
    def addRxns(self, rxnObjList: list[Reaction]):
        for rxn in rxnObjList:
            self.addRxn(rxn)


    # Adds or overrides a Component object in the ReactionNetwork.
    def addComponent(self, componentObj: Component, directFromRxn: bool = False):
        self._tryModifying()
        if componentObj.Name == "Time":
            raise ValueError("'Time' is a protected name used to reference time stamps for a solution; please use a different component name.")
        elif not componentObj.Name in self._Components and not directFromRxn:
            warnings.warn(f"No reaction currently in the network consumes or produces the component '{componentObj.Name}'.", UserWarning)
        self._Components[componentObj.Name] = componentObj


    # Adds a list of Component objects to ReactionNetwork.
    def addComponents(self, compObjList: list[Component]):
        for comp in compObjList:
            self.addComponent(comp)


    def getCompiledCRNMatricies(self, pruneDisabledBackward: bool = True):
        compCount = len(self._Components)
        if not self._CompNameList:
            compNameList = list(self._Components.keys()) # Maintain a pre-set order of components when transferring the data into a linear vector for the ODE calculation, as to not mis-map output concentrations.
        else:
            compNameList = self._CompNameList

        compiledRateList = []
        compiledRxnMatrix = []
        compiledModifierMatrix = []

        def matrixAssembly(substrs, prods):
            subsList = [0]*(compCount)
            negSubsPosProdsList = [0]*(compCount)
            
            for substr in substrs:
                curInd = compNameList.index(substr)
                subsList[curInd] += 1
                negSubsPosProdsList[curInd] -= 1
            
            for prod in prods:
                curInd = compNameList.index(prod)
                negSubsPosProdsList[curInd] += 1
            
            compiledRxnMatrix.append(subsList)
            compiledModifierMatrix.append(negSubsPosProdsList)
        
        rxnNameList = [] # Pass along the list of reaction identifiers to keep track of the pre-set order
        for rxn in self._Reactions.values():
            compiledRateList.append(rxn.ForwardRate)
            rxnNameList.append((rxn.RxnStr, "ForwardRate"))
            matrixAssembly(rxn.Substrates, rxn.Products)

            if not pruneDisabledBackward or rxn.BackwardRate > 0:
                compiledRateList.append(rxn.BackwardRate)
                rxnNameList.append((rxn.RxnStr, "BackwardRate"))
                matrixAssembly(rxn.Products, rxn.Substrates)
        
        return compNameList, rxnNameList, compiledRateList, compiledRxnMatrix, compiledModifierMatrix
    
    
    def getCompiledReactionODEFunction(self):
        compNameList, _, compiledRateList, compiledRxnMatrix, compiledModifierMatrix = self.getCompiledCRNMatricies()
        compiledRateList = np.array(compiledRateList)
        compiledRxnMatrix = np.array(compiledRxnMatrix)
        compiledModifierMatrixTransposed = np.array(compiledModifierMatrix).T

        if np.max(compiledRxnMatrix) > 1: # Stoichiometries present > 1, so exponentiation is required
            def calculateDerivative(tVal, dVals):
                return np.matmul( # Distribute and sum the rates, with signs according to their status as a substrate or product of each reaction
                    compiledModifierMatrixTransposed,
                    np.multiply( # Multiply the kinetic rates by the np.prod result to obtain total rates for each reaction
                        compiledRateList,
                        np.prod( # Multiply concentrations of substrates consumed
                            np.power( # Distribute the substrates' concentrations into the sparse matrix format and exponentiate based on stoichiometry
                                dVals,
                                compiledRxnMatrix
                            ),
                            axis=1
                        )
                    )
                )
        else:
            compiledRxnMatrix = np.array(compiledRxnMatrix, dtype=bool)
            compiledRxnMatrixToggled = ~compiledRxnMatrix # Apply NOT filter

            def calculateDerivative(tVal, dVals):
                return np.matmul( # Distribute and sum the rates, with signs according to their status as a substrate or product of each reaction
                    compiledModifierMatrixTransposed,
                    np.multiply( # Multiply the kinetic rates by the np.prod result to obtain total rates for each reaction
                        compiledRateList,
                        np.prod( # Multiply concentrations of substrates consumed (all are to the power of one so we can use this trick to mimic np.power at a fraction of the compute cost)
                            compiledRxnMatrix * dVals + compiledRxnMatrixToggled,
                            axis=1
                        )
                    )
                )
            
        return compNameList, calculateDerivative
    

    # Simulates the reaction
    def simulateReactionFn(
        self,
        simTime: float,
        continueFromLast: bool = True,
        method: str = "LSODA",
        simDataResolution: int = 0,
        evalPts: list[float] = []
    ):
        """
        Run an ODE simulation of the reaction network.
        
        Integrates the system of ODEs defined by the reaction network using
        scipy's solve_ivp. Results are stored in SimResults and the final
        state is saved in SimLastPoint.
        
        Args:
            simTime: Total simulation time.
            continueFromLast: Chooses whether to continue from the last simulation's final state or from initial component concentrations.
            method: ODE solver method (default "LSODA" handles stiff systems well).
            simDataResolution: Number of evenly-spaced time points to record.
            evalPts: Explicit list of time points to record. Overrides simDataResolution.
        
        Note:
            Running a simulation locks the network structure (Modifiable becomes False).
        """
        if simDataResolution == 0 and len(evalPts) == 0:
            raise ValueError("'evalPts' must have at least one time entry or 'simDataresolution' must be > 0 to record simulation results.")
        elif len(evalPts) == 0:
            evalPts = np.linspace(0, simTime, simDataResolution)
        else:
            for x in evalPts:
                if x > simTime:
                    raise ValueError("All evaluated time points in 'evalPts' must be within the designated simulation time.")

        compNameList, reactionODEFn = self.getCompiledReactionODEFunction()
        if self._Modifiable:
            self.disableModifying()
        if not self._CompNameList:
            self._CompNameList = compNameList
        
        freshStart = False
        if not continueFromLast or not self._SimLastPoint:
            freshStart = True
            self._SimLastPoint = ComponentGroup({comp.Name: comp.Concentration for comp in self._Components.values()})

        rawSolution = spi.solve_ivp( # Run scipy's solve_ivp function to integrate the ODE system over time. It is recommended that the method chosen should be capable of handling stiff ODE systems (default should handle most cases)
            reactionODEFn,
            (0, simTime),
            [self._SimLastPoint._Components[compName] for compName in compNameList],
            method=method,
            t_eval=evalPts
        )

        self._SimResults.append({compNameList[i]: x for i,x in enumerate(rawSolution.y)})
        self._SimResults[-1]["Time"] = rawSolution.t + (0 if freshStart else self._SimResults[-2]["Time"][-1]) # If continuing from a prior simulation, increase the Time values based on previous simulation's last time point.
        self._SimLastPoint = ComponentGroup({compName: self._SimResults[-1][compName][-1] for compName in compNameList})


    # Allows a simulation result to have a modification made to the last simulation's concentration profile.
    # Useful if you want to simulate spiking a component in a solution (simulating a sudden manual/external adjustment in concentration)
    def adjustResultComponent(self, compName: str, compConc: float):
        if not self._SimLastPoint:
            raise ValueError("A prior simulation has not yet been recorded.")
        self._SimLastPoint._Components[compName] = compConc

    def clearSimResults(self, regainModifiability: bool = True):
        self._SimLastPoint = None
        self._SimResults = []
        self._Modifiable = regainModifiability
        self._CompNameList = None


# Implicitly creates a string of contents of component concentrations in the correct formatting while also maintaining the desired data structure.
class ComponentGroup:
    """
    Implicitly creates a string of contents of component concentrations in the correct formatting 
    while also maintaining the desired data structure. Used to store and display simulation state.

    -- Functions --
    __init__ (Initializing function)
        componentConcDict:
            Dictionary mapping component names (strings) to their concentrations (numbers).


    -- Properties --
    Components: Internal dictionary storing the component-concentration mapping.
    
    """
    def __init__(self, componentConcDict: dict[str, Component]):
        self._Components = componentConcDict

    def __str__(self):
        return "\n".join([
            f"{itm} = {conc:.4g}"
            for itm, conc in self._Components.items()
        ])