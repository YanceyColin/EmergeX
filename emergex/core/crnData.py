######## Chemical Reaction Network Data #######
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from dataclasses import dataclass

import jax.numpy as jnp

from ..crn import ReactionNetwork

###############################################

@dataclass
class InitialConditionInfo:
    RateList: jnp.array
    ConcentrationList: jnp.array
    ReactionsDictionary: dict
    ComponentsDictionary: dict

class CRNInfoHandler:
    def __init__(self, crn: ReactionNetwork):
        self.CRN = crn
        
        # Need to keep track of all rates, even ones initially set to zero, incase there is a linked parameter or interruption which modifies a zero backward rate.
        compNameList, rxnNameList, rateList, rxnMatrix, modifierMatrix = self.CRN.getCompiledCRNMatricies(pruneDisabledBackward = False)
        crn.disableModifying()

        self.CompNameList = compNameList
        self.RxnNameList = rxnNameList
        self.ReactionMatrix = jnp.array(rxnMatrix)
        self.ModifierMatrixTransposed = jnp.array(modifierMatrix).T

        self.InterpolatedReactions = range(len(rxnNameList))
        self.InterpolatedComponents = range(len(compNameList))

        rateList = jnp.array(rateList)
        concList = jnp.array([self.CRN.Components[compName].Concentration for compName in compNameList])

        self.DefaultInitialConditionInfo = InitialConditionInfo(
            rateList,
            concList,
            {rxnNameList[i]: rateList[i] for i in self.InterpolatedReactions},
            {compNameList[i]: concList[i] for i in self.InterpolatedComponents}
        )
    
    def convertTimeCourseResultToDict(self, results: jnp.array):
        return {compName: results[:, compIdx] for compIdx, compName in enumerate(self.CompNameList)}