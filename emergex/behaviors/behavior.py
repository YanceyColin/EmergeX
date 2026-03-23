################# Behavior Data ###############
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from typing import Callable

from ..core.simulate import TimeSpan
from ..utils.base import (
    DynamicFunctionStoring,
    ComponentCompositesHandler
)
from ..utils.constants import (
    ALLOWED_BEHAVIORS
)

###############################################


# Describe a desired behavior for a set percentile-based time interval
class Behavior:
    def __init__(self, behaviorType: str, startBound: float = None, endBound: float = None, weight: float = 1.0, resolution: int = 20, requiredMatch: float = 1.0):
        behaviorType = behaviorType.upper()
        if behaviorType in ALLOWED_BEHAVIORS:
            self.BehaviorType = behaviorType
        else:
            raise ValueError(f"'{behaviorType}' is not recognized as a legal behavior type.")
        
        if requiredMatch <= 0.0:
            raise ValueError("Invalid matching point fraction. Behaviors must have a requiredMatch value greater than 0.0 (0%).")
        elif requiredMatch > 1.0:
            raise ValueError("Invalid matching point fraction. Behaviors must have a requiredMatch value less than or equal to 1.0 (100%).")
        
        if startBound is None:
            startBound = 0
        if endBound is None:
            endBound = float('inf')
        if endBound <= startBound:
            raise ValueError("The starting time of your behavior must be less than the ending time of your behavior.")

        self.StartBound = startBound
        self.EndBound = endBound

        self.Weight = weight
        self.Resolution = resolution
        self.PointCount = max(1, int(jnp.round(requiredMatch*resolution)))

        self.EvaluationPoints = jnp.linspace(self.StartBound, self.EndBound, self.Resolution)
    

    def isInBounds(self, timePoint: float):
        return jnp.logical_and(timePoint >= self.StartBound, timePoint <= self.EndBound)


    def __str__(self):
        endBoundLabel = "\u221E" if self.EndBound == float('inf') else f"{self.EndBound:.4g}"
        return f"{self.BehaviorType} Behavior: [{self.StartBound:.4g}, {endBoundLabel})"


# Assign a list of behaviors with a given and a normalization method (by default, it is the fraction of the component's initial concentration)
class BehaviorGroup(ComponentCompositesHandler, DynamicFunctionStoring):
    def __init__(self, compNames: str | list[str], behaviors: list[Behavior], normalizeFn: Callable[[float | int, dict], float], compositesAllowed: bool = False, weight: float = 1.0):
        ComponentCompositesHandler.__init__(self, compNames, compositesAllowed)

        self.Weight = weight
        
        self.normalizeValue = normalizeFn
        
        if len(behaviors) < 1:
            raise ValueError("Must include at least one behavior when optimizing for behaviors.")
        
        self.Behaviors = sorted(behaviors, key = self._getBehaviorOrderPriority)

        for i in range(len(self.Behaviors)-1):
            curBehavior = self.Behaviors[i]
            nextBehavior = self.Behaviors[i+1]
            if curBehavior.EndBound > nextBehavior.StartBound:
                raise ValueError(f"The following behaviors overlap in time and cannot coexist in this behavior group:\n  {curBehavior}\n  {nextBehavior}")
        
        self.EvaluationPoints = jnp.concatenate([behavior.EvaluationPoints for behavior in self.Behaviors])
        self.EvaluationIsHigh = jnp.concatenate([jnp.array([True if behavior.BehaviorType == "HIGH" else False for _ in range(behavior.Resolution)], dtype = bool) for behavior in self.Behaviors])
        
        allBehaviorWeight = sum([b.Weight for b in self.Behaviors])
        self.EvaluationPointWeights = jnp.concatenate([jnp.full(behavior.Resolution, behavior.Weight / allBehaviorWeight / behavior.Resolution) for behavior in self.Behaviors])
        
    def _getBehaviorOrderPriority(self, x):
            return x.StartBound



class BehaviorTimeCourse:
    def __init__(self, timeCourse: list[TimeSpan], behaviorGroups: list[BehaviorGroup], weight = 1.0):
        self.TimeCourse = timeCourse
        self.BehaviorGroups = behaviorGroups
        self.Weight = weight

        self.UniqueEvaluationPoints = jnp.sort(jnp.unique(jnp.concatenate([bG.EvaluationPoints for bG in self.BehaviorGroups]))) # Create a list of every unique evaluation point.
        self.EvaluationPointMappingGroups = [jnp.searchsorted(self.UniqueEvaluationPoints, bG.EvaluationPoints) for bG in self.BehaviorGroups] # Locate all the positions of the final evaluation points for each of the evaluation point groups.
        
        self.BehaviorGroupWeights = jnp.array([bG.Weight for bG in self.BehaviorGroups])
        self.BehaviorGroupWeights /= jnp.sum(self.BehaviorGroupWeights)
