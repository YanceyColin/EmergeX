############### Behavior Manager ##############
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
import warnings

from .behavior import BehaviorTimeCourse
from ..core.crnData import (
    CRNInfoHandler
)
from ..core.simulate import (
    EvaluationPointsHandler,
    CRNSimulationRunner
)
from ..optimization.optimize import (
    CRNOptimizationFramework
)
from ..utils.helpers import (
    ManagerClass
)

###############################################


class OptimizeBehaviorsManager(ManagerClass):
    def __init__(self, crnInfo: CRNInfoHandler, behaviorTimeCourses: list[BehaviorTimeCourse], highMin:float = 0.8, boundMult:float = 8, stochasticity = 0.02):
        self.CRNInfo = crnInfo
        self.BehaviorTimeCourses = behaviorTimeCourses

        self.NormalizedWeights = jnp.array([bTC.Weight for bTC in behaviorTimeCourses])
        self.NormalizedWeights /= jnp.sum(self.NormalizedWeights)

        self.EvaluationPointsHandlers = [EvaluationPointsHandler(behaviorTimeCourseObj.UniqueEvaluationPoints) for behaviorTimeCourseObj in self.BehaviorTimeCourses]

        # If true that all resolutions match point counts, then no specialized point matching necessary
        self.NeedsStochasticity = not all(
            behavior.PointCount == behavior.Resolution
            for bTC in self.BehaviorTimeCourses
            for bG in bTC.BehaviorGroups
            for behavior in bG.Behaviors
        )
        if self.NeedsStochasticity:
            warnings.warn(
                f"A behavior has been detected to have a point count different from its resolution (likely from providing requiredMatch).\n" \
                f"  Using a stochasticity of {stochasticity}. The higher the stochasticity, the less the bias toward cost function evaluations that performed well.",
                UserWarning
            )
            self.Stochasticity = stochasticity
        
        if highMin < 0.2 or highMin > 0.99:
            raise ValueError("The minimum HIGH threshold must be set to a value between 0.2 and 0.99.")
        if boundMult < 2:
            raise ValueError("Bound multiple must be set to a value >= 2.")
        
        # Memoize all the precursor math for attached functions
        self._HighMin = highMin
        self._LowMax = self._HighMin/boundMult
        
        self._CoefLow = 2*(self._LowMax - 1)
        self._DisplaceLow = self._LowMax - self._CoefLow*jnp.log(1-self._LowMax)

        self._DivHigh = 1 - self._HighMin
        self._CoefHigh = 2*self._HighMin
        self._DisplaceHigh = self._DivHigh + self._CoefHigh*jnp.log(self._HighMin)
    
    def getHighMin(self):
        return self._HighMin

    def getLowMax(self):
        return self._LowMax
    
    def getBFLow(self, x):
        return jnp.where(
            x <= self._LowMax,
            x**2/self._LowMax,
            jnp.where(
                x < 1,
                self._CoefLow*jnp.log(1 - x) + self._DisplaceLow,
                1e6  # This value (1.0) should never be reached under normal use cases.
            )
        )
    
    def getBFHigh(self, x):
        return jnp.where(
            x <= 0,
            1e6,  # This value (0.0) should never be reached under normal use cases.
            jnp.where(
                x < self._HighMin,
                -self._CoefHigh*jnp.log(x) + self._DisplaceHigh,
                (1 - x)**2/self._DivHigh
            )
        )
    
    def getMinimumSum(self, pointCosts, pointCount: int):
        resolution = pointCosts.shape[0]
        selectionLogits = -pointCosts / self.Stochasticity # More stochasticity = flatter cost landscape, cost incorporates more of an average point blend but you get a smoother derivative landscape.

        def selectionStep(_, loopState):
            currentRemainingSelectionMass, currentMinimumCostSum = loopState
            maskedLogits = selectionLogits + jnp.log(currentRemainingSelectionMass + 1e-12)

            # Differentiable "pick one more low-cost point" distribution.
            selectionProbabilities = jax.nn.softmax(maskedLogits)

            # Add expected selected cost for this step.
            updatedMinimumCostSum = currentMinimumCostSum + jnp.sum(selectionProbabilities * pointCosts)

            # Reduce future probability mass for points already selected now.
            updatedRemainingSelectionMass = currentRemainingSelectionMass * (1.0 - selectionProbabilities)

            return updatedRemainingSelectionMass, updatedMinimumCostSum

        remainingSelectionMass = jnp.ones(resolution)
        minimumCostSum = jnp.array(0.0, dtype=pointCosts.dtype)

        # Rolls up a for loop that iterates over point counts to 
        _, minimumCostSum = jax.lax.fori_loop(0, pointCount, selectionStep, (remainingSelectionMass, minimumCostSum))
        return minimumCostSum / pointCount

    def getCostFn(self, optimizerFrameworkObj: CRNOptimizationFramework, simulationRunnerObj: CRNSimulationRunner):
        crnInfo = self.CRNInfo
        bfLow = self.getBFLow
        bfHigh = self.getBFHigh
        normedWeights = self.NormalizedWeights
        behaviorTimeCourses = self.BehaviorTimeCourses

        precomputedTimeCourseData = []
        for i, behaviorTimeCourseObj in enumerate(self.BehaviorTimeCourses):
            timeDomainData = self.EvaluationPointsHandlers[i].parseEvalPts(behaviorTimeCourseObj.TimeCourse)
            precomputedTimeCourseData.append(timeDomainData)

        if self.NeedsStochasticity:
            def getBehaviorGroupCost(behaviorGroup, resultOfInterest):
                groupCost = 0
                startInd = 0
                for behavior in behaviorGroup.Behaviors:
                    endInd = startInd + behavior.Resolution
                    segmentedCosts = jax.vmap(bfHigh if behavior.BehaviorType == "HIGH" else bfLow)(resultOfInterest[startInd : endInd])
                    groupCost += self.getMinimumSum(segmentedCosts, behavior.PointCount) * behavior.Weight
                    startInd = endInd

                return groupCost / sum(b.Weight for b in behaviorGroup.Behaviors)
        else:
            def getBehaviorGroupCost(behaviorGroup, resultOfInterest):
                lowCosts = jax.vmap(bfLow)(resultOfInterest)
                highCosts = jax.vmap(bfHigh)(resultOfInterest)
                return jnp.sum(
                    jnp.dot(
                        jnp.where(
                            behaviorGroup.EvaluationIsHigh,
                            highCosts,
                            lowCosts
                        ),
                        behaviorGroup.EvaluationPointWeights
                    )
                )
        
        
        def getAllBehaviorCost(y):
            initCondInfo = optimizerFrameworkObj.getStartingConditions(y, crnInfo)

            behaveCost = 0
            for i, behaviorTimeCourseObj in enumerate(behaviorTimeCourses):
                results = simulationRunnerObj.runTimeCourse(precomputedTimeCourseData[i], crnInfo, initCondInfo)
                courseCost = 0

                for j, mappingGroup in enumerate(behaviorTimeCourseObj.EvaluationPointMappingGroups):
                    behaviorGroup = behaviorTimeCourseObj.BehaviorGroups[j]
                    
                    resultOfInterest = behaviorGroup.getNormalizeVMAP(initCondInfo.ComponentsDictionary)(
                        jnp.sum(
                            jnp.atleast_2d(
                                results[mappingGroup, behaviorGroup.getRelevantComponentIndices(crnInfo.CompNameList)]
                            ),
                            axis = 0
                        )
                    )
                    courseCost += getBehaviorGroupCost(behaviorGroup, resultOfInterest) * behaviorTimeCourseObj.BehaviorGroupWeights[j]
                    
                behaveCost += courseCost * normedWeights[i]
            return behaveCost
        
        return getAllBehaviorCost