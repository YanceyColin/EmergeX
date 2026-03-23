############### CRN Optimization ##############
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

import jax
jax.config.update("jax_enable_x64", True) # Maintain 64-bit precision.
import jax.numpy as jnp
import numpy as np

from equinox import filter_jit
import optax

from typing import Optional, Union
import time
from datetime import datetime

from .parameters import (
    FreeParameter,
    LinkedParameter
)
from ..core.crnData import (
    CRNInfoHandler,
    InitialConditionInfo
)
from ..utils.base import _getMinSecStr

###############################################

class CRNOptimizationFramework:
    def __init__(
        self,
        parameters: list[FreeParameter],
        linkedParameters: Optional[list[LinkedParameter]] = None,
        rateBoundFactor: float = 1e3,
        concentrationBoundFactor: float = 1e3,
        concentrationUnits: str = "nM",
        timeUnits: str = "sec"
    ):
        checkedRxns = []
        checkedComps = []
        if len(parameters) < 1:
            raise ValueError("No parameters were provided to the optimization framework!")
        for param in parameters:
            param.disableModifying()
            if param.ValueName == "Concentration":
                if param.Name in checkedComps:
                    raise AttributeError(f"Each free parameter must be a unique component concentration or reaction rate. The component '{param.Name}' was found to be listed at least twice.")
                checkedComps.append(param.Name)
            else:
                if (param.Name, param.ValueName) in checkedRxns:
                    raise AttributeError(f"Each free parameter must be a unique component concentration or reaction rate. The {"forward rate " if param.ValueName == "ForwardRate" else "backward rate"} of the reaction '{param.Name}' was found to be listed at least twice.")
                checkedRxns.append((param.Name, param.ValueName))
        
        self.Parameters = parameters
        self.LinkedParameters = [] if linkedParameters is None else linkedParameters

        # RateBoundFactor and ConcentrationBoundFactor define optimization search space as [initialValue / factor, initialValue * factor]
        self.RateBoundFactor = rateBoundFactor
        self.ConcentrationBoundFactor = concentrationBoundFactor
        
        self.ConcentrationUnits = concentrationUnits
        self.TimeUnits = timeUnits

        relevantBoundFactorList = [concentrationBoundFactor if param.ValueName == "Concentration" else rateBoundFactor for param in self.Parameters]

        self.LnDefaultValueList = jnp.array([jnp.log(param.DefaultValue) for param in self.Parameters])
        self.LnLowerBoundList = jnp.array([jnp.log(param.DefaultValue/relevantBoundFactorList[i]) if param.LowerBound is None else jnp.log(param.LowerBound) for i, param in enumerate(self.Parameters)])
        self.LnUpperBoundList = jnp.array([jnp.log(param.DefaultValue*relevantBoundFactorList[i]) if param.UpperBound is None else jnp.log(param.UpperBound) for i, param in enumerate(self.Parameters)])

    def adjustParameterValue(self, param: Union[FreeParameter, LinkedParameter], paramValue, crnInfo: CRNInfoHandler, rateList: jnp.array, concList: jnp.array):
        if param.ValueName in ["ForwardRate", "BackwardRate"]:
            rateList = rateList.at[crnInfo.RxnNameList.index((param.Name, param.ValueName))].set(paramValue)
        elif param.ValueName == "Concentration":
            concList = concList.at[crnInfo.CompNameList.index(param.Name)].set(paramValue)
        
        return rateList, concList

    def getStartingConditions(self, paramValues: jnp.ndarray, crnInfo: CRNInfoHandler):
        modRateList, modConcList = crnInfo.DefaultInitialConditionInfo.RateList, crnInfo.DefaultInitialConditionInfo.ConcentrationList

        for i, optParam in enumerate(self.Parameters):
            modRateList, modConcList = self.adjustParameterValue(
                optParam,
                jnp.exp(paramValues[i]),
                crnInfo,
                modRateList,
                modConcList
            )
        modRxnDict = {crnInfo.RxnNameList[i]: modRateList[i] for i in crnInfo.InterpolatedReactions}
        modCompDict = {crnInfo.CompNameList[i]: modConcList[i] for i in crnInfo.InterpolatedComponents}
        for linkedParam in self.LinkedParameters:
            modRateList, modConcList = self.adjustParameterValue(
                linkedParam,
                linkedParam.applyConversion(
                    modRxnDict,
                    modCompDict
                ),
                crnInfo,
                modRateList,
                modConcList
            )
        return InitialConditionInfo(
            modRateList,
            modConcList,
            {crnInfo.RxnNameList[i]: modRateList[i] for i in crnInfo.InterpolatedReactions},
            {crnInfo.CompNameList[i]: modConcList[i] for i in crnInfo.InterpolatedComponents}
        )
    
    def printCurrentOptimizationState(self, iteration: int, totalExpectedIterations: int, loss, iterTime, barMaxSize = 20):
        barLen = (iteration * barMaxSize) // totalExpectedIterations
        print(
            f"[{"#"*barLen + "-"*(barMaxSize - barLen)}] Iteration {iteration:4d} / {totalExpectedIterations}: Loss = {loss:.8f}\n" \
            f"  Iteration time: {iterTime:.3f} sec"
        )
    
    def optimize(self, costFn, iterationCount: int = 1000, gradThreshold: float = 1e-6, optaxOptimizerObj = None, callbackFrequency: Optional[int] = None):
        curTime = time.time()
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(
                learning_rate=0.05,
                b1=0.9, # Momentum term
                b2=0.999
            ) if optaxOptimizerObj is None else optaxOptimizerObj
        )
        
        curVals = self.LnDefaultValueList
        optState = optimizer.init(curVals)
        lowerBounds, upperBounds = self.LnLowerBoundList, self.LnUpperBoundList

        @filter_jit
        def step(vals, optState):
            boundedVals = jnp.clip(vals, lowerBounds, upperBounds)
            loss, grads = jax.value_and_grad(costFn)(boundedVals)
            updates, optState = optimizer.update(grads, optState, boundedVals)
            vals = optax.apply_updates(boundedVals, updates)
            return vals, optState, loss, grads

        # Optimization loop with gradient-based and boundary-based early stopping
        losses = []
        paramHistory = []

        # Run optimization
        print("Starting optimization...")
        iterationList = np.arange(iterationCount) + 1
        callbackPts = [] if callbackFrequency is None or callbackFrequency <= 0 else [i for i in iterationList if i%callbackFrequency == 0]
        for i in iterationList:
            paramHistory.append(jnp.exp(curVals))

            iterStart = time.time()
            curVals, optState, loss, grads = step(curVals, optState)
            iterTime = time.time() - iterStart
            
            losses.append(float(loss))

            gradNorm = jnp.linalg.norm(grads)
            if gradNorm < gradThreshold:
                print(f"Converged at Iteration {i}: Gradient norm {gradNorm:.2e} is below threshold {gradThreshold:.2e}")
                break
            
            if i in callbackPts:
                self.printCurrentOptimizationState(i, iterationCount, loss, iterTime)
        
        return OptimizerRunResult(
            self,
            i,
            iterationCount,
            lossHist = losses,
            paramHist = paramHistory,
            startTime = curTime,
            finishTime = time.time()
        )


class OptimizerRunResult:
    def __init__(
        self,
        crnOptimizationFrameworkObj: CRNOptimizationFramework,
        lastIteration: int,
        iterationMax: int,
        lossHist: jnp.ndarray,
        paramHist: jnp.ndarray,
        startTime: float,
        finishTime: float
    ):
        self.CRNOptimizationFrameworkObject = crnOptimizationFrameworkObj

        self.LastIteration = lastIteration
        self.IterationMaximum = iterationMax

        self.LossHistory = lossHist
        self.ParameterHistory = paramHist

        self.StartTime = startTime
        self.FinishTime = finishTime

        self.StartDateTime = datetime.fromtimestamp(self.StartTime)
        self.FinishDateTime = datetime.fromtimestamp(self.FinishTime)

        self.TimeDelta = self.FinishTime - self.StartTime

        self.OutputStr = f"\n{"="*60}\n{" "*19}OPTIMIZATION RESULTS\n{"="*60}\n" \
        f"Optimization Time: {_getMinSecStr(self.TimeDelta)}\n" \
        f"Final Optimization Iteration: {self.LastIteration} / {self.IterationMaximum}\n\n" \
        f"Initial Loss: {self.LossHistory[0]:.4g}\n" \
        f"Final Loss: {self.LossHistory[-1]:.4g}\n" \
        f"Loss Reduction: {((self.LossHistory[0] - self.LossHistory[-1]) / self.LossHistory[0] * 100):.4g}%\n" \
        f"\nPARAMETER CHANGES:\n" \
        f"{"-"*40}\n"
        
        for i, param in enumerate(self.CRNOptimizationFrameworkObject.Parameters):
            iP = self.ParameterHistory[0][i]
            fP = self.ParameterHistory[-1][i]
            change = ((fP - iP) / iP * 100)
            
            self.OutputStr += f"{param}:\n" \
            f"  Initial: {iP:.4g}\n" \
            f"  Final:   {fP:.4g}\n" \
            f"  Change:  {change:+.2f}%\n"
    
    def __str__(self):
        return self.OutputStr
