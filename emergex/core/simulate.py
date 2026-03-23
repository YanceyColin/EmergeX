################ Initialization ###############
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

import diffrax

from .timeCourse import (
    TimeSpan
)
from .crnData import (
    CRNInfoHandler,
    InitialConditionInfo
)

###############################################

class EvaluationPointsHandler:
    def __init__(self, totalEvalPts: jnp.ndarray):
        self.TotalEvalPts = totalEvalPts
        self.ParsedTimeCourseData = {}
    
    def parseEvalPts(self, timeCourse: list[TimeSpan]):
        timeCourseKey = id(timeCourse)
        if not timeCourseKey in self.ParsedTimeCourseData:
            currentTime = 0
            timeCourseData = []

            for i, timeSpan in enumerate(timeCourse):
                if i+1 != len(timeCourse):
                    spanMask = (self.TotalEvalPts >= currentTime) & (self.TotalEvalPts < currentTime + timeSpan.Time)
                else:
                    spanMask = (self.TotalEvalPts >= currentTime) & (self.TotalEvalPts <= currentTime + timeSpan.Time) # Account for the final point if this is the final TimeSpan
                spanIndices = jnp.where(spanMask)[0] # Get list of relevant indices
                evalPts = jnp.zeros(len(spanIndices) + 1)
                evalPts = evalPts.at[-1].set(timeSpan.Time) # Make sure the last time is being saved regardless, as this point is the initial condition vector for the next timespan
                if len(spanIndices) > 0: # Put in the rest of the span index values
                    evalPts = evalPts.at[:-1].set(self.TotalEvalPts[spanIndices] - currentTime)
                
                timeCourseData.append((timeSpan, spanIndices, evalPts))
                currentTime += timeSpan.Time
            
            self.ParsedTimeCourseData[timeCourseKey] = timeCourseData
            
        return self.ParsedTimeCourseData[timeCourseKey]
    

class CRNSimulationRunner:
    def __init__(self, rateMinimum: float = 0, concentrationMinimum: float = 0, dt0Min: float = 1e-3, dtmin: float = 1e-15, rtol: float = 1e-4, atol: float = 1e-7):
        self.RateMinimum = rateMinimum
        self.ConcentrationMinimum = concentrationMinimum

        self.dt0Min = dt0Min
        self.dtmin = dtmin
        self.rtol = rtol
        self.atol = atol

    def runTimeCourse(self, timeCourseData, crnInfo: CRNInfoHandler, initCondInfo: InitialConditionInfo, t0Time = 0):
        allResults = []

        concList = initCondInfo.ConcentrationList
        rateList = initCondInfo.RateList

        def odeFn(t, y, args):
            rates = rateList * jnp.prod(jnp.maximum(y, self.ConcentrationMinimum) ** crnInfo.ReactionMatrix, axis=1)
            return jnp.dot(crnInfo.ModifierMatrixTransposed, rates)

        for timeSpan, spanIndices, evalPts in timeCourseData:
            for mod in timeSpan.Interruptions:
                newVal = mod.getValue(initCondInfo.ReactionsDictionary, initCondInfo.ComponentsDictionary)
                if mod.ValueName == "Concentration":
                    concList = mod.getModifiedList(concList, crnInfo.CompNameList.index(mod.Name), newVal, self.ConcentrationMinimum)
                else:
                    rateList = mod.getModifiedList(rateList, crnInfo.RxnNameList.index((mod.Name, mod.ValueName)), newVal, self.RateMinimum)

            try:
                solution = diffrax.diffeqsolve(
                    diffrax.ODETerm(odeFn),
                    solver = diffrax.Kvaerno5(),
                    t0 = t0Time,
                    t1 = timeSpan.Time,
                    dt0 = jnp.minimum(timeSpan.Time / 10000, self.dt0Min),
                    y0 = concList,
                    saveat = diffrax.SaveAt(ts = evalPts),
                    stepsize_controller = diffrax.PIDController(
                        rtol = self.rtol,
                        atol = self.atol,
                        dtmin = self.dtmin,
                        dtmax = timeSpan.Time/100,
                        safety = 0.9
                    ),
                    max_steps = 50000
                )
            except Exception as e:
                raise ValueError(f"Integration failure during {timeSpan}:\n {e}")
            
            if len(spanIndices) > 0:
                allResults.append(solution.ys[:-1, :])
            concList = solution.ys[-1, :]
        
        return jnp.concatenate(allResults)
