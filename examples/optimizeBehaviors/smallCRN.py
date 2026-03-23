######## Small CRN Behavior Optimization ######
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 03/02/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from emergex import *
import jax.numpy as jnp
from pathlib import Path

###############################################

DATA_STORE = Path(__file__).resolve().parent.parent / "output"


combineAB = Reaction(5e-5, "A + B -> C") # In units of 1/sec/uM
cdComplex = Reaction(1e-2, "C + D <-> CD", backwardRate = 1e0) # In units of 1/sec/uM, and in units of 1/sec
makeE = Reaction(3e-2, "CD -> D + E") # In units of 1/sec

compA = Component("A", 100) # In units of uM
compB = Component("B", 80) # In units of uM
compD = Component("D", 5) # In units of uM

newNetwork = ReactionNetwork()
newNetwork.addRxns([combineAB, cdComplex, makeE])
newNetwork.addComponents([compA, compB, compD])

freeParams = [
    FreeParameter(combineAB),
    FreeParameter(cdComplex),
    FreeParameter(makeE),
    FreeParameter(compB)
]

addlMaterial = 40
interruptB = Interruption(compB, addlMaterial, interruptionType="ADD") # Some extra "B" added to the solution potentially to keep things going
timeCourse = [TimeSpan(3600), TimeSpan(3600, [interruptB])]


def normalizeByD(x, concs):
    return x / concs["D"]

# Make sure to incorporate any internally used variables in a class envelope to ensure the property is passed on to the save file.
class NormalizeByAOrB:
    def __init__(self, addlMaterial):
        self.AddlMaterial = addlMaterial
    
    def __call__(self, x, concs):
        return x / (jnp.minimum(concs["A"], concs["B"] + self.AddlMaterial))


behaviorTimeCourse1 = BehaviorTimeCourse(
    timeCourse,
    [
        BehaviorGroup(
            "CD",
            [
                Behavior("HIGH", 0.1*7200, 0.4*7200), # HIGH behavior from 20% to 25% of simulation time
                Behavior("LOW", 0.5*7200, 1*7200)     # LOW behavior from 50% to 100% of simulation time
            ],
            normalizeFn = normalizeByD
        ),
        BehaviorGroup(
            "E",
            [
                Behavior("HIGH", 0.5*7200, 0.6*7200) # HIGH behavior from 50% to 60% of simulation time
            ],
            normalizeFn = NormalizeByAOrB(addlMaterial)
        )
    ]
)

interrupt = Interruption(compA, 500, interruptionType="ADD")

behaviorTimeCourse2 = BehaviorTimeCourse(
    [TimeSpan(10000, [interrupt])],
    [
        BehaviorGroup(
            "CD",
            [
                Behavior("LOW", 0.8*10000, 0.9*10000) # LOW behavior from 80% to 90% of simulation time
            ],
            normalizeFn = normalizeByD
        )
    ]
)

behaveManager = OptimizeBehaviorsManager(
    CRNInfoHandler(newNetwork),
    [
        behaviorTimeCourse1,
        behaviorTimeCourse2
    ]
)
simulationRunnerObj = CRNSimulationRunner()
optimizerFramework = CRNOptimizationFramework(freeParams)


saveBehaviorLandscapeVisualization(
    behaveManager,
    fileLocation = DATA_STORE,
    fileName = "smallCRN_landscape",
    timeUnits = "min"
)

dataResult = CompiledOptimizationData(
    behaveManager,
    simulationRunnerObj,
    optimizerFramework,
    iterationCount = 100,
    callbackFrequency = 4
)
dataResult.save(DATA_STORE, "smallCRNOptResult")

iterSample = jnp.arange(1, dataResult.OptimizerRunResultObject.LastIteration + 1, dataResult.OptimizerRunResultObject.LastIteration // 31) # I want to see at least 30 plots

saveFreeParametersVisualization(
    dataResult,
    iterationList = iterSample,
    fileLocation = DATA_STORE,
    fileName = "smallCRN_parameters",
    fps = 6
)

saveBehaviorResultsVisualization(
    dataResult,
    iterationList = iterSample,
    fileLocation = DATA_STORE,
    fileName = "smallCRN_behaviors",
    fps = 6,
    timeUnits = "min"
)