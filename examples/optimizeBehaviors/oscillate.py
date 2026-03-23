## Tweaked Brusselator Behavior Optimization ##
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

import matplotlib.pyplot as plt

###############################################

"""
Optimizes the initial conditions of a version of the Brusselator that assumes an enzyme concentration gauge for fuel
release into the system. This change reflects the behaviors observed in certain enzymatically gated reactions or drug
release systems where a finite amount of some resource is released over time at a roughly constant rate
(i.e., zero-order kinetics), but eventually exhibits first-order kinetics as the concentration of the fuel precursor
gets exceedingly low.

To act as a "cap" to the amount of X and Y observed, I track a downstream "reporter" in the form of a molecule that spuriously
bind and unbinds with X and Y.
"""

DATA_STORE = Path(__file__).resolve().parent.parent / "output"

# Brusselator Scheme
aConv   = Reaction(1e-4, "A -> A + X")
createY = Reaction(1e-5, "B + X -> B + Y")
createX = Reaction(1e-8, "X + X + Y + C -> X + X + X + C") # Stoichiometry is not built-in directly, must represent using multiple copies of the same material
destroyX= Reaction(1e-5, "X + D -> D") # System "release valve"

# Measurement reactions
spuriousX = Reaction(1, "X + X_Reporter <-> X:X_Reporter", 50)
spuriousY = Reaction(1, "Y + Y_Reporter <-> Y:Y_Reporter", 50)

aComp = Component("A", 500)
bComp = Component("B", 300)
cComp = Component("C", 200)
dComp = Component("D", 100)
xComp = Component("X", 1)
yComp = Component("Y", 99)

xReporterComp = Component("X_Reporter", 1)
yReporterComp = Component("Y_Reporter", 1)


rN = ReactionNetwork()
rN.addRxns([
    aConv,
    createY,
    createX,
    destroyX,
    spuriousX,
    spuriousY
])

rN.addComponents([
    aComp,
    bComp,
    cComp,
    dComp,
    xComp,
    yComp,
    xReporterComp,
    yReporterComp
])

simTime = 10*3600
switchPeriod = 3600
enforcementWindow = 20*60
displaceTime = 7200


# Quick pre-optimization preview (blocks at show so results can be inspected).
rN.simulateReactionFn(simTime, simDataResolution = 1001)
defaultResult = rN.SimResults[-1]

timeHours = defaultResult["Time"] / 3600
fig, axes = plt.subplots(2, 4, figsize=(11, 7), sharex = True)
plotInfo = [
    ("A", "Remainder [A]"),
    ("B", "Remainder [B]"),
    ("X", "[X]"),
    ("Y", "[Y]"),
    ("X:X_Reporter", "[X:Rep]"),
    ("Y:Y_Reporter", "[Y:Rep]")
]

for ax, (compName, titleStr) in zip(axes.ravel(), plotInfo):
    ax.plot(timeHours, defaultResult[compName], linewidth = 1.6)
    ax.set_title(titleStr)
    ax.set_ylabel("Concentration")
    ax.set_ylim(ymin = 0)
    ax.grid(True, linestyle = "--", alpha = 0.3)

for ax in axes[1]:
    ax.set_xlabel("Time (hours)")

fig.tight_layout()
plt.show()





def normalizeByXReporter(x, concs):
    return x / concs["X_Reporter"]

def normalizeByYReporter(y, concs):
    return y / concs["Y_Reporter"]

xReporterBehaviors = []
yReporterBehaviors = []
for i in range(4):
    center = (i + 0.5)*switchPeriod + displaceTime
    startBound = center - enforcementWindow/2
    endBound = center + enforcementWindow/2

    if i%2 == 0:
        xReporterBehaviors.append(Behavior("HIGH", startBound, endBound, requiredMatch = 0.5))
        yReporterBehaviors.append(Behavior("LOW", startBound, endBound, requiredMatch = 0.5))
    else:
        xReporterBehaviors.append(Behavior("LOW", startBound, endBound, requiredMatch = 0.5))
        yReporterBehaviors.append(Behavior("HIGH", startBound, endBound, requiredMatch = 0.5))

behaviorTimeCourse = BehaviorTimeCourse(
    [TimeSpan(simTime)],
    [
        BehaviorGroup("X_Reporter", xReporterBehaviors, normalizeFn = normalizeByXReporter),
        BehaviorGroup("Y_Reporter", yReporterBehaviors, normalizeFn = normalizeByYReporter)
    ]
)

freeParams = [
    FreeParameter(aComp),
    FreeParameter(bComp),
    FreeParameter(cComp),
    FreeParameter(dComp),
    FreeParameter(xComp, lowerBound = 1, upperBound = 99),
    FreeParameter(xReporterComp, lowerBound = 1, upperBound = 10)
]

linkedParams = [
    LinkedParameter(
        yReporterComp,
        conversionFn = lambda _, concs: concs["X_Reporter"]
    ),
    LinkedParameter(
        yComp,
        conversionFn = lambda _, concs: 100 - concs["X"]
    )
]

behaveManager = OptimizeBehaviorsManager(
    CRNInfoHandler(rN),
    [behaviorTimeCourse],
    highMin = 0.6,
    boundMult = 3
)

saveBehaviorLandscapeVisualization(
    behaveManager,
    fileLocation = DATA_STORE,
    fileName = "oscillate_landscape",
    timeUnits = "min"
)

# simulationRunnerObj = CRNSimulationRunner(rtol = 1e-6, atol = 1e-9)
# optimizerFramework = CRNOptimizationFramework(
#     freeParams,
#     linkedParameters = linkedParams
# )

# dataResult = CompiledOptimizationData(
#     behaveManager,
#     simulationRunnerObj,
#     optimizerFramework,
#     iterationCount = 500,
#     callbackFrequency = 10
# )
# print(dataResult.OptimizerRunResultObject)
# dataResult.save(DATA_STORE, "oscillateResult")


# iterSample = jnp.arange(1, dataResult.OptimizerRunResultObject.LastIteration + 1, dataResult.OptimizerRunResultObject.LastIteration // 61) # I want to see at least 60 plots

# saveFreeParametersVisualization(
#     dataResult,
#     iterationList = iterSample,
#     fileLocation = DATA_STORE,
#     fileName = "oscillateParameters",
#     fps = 10
# )

# saveBehaviorResultsVisualization(
#     dataResult,
#     iterationList = iterSample,
#     fileLocation = DATA_STORE,
#     fileName = "oscillateBehaviors",
#     fps = 10,
#     timeUnits = "min"
# )

