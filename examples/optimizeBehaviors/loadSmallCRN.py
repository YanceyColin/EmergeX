######## Small CRN Behavior Optimization ######
#                                             #
#               By: Colin Yancey              #
#              Created March 2026             #
#            Last Edited: 03/11/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from emergex import *
import jax.numpy as jnp
from pathlib import Path

###############################################

"""
Must run optimizeBehaviors/smallCRN.py first (needs to have generated smallCRNOptResult.pkl in the output folder). Saving a file saves all
inherited details relevant to the optimization run, including dynamic functions such as those that control interruptions or linked parameters,
just to name a few classes. Load this data using the internal load command of the class, and pick up where you last left off with a given
optimization run result.
"""

DATA_STORE = Path(__file__).resolve().parent.parent / "output"

dataResult = CompiledOptimizationData.load(DATA_STORE / "smallCRNOptResult.pkl")

iterSample = jnp.arange(1, dataResult.OptimizerRunResultObject.LastIteration + 1, dataResult.OptimizerRunResultObject.LastIteration // 31) # I want to see at least 30 plots

# Obtain the behaviors across iterations image again but stack the images this time in the animation.
saveBehaviorResultsVisualization(
    dataResult,
    iterationList = iterSample,
    fileLocation = DATA_STORE,
    fileName = "smallCRN_behaviors_stackedImages",
    fps = 6,
    timeUnits = "min",
    stackImages = True
)