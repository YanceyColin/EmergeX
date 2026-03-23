############ EmergeX Initialization ###########
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 03/04/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

# CRN Constructor and Wrappers
from emergex.crn import (
    Reaction,
    Component,
    ReactionNetwork
)

# Core
from emergex.core import (
    InitialConditionInfo,
    CRNInfoHandler,

    EvaluationPointsHandler,
    CRNSimulationRunner,

    Interruption,
    TimeSpan
)

# Optimization
from emergex.optimization import (
    CRNOptimizationFramework,
    OptimizerRunResult,

    FreeParameter,
    LinkedParameter
)

# Behaviors
from emergex.behaviors import (
    Behavior,
    BehaviorGroup,
    BehaviorTimeCourse,
    OptimizeBehaviorsManager
)

# Experiments
from emergex.experiments import (
    Signal,
    Experiment,
    ExperimentGroup,
    OptimizeExperimentsManager
)

# Results
from emergex.results import (
    CompiledOptimizationData
)

# Visualization
from emergex.visualization import (
    saveFreeParametersVisualization,
    saveBehaviorResultsVisualization,
    saveBehaviorLandscapeVisualization,
    saveExperimentResultsVisualization
)

# Utils, use these at your own risk
# Public-facing 
from emergex.utils import (
    DynamicFunctionStoring,
    getComponentHosts
)

__all__ = [
    # CRN Constructor
    "Reaction",
    "Component",
    "ReactionNetwork",

    # Core
    "InitialConditionInfo",
    "CRNInfoHandler",
    "EvaluationPointsHandler",
    "CRNSimulationRunner",
    "Interruption",
    "TimeSpan",

    # Optimization
    "CRNOptimizationFramework",
    "OptimizerRunResult",
    "FreeParameter",
    "LinkedParameter",

    # Behaviors
    "Behavior",
    "BehaviorGroup",
    "BehaviorTimeCourse",
    "OptimizeBehaviorsManager",

    # Experiments
    "Signal",
    "Experiment",
    "ExperimentGroup",
    "OptimizeExperimentsManager",

    # Results
    "CompiledOptimizationData",

    # Visualization
    "saveFreeParametersVisualization",
    "saveBehaviorResultsVisualization",
    "saveBehaviorLandscapeVisualization",
    "saveExperimentResultsVisualization",

    # Utils
    "DynamicFunctionStoring",
    "getComponentHosts"
]
