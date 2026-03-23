########## Optimizer Results Handling #########
#                                             #
#               By: Colin Yancey              #
#             Created October 2025            #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from ..utils.base import DataStore
from ..utils.helpers import ManagerClass
from ..core.simulate import CRNSimulationRunner
from ..optimization.optimize import CRNOptimizationFramework

###############################################

# Class for handling all relevant context needed to uniquely identify an optimization run result (the input parameters that
# led to the result generation and the result itself). Can save optimization runs, and can load previously saved runs.
class CompiledOptimizationData(DataStore):
    def __init__(
        self,
        managerObj: ManagerClass,
        simulationRunnerObj: CRNSimulationRunner,
        optimizerFramework: CRNOptimizationFramework,
        iterationCount: int = 1000,
        callbackFrequency: int = None
    ):
        self.ManagerObject = managerObj
        self.SimulationRunnerObject = simulationRunnerObj

        managerCostFn = managerObj.getCostFn(optimizerFramework, simulationRunnerObj)
        optimizedResult = optimizerFramework.optimize(managerCostFn, iterationCount = iterationCount, callbackFrequency = callbackFrequency)

        self.OptimizerRunResultObject = optimizedResult
