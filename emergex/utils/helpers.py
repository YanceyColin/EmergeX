########### Helper Functions/Classes ##########
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 03/03/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from abc import ABC, abstractmethod

from ..core.simulate import (
    CRNSimulationRunner
)
from ..optimization.optimize import (
    CRNOptimizationFramework
)

###############################################

class ManagerClass(ABC):
    @abstractmethod
    def getCostFn(self, optimizerFrameworkObj: CRNOptimizationFramework, simulationRunnerObj: CRNSimulationRunner):
        pass


# Used to debug visualization code errors where the wrong manager class is provided for a given visualizer.
def _checkManagerType(managerObj: ManagerClass, desiredClass: ManagerClass):
    if not isinstance(managerObj, desiredClass):
        raise TypeError(f"Expected the following manager class: '{desiredClass.__name__}'\n  Got class '{type(managerObj).__name__}'")
