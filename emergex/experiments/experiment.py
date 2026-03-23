############### Experiment Data ###############
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

import jax.numpy as jnp
from scipy.signal import savgol_filter

from typing import Callable
import warnings

from ..core.crnData import (
    CRNInfoHandler
)
from ..core.simulate import (
    TimeSpan,
    EvaluationPointsHandler
)
from ..crn import (
    ReactionNetwork
)
from ..utils.base import (
    ComponentCompositesHandler,
    DynamicFunctionStoring,
    DataStore
)

###############################################

# Stores data points that represent the fraction of maximum relevant fluorescence within an experiment alongside the model for obtaining this fluorescence from the CRN results.
class Signal(ComponentCompositesHandler, DynamicFunctionStoring):
    def __init__(
        self,
        compNames: str | list[str],
        maxSignalConc: float | int | Callable[[dict], float],
        dataPoints: jnp.ndarray,
        weight: float = 1.0, # Must be >= 0 (although, why bother including it if it is 0?)
        gradientImpactRadius: int = 5, # Must be >= 0
        gradientImpactSigma: float = 2.0, # Must be > 0
        compositesAllowed: bool = False
    ):
        ComponentCompositesHandler.__init__(self, compNames, compositesAllowed)

        if callable(maxSignalConc):
            self.getMaxSignal = maxSignalConc
        else:
            self.MaxSignal = maxSignalConc
        
        self.DataPoints = dataPoints

        self.Weight = weight
        # Use absolute gradient magnitude so rising/falling regions both increase impact.
        weights = jnp.abs(
            savgol_filter(
                self.DataPoints,
                window_length = gradientImpactRadius,
                polyorder = 1,
                deriv=1
            )
        )
        totalWeight = jnp.sum(weights)
        
        if totalWeight > 0:
            if gradientImpactRadius > 0:
                # Spreads any strong signal gradients to nearby points so "near-gradient" regions also retain impactful weights.
                offsets = jnp.arange(-gradientImpactRadius, gradientImpactRadius + 1)
                kernel = jnp.exp(-0.5 * jnp.square(offsets / gradientImpactSigma))
                kernel /= jnp.sum(kernel)
                weights = jnp.convolve(weights, kernel, mode="same")

            totalWeight = jnp.sum(weights)
            self.GradientImpactVector = weights / totalWeight
        else:
            self.GradientImpactVector = jnp.ones_like(weights) / weights.size
    
    def getMaxSignal(self, _):
        return self.MaxSignal
    
    def normalizeValue(self, valToNorm, concs):
        return valToNorm / self.getMaxSignal(concs)
        


# Stores a given experiment's progression through the time course and defines what signals were collected.
class Experiment:
    def __init__(self, timeCourse: list[TimeSpan], signals: list[Signal], weight: float = 1.0):
        self.TimeCourse = timeCourse

        if len(signals) == 0:
            raise ValueError("An Experiment object must have at least one Signal object.")
        standardDPNum = signals[0]
        for signal in signals[1:]:
            if len(signal.DataPoints) != standardDPNum:
                raise ValueError("All Signal objects in an Experiment object must have the same number of data points.")
        self.Signals = signals
        self.Weight = weight

        self.SignalWeights = jnp.array([signal.Weight * signal.GradientImpactVector for signal in self.Signals])
        self.SignalWeights /= jnp.sum(jnp.array([signal.Weight for signal in self.Signals]))


# Stores experiments that were run simultaneously, thus sharing data recording times and ideally CRN charactertistics. Differences in CRN initial conditions are handled by
# each individual experiment's time course interruptions.
class ExperimentGroup(DataStore):
    def __init__(self, timePointsHandler: EvaluationPointsHandler, crn: ReactionNetwork, experiments: list[Experiment], weight: float = 1.0):
        self.TimePointsHandler = timePointsHandler
        self.CRNInfo = CRNInfoHandler(crn)
        self.Experiments = experiments
        self.Weight = weight

        standardDPNum = len(self.TimePointsHandler.TotalEvalPts)
        for experiment in self.Experiments:
            for signal in experiment.Signals:
                if len(signal.DataPoints) != standardDPNum:
                    raise ValueError(
                        "All Experiment objects in an ExperimentGroup object must have the same number of data points in each of their Signal " \
                        "objects as the given list of associated times in the EvaluationPointsHandler."
                    )

        self.ExperimentWeights = jnp.array([expt.Weight for expt in self.Experiments])
        self.ExperimentWeights /= jnp.sum(self.ExperimentWeights)*len(self.TimePointsHandler.TotalEvalPts)
    

    # This class specifically has a weight adjustment to accomodate the cumulative data curation process. Weights should generally be decided on instance generation for maximum safety.
    def adjustWeight(self, weight: float):
        warnings.warn(f"ExperimentGroup weight adjustments should be made before incorporating them into an OptimizeExperimentsManager instance.", UserWarning)
        self.Weight = weight
