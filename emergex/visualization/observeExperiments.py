###### Experiment Optimization Visualizer #####
#                                             #
#               By: Colin Yancey              #
#             Created March 2026              #
#            Last Edited: 03/04/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from pathlib import Path
from typing import Optional
import numpy as np
import jax.numpy as jnp
import time

import imageio.v2 as iio
import os

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from ..results.optimizerResults import CompiledOptimizationData
from ..experiments.manager import OptimizeExperimentsManager
from ..utils.constants import UNIT_WEIGHTS
from ..utils.base import _getMinSecStr
from ..utils.helpers import _checkManagerType
from .plotting import *

###############################################


def saveExperimentResultsVisualization(
    compiledOptDataObject: CompiledOptimizationData,
    experimentSignalWhitelist: Optional[list[int | tuple[int, int] | tuple[int, int, int]]] = None,
    iterationList: Optional[list[int]] = None,
    fileLocation: Path = Path('./'),
    fileName: str = 'experimentResults',
    fps: int = 1,
    stackImages: bool = False,
    keepIndividualImages: bool = False,
    startLineColor: str = "#202020",
    endLineColor: str = "#30E0F0",
    dataPointColor: str = "#A0A0A0",
    dataPointAlpha: float = 0.5,
    dataPointSize: float = 14,
    timeUnits: str = "sec"
):
    _checkManagerType(compiledOptDataObject.ManagerObject, OptimizeExperimentsManager)

    if timeUnits not in UNIT_WEIGHTS:
        raise ValueError(f"Unknown timeUnits '{timeUnits}'. Available units: {list(UNIT_WEIGHTS.keys())}")

    optRes = compiledOptDataObject.OptimizerRunResultObject
    simRunner = compiledOptDataObject.SimulationRunnerObject
    optFrameworkObj = optRes.CRNOptimizationFrameworkObject
    exptGroups = compiledOptDataObject.ManagerObject.ExperimentGroups

    totalIter = optRes.LastIteration
    if iterationList is None:
        iterationList = np.arange(totalIter)
    else:
        iterationList = np.array(iterationList) - 1

    if len(iterationList) == 0:
        raise ValueError("iterationList must contain at least one iteration.")
    if np.any(iterationList < 0) or np.any(iterationList >= totalIter):
        raise ValueError(f"All entries in iterationList must be between 1 and {totalIter}, inclusive.")

    panelInfo = []

    def _add_all_signals(groupIdx: int, exptIdx: int):
        exptObj = exptGroups[groupIdx].Experiments[exptIdx]
        for signalIdx in range(len(exptObj.Signals)):
            candidate = (groupIdx, exptIdx, signalIdx)
            if candidate not in panelInfo:
                panelInfo.append(candidate)

    def _add_all_experiments(groupIdx: int):
        for exptIdx in range(len(exptGroups[groupIdx].Experiments)):
            _add_all_signals(groupIdx, exptIdx)

    if experimentSignalWhitelist is None:
        for groupIdx in range(len(exptGroups)):
            _add_all_experiments(groupIdx)
    elif len(experimentSignalWhitelist) == 0:
        raise ValueError(
            "If providing experimentSignalWhitelist, include at least one entry selecting a group, experiment, or signal."
        )
    else:
        for selection in experimentSignalWhitelist:
            if isinstance(selection, int):
                groupIdx = selection - 1
                if groupIdx < 0 or groupIdx >= len(exptGroups):
                    raise ValueError(f"Experiment Group {selection} does not exist.")
                _add_all_experiments(groupIdx)
                continue

            if len(selection) == 2:
                groupIdx = selection[0] - 1
                exptIdx = selection[1] - 1

                if groupIdx < 0 or groupIdx >= len(exptGroups):
                    raise ValueError(f"Experiment Group {selection[0]} does not exist.")
                if exptIdx < 0 or exptIdx >= len(exptGroups[groupIdx].Experiments):
                    raise ValueError(f"Experiment {selection[1]} is not contained within Experiment Group {selection[0]}.")

                _add_all_signals(groupIdx, exptIdx)
                continue

            if len(selection) == 3:
                groupIdx = selection[0] - 1
                exptIdx = selection[1] - 1
                signalIdx = selection[2] - 1

                if groupIdx < 0 or groupIdx >= len(exptGroups):
                    raise ValueError(f"Experiment Group {selection[0]} does not exist.")
                if exptIdx < 0 or exptIdx >= len(exptGroups[groupIdx].Experiments):
                    raise ValueError(f"Experiment {selection[1]} is not contained within Experiment Group {selection[0]}.")
                if signalIdx < 0 or signalIdx >= len(exptGroups[groupIdx].Experiments[exptIdx].Signals):
                    raise ValueError(
                        f"Signal {selection[2]} is not contained within Experiment {selection[1]} of Experiment Group {selection[0]}."
                    )

                candidate = (groupIdx, exptIdx, signalIdx)
                if candidate not in panelInfo:
                    panelInfo.append(candidate)
                continue

            raise ValueError(
                "Each whitelist entry must be either an int (group), a tuple of length 2 (group, experiment), or a tuple of length 3 (group, experiment, signal)."
            )

    if len(panelInfo) == 0:
        raise ValueError("No panels were selected for plotting.")

    uniqueExpts = sorted({(groupIdx, exptIdx) for groupIdx, exptIdx, _ in panelInfo})
    exptsByGroup = {}
    for groupIdx, exptIdx in uniqueExpts:
        if groupIdx not in exptsByGroup:
            exptsByGroup[groupIdx] = []
        exptsByGroup[groupIdx].append(exptIdx)

    parsedTimeCourseData = {}
    evalPts = {}
    for groupIdx, exptIdx in uniqueExpts:
        groupObj = exptGroups[groupIdx]
        exptObj = groupObj.Experiments[exptIdx]
        parsedTimeCourseData[(groupIdx, exptIdx)] = groupObj.TimePointsHandler.parseEvalPts(exptObj.TimeCourse)
        evalPts[(groupIdx, exptIdx)] = np.array(groupObj.TimePointsHandler.TotalEvalPts) / UNIT_WEIGHTS[timeUnits]

    print(f"Computing integration solutions...\n{"#"*40}")

    initCondCompsByGroup = {groupIdx: [] for groupIdx in exptsByGroup.keys()}
    exptSolutions = {(groupIdx, exptIdx): [] for groupIdx, exptIdx in uniqueExpts}

    startTime = time.time()

    firstIteration = iterationList[0]
    curLnParams = jnp.log(optRes.ParameterHistory[firstIteration])
    for groupIdx, exptIndices in exptsByGroup.items():
        groupObj = exptGroups[groupIdx]
        initCondInfo = optFrameworkObj.getStartingConditions(curLnParams, groupObj.CRNInfo)
        initCondCompsByGroup[groupIdx].append(initCondInfo.ComponentsDictionary)

        for exptIdx in exptIndices:
            exptSolutions[(groupIdx, exptIdx)].append(
                simRunner.runTimeCourse(
                    parsedTimeCourseData[(groupIdx, exptIdx)],
                    groupObj.CRNInfo,
                    initCondInfo
                )
            )

    iterTime = time.time() - startTime
    completionTimeEstimate = iterTime * (len(iterationList) - 1)
    print(
        "Integration solutions for the first selected iteration were found.\n"
        f"  Time: {iterTime:.3f} sec\n"
        f"  Estimated time to completion: {_getMinSecStr(completionTimeEstimate)}"
    )
    if completionTimeEstimate > 120:
        print("  If this is too long, it is recommended to use a sparser iteration sampling.")

    for iteration in iterationList[1:]:
        curLnParams = jnp.log(optRes.ParameterHistory[iteration])

        for groupIdx, exptIndices in exptsByGroup.items():
            groupObj = exptGroups[groupIdx]
            initCondInfo = optFrameworkObj.getStartingConditions(curLnParams, groupObj.CRNInfo)
            initCondCompsByGroup[groupIdx].append(initCondInfo.ComponentsDictionary)

            for exptIdx in exptIndices:
                exptSolutions[(groupIdx, exptIdx)].append(
                    simRunner.runTimeCourse(
                        parsedTimeCourseData[(groupIdx, exptIdx)],
                        groupObj.CRNInfo,
                        initCondInfo
                    )
                )

    print(
        "All integration solutions found.\n"
        f"  Total Time (including first iteration): {_getMinSecStr(time.time() - startTime)}\n"
        "Creating plots..."
    )

    panelYValues = [[None] * len(iterationList) for _ in range(len(panelInfo))]

    for panelIdx, (groupIdx, exptIdx, signalIdx) in enumerate(panelInfo):
        groupObj = exptGroups[groupIdx]
        exptObj = groupObj.Experiments[exptIdx]
        signalObj = exptObj.Signals[signalIdx]
        relevantIndices = signalObj.getRelevantComponentIndices(groupObj.CRNInfo.CompNameList)

        for iterIdx in range(len(iterationList)):
            relevantResults = exptSolutions[(groupIdx, exptIdx)][iterIdx][:, relevantIndices]
            if len(relevantIndices) == 1:
                rawSignalVals = relevantResults.reshape(-1)
            else:
                rawSignalVals = jnp.sum(relevantResults, axis=1)

            curVals = signalObj.getNormalizeVMAP(initCondCompsByGroup[groupIdx][iterIdx])(rawSignalVals)
            panelYValues[panelIdx][iterIdx] = curVals

    workspaceName = fileLocation / 'mp4 Output'
    os.makedirs(workspaceName, exist_ok=True)
    framePaths = []

    panelNum = len(panelInfo)
    squareGridDim = int(np.ceil(np.sqrt(panelNum)))
    gridLen = 4 * squareGridDim

    fig, axes = plt.subplots(squareGridDim, squareGridDim, figsize=(gridLen, gridLen))
    if squareGridDim == 1:
        axes = np.array([[axes]])
    flatAxes = axes.ravel()

    for m in range(panelNum, squareGridDim**2):
        flatAxes[m].axis('off')

    # For each column, find the lowest row that still contains a visible panel.
    lowestVisibleRowByColumn = {}
    for columnIndex in range(squareGridDim):
        visibleRows = [
            rowIndex
            for rowIndex in range(squareGridDim)
            if (rowIndex * squareGridDim + columnIndex) < panelNum
        ]
        if len(visibleRows) > 0:
            lowestVisibleRowByColumn[columnIndex] = max(visibleRows)

    for panelIdx, (groupIdx, exptIdx, signalIdx) in enumerate(panelInfo):
        groupObj = exptGroups[groupIdx]
        exptObj = groupObj.Experiments[exptIdx]
        signalObj = exptObj.Signals[signalIdx]

        ax = flatAxes[panelIdx]
        x = evalPts[(groupIdx, exptIdx)]

        ax.scatter(
            x,
            signalObj.DataPoints,
            s=dataPointSize,
            color=dataPointColor,
            alpha=dataPointAlpha,
            zorder=2,
            label='Data'
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-0.03, 1.03)
        rowIndex = panelIdx // squareGridDim
        columnIndex = panelIdx % squareGridDim
        if rowIndex == lowestVisibleRowByColumn.get(columnIndex):
            ax.set_xlabel(f"Time ({timeUnits})")
        ax.set_ylabel("Normalized Signal")

        signalName = ":".join(signalObj.ComponentNames)
        ax.set_title(f"Group {groupIdx + 1}, Experiment {exptIdx + 1}, Signal {signalIdx + 1} [{signalName}]", fontsize=9)

    startRGB = np.array(mcolors.hex2color(startLineColor))
    endRGB = np.array(mcolors.hex2color(endLineColor))
    colorDenom = max(len(iterationList) - 1, 1)

    for iterIdx, iteration in enumerate(iterationList):
        curPlots = []
        colorFrac = iterIdx / colorDenom
        curPlotColor = mcolors.rgb2hex(startRGB + (endRGB - startRGB) * colorFrac)

        for panelIdx, (groupIdx, exptIdx, _) in enumerate(panelInfo):
            ax = flatAxes[panelIdx]
            x = evalPts[(groupIdx, exptIdx)]
            y = panelYValues[panelIdx][iterIdx]
            curPlots.append(ax.plot(x, y, color=curPlotColor, linewidth=1.8, zorder=3))

        iterTracker = addIterationTracker(fig, iteration + 1, optRes.LastIteration)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        framePath = workspaceName / f"{fileName}_{iteration + 1}.png"
        plt.savefig(framePath, dpi=100)
        framePaths.append(framePath)

        iterTracker.remove()
        if not stackImages:
            for curPlot in curPlots:
                curPlot[0].remove()

    with iio.get_writer(
        workspaceName / f"{fileName}.mp4",
        fps=fps,
        codec='libx264',
        format='FFMPEG',
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    ) as writer:
        for framePath in framePaths:
            image = iio.imread(framePath)
            writer.append_data(image)

    if not keepIndividualImages:
        for framePath in framePaths:
            try:
                os.remove(framePath)
            except OSError:
                pass
