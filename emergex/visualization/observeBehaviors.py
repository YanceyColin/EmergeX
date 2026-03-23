####### Behavior Optimization Visualizer ######
#                                             #
#               By: Colin Yancey              #
#             Created January 2026            #
#            Last Edited: 02/24/2026          #
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
from ..behaviors.manager import OptimizeBehaviorsManager
from ..utils.constants import UNIT_WEIGHTS
from ..utils.base import _getMinSecStr
from ..utils.helpers import _checkManagerType
from .plotting import *

from ..core.simulate import EvaluationPointsHandler

###############################################

# Returns an animated visualization of the tracked behavior functions and how they changed over the optimization process.
def saveBehaviorResultsVisualization(
    compiledOptDataObject: CompiledOptimizationData,
    timeCourseBehaviorGroupWhitelist: Optional[list[int | tuple[int, int]]] = None,
    iterationList: Optional[list[int]] = None, # If provided, plot will only contain the selected iterations in the list, each denoted by either a number indicating the time course number, or a tuple containing the time course number and the behavior group number within that time course.
    fileLocation: Path = Path('./'),
    fileName: str = 'behaviorResults',
    fps: int = 1,
    stackImages: bool = False,
    keepIndividualImages: bool = False,
    startLineColor = "#202020",
    endLineColor = "#30E0F0",
    timeUnits = "sec",
    dataResolution: int = 1001
):
    _checkManagerType(compiledOptDataObject.ManagerObject, OptimizeBehaviorsManager)
    
    timeCourses = compiledOptDataObject.ManagerObject.BehaviorTimeCourses
    simRunner = compiledOptDataObject.SimulationRunnerObject
    crnInfo = compiledOptDataObject.ManagerObject.CRNInfo
    optRes = compiledOptDataObject.OptimizerRunResultObject
    optFrameworkObj = optRes.CRNOptimizationFrameworkObject

    totalIter = optRes.LastIteration
    if iterationList is None:
        iterationList = np.arange(totalIter)
    else:
        iterationList = np.array(iterationList) - 1

    if len(iterationList) == 0:
        raise ValueError("iterationList must contain at least one iteration.")
    if np.any(iterationList < 0) or np.any(iterationList >= totalIter):
        raise ValueError(f"All entries in iterationList must be between 1 and {totalIter}, inclusive.")
    
    timeCoursesLen = len(timeCourses)

    def raiseTimeCourseLenError(selectionNum):
        raise ValueError(f"Since only {timeCoursesLen} time courses were provided in this optimization run, Time Course {selectionNum} does not exist.")

    timeCoursesToSolve = set()
    timeCourseSolutions = [0]*timeCoursesLen
    if timeCourseBehaviorGroupWhitelist is None:
        timeCourseBehaviorGroupWhitelist = []
        for i, timeCourse in enumerate(timeCourses):
            timeCoursesToSolve.add(i)
            for j in range(len(timeCourse.BehaviorGroups)):
                timeCourseBehaviorGroupWhitelist.append((i + 1, j + 1))
    elif len(timeCourseBehaviorGroupWhitelist) == 0:
        raise ValueError("If providing a whitelist, the behavior evolution visualization must contain at least one tuple indicating which time course's behavior group to plot.")
    else:
        for i, selection in timeCourseBehaviorGroupWhitelist:
            if isinstance(selection, int):
                newSelection = selection - 1
                timeCoursesToSolve.add(newSelection)
            else:
                newSelection = selection[0] - 1
                timeCoursesToSolve.add(newSelection)
                if selection[0] < 1 or selection[0] > timeCoursesLen:
                    raiseTimeCourseLenError(selection[0])
                if selection[1] < 1 or selection[1] > len(timeCourses[newSelection].BehaviorGroups):
                    raise ValueError(f"Behavior Group {selection[1]} is not contained within Time Course {selection[0]}")
    
    evalPts = [0]*timeCoursesLen
    evalPtsParsed = [0]*timeCoursesLen

    for timeCourseNum in timeCoursesToSolve:
        if timeCourseNum < 0 or timeCourseNum >= timeCoursesLen:
            raiseTimeCourseLenError(timeCourseNum + 1)
        
        evalPts[timeCourseNum] = np.linspace(0, sum([timeSpan.Time for timeSpan in timeCourses[timeCourseNum].TimeCourse]), dataResolution)
        evalPtsParsed[timeCourseNum] = EvaluationPointsHandler(evalPts[timeCourseNum]).parseEvalPts(timeCourses[timeCourseNum].TimeCourse)

    print(f"Computing integration solutions...\n{"#"*40}")
    startTime = time.time()

    initCondComps = [None] * len(iterationList)

    firstIter = iterationList[0]
    initCondInfo = optFrameworkObj.getStartingConditions(jnp.log(optRes.ParameterHistory[firstIter]), crnInfo)
    initCondComps[0] = initCondInfo.ComponentsDictionary
    for timeCourseNum in timeCoursesToSolve:
        timeCourseSolutions[timeCourseNum] = [
            simRunner.runTimeCourse(evalPtsParsed[timeCourseNum], crnInfo, initCondInfo)
        ]
    
    iterTime = time.time() - startTime
    completionTimeEstimate = iterTime*(len(iterationList)-1)
    print(
        "Integration solutions for the first iteration of the chosen time course behavior group pairs have been found.\n" \
        f"  Time: {iterTime:.3f} sec\n" \
        f"  Estimated time to completion: {_getMinSecStr(completionTimeEstimate)}"
    )
    if completionTimeEstimate > 120:
        print(f"  If this is too long, it is recommended to use a smaller sample of time courses or a sparser iteration sampling.")
    
    for iterIdx, iter in enumerate(iterationList[1:], start = 1):
        initCondInfo = optFrameworkObj.getStartingConditions(jnp.log(optRes.ParameterHistory[iter]), crnInfo)
        initCondComps[iterIdx] = initCondInfo.ComponentsDictionary
        for timeCourseNum in timeCoursesToSolve:
            timeCourseSolutions[timeCourseNum].append(
                simRunner.runTimeCourse(evalPtsParsed[timeCourseNum], crnInfo, initCondInfo)
            )
    
    print(
        "All integration solutions found.\n" \
        f"  Total Time (including first iteration): {_getMinSecStr(time.time() - startTime)}\n"
        "Creating plots..."
    )

    # Prepare output workspace
    workspaceName = fileLocation / 'mp4 Output'
    os.makedirs(workspaceName, exist_ok=True)
    framePaths = []

    startRGB = np.array(mcolors.hex2color(startLineColor))
    endRGB = np.array(mcolors.hex2color(endLineColor))
    startIteration, endIteration = iterationList[0], iterationList[-1]
    
    panelNum = len(timeCourseBehaviorGroupWhitelist)
    squareGridDim = int(np.ceil(np.sqrt(panelNum))) # Square grid of the panel count
    gridLen = 4*squareGridDim
    lowMax = compiledOptDataObject.ManagerObject.getLowMax()
    highMin = compiledOptDataObject.ManagerObject.getHighMin()

    fig, axes = plt.subplots(squareGridDim, squareGridDim, figsize=(gridLen, gridLen))
    if squareGridDim == 1: # Fix annoying auto-axis size leading to dimension reduction
        axes = np.array([[axes]])
    flatAxes = axes.ravel()

    for m in range(panelNum, squareGridDim**2):
        flatAxes[m].axis('off')

    lowestVisibleRowByColumn = {}
    for columnIndex in range(squareGridDim):
        visibleRows = [
            rowIndex
            for rowIndex in range(squareGridDim)
            if (rowIndex * squareGridDim + columnIndex) < panelNum
        ]
        if len(visibleRows) > 0:
            lowestVisibleRowByColumn[columnIndex] = max(visibleRows)

    for k, panelInfo in enumerate(timeCourseBehaviorGroupWhitelist):
        tCIdx = panelInfo[0] - 1
        timeCourseObj = timeCourses[tCIdx]
        x = evalPts[tCIdx]/UNIT_WEIGHTS[timeUnits]

        ax = flatAxes[k]
        missingLowLabel = True
        missingHighLabel = True

        for behavior in timeCourseObj.BehaviorGroups[panelInfo[1] - 1].Behaviors:
            startBound = behavior.StartBound/UNIT_WEIGHTS[timeUnits]
            endBound = (behavior.EndBound if behavior.EndBound != float('inf') else sum([ts.Time for ts in timeCourseObj.TimeCourse]))/UNIT_WEIGHTS[timeUnits]
            
            curLabel = None
            if behavior.BehaviorType == "LOW" and missingLowLabel:
                curLabel = "LOW target"
                missingLowLabel = False
            elif behavior.BehaviorType == "HIGH" and missingHighLabel:
                curLabel = "HIGH target"
                missingHighLabel = False
            
            if behavior.BehaviorType == "LOW":
                ax.plot([startBound, endBound], [lowMax, lowMax], color = "#FF4040", linestyle = '--', alpha = 0.7, label = 'LOW threshold')
                ax.axvspan(startBound, endBound, ymax = (lowMax + 0.03) / 1.06, alpha = 0.2, color = "#FF4040", label = curLabel)
            elif behavior.BehaviorType == "HIGH":
                ax.plot([startBound, endBound], [highMin, highMin], color = "#228B22", linestyle='--', alpha=0.7, label = 'HIGH threshold')
                ax.axvspan(startBound, endBound, ymin = (highMin + 0.03) / 1.06, alpha = 0.2, color = "#228B22", label = curLabel)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha = 0.3)
        ax.set_xlim(x[0], x[-1])
        rowIndex = k // squareGridDim
        columnIndex = k % squareGridDim
        if rowIndex == lowestVisibleRowByColumn.get(columnIndex):
            ax.set_xlabel(f"Time ({timeUnits})")

        ax.set_ylim(-0.03, 1.03)

    panelYValues = [[None] * len(iterationList) for _ in range(panelNum)]
    for k, panelInfo in enumerate(timeCourseBehaviorGroupWhitelist):
        tCIdx = panelInfo[0] - 1
        behaviorGroup = timeCourses[tCIdx].BehaviorGroups[panelInfo[1] - 1]
        relevantIndices = behaviorGroup.getRelevantComponentIndices(crnInfo.CompNameList)

        for iterIdx in range(len(iterationList)):
            relevantResults = timeCourseSolutions[tCIdx][iterIdx][:, relevantIndices]
            if len(relevantIndices) == 1:
                rawSignalVals = relevantResults.reshape(-1)
            else:
                rawSignalVals = jnp.sum(relevantResults, axis = 1)

            panelYValues[k][iterIdx] = behaviorGroup.getNormalizeVMAP(initCondComps[iterIdx])(rawSignalVals)

    # Generate a frame for each iteration in iterationList
    for iterIdx, iteration in enumerate(iterationList):
        curPlots = []
        for k, panelInfo in enumerate(timeCourseBehaviorGroupWhitelist):
            tCIdx = panelInfo[0] - 1
            timeCourseObj = timeCourses[tCIdx]
            ax = flatAxes[k]
            x = evalPts[tCIdx]/UNIT_WEIGHTS[timeUnits]
            behaviorGroup = timeCourseObj.BehaviorGroups[panelInfo[1] - 1]
            y = panelYValues[k][iterIdx]
            curPlotColor = mcolors.rgb2hex(startRGB + (endRGB - startRGB) * ((iteration - startIteration) / (endIteration - startIteration + 1)))
            curPlots.append(
                ax.plot(x, y, color = curPlotColor, linewidth = 1.8)
            )
            ax.set_title(fr"Time Course {tCIdx + 1}, $\mathbf{{f}}\left(\text{{[{behaviorGroup.ComponentNames[0]}]}}\right)$", fontsize=9)

        iterTracker = addIterationTracker(fig, iteration + 1, optRes.LastIteration)

        fig.tight_layout(rect = [0, 0, 1, 0.98])
        framePath = workspaceName / f"{fileName}_{iteration}.png"
        plt.savefig(framePath, dpi=100)
        framePaths.append(framePath)
        
        iterTracker.remove()
        if not stackImages:
            for curPlot in curPlots:
                curPlot[0].remove()

    # Write MP4
    with iio.get_writer(workspaceName / f"{fileName}.mp4", fps = fps, codec = 'libx264', format = 'FFMPEG', ffmpeg_params = ['-pix_fmt', 'yuv420p']) as writer:
        for framePath in framePaths:
            image = iio.imread(framePath)
            writer.append_data(image)

    if not keepIndividualImages:
        for framePath in framePaths:
            try:
                os.remove(framePath)
            except OSError:
                pass


def saveBehaviorLandscapeVisualization(
    optBehaviorManagerObject: OptimizeBehaviorsManager,
    timeCourseBehaviorGroupWhitelist: Optional[list[int | tuple[int, int]]] = None,
    fileLocation: Path = Path('./'),
    fileName: str = 'behaviorLandscapeScaffold',
    timeUnits: str = "sec"
):
    if timeUnits not in UNIT_WEIGHTS:
        raise ValueError(f"Unknown timeUnits '{timeUnits}'. Available units: {list(UNIT_WEIGHTS.keys())}")

    timeCourses = optBehaviorManagerObject.BehaviorTimeCourses
    timeCoursesLen = len(timeCourses)

    def raiseTimeCourseLenError(selectionNum):
        raise ValueError(f"Since only {timeCoursesLen} time courses were provided in this optimization run, Time Course {selectionNum} does not exist.")

    panelInfo = []
    if timeCourseBehaviorGroupWhitelist is None:
        for i, timeCourseObj in enumerate(timeCourses):
            for j in range(len(timeCourseObj.BehaviorGroups)):
                panelInfo.append((i + 1, j + 1))
    elif len(timeCourseBehaviorGroupWhitelist) == 0:
        raise ValueError("If providing a whitelist, it must contain at least one entry.")
    else:
        for selection in timeCourseBehaviorGroupWhitelist:
            if isinstance(selection, int):
                timeCourseIdx = selection - 1
                if timeCourseIdx < 0 or timeCourseIdx >= timeCoursesLen:
                    raiseTimeCourseLenError(selection)
                for behaviorGroupIdx in range(len(timeCourses[timeCourseIdx].BehaviorGroups)):
                    panelInfo.append((selection, behaviorGroupIdx + 1))
            else:
                if len(selection) != 2:
                    raise ValueError("Each whitelist tuple must contain exactly (timeCourseNumber, behaviorGroupNumber).")
                timeCourseNum, behaviorGroupNum = selection
                timeCourseIdx = timeCourseNum - 1
                if timeCourseIdx < 0 or timeCourseIdx >= timeCoursesLen:
                    raiseTimeCourseLenError(timeCourseNum)
                if behaviorGroupNum < 1 or behaviorGroupNum > len(timeCourses[timeCourseIdx].BehaviorGroups):
                    raise ValueError(f"Behavior Group {behaviorGroupNum} is not contained within Time Course {timeCourseNum}.")
                panelInfo.append((timeCourseNum, behaviorGroupNum))

    panelNum = len(panelInfo)
    if panelNum == 0:
        raise ValueError("No behavior panels selected to scaffold.")

    squareGridDim = int(np.ceil(np.sqrt(panelNum)))
    gridLen = 4 * squareGridDim
    lowMax = optBehaviorManagerObject.getLowMax()
    highMin = optBehaviorManagerObject.getHighMin()

    fig, axes = plt.subplots(squareGridDim, squareGridDim, figsize=(gridLen, gridLen))
    if squareGridDim == 1:
        axes = np.array([[axes]])
    flatAxes = axes.ravel()

    for m in range(panelNum, squareGridDim**2):
        flatAxes[m].axis('off')

    lowestVisibleRowByColumn = {}
    for columnIndex in range(squareGridDim):
        visibleRows = [
            rowIndex
            for rowIndex in range(squareGridDim)
            if (rowIndex * squareGridDim + columnIndex) < panelNum
        ]
        if len(visibleRows) > 0:
            lowestVisibleRowByColumn[columnIndex] = max(visibleRows)

    for panelIdx, panelPair in enumerate(panelInfo):
        timeCourseIdx = panelPair[0] - 1
        behaviorGroupIdx = panelPair[1] - 1
        timeCourseObj = timeCourses[timeCourseIdx]
        behaviorGroupObj = timeCourseObj.BehaviorGroups[behaviorGroupIdx]

        totalTime = sum([timeSpan.Time for timeSpan in timeCourseObj.TimeCourse]) / UNIT_WEIGHTS[timeUnits]
        ax = flatAxes[panelIdx]

        missingLowLabel = True
        missingHighLabel = True
        for behavior in behaviorGroupObj.Behaviors:
            startBound = behavior.StartBound / UNIT_WEIGHTS[timeUnits]
            endBound = (behavior.EndBound if behavior.EndBound != float('inf') else sum([ts.Time for ts in timeCourseObj.TimeCourse])) / UNIT_WEIGHTS[timeUnits]

            curLabel = None
            if behavior.BehaviorType == "LOW" and missingLowLabel:
                curLabel = "LOW target"
                missingLowLabel = False
            elif behavior.BehaviorType == "HIGH" and missingHighLabel:
                curLabel = "HIGH target"
                missingHighLabel = False

            if behavior.BehaviorType == "LOW":
                ax.plot([startBound, endBound], [lowMax, lowMax], color="#FF4040", linestyle='--', alpha=0.7, label='LOW threshold')
                ax.axvspan(startBound, endBound, ymax=(lowMax + 0.03) / 1.06, alpha=0.2, color="#FF4040", label=curLabel)
            elif behavior.BehaviorType == "HIGH":
                ax.plot([startBound, endBound], [highMin, highMin], color="#228B22", linestyle='--', alpha=0.7, label='HIGH threshold')
                ax.axvspan(startBound, endBound, ymin=(highMin + 0.03) / 1.06, alpha=0.2, color="#228B22", label=curLabel)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(0, totalTime)
        ax.set_ylim(-0.03, 1.03)

        rowIndex = panelIdx // squareGridDim
        columnIndex = panelIdx % squareGridDim
        if rowIndex == lowestVisibleRowByColumn.get(columnIndex):
            ax.set_xlabel(f"Time ({timeUnits})")

        ax.set_title(
            fr"Time Course {timeCourseIdx + 1}, $\mathbf{{f}}\left(\text{{[{behaviorGroupObj.ComponentNames[0]}]}}\right)$",
            fontsize=9
        )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(fileLocation, exist_ok=True)
    plt.savefig(fileLocation / f"{fileName}.png", dpi=100)
    plt.close(fig)

