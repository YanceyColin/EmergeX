####### Parameter Optimization Visualizer #####
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
from numpy.typing import NDArray
import numpy as np
import jax.numpy as jnp

import imageio.v2 as iio
import os

import matplotlib.pyplot as plt

from ..results.optimizerResults import CompiledOptimizationData
from .plotting import *

###############################################

def saveFreeParametersVisualization(
    compiledOptDataObject: CompiledOptimizationData,
    iterationList: Optional[list[int]] = None, # If provided, plot will only contain the selected iterations in the list, each denoted by either a number indicating the time course number, or a tuple containing the time course number and the behavior group number within that time course.
    fileLocation: Path = Path('./'),
    fileName: str = 'freeParameters',
    fps: int = 10,
    keepIndividualImages: bool = False,
    lossLineAccentColor: NDArray = np.array([0.122, 0.467, 0.706]), # Coloring the prior iterations in the loss line plot
    baseColor: NDArray = np.array([0.75, 0.75, 0.75]), # Baseline bar color (default is neutral gray)
    posTargetColor: NDArray = np.array([0.5, 0.95, 0.5]), # Bar color when parameter is increasing rapidly (default is green)
    negTargetColor: NDArray = np.array([0.95, 0.5, 0.5]) # Bar color when parameter is decreasing rapidly (default is red)
):
    optRes = compiledOptDataObject.OptimizerRunResultObject
    paramHist = optRes.ParameterHistory
    totalIter = optRes.LastIteration
    parameters = optRes.CRNOptimizationFrameworkObject.Parameters
    paramNames = [str(param) for param in parameters]

    if iterationList is None:
        iterationList = np.arange(totalIter)
    else:
        iterationList = np.array(iterationList) - 1

    workspaceName = fileLocation / 'mp4 Output'
    os.makedirs(workspaceName, exist_ok=True)

    lossHist = optRes.LossHistory
    minLoss = jnp.min(jnp.array(lossHist)).item()
    maxLoss = jnp.max(jnp.array(lossHist)).item()
    multiplierLoss = (maxLoss / minLoss)**(1 / 20)
    xLoss = np.arange(totalIter)

    framePaths = []
    minValue = jnp.min(jnp.stack(paramHist)).item()
    maxValue = jnp.max(jnp.stack(paramHist)).item()
    multiplier = (maxValue / minValue)**(1 / 20)

    for iteration in iterationList:
        params = paramHist[iteration]

        fig, (ax_loss, ax) = plt.subplots(
            2, 1, figsize=(12, 8),
            gridspec_kw={'height_ratios': [1, 4]}
        )

        for a in (ax_loss, ax):
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.grid(True, which='major', axis='y', linestyle='--', color='0.85', alpha=0.5)
            a.grid(True, which='minor', axis='y', linestyle=':', color='0.85', alpha=0.35)
            a.set_axisbelow(True)

        ax_loss.plot(xLoss[:iteration + 1], lossHist[:iteration + 1], color=lossLineAccentColor, linewidth=2.0) # Prior loss
        ax_loss.plot(xLoss[iteration:], lossHist[iteration:], color='0.6', linestyle='--', linewidth=1.5) # Future loss
        #ax_loss.fill_between(xLoss[:iteration + 1], lossHist[:iteration + 1], color=accentColor, alpha=0.08, step=None)
        ax_loss.plot(xLoss[iteration], lossHist[iteration], marker='o', markersize=5, markerfacecolor='white', markeredgecolor=lossLineAccentColor, markeredgewidth=1.5, zorder=5) # Add marker for current iteration

        ax_loss.set_ylabel('Loss', fontsize=11)
        ax_loss.set_xticks([])

        # Top: Loss history plot (log space)
        ax_loss.set_ylim(minLoss / multiplierLoss, maxLoss * multiplierLoss)
        ax_loss.set_yscale('log')
        ax_loss.set_axisbelow(True)
        ax_loss.set_xlim(0, totalIter - 1)

        ax_loss.set_ylabel('Loss')
        ax_loss.set_xticks([])  # keep top plot clean

        bars = ax.bar(np.arange(len(params)), params, tick_label = paramNames, color = baseColor, edgecolor = '0.25', linewidth = 1.5)
        for i, bar in enumerate(bars):
            x = bar.get_x()
            w = bar.get_width()
            ax.hlines(y=paramHist[0][i], xmin=x + w/3, xmax=x + 2*w/3, colors='0.25', linewidth=2.0, zorder=3) # Slightly thicker than the bar edge to distinguish it

        if iteration > 0:
            pct_change = (paramHist[iteration] - paramHist[iteration - 1]) / paramHist[iteration - 1]
            for i, bar in enumerate(bars):
                pc = pct_change[i]
                if pc >= 0:
                    finalColor = baseColor + (posTargetColor - baseColor) * min(pc, 0.25) / 0.25 # map [0, 0.25] -> [0, 1]
                else:
                    finalColor = baseColor + (negTargetColor - baseColor) * -max(pc, -0.2) / 0.2 # map [-0.2, 0] -> [1, 0]
                bar.set_facecolor(tuple(float(x) for x in finalColor))

        ax.set_xlabel('Free Parameters', fontsize=11)
        ax.set_ylabel('Parameter Value', fontsize=11)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xticklabels(paramNames, rotation=30, ha='right', fontsize=9)

        ax.set_ylim(minValue / multiplier, maxValue * multiplier)
        ax.set_yscale('log')
        ax.set_axisbelow(True)

        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.35)

        addIterationTracker(fig, iteration + 1, totalIter)

        fig.tight_layout()
        framePath = workspaceName / f"{fileName}_{iteration}.png"
        plt.savefig(framePath, dpi=100)
        plt.close(fig)
        framePaths.append(framePath)

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
