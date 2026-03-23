################ Figure Elements ##############
#                                             #
#               By: Colin Yancey              #
#             Created January 2026            #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

from matplotlib.figure import Figure as figureClass

###############################################

def addIterationTracker(fig: figureClass, iteration: int, totalIter: int, posX: float = 0.5, posY: float = 0.99):
    txt = fig.text(
        posX,
        posY,
        f"Iteration {iteration} / {totalIter}",
        transform=fig.transFigure,
        fontsize=10,
        ha='center',
        va='top',
        bbox=dict(facecolor='white', alpha=0.5)
    )
    return txt