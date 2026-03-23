###### Chemical Reaction Network Example ######
#                                             #
#               By: Colin Yancey              #
#             Created February 2025           #
#            Last Edited: 03/05/2026          #
#                 Schulman Lab                #
#           Johns Hopkins University          #
#                                             #
###############################################

"""
The CRN Constructor module was designed first, running independently from all other
EmergeX modules. Below is an example of how to make a simple reaction system without any other
EmergeX tools and run a reaction with components added intermittently.

The CRN Constructor tool assumes no specific units of concentration or time. It is up to
the user to keep track of these.
"""

import matplotlib.pyplot as plt
import time

from emergex.crn import *

###############################################

curTime = time.time()

# Units are arbitrary, so long as they are INTERNALLY CONSISTENT
# (i.e., if your second-order rates use 1/sec/uM, all concentrations should be in uM)

simTime = 600 # In units of sec
simResolution = 201 # Number of points retrieved

rxn1 = Reaction(5e-5, "A + B -> C") # In units of 1/sec/uM
rxn2 = Reaction(1e-2, "C + D <-> CD", backwardRate = 1e-0) # In units of 1/sec/uM, and in units of 1/sec
rxn3 = Reaction(8e-2, "CD -> D + E") # In units of 1/sec

compA = Component("A", 100) # In units of uM
compB = Component("B", 80) # In units of uM
compD = Component("D", 5) # In units of uM

newNetwork = ReactionNetwork()
newNetwork.addRxns([rxn1, rxn2, rxn3])
newNetwork.addComponents([compA, compB, compD])
newNetwork.simulateReactionFn(simTime, simDataResolution = simResolution)

# By default the last time point is held in memory. To wipe this, you may call clearSimResults() or set
# the continueFromLast flag in your simulateReactionFn call to False. Make sure when clearing that you
# save any data you would like to plot afterwards by assigning entries to a local variable,
# i.e. firstData = newNetwork.SimResults[0].
 
# Change the concentrations of A and B, stimulating the reaction to continue.
newNetwork.adjustResultComponent("A", 40)
newNetwork.adjustResultComponent("B", 50)

newNetwork.simulateReactionFn(simTime / 2, simDataResolution = simResolution // 2)


components = ["A", "B", "C", "D", "CD", "E"] # Add or remove what you would like plotted

print(f"Time to Simulate: {(time.time() - curTime):.4g} sec")
print("--- Final Concentrations ---\n" + str(newNetwork.SimLastPoint))

# Plot each simulation result vs. the time points recorded
for i, component in enumerate(components):
    # Iterate through the periods that results were computed. (SimResults holds reach simulation run in order from oldest to newest)
    noColorLegend = True
    for results in newNetwork.SimResults:
        plt.plot(
            results["Time"],
            results[component],
            color = f"C{i}",
            label = components[i] if noColorLegend else None
        )
        noColorLegend = False

    plt.xlabel("Time (seconds)")
    plt.ylabel("Concentration (uM)")
    plt.legend()

plt.show()