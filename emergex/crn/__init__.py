################ Initialization ###############
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 02/24/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

"""
Modules:
    crnConstructor: Core classes for defining reactions, components, and chemical reaction networks.
    wrappers: Domain-specific network constructors.

Example:
    >>> rxn1 = Reaction(5e-5, "A + B -> C")
    >>> rxn2 = Reaction(1e-2, "C + D <-> CD", backwardRate = 1e-0)
    >>> rxn3 = Reaction(3e-2, "CD -> D + E")

    >>> compA = Component("A", 100)
    >>> compB = Component("B", 80)
    >>> compD = Component("D", 5)

    >>> newNetwork = ReactionNetwork()
    >>> newNetwork.addRxns([rxn1, rxn2, rxn3])
    >>> newNetwork.addComponents([compA, compB, compD])
    >>> newNetwork.simulateReactionFn(1000)
"""

from .crnConstructor import (
    Reaction, 
    Component,
    ReactionNetwork,

    # To be used internally
    _getRxnStr
)

__all__ = [
    "Reaction",
    "Component", 
    "ReactionNetwork",

    "_getRxnStr"
]