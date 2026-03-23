############ Configuration Variables ##########
#                                             #
#               By: Colin Yancey              #
#            Created September 2025           #
#            Last Edited: 03/02/2026          #
#                  Schulman Lab               #
#            Johns Hopkins University         #
#                                             #
###############################################

# Some relevant unit conversions for CRNs.
UNIT_WEIGHTS = {
    "M": 1,
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,

    "hours": 3600, # Avoiding h, m, s to avoid potential unit ambiguities
    "hr": 3600,
    "minutes": 60,
    "min": 60,
    "seconds": 1,
    "sec": 1
}

# Possible behavior states.
ALLOWED_BEHAVIORS = ["LOW", "HIGH"]

# Possible interruption states.
ALLOWED_INTERRUPTIONS = ["SUBSTITUTE", "ADD", "SUBTRACT"]