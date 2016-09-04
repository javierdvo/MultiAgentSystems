import numpy as np

import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from plotting import plotPiV
from gridworld import Gridworld

# V = ...
# Q = ...

# pi = ...
pi_converged = False

gamma = .8

while not pi_converged:

    # policy evaluation
    V_converged = False
    while not V_converged:
        # compute Q function
        #  ...

        # compute V function
        # ...

        V_diff = V - V_old
        V_diff = np.sum(np.absolute(V_diff).flat)
        if V_diff < 0.01:
            V_converged = True

    # policy improvement (compute greedy policy)
    # ...

    # test for convergence
    pi_diff = pi - pi_old
    pi_diff = np.sum(np.absolute(pi_diff).flat)
    if pi_diff < 0.01:
        pi_converged = True

# convert policy for plot
pi = Gridworld.actions[0][Q.argmax(2).flat].reshape((V.shape[0], V.shape[1],
                                                     2))
plotPiV(pi, V)
