import numpy as np

import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from plotting import plotPiV
from gridworld import Gridworld

# V = ...
# Q = ...
V_converged = False
w = Gridworld()
gamma = .8
# init V at random
V = np.random.rand(Gridworld.states.shape[0])
V_old = 0
while not V_converged:

    # compute Q function
    # ...
    Q = Gridworld.reward + gamma * V
    # compute V function
    # ...

    V_diff = V - V_old
    V_diff = np.sum(np.absolute(V_diff).flat)
    if V_diff < 0.01:
        V_converged = True
    V = V_old
# convert policy for plot
pi = Gridworld.actions[0][Q.argmax(2).flat].reshape((V.shape[0], V.shape[1], 2))
plotPiV(pi, V)
