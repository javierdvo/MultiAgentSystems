import numpy as np

import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from plotting import plotPiV
from gridworld import Gridworld

alpha = .7
beta = 10
gamma = .8
numEpisodes = 1000

# IMPLEMENT SARSA BEFORE Q-LEARNING

# initialize Q function
# Q = ...

# for each iteration
for e in range(numEpisodes):
    # initialize gridworld (with initial distribution)
    # gridworld = ...

    # initialize reward
    # R = ...

    # any state that gives a reward is a terminal state
    while R == 0:
        # ...
        pass

    # ...

    # decrease beta
    beta *= .999

# compute value function and policy for plot
V = Q.max(2)
pi = Gridworld.actions[0][Q.argmax(2).flat].reshape((V.shape[0], V.shape[1], 2))
plotPiV(pi, V)
