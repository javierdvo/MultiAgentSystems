import numpy as np

import sys
import os

sys.path.append(os.path.join(sys.path[0], '..'))

import plotting
import gridworld

alpha = .7
beta = 10
gamma = .8
numEpisodes = 1000

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
        # get state from gridworld (we have to copy since we will modify it)
        s_t = gridworld.state.copy()

        # get action from soft-max policy
        # ...
        # a_t = ...

        # apply action and get reward
        # R = ...

        # next state
        # s_next = ...

        # next actions
        # ...
        # a_next = ...

        # perform sarsa update
        # Q[s_t[0], s_t[1], a_t] = ...

    # in the terminal states we don't want to consider any other action
    # besides staying where we are
    Q[s_next[0], s_next[1], 1:] = -1000
    # update the state action value function for the terminal state
    # Q[s_next[0], s_next[1], 0] = ...

    # decrease beta
    beta *= .999

# compute value function and policy for plot
V = Q.max(2)
pi = gridworld.Gridworld.actions[0][Q.argmax(2).flat].reshape((V.shape[0], V.shape[1], 2))
plotting.plotPiV(pi, V)
