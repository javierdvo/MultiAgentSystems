import numpy as np
from rbf import rbf

X_2, X_1 = np.meshgrid(np.arange(.5, 7, 1), np.arange(.5, 7, 1))


class Gridworld:
    reward = np.array([
        [ 0,  0,  0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0, -3,  0, -3],
        [-3,  0,  0,  0, -3,  0, -3],
        [ 0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0],
    ])

    actions = np.array([
        [[0, 1, 1, 0, -1, -1, -1,  0,  1],
         [0, 0, 1, 1,  1,  0, -1, -1, -1]],

        [[0,  1, 1, 1, 0, -1, -1, -1,  0],
         [0, -1, 0, 1, 1,  1,  0, -1, -1]],

        [[0, 1, 0, -1, -1, -1,  0,  1, 1],
         [0, 1, 1,  1,  0, -1, -1, -1, 0]],
    ]).swapaxes(1, 2)

    states = np.array([(x, y) for x in range(7) for y in range(7)])

    perturbation_probs = np.array([.8, .1, .1])
    perturbation_probs_cs = perturbation_probs.cumsum()

    def __init__(self, init_state=(6, 5)):
        self.state = np.array(init_state)
        self.state = np.maximum(self.state, (0, 0))
        self.state = np.minimum(self.state, np.array(self.reward.shape) - 1)

    def move(self, action_idx):
        perturbation = np.argwhere(self.perturbation_probs_cs >=
                                   np.random.random())[0]
        a = self.actions[perturbation, action_idx].squeeze()
        self.state += a
        self.state = np.maximum(self.state, (0, 0))
        self.state = np.minimum(self.state, np.array(self.reward.shape) - 1)

        return self.reward[self.state[0], self.state[1]]

    def testMove(state, action):
        newState = state + action
        newState = np.maximum(newState, (0, 0))
        newState = np.minimum(newState, np.array(Gridworld.reward.shape) - 1)

        return newState

    # a reward function for the continuous world. The functions rbf
    # (with bw=.2), np.where (and others) and the grids X1, X2 might be useful.
    def rewardFunction(state):
        reward = Gridworld.reward
        # ...

        return r
