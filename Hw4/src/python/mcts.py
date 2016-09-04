import numpy as np
from mcts_classes import POMDP, HistoryNode
import pudb

# time horizon
T = 0

# Observation probabilities (state x action x observation -> probability)
O = np.zeros((2, 3, 3))
# TODO: add observation probabilities

# Reward (state x action -> reward)
R = np.zeros((2, 3))
# TODO: add reward

# Transition probabilities (state x action x nextState -> probability)
P = np.zeros((2, 3, 2))
# TODO: add transition probabilities


terminalActions = np.array(())
pomdp = POMDP(T, O, R, P, terminalActions)
history = []

rootNode = HistoryNode(None, history,
                       pomdp.getBeliefFromHistory(history, np.array((.5, .5))),
                       pomdp)

for i in range(int(1e5)):
    # select
    selected, state, rewards, alreadyExpanded = rootNode.select(None, [])

    # expand
    if not alreadyExpanded:
        expanded, state, immediateReward = selected.expand(state)
        rewards.append(immediateReward)
    else:
        expanded = selected

    # simulate
    rewards.append(expanded.rollout(state))

    # update
    expanded.update(rewards, len(rewards)-1)

rootNode.getBestPath().printHistory()
pudb.set_trace()
