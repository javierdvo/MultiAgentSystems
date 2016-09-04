import numpy as np


class POMDP:
    """docstring for POMDP"""
    def __init__(self, T, O, R, P, terminalActions):
        self.T = T
        self.O = O
        self.R = R
        self.P = P
        self.terminalActions = terminalActions
        self.numObservations = O.shape[1]
        self.numStates, self.numActions, _ = P.shape

    def isTerminalAction(self, action):
        return np.any(self.terminalActions == action)

    def sampleNextState(self, state, action):
        p_next = self.P[state, action]
        r = np.random.rand()
        return np.where(p_next.cumsum() > r)[0][0]

    def sampleObservation(self, state, action):
        p_observation = self.O[state, action]
        r = np.random.rand()
        return np.where(p_observation.cumsum() > r)[0][0]

    def getBeliefFromHistory(self, history, initialBelief):
        belief = initialBelief
        for i in range(len(history)):
            belief = self.updateBelief(belief, history[i].action,
                                       history[i].observation)

        return belief

    def updateBelief(self, oldBelief, action, observation):
        newBelief = self.O[:, action, observation] * \
            (self.P[:, action, :].dot(oldBelief))
        newBelief = newBelief / newBelief.sum()
        return newBelief

    def getImmediateReward(self, state, action):
        return self.R[state, action]


class TreeNode:
    """docstring for TreeNode"""
    def __init__(self, parent, pomdp):
        self.pomdp = pomdp
        self.parent = parent
        self.children = []
        self.isFullyExpanded = False
        self.visitationCount = 0
        self.reward = 0

    def getBestPath(self):
        if len(self.children) == 0:
            return self

        maxVisits = 0
        idx = 0
        for i in range(len(self.children)):
            if issubclass(self.children[i].__class__, TreeNode) and \
                    self.children[i].visitationCount >= maxVisits:
                maxVisits = self.children[i].visitationCount
                idx = i

        return self.children[idx].getBestPath()


class HistoryNode(TreeNode):
    """docstring for HistoryNode"""
    def __init__(self, parent, history, beliefState, pomdp):
        super(HistoryNode, self).__init__(parent, pomdp)
        self.beliefState = beliefState
        self.history = history
        self.isInitialized = True
        if len(history) > 0 and (len(history) >= pomdp.T or
                                 pomdp.isTerminalAction(history[-1]['action'])):
            self.isTerminal = True
        else:
            self.isTerminal = False

    def rollout(self, sampledState):
        reward = 0
        if self.isTerminal:
            return reward

        s = sampledState
        for i in range(self.pomdp.T):
            # TODO: implement
            pass

        return reward

    def expand(self, sampledState):
        if self.isFullyExpanded:
            raise AssertionError('Node is already fully expanded')

        nextAction = len(self.children)
        actionNode = ActionNode(self, nextAction, self.pomdp)
        self.children.append(actionNode)
        reward = self.pomdp.getImmediateReward(sampledState, nextAction)
        if len(self.children) == self.pomdp.numActions:
            self.isFullyExpanded = True

        newChild, state, reward, _ = actionNode.select(sampledState, reward)
        return newChild, state, reward

    def select(self, sampledState, pastReward):
        alreadyExpanded = False

        if sampledState is None:
            if np.random.rand() <= self.beliefState[0]:
                sampledState = 0
            else:
                sampledState = 1

        if self.isTerminal:
            return (self, sampledState, pastReward, True)
        elif not self.isFullyExpanded:
            return (self, sampledState, pastReward, False)

        bestUcbIndex = 0
        # TODO: implement

        pastReward.append(self.pomdp.getImmediateReward(sampledState,
                                                        bestUcbIndex))
        (treeNode, state, pastReward, alreadyExpanded) = \
            self.children[bestUcbIndex].select(sampledState, pastReward)

        return (treeNode, state, pastReward, alreadyExpanded)

    def update(self, reward, idx):
        self.reward += np.sum(reward[idx:])
        self.visitationCount += 1
        if not self.parent is None:
            self.parent.update(reward, idx-1)

    def printHistory(self):
        actionStrings = {
            0: 'opened the left door',
            1: 'listened',
            2: 'opened the right door'
        }
        observationStrings = {
            0: 'heard nothing',
            1: 'heard the tiger on the left',
            2: 'heard the tiger on the right'
        }
        for i in range(len(self.history)):
            print('-> I {action} and {observation}.'.format(
                action=actionStrings[self.history[i]['action']],
                observation=observationStrings[self.history[i]['observation']]))


class ActionNode(TreeNode):
    """docstring for ActionNode"""
    def __init__(self, parent, action, pomdp):
        super(ActionNode, self).__init__(parent, pomdp)
        self.action = action
        self.children = [None for i in range(pomdp.numObservations)]

    def select(self, sampledState, reward):
        a = self.action
        s = self.pomdp.sampleNextState(sampledState, a)
        o = self.pomdp.sampleObservation(s, a)

        if not self.children[o] is None:
            # the observation is already in the tree
            return self.children[o].select(s, reward)
        else:
            # the observation is new, hence a new HistoryNode has to be created
            alreadyExpanded = True
            newBelief = self.pomdp.updateBelief(self.parent.beliefState,
                                                self.action, o)
            newHistory = self.parent.history.copy()
            newHistory.append({'action': a, 'observation': o})
            treeNode = HistoryNode(self, newHistory, newBelief, self.pomdp)
            self.children[o] = treeNode

            return treeNode, s, reward, alreadyExpanded

    def update(self, reward, idx):
        self.reward += np.sum(reward[idx:])
        self.visitationCount += 1
        if not self.parent is None:
            self.parent.update(reward, idx)

