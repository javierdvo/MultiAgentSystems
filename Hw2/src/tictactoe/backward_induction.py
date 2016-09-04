import numpy
def backwardInduction(game):
    """
    :Parameters:
    game - an instance of TicTacToe

    :Returns:
    bestUtil - best utility that can be achieved against a perfect opponent
               from the given state of the game
    bestAction - list of actions that lead to an outcome with bestUtil as
                 utility
    """

    bestAction = numpy.empty(1,dtype=numpy.int)
    player = game.getPlayer()
    opponent = -player
    bestUtility = -1
    terminalState, winner = game.getTerminalStateAndWinner()
    if(terminalState):
        if winner == player:
            return 1,bestAction
        elif winner == opponent:
            return -1,bestAction
        else:
            return 0,bestAction

    actions=game.getActions()

    for a in actions:
        copy = game.copy()
        copy.performAction(a)
        cutil,ca = backwardInduction(copy)
        if(cutil>=bestUtility):
            bestUtility,bestAction=cutil,numpy.append(bestAction,ca)
        if (cutil==1):
            break

    return bestUtility,bestAction


