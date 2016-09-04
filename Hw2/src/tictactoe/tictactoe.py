import numpy as np
from gtk._gtk import widget_class_find_style_property


class TicTacToe:
    """The tictactoe game"""
    P1 = 1
    P2 = -1

    def __init__(self):
        self._state = np.zeros((3, 3))

    def getTerminalStateAndWinner(self):
        """ returns if the state is in a terminal state. And if so, which
        player has won the game.

        :Returns:
        terminalState - True if the game is in a terminal state, False if not.
        winner -  1 if the first player has won the game,
                 -1 if the seond player has won the game,
                  0 if its a draw or the game is not in a terminal state.
        """
        return self.inTerminalState(),self.winner()


    def winner(self):
        rows = np.split(np.ravel(self._state),3,axis=0)
        column = np.split(np.ravel(np.transpose(self._state)),3,axis=0)

        rows = np.array(rows)
        column = np.array(column)

        # Diagonals
        d1 = np.array(np.diagonal(self._state))
        d2 = np.array(np.diagonal(np.flipud(self._state)))
        d = np.array(np.stack((d1,d2)))

        stack = np.concatenate((rows,column,d),axis=0)

        cond1=np.full(3,TicTacToe.P1,dtype=np.int)==stack
        cond1 = np.all(cond1,axis=1)

        cond2 = np.full(3, TicTacToe.P2, dtype=np.int) == stack
        cond2 = np.all(cond2, axis=1)

        if np.any(cond1):
            return TicTacToe.P1
        elif np.any(cond2):
            return  TicTacToe.P2
        else:
            return 0

    def inTerminalState(self):
        """ returns if the state is a terminal state.

        :Returns:
        terminalState - True if the game is in a terminal state, False if not.
        """
        # Terminal state when no more moves are possible or when one of the players has won
        if np.count_nonzero(self._state) == 9:
            return True
        else:
            winner = self.winner()
            return winner==TicTacToe.P1 or winner==TicTacToe.P2


    def getPlayer(self):
        """ returns which player's turn it is.

        :Returns:
        player - 0 for the first player and 1 for the second player.
        """
        # We just cound the number of zeros
        # if it's odd then it is players1
        # otherwise player 2

        zero=9-np.count_nonzero(self._state)
        if zero % 2 == 1:
            return TicTacToe.P1
        else:
            return TicTacToe.P2

    def getActions(self):
        """ returns the possible actions for the current state of the game.

        :Returns:
        actions - array with the indices of the possible actions

        :Example:
        for the state s = [[-1 0 1], [-1 0 0], [0 0 0]], the function
%       getActions should return [1, 4, 5, 6, 7, 8].
        """
        # the available actions are the indices i s.t s[i]==0
        state = self._state
        state = np.ravel(state)
        cond = np.array(state==0)
        return np.flatnonzero(cond)

    def checkAction(self, action):
        """ checks if a given action is valid for the current state of the
        game.

        :Parameters:
        action - the action as integer between 0 and 8, as index for the
                 flattened game matrix.

        :Returns:
        actionValid - True if the given action can be performed, and False
                      otherwise
        """
        # we just see if the action is part of the available action
        if(type(action) is not int):
            return False
        cond = np.array(self.getActions() == action)
        return np.any(cond)

    def performAction(self, action):
        """ performs the given action on the current state of the game.

        :Parameters:
        action - the action as integer between 0 and 8, as index for the
                 flattened game matrix.

        :Returns:
        actionPerformed - True if the action has been performed, and False
                          otherwise.
        """

        # first we check the action
        # we apply it to the current player if ok
        if(self.checkAction(action)):
            player=self.getPlayer()
            self._state[action//3][action%3]=player
            return True

        return False

    def copy(self):
        return self.__deepcopy__()

    def __repr__(self):
        return "TicTacToe()"

    def __str__(self):
        if not self.inTerminalState():
            actions = np.arange(9, dtype=np.int8).reshape((3, 3))
            actions = actions.astype(np.str_)
        else:
            actions = np.zeros((3, 3), np.str_)
            actions[:] = ' '

        actions[self._state == TicTacToe.P1] = 'X'
        actions[self._state == TicTacToe.P2] = '@'

        ret = '\n|-----------|\n'
        for line in actions:
            ret += '| '
            for field in line:
                ret += ' ' + field + ' '
            ret += ' |\n'
            ret += '|-----------|\n'
        ret += '\n'

        return ret

    def __copy__(self):
        newGame = TicTacToe()
        newGame._state = self._state
        return newGame

    def __deepcopy__(self):
        newGame = TicTacToe()
        newGame._state[:] = self._state
        return newGame
