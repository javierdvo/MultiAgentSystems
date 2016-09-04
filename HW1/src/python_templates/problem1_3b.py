import games
import numpy as np
import pudb
from ip_lcp import ip_lcp


def lcp_nash(utility_table):
    """ computes a Nash equilibrium

    :Parameters:
    utility_table: [2 x m x n] array with the pay-offs for both players,
        with m, the number of actions for player 1 and n, the number
        of actions for player 2.

    :Returns:
    v: the utilities for both players
    s: the strategies for both players
    g: the slack variables
    """
    A = np.array([])
    b = np.array([])

    # the initial state
    x0 = np.zeros(b.size)

    # experiment with the maximum number of iterations
    max_iter = 10

    pudb.set_trace()

    # call the interior point lcp solver. Have a look at the documentation of
    # the solver. You can find it in the python script.
    r = ip_lcp(A, b, x0, max_iter)

    v = []
    s = [[], []]
    g = []

    return v, s, g


if __name__ == '__main__':
    lcp_nash(games.UBattleOfSexes)
