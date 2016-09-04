import games
import numpy as np
from scipy.optimize import linprog
import pudb


def minmax(utility_table):
    """runs the minmax algorithm on the game defined by the matrix P.

    :Parameters:
    utility_table: numpy array of size [m x n] with the payoff for player 1.
    The actions of player 1 correspond to the rows of P.

    :Returns:
    v: minmax value
    s: minmax strategy
    """
    # f = []
    # A = []
    # b = []
    # Aeq = []
    # beq = []

    x = linprog(f, A, b, Aeq, beq)

    # v = 
    # s = 

    return v, s


def maxmin(utility_table):
    """runs the maxmin algorithm on the game defined by the matrix P.

    :Parameters:
    utility_table: numpy array of size [m x n] with the payoff for player 1.
    The actions of player 1 correspond to the rows of P.

    :Returns:
    v: maxmin value
    s: maxmin strategy
    """
    # f = []
    # A = []
    # b = []
    # Aeq = []
    # beq = []

    return v, s


def minmax_slack(utility_table):
    """runs the minmax algorithm with slack variables on the game defined by
    the matrix P.

    :Parameters:
    utility_table: numpy array of size [m x n] with the payoff for player 1.
    The actions of player 1 correspond to the rows of P.

    :Returns:
    v: maxmin value
    s: maxmin strategy
    g: slack variables
    """
    # f = []
    # A = []
    # b = []
    # Aeq = []
    # beq = []

    return v, s, g


def maxmin_slack(utility_table):
    """runs the maxmin algorithm with slack variables on the game defined by
    the utility_table.

    :Parameters:
    utility_table: numpy array of size [m x n] with the payoff for player 1.
    The actions of player 1 correspond to the rows of P.

    :Returns:
    v: maxmin value
    s: maxmin strategy
    g: slack variables
    """
    # f = []
    # A = []
    # b = []
    # Aeq = []
    # beq = []

    return v, s, g


if __name__ == '__main__':
    utility_table = games.UPennyGame

    v_minmax, s_minmax = minmax(utility_table)
    v_maxmin, s_maxmin = maxmin(utility_table)
    v_minmax_slack, s_minmax_slack, g_minmax = minmax_slack(utility_table)
    v_maxmin_slack, s_maxmin_slack, g_maxmin = maxmin_slack(utility_table)

    pudb.set_trace()
