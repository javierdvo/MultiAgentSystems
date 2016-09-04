import numpy as np
from scipy.spatial.distance import pdist

import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from plotting import plotPiV
from gridworld import Gridworld
from rbf import rbf


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def policy(s, feature, w, epsilon):
    angles = 8
    lengths = 3

    # Lengths
    a1 = np.ones((len(s), 1)) / lengths
    a1 = a1.dot(np.arange(1, lengths+1).reshape(1, -1))
    a1 = np.tile(np.expand_dims(a1, 2), (1, 1, angles))

    # Angles
    a2 = 2 * np.pi * np.ones((len(s), 1)) / angles
    a2 = a2.dot(np.arange(1, angles+1).reshape(1, -1))
    a2 = np.tile(np.expand_dims(a2, 2), (1, 1, lengths))
    a2 = np.swapaxes(a2, 1, 2)

    a1c = np.hstack((np.zeros((len(s), 1)), a1.reshape(-1, angles*lengths)))
    a2c = np.hstack((np.zeros((len(s), 1)), a2.reshape(-1, angles*lengths)))

    # get all combinations of lengths and angles
    a = np.hstack((a1c[0, :, np.newaxis], a2c[0, :, np.newaxis]))

    # get actions in cartesian space
    A1, A2 = pol2cart(a1c.reshape(-1, 1), a2c.reshape(-1, 1))

    # get the state action values in the form [|s|x(lengths * angles)] so
    # that in row i we have the state action values for all action
    # combinations with state s_i.
    # Q = ...

    # get the index of the maximum Q value for each state
    # maxQi = ...

    # select the action a for each state s_i that maximize Q(s_i,a)
    # a = ...

    # select a random action with probability epsilon
    # a = ...

    return a


def sampleData(gridworld, pi, initialDistribution, numEpisodes,
               maxEpisodeLength):
    """
    :param terminal: [] matrix of the terminal states :param reward: reward
    function handle; takes n states and returns the rewards.
    :param pi: policy function handle; takes n states and returns selected
    actions
    :param initialDistribution: function handle; takes a number n and returns
    n samples of the initial Distribution
    :param numEpisodes: number of episodes to sample
    :param maxEpisodeLength: maximum number of steps to sample before
    aborting the episode (if the episode was not terminated before becuase of
    reaching a terminal state)
    :return:
       s1:         [mx2] states s1
       a:          [mx2] selected actions a
       r:          [mx1] observed reward
       s2:         [mx2] next states s2
    m is the number of sampled steps.
    """
    s1 = np.empty((numEpisodes, 2, maxEpisodeLength))
    s1[:] = np.NAN

    a = np.empty((numEpisodes, 2, maxEpisodeLength))
    a[:] = np.NAN

    r = np.empty((numEpisodes, maxEpisodeLength))
    r[:] = np.NAN

    s2 = np.empty((numEpisodes, 2, maxEpisodeLength))
    s2[:] = np.NAN

    term = np.zeros(numEpisodes, dtype=bool)
    terminal = gridworld.reward != 0

    def sub2ind(matrix, shape, order='F'):
        return np.ravel_multi_index(np.shape(matrix),
                                    dims=shape, order=order)

    X0max = gridworld.reward.shape[0] - 0.0001
    X1max = gridworld.reward.shape[1] - 0.0001

    for t in range(maxEpisodeLength):
        # initialize at first time step
        if t is 0:
            s1[:, :, 0] = initialDistribution(numEpisodes)
            s1[:, 0, 0] = np.maximum(np.minimum(s1[:, 0, 0], X0max), 0.0001)
            s1[:, 1, 0] = np.maximum(np.minimum(s1[:, 1, 0], X1max), 0.0001)

        # check for termination in remaining time steps
        else:
            # idx = 1 if not t
            s1d = np.array(np.floor(s1[np.invert(term), :, t-1]), dtype=np.int)
            term_ = terminal[s1d[:, 0], s1d[:, 1]]
            term[np.invert(term)] = term_

            s1[np.invert(term), :, t] = s2[np.invert(term), :, t-1]

        if np.all(term):
            break

        # get action from policy
        a[np.invert(term), :, t] = pi(s1[np.invert(term), :, t])

        # compute next state
        s2[np.invert(term), :, t] = transitionFunction(
            s1[np.invert(term), :, t], a[np.invert(term), :, t], X0max, X1max)

        # get reward from gridworld
        r[np.invert(term), t] = \
            Gridworld.rewardFunction(s2[np.invert(term), :, t])

    # reshape to two dimensional matrix
    s1 = s1.swapaxes(1, 2).reshape(-1, 2)
    # remove all nan values
    valid = np.invert(np.isnan(s1[:, 1]))
    s1 = s1[valid, :]

    a = a.swapaxes(1, 2).reshape(-1, 2)
    a = a[valid, :]

    r = r.reshape(-1, 1)
    r = r[valid]

    s2 = s2.swapaxes(1, 2).reshape(-1, 2)
    s2 = s2[valid, :]

    return s1, a, r, s2


def transitionFunction(s1, a, X0max, X1max):
    sigma_length = .1
    sigma_angle = .1

    noise = np.random.randn(a.shape[0], a.shape[1])
    noise[:, 0] = noise[:, 0] * sigma_length * a[:, 0]
    a[:, 0] = a[:, 0] + noise[:, 0]
    noise[:, 1] = noise[:, 1] * sigma_angle * a[:, 0]
    a[:, 1] = a[:, 1] + noise[:, 1]

    a_cart_x0, a_cart_x1 = pol2cart(a[:, 0], a[:, 1])
    a_cart = np.array([a_cart_x0, a_cart_x1]).T

    s2 = s1 + a_cart

    s2[:, 0] = np.maximum(np.minimum(s2[:, 0], X0max), 0.0)
    s2[:, 1] = np.maximum(np.minimum(s2[:, 1], X1max), 0.0)

    return s2


def lspi():
    gridworld = Gridworld()

    gamma = .8
    beta = 1
    numEpisodes = 10
    beta_factor = 1 - 5 * numEpisodes/10000

    # Sample the state space with different grid sizes
    X_1, X_2 = np.meshgrid(np.arange(.5, 7, 1), np.arange(.5, 7, 1),
                           indexing='ij')
    X_1m, X_2m = np.meshgrid(np.arange(.5, 7, 0.5), np.arange(.5, 7, 0.5),
                             indexing='ij')

    # initialize the policy randomly. should give for each state in s (nx2) a
    # random action of the form (r,a), where r ~ Unif[0,1] and a ~ Unif[0,2pi].
    # pi = lambda s: ...

     # samples from initial distribution n starting positions (you can start
     # with a random initialization in the entire gridworld)
    # initialDistribution = lambda n: ...

    converged = False

    # generate an ndgrid over the state space for the centers of the basis
    # functions
    X1, X2, A1, A2 = np.meshgrid(np.arange(.5, 7, 1), np.arange(.5, 7, 1),
                                 np.arange(-1, 2), np.arange(-1, 2))
    # NOTE: the policy returns the action in polar coordinates while the basis
    # functions use cartesian coordinates!!! You have to convert between these
    # representations.

    # matrix of the centers
    c = np.column_stack((np.transpose(X1.flatten()),
                         np.transpose(X2.flatten()),
                         np.transpose(A1.flatten()),
                         np.transpose(A2.flatten())))

    # number of basis functions
    # k = ...

    # initialize weights
    # w = ...

    # compute bandwiths with median trick
    bw = np.zeros(4)
    for i in range(4):
        dist = pdist(c[:,[i]])
        bw[i] = np.sqrt(np.median(dist**2)) * .4

    # feature function (making use of rbf)
    # feature = lambda x_: ...

    # time step
    t = 0

    # initialize A and b
    # A = ...
    # b = ...

    while not converged:
        # Policy evaluation
        # sample data
        s1, a, r, s2 = sampleData(gridworld, pi, initialDistribution,
                                  numEpisodes, 50)

        # compute actions in cartesian space
        ac1, ac2 = pol2cart(a[:, 0, np.newaxis], a[:, 1, np.newaxis])

        # compute PHI
        # PHI = ...

        # compute PPI
        # PPI = ...

        # update A and b
        # A = ...
        # b = ...

        # compute new w
        w_old = w
        # w = ...

        # Policy improvement
        # pi = ...

        beta = beta_factor * beta
        t = t + 1

        # Check for convergence
        if np.abs(w-w_old).sum() / len(w) < 0.05:
            converged = True

        print(t, ' - ', beta, ' - ', np.abs(w-w_old).sum() / len(w))

        ### plotting
        a = policy(np.hstack((X_1m.reshape(-1, 1),
                              X_2m.reshape(-1, 1))),
                   feature, w, 0)

        ax1, ax2 = pol2cart(a[:, 0].reshape(-1, 1), a[:, 1].reshape(-1, 1))
        phi = rbf(np.hstack((X_1m.reshape(-1, 1),
                             X_2m.reshape(-1, 1), ax1, ax2)), c, bw)
        Q = phi.dot(w)
        n_plot = len(X_1m)

        plot_a = np.hstack((ax1, ax2)).reshape((n_plot, n_plot, 2))
        plot_V = Q.reshape((n_plot, n_plot))

        plotPiV(plot_a, plot_V, vmin=-5, vmax=5, block=False)

    plotPiV(plot_a, plot_V, vmin=-5, vmax=5)


if __name__ == '__main__':
    lspi()
