import matplotlib.pyplot as plt
import numpy as np


def plotPiV(p, V, vmin=None, vmax=None, block=True):
    (m, n, _) = p.shape

    plt.close(plt.gcf())
    # plt.clf()
    plt.matshow(V, fignum=1, vmin=None, vmax=None)

    X, Y = np.meshgrid(np.arange(0, m), np.arange(0, n))
    plt.quiver(X, Y, p[:, :, 1], -p[:, :, 0],
               pivot='mid', color='k')

    plt.draw()
    plt.show(block=block)
