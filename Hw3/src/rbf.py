import numpy as np


def rbf(X, Mu, bw):
    if np.isscalar(bw):
        bw = np.ones(X.shape[1]) * bw
    Q = np.diag(1/(bw**2))
    XQ = X.dot(Q)
    MuQ = Mu.dot(Q)

    sqdist = np.tile(np.sum(XQ * X, 1, keepdims=True), (1, Mu.shape[0])) + \
        np.tile(np.sum(MuQ * Mu, 1, keepdims=True).T, (X.shape[0], 1)) - \
        2 * XQ.dot(Mu.T)

    Phi = np.exp(-0.5 * sqdist)

    Phi = Phi / np.sqrt((bw**2).prod() * (2 * np.pi)**X.shape[1])

    return Phi
