import numpy as np
from numba import jit, guvectorize, njit, prange


@jit("f8[:, :](f8[:, :, :], f8[:, :, :])", nopython=True)
def fast_sum(x, ty):
    res = np.sum((x - ty) ** 2, axis=2)
    return res

@njit(parallel=True)
def fast_exp(p, s):
    for i in prange(p.shape[0]):
        for j in prange(p.shape[1]):
            p[i, j] = np.exp(-p[i, j] / (2 * s))
    return p


def initialize_sigma2(X, Y):
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    return np.sum(err) / (D * M * N)


def register_fast(X, Y, max_iterations=45):
    sigma2 = initialize_sigma2(X, Y)
    w = 0.0
    (N, D) = X.shape
    (M, _) = Y.shape
    B = np.eye(D)
    t = np.atleast_2d(np.zeros((1, D)))
    TY = np.dot(Y, B) + np.tile(t, (M, 1))
    iteration = 0
    tolerance = 0.001
    q = np.inf

    while iteration < max_iterations:
        iteration += 1

        # expectation
        P = fast_sum(X[None, :, :], TY[:, None, :])

        c = (2 * np.pi * sigma2) ** (D / 2)
        c = c * w / (1 - w)
        c = c * M / N

        P = fast_exp(P, sigma2)

        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        P = np.divide(P, den)
        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)

        # update transform
        muX = np.divide(np.sum(np.dot(P, X), axis=0), Np)
        muY = np.divide(
            np.sum(np.dot(np.transpose(P), Y), axis=0), Np)

        X_hat = X - np.tile(muX, (N, 1))
        Y_hat = Y - np.tile(muY, (M, 1))

        A = np.dot(np.transpose(X_hat), np.transpose(P))
        A = np.dot(A, Y_hat)

        YPY = np.dot(np.transpose(Y_hat), np.diag(P1))
        YPY = np.dot(YPY, Y_hat)

        B = np.linalg.solve(np.transpose(YPY), np.transpose(A))
        t = np.transpose(
            muX) - np.dot(np.transpose(B), np.transpose(muY))

        # transform pc
        TY = np.dot(Y, B) + np.tile(t, (M, 1))

        # update variance
        qprev = q

        trAB = np.trace(np.dot(A, B))
        xPx = np.dot(np.transpose(Pt1), np.sum(
            np.multiply(X_hat, X_hat), axis=1))
        trBYPYP = np.trace(np.dot(np.dot(B, YPY), B))
        q = (xPx - 2 * trAB + trBYPYP) / (2 * sigma2) + \
                 D * Np / 2 * np.log(sigma2)
        diff = np.abs(q - qprev)

        sigma2 = (xPx - trAB) / (Np * D)

        if sigma2 <= 0:
            sigma2 = tolerance / 10

    return TY, B, t


def register(X, Y, max_iterations=45):
    sigma2 = initialize_sigma2(X, Y)
    w = 0.0
    (N, D) = X.shape
    (M, _) = Y.shape
    B = np.eye(D)
    t = np.atleast_2d(np.zeros((1, D)))
    TY = np.dot(Y, B) + np.tile(t, (M, 1))
    iteration = 0
    tolerance = 0.001
    q = np.inf

    while iteration < max_iterations:
        iteration += 1

        # expectation
        P = np.sum((X[None, :, :] - TY[:, None, :]) ** 2, axis=2)

        c = (2 * np.pi * sigma2) ** (D / 2)
        c = c * w / (1 - w)
        c = c * M / N

        P = np.exp(-P / (2 * sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        P = np.divide(P, den)
        Pt1 = np.sum(P, axis=0)
        P1 = np.sum(P, axis=1)
        Np = np.sum(P1)

        # update transform
        muX = np.divide(np.sum(np.dot(P, X), axis=0), Np)
        muY = np.divide(
            np.sum(np.dot(np.transpose(P), Y), axis=0), Np)

        X_hat = X - np.tile(muX, (N, 1))
        Y_hat = Y - np.tile(muY, (M, 1))

        A = np.dot(np.transpose(X_hat), np.transpose(P))
        A = np.dot(A, Y_hat)

        YPY = np.dot(np.transpose(Y_hat), np.diag(P1))
        YPY = np.dot(YPY, Y_hat)

        B = np.linalg.solve(np.transpose(YPY), np.transpose(A))
        t = np.transpose(
            muX) - np.dot(np.transpose(B), np.transpose(muY))

        # transform pc
        TY = np.dot(Y, B) + np.tile(t, (M, 1))

        # update variance
        qprev = q

        trAB = np.trace(np.dot(A, B))
        xPx = np.dot(np.transpose(Pt1), np.sum(
            np.multiply(X_hat, X_hat), axis=1))
        trBYPYP = np.trace(np.dot(np.dot(B, YPY), B))
        q = (xPx - 2 * trAB + trBYPYP) / (2 * sigma2) + \
                 D * Np / 2 * np.log(sigma2)
        diff = np.abs(q - qprev)

        sigma2 = (xPx - trAB) / (Np * D)

        if sigma2 <= 0:
            sigma2 = tolerance / 10

    return TY, B, t