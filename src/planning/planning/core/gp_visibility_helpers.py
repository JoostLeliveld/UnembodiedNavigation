"""Shared GP utilities for empirical visibility fitting and lookup."""

from __future__ import annotations

import numpy as np


def clip_prob(p: np.ndarray | float, eps: float) -> np.ndarray | float:
    return np.clip(p, eps, 1.0 - eps)


def sigmoid(x):
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-arr))


def logit(p):
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


class SimpleRBFGP:
    """Lightweight RBF GP regressor used for visibility-field fitting."""

    def __init__(self, *, length_scale=1.5, signal_var=1.0, noise_var=5e-2, jitter=1e-8):
        self.length_scale = float(length_scale)
        self.signal_var = float(signal_var)
        self.noise_var = float(noise_var)
        self.jitter = float(jitter)
        self.X_train = None
        self.y_mean = None
        self.L = None
        self.alpha = None

    def _kernel(self, Xa, Xb):
        Xa = np.asarray(Xa, dtype=float)
        Xb = np.asarray(Xb, dtype=float)
        d2 = np.sum((Xa[:, None, :] - Xb[None, :, :]) ** 2, axis=2)
        ls2 = max(self.length_scale ** 2, 1e-12)
        return self.signal_var * np.exp(-0.5 * d2 / ls2)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.X_train = X
        self.y_mean = float(y.mean())
        y0 = y - self.y_mean

        K = self._kernel(X, X)
        K = K + (self.noise_var + self.jitter) * np.eye(X.shape[0])
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y0))
        return self

    def predict_mean_std(self, X):
        X = np.asarray(X, dtype=float)
        Ks = self._kernel(X, self.X_train)
        mean = self.y_mean + Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.signal_var - np.sum(v * v, axis=0), 1e-12)
        return mean, np.sqrt(var)
