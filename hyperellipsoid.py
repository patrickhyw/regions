from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import chi2


class Ellipsoid(NamedTuple):
    center: np.ndarray  # (d,) unit-norm mean
    alpha: float  # shrinkage * mu, the identity coefficient
    X: np.ndarray  # (n, d) centered data, or (0, d) for fallback
    M_inv: np.ndarray  # (n, n) Woodbury core inverse, or (0, 0)
    threshold: float  # chi2.ppf(0.95, d)

    def contains(self, vec: list[float]) -> bool:
        """Check whether a vector falls within this hyperellipsoid.

        Uses the Woodbury formula to compute the Mahalanobis distance
        without forming a d x d precision matrix.
        """
        diff = np.asarray(vec) - self.center
        mahal = float(diff @ diff) / self.alpha
        if self.X.shape[0] > 0:
            z = self.X @ diff
            mahal -= float(z @ self.M_inv @ z) / self.alpha**2
        return mahal <= self.threshold


def _ledoit_wolf_shrinkage_gram(X: np.ndarray, G: np.ndarray) -> float:
    """Compute Ledoit-Wolf shrinkage from centered data X and Gram
    matrix G = X @ X.T, without forming any d x d matrix.

    Reproduces sklearn's ledoit_wolf_shrinkage(X,
    assume_centered=True).
    """
    n, d = X.shape
    # mu = trace(S) / d where S = X.T @ X / n
    trace_G = np.trace(G)
    mu = trace_G / (n * d)
    # delta = ||S - mu*I||_F^2 / d
    G_frob_sq = float((G**2).sum())
    delta = G_frob_sq / (n**2 * d) - mu**2
    if delta == 0:
        return 1.0
    # beta from fourth-moment terms, all via G
    diag_G_sq_sum = float((np.diag(G) ** 2).sum())
    beta = (diag_G_sq_sum / n - G_frob_sq / n**2) / (d * n)
    beta = min(beta, delta)
    return 0.0 if beta == 0 else beta / delta


def _identity_ellipsoid(center: np.ndarray, d: int, threshold: float) -> Ellipsoid:
    """Return an identity-precision Ellipsoid (fallback case)."""
    return Ellipsoid(
        center=center,
        alpha=1.0,
        X=np.empty((0, d)),
        M_inv=np.empty((0, 0)),
        threshold=threshold,
    )


def hyperellipsoid(vecs: list[list[float]]) -> Ellipsoid:
    """Compute the hyperellipsoid for a set of vectors.

    Returns a factored Ellipsoid using the Woodbury identity so that
    only n x n matrices are formed (instead of d x d), where n is the
    number of samples.
    """
    arr = np.array(vecs)
    mean = arr.mean(axis=0)
    center = mean / np.linalg.norm(mean)
    centered = arr - center
    n, d = centered.shape
    threshold = float(chi2.ppf(0.95, df=d))
    if n < 3:
        return _identity_ellipsoid(center, d, threshold)
    # Gram matrix (n x n)
    G = centered @ centered.T
    # Shrinkage coefficient
    mu = float(np.trace(G)) / (n * d)
    if mu == 0:
        return _identity_ellipsoid(center, d, threshold)
    s = _ledoit_wolf_shrinkage_gram(centered, G)
    if not (0.0 < s < 1.0):
        return _identity_ellipsoid(center, d, threshold)
    # Factored covariance: Sigma = alpha * I + gamma * X.T @ X
    alpha = s * mu
    gamma = (1.0 - s) / n
    # Woodbury core: M = (1 / gamma) * I_n + (1 / alpha) * G
    M = (1.0 / gamma) * np.eye(n) + (1.0 / alpha) * G
    M_inv = np.linalg.inv(M)
    return Ellipsoid(
        center=center,
        alpha=alpha,
        X=centered,
        M_inv=M_inv,
        threshold=threshold,
    )
