from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import chi2

# Word embeddings live in high-dimensional space where d >> n. Forming
# d x d covariance matrices would be singular and expensive, so we work
# with n x n Gram matrices and use Ledoit-Wolf shrinkage to regularize.


class Hyperellipsoid(NamedTuple):
    center: np.ndarray  # (d,) mean
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
        # Woodbury expansion of diff @ Sigma^{-1} @ diff:
        # (1/alpha)*||diff||^2 - (1/alpha^2)*z @ M^{-1} @ z.
        mahal = float(diff @ diff) / self.alpha
        if self.X.shape[0] > 0:
            z = self.X @ diff
            mahal -= float(z @ self.M_inv @ z) / self.alpha**2
        return mahal <= self.threshold

    @classmethod
    def fit(cls, vecs: list[list[float]], confidence: float = 0.95) -> Hyperellipsoid:
        """Compute the hyperellipsoid for a set of vectors.

        Returns a factored Hyperellipsoid using the Woodbury identity so
        that only n x n matrices are formed (instead of d x d), where
        n is the number of samples.
        """
        vecs_arr = np.asarray(vecs)
        center = vecs_arr.mean(axis=0)
        centered = vecs_arr - center
        n, d = centered.shape
        # Under a Gaussian model, the 95th-percentile chi-squared
        # value gives a region containing ~95% of the probability
        # mass.
        threshold = float(chi2.ppf(confidence, df=d))
        # Ledoit-Wolf needs at least 3 samples for a meaningful
        # shrinkage estimate.
        if n < 3:
            return _identity_ellipsoid(center, d, threshold)
        # Gram matrix (n x n)
        G = centered @ centered.T
        # Shrinkage coefficient
        mu = float(np.trace(G)) / (n * d)
        # Zero average eigenvalue means all points coincide after
        # centering.
        if mu == 0:
            return _identity_ellipsoid(center, d, threshold)
        s = _ledoit_wolf_shrinkage_gram(centered, G)
        # s=1 means pure identity (no data information); s=0 means
        # no regularization (singular when d > n). Both fall back to
        # identity.
        if not (0.0 < s < 1.0):
            return _identity_ellipsoid(center, d, threshold)
        # Sigma = s*mu*I + (1-s)/n * X^T X. Naming alpha (identity
        # weight) and gamma (data weight) keeps the Woodbury formula
        # clean.
        alpha = s * mu
        gamma = (1.0 - s) / n
        # Woodbury identity: invert this n x n matrix instead of the
        # d x d Sigma, which is cheap when n << d.
        M = (1.0 / gamma) * np.eye(n) + (1.0 / alpha) * G
        M_inv = np.linalg.inv(M)
        return Hyperellipsoid(
            center=center,
            alpha=alpha,
            X=centered,
            M_inv=M_inv,
            threshold=threshold,
        )


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
    # The sample covariance is already a scaled identity.
    if delta == 0:
        return 1.0
    # beta from fourth-moment terms, all via G
    diag_G_sq_sum = float((np.diag(G) ** 2).sum())
    beta = (diag_G_sq_sum / n - G_frob_sq / n**2) / (d * n)
    beta = min(beta, delta)  # Clamp keeps shrinkage in [0, 1].
    return 0.0 if beta == 0 else beta / delta


def _identity_ellipsoid(center: np.ndarray, d: int, threshold: float) -> Hyperellipsoid:
    """Return an identity-precision Hyperellipsoid (fallback case).

    Identity precision reduces Mahalanobis to scaled Euclidean distance.
    Empty X and M_inv skip the Woodbury correction term.
    """
    return Hyperellipsoid(
        center=center,
        alpha=1.0,
        X=np.empty((0, d)),
        M_inv=np.empty((0, 0)),
        threshold=threshold,
    )
