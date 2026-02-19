from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import chi2


class Hypersphere(NamedTuple):
    center: np.ndarray  # (d,) mean
    variance: float  # average per-dimension variance (isotropic)
    threshold: float  # chi2.ppf(0.95, d)

    def contains(self, vec: list[float]) -> bool:
        """Check whether a vector falls within this hypersphere."""
        diff = np.asarray(vec) - self.center
        return float(diff @ diff) / self.variance <= self.threshold

    @classmethod
    def fit(cls, vecs: list[list[float]]) -> Hypersphere:
        """Fit an isotropic hypersphere to a set of vectors.

        Under an isotropic Gaussian, ||x - center||^2 / sigma^2
        follows chi^2(d), so the 95th percentile gives the
        containment threshold.
        """
        vecs_arr = np.asarray(vecs)
        center = vecs_arr.mean(axis=0)
        n, d = vecs_arr.shape
        sq_dists = np.sum((vecs_arr - center) ** 2, axis=1)
        variance = float(sq_dists.mean()) / d
        threshold = float(chi2.ppf(0.95, df=d))
        return Hypersphere(center=center, variance=variance, threshold=threshold)
