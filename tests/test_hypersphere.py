import numpy as np
import pytest
from scipy.stats import chi2

from hypersphere import Hypersphere


class TestHypersphere:
    # --- Hypersphere.fit() ---

    def test_returns_hypersphere(self, vecs: list[list[float]]) -> None:
        """Returns a Hypersphere NamedTuple."""
        result = Hypersphere.fit(vecs)
        assert isinstance(result, Hypersphere)

    def test_center_is_mean(self, vecs: list[list[float]]) -> None:
        """Center is the raw arithmetic mean of input vectors."""
        result = Hypersphere.fit(vecs)
        expected = np.array(vecs).mean(axis=0)
        np.testing.assert_allclose(result.center, expected)

    def test_variance_is_average_per_dim(self, vecs: list[list[float]]) -> None:
        """Variance = mean(squared distances from center) / d."""
        result = Hypersphere.fit(vecs)
        arr = np.array(vecs)
        center = arr.mean(axis=0)
        d = arr.shape[1]
        sq_dists = np.sum((arr - center) ** 2, axis=1)
        expected = float(sq_dists.mean()) / d
        assert result.variance == pytest.approx(expected)

    @pytest.mark.parametrize(
        "confidence",
        [pytest.param(0.95, id="95"), pytest.param(0.99, id="99")],
    )
    def test_threshold_is_chi_squared(
        self, vecs: list[list[float]], confidence: float
    ) -> None:
        """Threshold equals chi2.ppf(confidence, d)."""
        result = Hypersphere.fit(vecs, confidence=confidence)
        d = 3  # conftest vectors are 3-dimensional
        assert result.threshold == pytest.approx(chi2.ppf(confidence, d))

    # --- Hypersphere.contains() ---

    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            pytest.param([0.0, 0.0], True, id="center_inside"),
            pytest.param([1.0, 0.0], True, id="boundary_inside"),
            pytest.param([1.1, 0.0], False, id="outside"),
        ],
    )
    def test_contains(
        self,
        point: list[float],
        expected: bool,
    ) -> None:
        """Points at/within threshold are inside; beyond are outside."""
        # variance=1.0 and threshold=1.0: contains iff ||point||² <= 1.
        sphere = Hypersphere(
            center=np.array([0.0, 0.0]),
            variance=1.0,
            threshold=1.0,
        )
        assert sphere.contains(point) == expected

    def test_center_always_inside(self, vecs: list[list[float]]) -> None:
        """The center of a fitted hypersphere is always contained."""
        sphere = Hypersphere.fit(vecs)
        assert sphere.contains(sphere.center.tolist())
