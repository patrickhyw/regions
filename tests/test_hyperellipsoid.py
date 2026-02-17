import numpy as np
import pytest
from analytics.hyperellipsoid import (
    Ellipsoid,
    _identity_ellipsoid,
    _ledoit_wolf_shrinkage_gram,
    hyperellipsoid,
)
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf


class TestHyperellipsoid:
    # --- Ledoit-Wolf shrinkage gram ---

    @pytest.mark.parametrize(
        ("n", "d"),
        [(5, 3), (10, 4), (20, 8), (50, 10)],
        ids=["5x3", "10x4", "20x8", "50x10"],
    )
    def test_matches_sklearn(self, rng: np.random.Generator, n: int, d: int) -> None:
        """Shrinkage coefficient matches sklearn's LedoitWolf."""
        X = rng.standard_normal((n, d))
        X -= X.mean(axis=0)
        G = X @ X.T
        shrinkage = _ledoit_wolf_shrinkage_gram(X, G)
        lw = LedoitWolf(assume_centered=True).fit(X)
        assert shrinkage == pytest.approx(lw.shrinkage_, abs=1e-10)

    def test_identical_rows_returns_one(self, rng: np.random.Generator) -> None:
        """When all rows are identical (zero variance after centering),
        returns 1.0."""
        row = rng.standard_normal(5)
        X = np.tile(row, (4, 1))
        X -= X.mean(axis=0)
        G = X @ X.T
        assert _ledoit_wolf_shrinkage_gram(X, G) == 1.0

    # --- Factored Mahalanobis ---

    def test_matches_direct(self) -> None:
        """Woodbury Mahalanobis matches diff @ precision @ diff."""
        # Seed 123 produces well-conditioned data (shrinkage in (0, 1)).
        rng = np.random.default_rng(seed=123)
        n, d = 8, 4
        random_vecs = rng.standard_normal((n, d)).tolist()
        ellipsoid = hyperellipsoid(random_vecs)
        # Reconstruct LW precision the old way.
        arr = np.array(random_vecs)
        mean = arr.mean(axis=0)
        center = mean / np.linalg.norm(mean)
        centered = arr - center
        lw = LedoitWolf(assume_centered=True).fit(centered)
        # Test several query points.
        for _ in range(10):
            query = rng.standard_normal(d)
            diff = query - ellipsoid.center
            direct = float(diff @ lw.precision_ @ diff)
            z = ellipsoid.X @ diff
            factored = float(
                diff @ diff / ellipsoid.alpha
                - z @ ellipsoid.M_inv @ z / ellipsoid.alpha**2
            )
            assert factored == pytest.approx(direct, rel=1e-8)

    # --- hyperellipsoid() ---

    def test_returns_ellipsoid(self, vecs: list[list[float]]) -> None:
        """Returns an Ellipsoid NamedTuple."""
        result = hyperellipsoid(vecs)
        assert isinstance(result, Ellipsoid)

    def test_center_is_normalized_mean(self, vecs: list[list[float]]) -> None:
        """Center is the unit-norm mean of all subtree embeddings."""
        result = hyperellipsoid(vecs)
        # Mean of all 5 vectors: [3/5, 3/5, 3/5]
        raw_mean = np.array([0.6, 0.6, 0.6])
        expected = raw_mean / np.linalg.norm(raw_mean)
        np.testing.assert_allclose(result.center, expected)

    @pytest.mark.parametrize(
        "input_vecs",
        [
            pytest.param([[3.0, 4.0]], id="single_node"),
            pytest.param([[1.0, 0.0], [0.0, 1.0]], id="two_points"),
            pytest.param([[1.0, 0.0, 0.0]] * 3, id="identical_points"),
        ],
    )
    def test_hyperellipsoid_identity_fallback(
        self,
        input_vecs: list[list[float]],
    ) -> None:
        """Degenerate subtrees use the identity fallback."""
        d = len(input_vecs[0])
        result = hyperellipsoid(input_vecs)
        assert result.alpha == 1.0
        assert result.X.shape == (0, d)
        assert result.M_inv.shape == (0, 0)

    def test_m_inv_is_symmetric_positive_definite(
        self, vecs: list[list[float]]
    ) -> None:
        """M_inv matrix is symmetric and positive definite."""
        result = hyperellipsoid(vecs)
        assert result.M_inv.shape[0] > 0
        np.testing.assert_allclose(result.M_inv, result.M_inv.T, atol=1e-10)
        eigenvalues = np.linalg.eigvalsh(result.M_inv)
        assert all(eigenvalues > 0)

    def test_threshold_is_chi_squared_95(self, vecs: list[list[float]]) -> None:
        """Threshold equals chi2.ppf(0.95, d) where d is the
        dimensionality of the embeddings."""
        result = hyperellipsoid(vecs)
        d = 3  # conftest vectors are 3-dimensional
        assert result.threshold == pytest.approx(chi2.ppf(0.95, d))

    # --- Ellipsoid.contains() ---

    @pytest.mark.parametrize(
        ("center", "point", "expected"),
        [
            pytest.param([1.0, 2.0], [1.0, 2.0], True, id="center_inside"),
            pytest.param([0.0, 0.0], [1.0, 0.0], True, id="boundary_inside"),
            pytest.param([0.0, 0.0], [1.1, 0.0], False, id="outside"),
        ],
    )
    def test_contains(
        self,
        center: list[float],
        point: list[float],
        expected: bool,
    ) -> None:
        """Points at/within threshold are inside; beyond are outside."""
        ellipsoid = _identity_ellipsoid(np.array(center), 2, threshold=1.0)
        assert ellipsoid.contains(point) == expected

    def test_directional_scaling(self, rng: np.random.Generator) -> None:
        """An ellipsoid from anisotropic data is tighter in the
        low-variance direction."""
        n = 50
        # Variance ratio ~100:1 (y vs x). Small offset so mean is
        # nonzero but doesn't dominate the variance.
        arr = rng.standard_normal((n, 2)) * [1.0, 10.0]
        arr += [0.5, 0.5]
        aniso_vecs = arr.tolist()
        ellipsoid = hyperellipsoid(aniso_vecs)
        center = ellipsoid.center
        # Same offset in both directions: tighter along x.
        offset = 5.0
        in_x = ellipsoid.contains((center + [offset, 0.0]).tolist())
        in_y = ellipsoid.contains((center + [0.0, offset]).tolist())
        assert in_y and not in_x
