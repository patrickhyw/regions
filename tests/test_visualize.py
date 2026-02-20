import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.decomposition import PCA

from hyperellipsoid import Hyperellipsoid
from visualize import ellipsoid_surface, marker_size


class TestMarkerSize:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1, 40.0),
            (8, 20.0),
            (1000, 4.0),
            (8_000_000, 0.2),
        ],
    )
    def test_scales_with_cube_root(self, n: int, expected: float) -> None:
        """Marker size follows 40 / n^(1/3), floored at 2."""
        assert marker_size(n) == pytest.approx(max(2.0, expected), rel=1e-6)

    def test_decreases_as_n_grows(self) -> None:
        """More points should produce smaller markers."""
        sizes = [marker_size(n) for n in [10, 100, 1000]]
        assert sizes == sorted(sizes, reverse=True)

    def test_floor_at_one(self) -> None:
        """Marker size never drops below 2."""
        assert marker_size(10**9) == 2.0


class TestHyperellipsoidSurface:
    @pytest.fixture()
    def pca_10d(self) -> PCA:
        """Fit a PCA(3) on 20 random 10-dimensional vectors."""
        X = np.random.standard_normal((20, 10))
        pca = PCA(n_components=3)
        pca.fit(X)
        return pca

    @pytest.fixture()
    def isotropic_ell(self) -> Hyperellipsoid:
        """Hyperellipsoid from isotropic 10-d data (many samples)."""
        data = np.random.standard_normal((60, 10)) * 0.1 + 1.0
        return Hyperellipsoid.fit(data)

    def test_returns_surface_with_correct_color_and_opacity(
        self,
        isotropic_ell: Hyperellipsoid,
        pca_10d: PCA,
    ) -> None:
        """Returns a go.Surface with the requested color and opacity."""
        surf = ellipsoid_surface(isotropic_ell, pca_10d, color="orange", opacity=0.3)
        assert isinstance(surf, go.Surface)
        assert surf.opacity == 0.3
        assert surf.colorscale == ((0, "orange"), (1, "orange"))

    def test_surface_centered_on_projected_center(
        self,
        isotropic_ell: Hyperellipsoid,
        pca_10d: PCA,
    ) -> None:
        """Surface median coordinates match the PCA-projected center."""
        surf = ellipsoid_surface(isotropic_ell, pca_10d, color="blue")
        projected = pca_10d.transform(isotropic_ell.center.reshape(1, -1))[0]
        # The median of a parametric ellipsoid surface is its center.
        median = [np.median(surf.x), np.median(surf.y), np.median(surf.z)]
        np.testing.assert_allclose(median, projected, atol=0.1)

    def test_identity_fallback_produces_sphere(
        self,
        pca_10d: PCA,
    ) -> None:
        """An identity-fallback ellipsoid (empty X) projects to a
        sphere with equal extents in all 3 axes."""
        # Two points trigger the identity fallback (n < 3).
        ell = Hyperellipsoid.fit(np.array([[1.0] * 10, [1.1] * 10]))
        assert ell.X.shape[0] == 0, "Expected identity fallback"
        surf = ellipsoid_surface(ell, pca_10d, color="red")
        ranges = [np.ptp(surf.x), np.ptp(surf.y), np.ptp(surf.z)]
        np.testing.assert_allclose(ranges[0], ranges[1:], rtol=0.05)

    def test_anisotropic_data_produces_elongated_ellipsoid(
        self,
    ) -> None:
        """Data with one dominant direction produces an ellipsoid
        elongated along that direction after PCA projection."""
        d = 10
        n = 60
        # Variance concentrated in the first dimension.
        data = np.random.standard_normal((n, d)) * 0.1
        data[:, 0] += np.random.standard_normal(n) * 5.0
        data += 1.0
        pca = PCA(n_components=3)
        pca.fit(data)
        ell = Hyperellipsoid.fit(data)
        surf = ellipsoid_surface(ell, pca, color="green")
        # PC1 should capture the high-variance direction, so the
        # surface extent along x (PC1) should exceed y and z.
        x_range = np.ptp(surf.x)
        y_range = np.ptp(surf.y)
        z_range = np.ptp(surf.z)
        assert x_range > 2 * y_range
        assert x_range > 2 * z_range
