import numpy as np
import pytest
from scipy.spatial import ConvexHull

from convexhull import (
    Hull,
    _fallback_ball,
    _halfplane_contains,
    _recompute_equations,
    convex_hull,
)
from tree import KnowledgeNode
from util import set_seed


@pytest.fixture()
def triangle_hull() -> Hull:
    """Convex hull of a 2D triangle with vertices (0,0), (4,0), (0,4)."""
    return convex_hull(
        KnowledgeNode(
            concept="r",
            children=[
                KnowledgeNode(concept="a"),
                KnowledgeNode(concept="b"),
            ],
        ),
        {
            "r": [0.0, 0.0],
            "a": [4.0, 0.0],
            "b": [0.0, 4.0],
        },
    )


@pytest.fixture()
def unit_segment_hull() -> Hull:
    """Fallback ball from segment [(0,0), (2,0)]: center (1,0), radius 1."""
    return _fallback_ball(np.array([[0.0, 0.0], [2.0, 0.0]]))


@pytest.fixture()
def high_dim_triangle_vecs() -> np.ndarray:
    """Three points forming a triangle in 10D."""
    vecs = np.zeros((3, 10))
    vecs[1, 0] = 4.0
    vecs[2, 1] = 4.0
    return vecs


@pytest.fixture()
def collinear_vecs() -> np.ndarray:
    """Three collinear points in 3D."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )


@pytest.fixture()
def recomputed_equations() -> tuple[np.ndarray, np.ndarray]:
    """Recomputed halfplane equations for a random 10D point cloud.

    Returns (equations, points) where equations are recomputed from a
    ConvexHull built on points.
    """
    points = np.random.standard_normal((20, 10))
    hull = ConvexHull(points)
    interior = points.mean(axis=0)
    equations = _recompute_equations(hull.simplices, points, interior)
    return equations, points


@pytest.fixture()
def tree() -> KnowledgeNode:
    """Tree with 6 concepts for 2D hull tests (n > d)."""
    return KnowledgeNode(
        concept="r",
        children=[
            KnowledgeNode(concept="a"),
            KnowledgeNode(concept="b"),
            KnowledgeNode(concept="c"),
            KnowledgeNode(concept="d"),
            KnowledgeNode(concept="e"),
        ],
    )


@pytest.fixture()
def representations() -> dict[str, list[float]]:
    """2D representations for the tree fixture (6 non-degenerate points)."""
    return {
        "r": [0.0, 0.0],
        "a": [4.0, 0.0],
        "b": [0.0, 4.0],
        "c": [2.0, 1.0],
        "d": [1.0, 3.0],
        "e": [3.0, 2.0],
    }


@pytest.fixture()
def sampling_tree() -> KnowledgeNode:
    """Tree with 20 concepts for sampling tests."""
    return KnowledgeNode(
        concept="root",
        children=[KnowledgeNode(concept=f"s{i}") for i in range(19)],
    )


@pytest.fixture()
def sampling_reps() -> dict[str, list[float]]:
    """2D representations for the sampling tree fixture."""
    vecs = np.random.standard_normal((20, 2))
    concepts = ["root"] + [f"s{i}" for i in range(19)]
    return {c: v.tolist() for c, v in zip(concepts, vecs)}


class TestConvexHull:
    # Hull.contains tests

    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            pytest.param([4 / 3, 4 / 3], True, id="interior"),
            pytest.param([0.0, 0.0], True, id="vertex"),
            pytest.param([3.0, 3.0], False, id="outside"),
        ],
    )
    def test_triangle_hull_contains(
        self, triangle_hull: Hull, point: list[float], expected: bool
    ) -> None:
        """Triangle hull correctly classifies interior, boundary,
        and exterior points."""
        assert triangle_hull.contains(point) == expected

    def test_fallback_center_is_inside(self) -> None:
        """Fallback ball center is inside."""
        hull = _fallback_ball(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert hull.contains(hull.center.tolist())

    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            pytest.param([2.0, 0.0], True, id="boundary"),
            pytest.param([3.0, 0.0], False, id="outside"),
        ],
    )
    def test_fallback_ball_contains(
        self, unit_segment_hull: Hull, point: list[float], expected: bool
    ) -> None:
        """Fallback ball correctly classifies boundary and exterior
        points. Center is (1,0), radius is 1."""
        assert unit_segment_hull.contains(point) == expected

    # _fallback_ball tests

    def test_single_point(self) -> None:
        """Single point: center equals point, radius is 0."""
        hull = _fallback_ball(np.array([[3.0, 4.0]]))
        np.testing.assert_allclose(hull.center, [3.0, 4.0])
        assert hull.radius == 0.0

    def test_two_points(self) -> None:
        """Two points: center is midpoint, radius is half-distance."""
        hull = _fallback_ball(np.array([[0.0, 0.0], [4.0, 0.0]]))
        np.testing.assert_allclose(hull.center, [2.0, 0.0])
        assert hull.radius == pytest.approx(2.0)

    def test_all_points_contained(self) -> None:
        """All input points satisfy contains."""
        vecs = np.random.standard_normal((10, 3))
        hull = _fallback_ball(vecs)
        for v in vecs:
            assert hull.contains(v.tolist())

    def test_equations_empty(self) -> None:
        """Equations shape is (0, d+1)."""
        hull = _fallback_ball(np.array([[1.0, 2.0, 3.0]]))
        assert hull.equations.shape == (0, 4)

    # convex_hull tests

    def test_returns_hull(
        self,
        tree: KnowledgeNode,
        representations: dict[str, list[float]],
    ) -> None:
        """Returns a Hull instance."""
        result = convex_hull(tree, representations)
        assert isinstance(result, Hull)

    def test_single_point_uses_fallback(self) -> None:
        """A single point produces fallback (empty equations)."""
        node = KnowledgeNode(concept="x")
        reps = {"x": [1.0, 2.0, 3.0]}
        result = convex_hull(node, reps)
        assert result.equations.shape[0] == 0

    def test_valid_hull_has_equations(
        self,
        tree: KnowledgeNode,
        representations: dict[str, list[float]],
    ) -> None:
        """Sufficient non-degenerate points produce non-empty equations."""
        result = convex_hull(tree, representations)
        assert result.equations.shape[0] > 0

    def test_all_original_points_contained(
        self,
        tree: KnowledgeNode,
        representations: dict[str, list[float]],
    ) -> None:
        """All subtree points are inside the hull."""
        result = convex_hull(tree, representations)
        for concept in tree.concepts():
            assert result.contains(representations[concept])

    @pytest.mark.parametrize(
        "k",
        [pytest.param(None, id="none"), pytest.param(1.0, id="1.0")],
    )
    def test_k_uses_all_points(
        self,
        tree: KnowledgeNode,
        representations: dict[str, list[float]],
        k: float | None,
    ) -> None:
        """k=None and k=1.0 both give the same result as the default."""
        default = convex_hull(tree, representations)
        explicit = convex_hull(tree, representations, k=k)
        np.testing.assert_allclose(explicit.equations, default.equations)
        np.testing.assert_allclose(explicit.center, default.center)
        assert explicit.radius == pytest.approx(default.radius)

    def test_k_sampling_uses_subset(
        self,
        sampling_tree: KnowledgeNode,
        sampling_reps: dict[str, list[float]],
    ) -> None:
        """With k < 1.0, only a fraction of points are sampled."""
        all_result = convex_hull(sampling_tree, sampling_reps)
        k_result = convex_hull(sampling_tree, sampling_reps, k=0.4)
        shapes_differ = k_result.equations.shape != all_result.equations.shape
        center_differs = not np.allclose(k_result.center, all_result.center)
        assert shapes_differ or center_differs

    def test_k_sampling_is_deterministic(
        self,
        sampling_tree: KnowledgeNode,
        sampling_reps: dict[str, list[float]],
    ) -> None:
        """Two calls produce identical results (seed is reset)."""
        set_seed()
        r1 = convex_hull(sampling_tree, sampling_reps, k=0.4)
        set_seed()
        r2 = convex_hull(sampling_tree, sampling_reps, k=0.4)
        np.testing.assert_allclose(r1.equations, r2.equations)
        np.testing.assert_allclose(r1.center, r2.center)
        assert r1.radius == pytest.approx(r2.radius)

    def test_hull_shape_reflects_distribution(self) -> None:
        """An elongated point set contains a point along the long axis
        but excludes a same-distance point along the short axis."""
        n = 50
        concepts = [f"c{i}" for i in range(n)]
        root = KnowledgeNode(
            concept=concepts[0],
            children=[KnowledgeNode(concept=c) for c in concepts[1:]],
        )
        # Elongated in y-direction (scale 10) vs x-direction (scale 1).
        vecs = np.random.standard_normal((n, 2)) * [1.0, 10.0]
        vecs += [0.5, 0.5]
        reps = {c: v.tolist() for c, v in zip(concepts, vecs)}
        hull = convex_hull(root, reps)
        center = hull.center
        offset = 5.0
        in_x = hull.contains((center + [offset, 0.0]).tolist())
        in_y = hull.contains((center + [0.0, offset]).tolist())
        assert in_y and not in_x

    # PCA projection tests

    def test_basis_not_none_when_n_le_d(
        self, high_dim_triangle_vecs: np.ndarray
    ) -> None:
        """When n <= d, PCA projection sets basis."""
        hull = Hull.fit(high_dim_triangle_vecs)
        assert hull.basis is not None
        assert hull.basis.shape == (2, 10)

    def test_basis_none_when_n_gt_d(
        self,
        tree: KnowledgeNode,
        representations: dict[str, list[float]],
    ) -> None:
        """When n > d, no PCA projection is used."""
        hull = convex_hull(tree, representations)
        assert hull.basis is None

    def test_high_dim_triangle_points_contained(
        self, high_dim_triangle_vecs: np.ndarray
    ) -> None:
        """All vertices of a triangle in 10D are contained."""
        hull = Hull.fit(high_dim_triangle_vecs)
        for v in high_dim_triangle_vecs:
            assert hull.contains(v.tolist())

    def test_high_dim_triangle_interior_contained(
        self, high_dim_triangle_vecs: np.ndarray
    ) -> None:
        """A convex combination on the plane is inside."""
        hull = Hull.fit(high_dim_triangle_vecs)
        interior = high_dim_triangle_vecs.mean(axis=0)
        assert hull.contains(interior.tolist())

    def test_point_off_subspace_not_contained(
        self, high_dim_triangle_vecs: np.ndarray
    ) -> None:
        """A point displaced off the affine plane is rejected."""
        hull = Hull.fit(high_dim_triangle_vecs)
        interior = high_dim_triangle_vecs.mean(axis=0)
        off_plane = interior.copy()
        off_plane[2] = 1.0
        assert not hull.contains(off_plane.tolist())

    def test_collinear_points_contained(self, collinear_vecs: np.ndarray) -> None:
        """Collinear points are all contained, plus midpoint."""
        hull = Hull.fit(collinear_vecs)
        for v in collinear_vecs:
            assert hull.contains(v.tolist())
        assert hull.contains([1.0, 0.0, 0.0])

    def test_collinear_outside_not_contained(self, collinear_vecs: np.ndarray) -> None:
        """A point beyond the collinear segment is rejected."""
        hull = Hull.fit(collinear_vecs)
        assert not hull.contains([3.0, 0.0, 0.0])

    def test_two_points_in_high_dim_contained(self) -> None:
        """Endpoints and midpoint of a segment in high-D
        are contained."""
        vecs = np.zeros((2, 10))
        vecs[0, 0] = 1.0
        vecs[1, 0] = 3.0
        hull = Hull.fit(vecs)
        assert hull.contains(vecs[0].tolist())
        assert hull.contains(vecs[1].tolist())
        mid = (vecs[0] + vecs[1]) / 2.0
        assert hull.contains(mid.tolist())

    def test_single_point_high_dim(self) -> None:
        """Only the exact point is contained for a single-point
        hull."""
        vecs = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        hull = Hull.fit(vecs)
        assert hull.contains(vecs[0].tolist())
        other = vecs[0].copy()
        other[0] += 0.1
        assert not hull.contains(other.tolist())

    # Hull.fit tests

    def test_matches_convex_hull(
        self,
        tree: KnowledgeNode,
        representations: dict[str, list[float]],
    ) -> None:
        """Hull.fit(vecs) produces the same result as
        convex_hull(node, reps) for equivalent data."""
        concepts = tree.concepts()
        vecs = np.array([representations[c] for c in concepts])
        from_fit = Hull.fit(vecs)
        from_hull = convex_hull(tree, representations)
        np.testing.assert_allclose(from_fit.equations, from_hull.equations)
        np.testing.assert_allclose(from_fit.center, from_hull.center)
        assert from_fit.radius == pytest.approx(from_hull.radius)
        assert from_fit.basis is None
        assert from_hull.basis is None

    @pytest.mark.parametrize(
        "vecs",
        [
            pytest.param(
                np.array([[3.0, 4.0, 5.0]]),
                id="single_vector",
            ),
            pytest.param(
                np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]),
                id="two_vectors",
            ),
            pytest.param(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                id="collinear",
            ),
        ],
    )
    def test_fallback(self, vecs: np.ndarray) -> None:
        """Degenerate inputs produce fallback behavior with empty
        equations."""
        result = Hull.fit(vecs)
        assert result.equations.shape[0] == 0

    # _recompute_equations tests

    def test_recomputed_contains_all_points(
        self, recomputed_equations: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A hull with recomputed equations contains all its input
        points."""
        equations, points = recomputed_equations
        for p in points:
            assert _halfplane_contains(equations, p)

    def test_recomputed_excludes_outside(
        self, recomputed_equations: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A point far outside the hull is rejected by recomputed
        equations."""
        equations, points = recomputed_equations
        outside = np.full(points.shape[1], 100.0)
        assert not _halfplane_contains(equations, outside)

    def test_separable_clusters_no_overlap(self) -> None:
        """Two well-separated clusters in high-D have no overlap
        when using recomputed equations."""
        d = 50
        n = d + 10
        cluster_a = np.random.standard_normal((n, d)) + 100.0
        cluster_b = np.random.standard_normal((n, d)) - 100.0
        hull_a = Hull.fit(cluster_a)
        hull_b = Hull.fit(cluster_b)
        for p in cluster_b:
            assert not hull_a.contains(p.tolist()), (
                "Point from cluster B found inside hull A"
            )
        for p in cluster_a:
            assert not hull_b.contains(p.tolist()), (
                "Point from cluster A found inside hull B"
            )
