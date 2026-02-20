from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull as ScipyConvexHull
from scipy.spatial import QhullError

from tree import KnowledgeNode

# QHull needs n > d, but word embeddings typically have n <= d. For
# n <= d we use QP distance to the convex hull with an OAS-derived
# tolerance. For n > d, QHull works directly.

# Slack for halfplane checks so boundary vertices aren't rejected by
# floating-point noise.
_HALFPLANE_TOL: float = 1e-12
# Z-score multiplier for OAS tolerance (~95% coverage under Gaussian
# assumption).
_OAS_Z: float = 2.0


def _oas_epsilon(S: np.ndarray, n: int, d: int, rank: int) -> float:
    """Compute OAS shrinkage tolerance from SVD singular values.

    Uses the Oracle Approximating Shrinkage formula (Chen et al., 2010)
    to estimate a data-driven tolerance for convex hull containment
    in the directions orthogonal to the data subspace.

    Returns 0.0 when n <= 1, rank = 0, or d = rank (no null space).
    """
    if n <= 1 or rank == 0 or d == rank:
        return 0.0
    eigenvalues = S**2 / (n - 1)
    tr_S = float(np.sum(eigenvalues))
    tr_S2 = float(np.sum(eigenvalues**2))
    mu = tr_S / d
    rho_num = (1 - 2 / d) * tr_S2 + tr_S**2
    rho_den = (n + 1 - 2 / d) * (tr_S2 - tr_S**2 / d)
    if rho_den == 0:
        return 0.0
    rho = rho_num / rho_den
    return float(_OAS_Z * math.sqrt((d - rank) * rho * mu))


def _qp_distance(vertices: np.ndarray, point: np.ndarray) -> float:
    """Find the distance from a point to the convex hull of vertices.

    Solves min ||p - V^T lambda|| s.t. lambda >= 0, sum(lambda) = 1
    via SLSQP. vertices is (n, d), point is (d,).
    """
    n = vertices.shape[0]
    if n == 1:
        return float(np.linalg.norm(point - vertices[0]))

    def objective(lam: np.ndarray) -> float:
        diff = point - lam @ vertices
        return float(diff @ diff)

    def gradient(lam: np.ndarray) -> np.ndarray:
        diff = point - lam @ vertices
        return -2.0 * (vertices @ diff)

    lam0 = np.full(n, 1.0 / n)
    result = minimize(
        objective,
        lam0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints={"type": "eq", "fun": lambda lam: np.sum(lam) - 1.0},
    )
    return float(math.sqrt(max(result.fun, 0.0)))


def _recompute_equations(
    simplices: np.ndarray,
    points: np.ndarray,
    interior: np.ndarray,
) -> np.ndarray:
    """Recompute facet halfplane equations via SVD for numerical precision.

    QHull's equations can be imprecise in high dimensions. This keeps
    QHull's topology (which points form each facet) but recomputes each
    facet's normal and offset independently.

    Returns an (nfacet, d+1) array in the same format as
    scipy's ConvexHull.equations: each row is [normal..., offset] such that
    normal @ x + offset <= 0 for points inside the hull.
    """
    d = points.shape[1]
    nfacet = simplices.shape[0]
    equations = np.empty((nfacet, d + 1))
    for i, facet_indices in enumerate(simplices):
        verts = points[facet_indices]
        centered = verts[1:] - verts[0]
        # The last right singular vector is orthogonal to all edge
        # vectors, giving the facet normal. More numerically stable
        # than QHull's internal equations in high dimensions.
        _, _, Vt = np.linalg.svd(centered, full_matrices=True)
        normal = Vt[-1]
        offset = -normal @ verts[0]
        # Orient so interior satisfies normal @ x + offset <= 0.
        if normal @ interior + offset > 0:
            normal = -normal
            offset = -offset
        equations[i, :-1] = normal
        equations[i, -1] = offset
    return equations


def _halfplane_contains(equations: np.ndarray, point: np.ndarray) -> bool:
    """Check whether a point satisfies all halfplane equations.

    Uses a small positive tolerance so that boundary vertices (which
    should be inside) aren't rejected by floating-point rounding.
    """
    return bool(np.all(equations[:, :-1] @ point + equations[:, -1] <= _HALFPLANE_TOL))


class ConvexHull(NamedTuple):
    equations: np.ndarray  # (nfacet, d+1) halfplanes, or (0, ?) fallback
    center: np.ndarray  # (d,) mean of points in ambient space
    radius: float  # max distance from center (for fallback ball)
    vertices: np.ndarray | None = None  # (n, d) original vecs for QP path
    epsilon: float = 0.0  # OAS tolerance for QP containment

    def contains(self, vec: list[float]) -> bool:
        """Check whether a vector falls within this hull.

        For n <= d (vertices set): uses QP distance with OAS
        tolerance. For n > d: uses halfplane equations, falling back
        to a ball check.
        """
        point = np.asarray(vec)
        if self.vertices is not None:
            # Quick reject: point outside bounding ball + tolerance.
            if float(np.linalg.norm(point - self.center)) > self.radius + self.epsilon:
                return False
            return _qp_distance(self.vertices, point) <= self.epsilon
        if self.equations.shape[0] > 0:
            return _halfplane_contains(self.equations, point)
        return float(np.linalg.norm(point - self.center)) <= self.radius

    @classmethod
    def fit(cls, vecs: list[list[float]], confidence: float = 0.95) -> ConvexHull:
        """Fit a convex hull to a list of row vectors."""
        vecs_arr = np.asarray(vecs)
        n, d = vecs_arr.shape
        # When n <= d, use QP distance with OAS tolerance instead of
        # QHull (which needs n > d).
        if n <= d:
            center = vecs_arr.mean(axis=0)
            centered = vecs_arr - center
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            # Standard numerical rank threshold, matching numpy's
            # convention.
            tol = S[0] * max(n, d) * np.finfo(vecs_arr.dtype).eps
            rank = int(np.sum(S > tol))
            radius = float(np.linalg.norm(centered, axis=1).max())
            epsilon = _oas_epsilon(S, n, d, rank)
            return ConvexHull(
                equations=np.empty((0, d + 1)),
                center=center,
                radius=radius,
                vertices=vecs_arr,
                epsilon=epsilon,
            )
        # n > d: QHull can work directly in the ambient space.
        try:
            hull = ScipyConvexHull(vecs_arr)
        except QhullError:
            return _fallback_ball(vecs_arr)
        center = vecs_arr.mean(axis=0)
        radius = float(np.linalg.norm(vecs_arr - center, axis=1).max())
        # Recompute normals via SVD even in full rank; QHull's
        # internal equations lose precision in high dimensions.
        equations = _recompute_equations(hull.simplices, vecs_arr, center)
        return ConvexHull(equations=equations, center=center, radius=radius)


def _fallback_ball(vecs: np.ndarray) -> ConvexHull:
    """Return a ConvexHull with empty equations, using a ball fallback.

    Conservative approximation when QHull fails: the bounding ball
    centered at the mean always contains the original points.
    """
    center = vecs.mean(axis=0)
    d = vecs.shape[1]
    radius = float(np.linalg.norm(vecs - center, axis=1).max())
    return ConvexHull(
        equations=np.empty((0, d + 1)),
        center=center,
        radius=radius,
    )


def convex_hull(
    node: KnowledgeNode,
    representations: dict[str, list[float]],
    k: float | None = None,
) -> ConvexHull:
    """Compute the convex hull for a subtree.

    If k is given (a fraction between 0 and 1), randomly sample
    ceil(k * num_concepts) concepts.
    """
    concepts = node.concepts()
    # Subsampling reduces QHull cost.
    if k is not None:
        sample_size = math.ceil(k * len(concepts))
        if sample_size < len(concepts):
            indices = np.random.choice(len(concepts), size=sample_size, replace=False)
            concepts = [concepts[i] for i in indices]
    return ConvexHull.fit([representations[c] for c in concepts])
