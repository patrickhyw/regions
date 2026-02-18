from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from tree import KnowledgeNode

# QHull needs n > d, but word embeddings typically have n <= d. We use
# PCA to project points into the subspace they span, build the hull
# there, then test containment by projecting back.

# Maximum residual norm for a point to count as "on the subspace."
_SUBSPACE_TOL: float = 1e-10
# Slack for halfplane checks so boundary vertices aren't rejected by
# floating-point noise.
_HALFPLANE_TOL: float = 1e-12


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
    ConvexHull.equations: each row is [normal..., offset] such that
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


class Hull(NamedTuple):
    equations: np.ndarray  # (nfacet, k+1) in projected space, or (0, ?) fallback
    center: np.ndarray  # (d,) mean of points in ambient space
    radius: float  # max distance from center (for fallback ball)
    basis: np.ndarray | None = None  # (k, d) orthonormal basis, or None for ambient

    def contains(self, vec: list[float]) -> bool:
        """Check whether a vector falls within this hull.

        Uses halfplane equations when available, otherwise falls back
        to a ball check. When basis is set, projects to the subspace
        first and rejects points off the subspace.
        """
        point = np.asarray(vec)
        if self.basis is not None:
            diff = point - self.center
            projected = self.basis @ diff
            residual = diff - self.basis.T @ projected
            if float(np.linalg.norm(residual)) > _SUBSPACE_TOL:
                return False
            if self.equations.shape[0] > 0:
                return _halfplane_contains(self.equations, projected)
            return float(np.linalg.norm(projected)) <= self.radius
        if self.equations.shape[0] > 0:
            return _halfplane_contains(self.equations, point)
        return float(np.linalg.norm(point - self.center)) <= self.radius

    @classmethod
    def fit(cls, vecs: np.ndarray) -> Hull:
        """Fit a convex hull to a matrix of row vectors."""
        n, d = vecs.shape
        # When n <= d, points span a subspace of R^d. Project via
        # SVD/PCA into that subspace so QHull has enough points
        # relative to d.
        if n <= d:
            center = vecs.mean(axis=0)
            centered = vecs - center
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Standard numerical rank threshold, matching numpy's
            # convention.
            tol = S[0] * max(n, d) * np.finfo(vecs.dtype).eps
            rank = int(np.sum(S > tol))
            # All points are identical (or nearly so).
            if rank == 0:
                return Hull(
                    equations=np.empty((0, 1)),
                    center=center,
                    radius=0.0,
                    basis=np.empty((0, d)),
                )
            basis = Vt[:rank]
            projected = centered @ basis.T
            # QHull needs at least 2 dimensions. Rank 1 means
            # collinear points, which fall back to a ball/segment.
            if rank < 2:
                return _subspace_hull(center, basis, projected)
            try:
                hull = ConvexHull(projected)
            except QhullError:
                return _subspace_hull(center, basis, projected)
            # Projected points are centered, so the origin is
            # interior.
            equations = _recompute_equations(hull.simplices, projected, np.zeros(rank))
            return _subspace_hull(center, basis, projected, equations)
        # n > d: QHull can work directly in the ambient space.
        try:
            hull = ConvexHull(vecs)
        except QhullError:
            return _fallback_ball(vecs)
        center = vecs.mean(axis=0)
        radius = float(np.linalg.norm(vecs - center, axis=1).max())
        # Recompute normals via SVD even in full rank; QHull's
        # internal equations lose precision in high dimensions.
        equations = _recompute_equations(hull.simplices, vecs, center)
        return Hull(equations=equations, center=center, radius=radius)


def _fallback_ball(vecs: np.ndarray) -> Hull:
    """Return a Hull with empty equations, using a ball fallback.

    Conservative approximation when QHull fails: the bounding ball
    centered at the mean always contains the original points.
    """
    center = vecs.mean(axis=0)
    d = vecs.shape[1]
    radius = float(np.linalg.norm(vecs - center, axis=1).max())
    return Hull(
        equations=np.empty((0, d + 1)),
        center=center,
        radius=radius,
    )


def _subspace_hull(
    center: np.ndarray,
    basis: np.ndarray,
    projected: np.ndarray,
    equations: np.ndarray | None = None,
) -> Hull:
    """Build a Hull in a PCA subspace.

    If equations are provided, uses them; otherwise falls back to a
    ball in the projected space.
    """
    rank = projected.shape[1]
    radius = float(np.linalg.norm(projected, axis=1).max())
    return Hull(
        equations=equations if equations is not None else np.empty((0, rank + 1)),
        center=center,
        radius=radius,
        basis=basis,
    )


def convex_hull(
    node: KnowledgeNode,
    representations: dict[str, list[float]],
    k: float | None = None,
) -> Hull:
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
    vecs = np.array([representations[c] for c in concepts])
    return Hull.fit(vecs)
