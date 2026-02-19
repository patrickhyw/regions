"""3D PCA visualization of bird vs mammal embeddings."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2
from sklearn.decomposition import PCA

from embedding import get_embeddings
from hyperellipsoid import Hyperellipsoid
from tree import build_named_tree
from util import set_seed


def marker_size(n: int) -> float:
    """Return a marker size that scales inversely with point count.

    Cube root scaling keeps total marker volume roughly constant as
    point count changes (volume ~ r^3, so r ~ n^(-1/3)).
    """
    return max(1.0, 20 / n ** (1 / 3))


def ellipsoid_surface(
    ell: Hyperellipsoid,
    pca: PCA,
    *,
    color: str,
    opacity: float = 0.3,
    resolution: int = 30,
) -> go.Surface:
    """Project a high-dimensional Hyperellipsoid into 3D PCA space and
    return a plotly Surface trace.

    The projected covariance is V @ Sigma @ V^T where V is the PCA
    components matrix (3 x d) and Sigma = alpha*I + gamma*X^T@X.
    """
    # Project center into PCA space.
    center_3d = pca.transform(ell.center.reshape(1, -1))[0]

    # Build the 3x3 projected covariance.
    V = pca.components_  # (3, d)
    if ell.X.shape[0] > 0:
        n = ell.X.shape[0]
        # Project X into PCA space and build the 3x3 covariance
        # Sigma_3d = alpha*I_3 + gamma * X_3d^T @ X_3d. The
        # Hyperellipsoid stores M_inv but not gamma, so recover gamma
        # from M = inv(M_inv) via the trace of
        # M = (1/gamma)*I_n + (1/alpha)*G.
        X_3d = ell.X @ V.T  # (n, 3)
        G_3d = X_3d.T @ X_3d  # (3, 3)
        M = np.linalg.inv(ell.M_inv)  # (n, n)
        G = ell.X @ ell.X.T  # (n, n)
        inv_gamma = (np.trace(M) - np.trace(G) / ell.alpha) / n
        gamma = 1.0 / inv_gamma
        sigma_3d = ell.alpha * np.eye(3) + gamma * G_3d
    else:
        sigma_3d = ell.alpha * np.eye(3)

    eigvals, eigvecs = np.linalg.eigh(sigma_3d)
    threshold_3d = chi2.ppf(0.95, df=3)

    # Generate parametric unit sphere.
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    # Stack into (resolution*resolution, 3) and scale by
    # sqrt(eigenvalue * threshold) along eigenvector directions.
    sphere = np.stack([sphere_x.ravel(), sphere_y.ravel(), sphere_z.ravel()], axis=1)
    radii = np.sqrt(eigvals * threshold_3d)
    scaled = sphere * radii  # broadcast (N, 3) * (3,)
    # Rotate into eigenvector frame and translate to center.
    transformed = scaled @ eigvecs.T + center_3d

    x = transformed[:, 0].reshape(resolution, resolution)
    y = transformed[:, 1].reshape(resolution, resolution)
    z = transformed[:, 2].reshape(resolution, resolution)

    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        surfacecolor=np.zeros((resolution, resolution)),
        showscale=False,
    )


if __name__ == "__main__":
    set_seed()

    tree = build_named_tree("animalmin")

    bird_node = tree.find("bird")
    mammal_node = tree.find("mammal")

    bird_concepts = bird_node.concepts()
    mammal_concepts = mammal_node.concepts()
    all_concepts = bird_concepts + mammal_concepts

    embeddings = get_embeddings(all_concepts, dimension=768)
    X = np.array(embeddings)

    pca = PCA(n_components=3)
    coords = pca.fit_transform(X)

    n_birds = len(bird_concepts)
    labels = ["bird"] * n_birds + ["mammal"] * len(mammal_concepts)

    fig = px.scatter_3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        color=labels,
        color_discrete_map={"bird": "orange", "mammal": "blue"},
        hover_name=all_concepts,
    )
    fig.update_traces(
        marker=dict(size=marker_size(len(all_concepts))),
        selector=dict(mode="markers"),
    )

    bird_ell = Hyperellipsoid.fit(X[:n_birds])
    mammal_ell = Hyperellipsoid.fit(X[n_birds:])
    fig.add_trace(ellipsoid_surface(bird_ell, pca, color="orange"))
    fig.add_trace(ellipsoid_surface(mammal_ell, pca, color="blue"))

    fig.show()
