import argparse
from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
from pydmodels.knowledge import KnowledgeNode

from convexhull import fit_hull
from embedding import get_embeddings
from hyperellipsoid import hyperellipsoid as fit_ellipsoid
from specificity import Shape
from tree import build_named_tree


class NodeResult(NamedTuple):
    concept: str
    contained: int
    total: int


def _spaceaug_concepts(concepts: list[str]) -> list[str]:
    """Generate spaceaug variants for each concept.

    For each concept, generates three whitespace variants:
    leading space, trailing space, and both.
    """
    return [variant for c in concepts for variant in (f" {c}", f"{c} ", f" {c} ")]


def _split_spaceaug(
    concepts: list[str],
    train_fraction: float,
) -> tuple[list[str], list[str]]:
    """Split spaceaug concepts into train/test by the given fraction."""
    rng = np.random.default_rng(seed=0)
    indices = rng.permutation(len(concepts))
    mid = int(len(concepts) * train_fraction)
    train = [concepts[i] for i in indices[:mid]]
    test = [concepts[i] for i in indices[mid:]]
    return train, test


def _collect_training_vectors(
    node: KnowledgeNode,
    original_reps: dict[str, list[float]],
    spaceaug_reps: dict[str, list[float]],
    train_spaceaug: set[str],
) -> np.ndarray:
    """Collect training vectors for a node.

    Includes all original subtree concepts plus train-split spaceaug
    concepts mapped to this subtree.
    """
    concepts = node.concepts()
    subtree_concepts = set(concepts)
    vecs: list[list[float]] = [original_reps[c] for c in concepts]
    for sa_concept in train_spaceaug:
        orig = sa_concept.strip()
        if orig in subtree_concepts:
            vecs.append(spaceaug_reps[sa_concept])
    return np.array(vecs)


def print_node_results(results: list[NodeResult]) -> None:
    """Print per-node containment rate and overall summary."""

    def _pct(contained: int, total: int) -> str:
        return f"{contained / total * 100:.1f}%"

    for r in results:
        if r.total == 0:
            continue
        print(
            f"{r.concept}  contained={r.contained}/{r.total}"
            f" ({_pct(r.contained, r.total)})"
        )
    overall_contained = sum(r.contained for r in results)
    overall_total = sum(r.total for r in results)
    if overall_total > 0:
        print(
            f"overall contained={overall_contained}/{overall_total}"
            f" ({_pct(overall_contained, overall_total)})"
        )


def sensitivity(
    shape: Literal["hyperellipsoid", "convexhull"],
    tree_name: str,
    dimension: int,
    train_fraction: float = 0.0,
) -> list[NodeResult]:
    """Run sensitivity analysis.

    Builds the tree, generates spaceaug concepts, fetches embeddings,
    fits shapes per node, and evaluates containment of held-out
    spaceaug vectors.
    """
    tree = build_named_tree(tree_name)
    concepts = tree.root.concepts()
    sa_concepts = _spaceaug_concepts(concepts)

    embeddings = get_embeddings(concepts + sa_concepts, dimension=dimension)
    original_reps = dict(zip(concepts, embeddings[: len(concepts)]))
    spaceaug_reps = dict(zip(sa_concepts, embeddings[len(concepts) :]))

    fit_fns: dict[str, Callable[[np.ndarray], Shape]] = {
        "hyperellipsoid": fit_ellipsoid,
        "convexhull": fit_hull,
    }
    fit_fn = fit_fns[shape]

    train_sa, test_sa = _split_spaceaug(sa_concepts, train_fraction)
    train_sa_set = set(train_sa)

    # Build mapping: original concept -> list of test spaceaug concepts.
    test_sa_by_orig: dict[str, list[str]] = {}
    for sa_concept in test_sa:
        orig = sa_concept.strip()
        test_sa_by_orig.setdefault(orig, []).append(sa_concept)

    # Fit shapes and evaluate per-node containment in a single DFS pass.
    results: list[NodeResult] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        vecs = _collect_training_vectors(
            node, original_reps, spaceaug_reps, train_sa_set
        )
        region = fit_fn(vecs)
        contained = 0
        total = 0
        for orig_concept in node.concepts():
            for sa_concept in test_sa_by_orig.get(orig_concept, []):
                if region.contains(spaceaug_reps[sa_concept]):
                    contained += 1
                total += 1
        results.append(
            NodeResult(
                concept=node.concept,
                contained=contained,
                total=total,
            )
        )
        stack.extend(node.children)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: fit shapes on training data"
        " and evaluate on held-out spaceaug data."
    )
    parser.add_argument(
        "tree_name",
        help="Name of the tree (e.g. 'manual_tiny').",
    )
    parser.add_argument(
        "dimension",
        type=int,
        help="Dimension of embeddings.",
    )
    parser.add_argument(
        "--shape",
        default="hyperellipsoid",
        choices=["hyperellipsoid", "convexhull"],
        help="Shape type. Default: hyperellipsoid.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.0,
        help="Fraction of spaceaug concepts used for training"
        " (0.0 to 1.0). Default: 0.0.",
    )
    args = parser.parse_args()

    print_node_results(
        sensitivity(
            shape=args.shape,
            tree_name=args.tree_name,
            dimension=args.dimension,
            train_fraction=args.train_fraction,
        )
    )
