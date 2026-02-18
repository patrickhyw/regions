import argparse
from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np

from convexhull import fit_hull
from embedding import get_embeddings
from hyperellipsoid import hyperellipsoid as fit_ellipsoid
from shape import Shape
from tree import KnowledgeNode, build_named_tree


class NodeResult(NamedTuple):
    concept: str
    contained: int
    total: int


def _spaceaug_concept(concept: str) -> list[str]:
    """Generate spaceaug variants for a concept.

    Generates three whitespace variants: leading space, trailing
    space, and both.
    """
    return [f" {concept}", f"{concept} ", f" {concept} "]


def _split_spaceaug(
    concepts: list[str],
    train_fraction: float,
) -> tuple[list[str], list[str]]:
    """Split spaceaug concepts into train/test by the given fraction."""
    indices = np.random.permutation(len(concepts))
    mid = int(len(concepts) * train_fraction)
    train = [concepts[i] for i in indices[:mid]]
    test = [concepts[i] for i in indices[mid:]]
    return train, test


def _collect_training_embeddings(
    node: KnowledgeNode,
    original_embeddings: dict[str, list[float]],
    spaceaug_embeddings: dict[str, list[float]],
    train_spaceaug: set[str],
) -> np.ndarray:
    """Collect training vectors for a node.

    Includes all original subtree concepts plus train-split spaceaug
    concepts mapped to this subtree.
    """
    concepts = node.concepts()
    subtree_concepts = set(concepts)
    embeddings: list[list[float]] = [original_embeddings[c] for c in concepts]
    for spaceaug_concept in train_spaceaug:
        orig = spaceaug_concept.strip()
        if orig in subtree_concepts:
            embeddings.append(spaceaug_embeddings[spaceaug_concept])
    return np.array(embeddings)


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
    spaceaug_concepts = [v for c in concepts for v in _spaceaug_concept(c)]

    embeddings = get_embeddings(concepts + spaceaug_concepts, dimension=dimension)
    original_embeddings = dict(zip(concepts, embeddings[: len(concepts)]))
    spaceaug_embeddings = dict(zip(spaceaug_concepts, embeddings[len(concepts) :]))

    fit_fns: dict[str, Callable[[np.ndarray], Shape]] = {
        "hyperellipsoid": fit_ellipsoid,
        "convexhull": fit_hull,
    }
    fit_fn = fit_fns[shape]

    train_spaceaug, test_spaceaug = _split_spaceaug(spaceaug_concepts, train_fraction)
    train_spaceaug_set = set(train_spaceaug)

    # Build mapping: original concept -> list of test spaceaug concepts.
    test_spaceaug_by_orig: dict[str, list[str]] = {}
    for spaceaug_concept in test_spaceaug:
        orig = spaceaug_concept.strip()
        test_spaceaug_by_orig.setdefault(orig, []).append(spaceaug_concept)

    # Fit shapes and evaluate per-node containment in a single DFS pass.
    results: list[NodeResult] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        train_embeddings = _collect_training_embeddings(
            node, original_embeddings, spaceaug_embeddings, train_spaceaug_set
        )
        region = fit_fn(train_embeddings)
        contained = 0
        total = 0
        for orig_concept in node.concepts():
            for spaceaug_concept in test_spaceaug_by_orig.get(orig_concept, []):
                if region.contains(spaceaug_embeddings[spaceaug_concept]):
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

    from util import set_seed

    set_seed()
    print_node_results(
        sensitivity(
            shape=args.shape,
            tree_name=args.tree_name,
            dimension=args.dimension,
            train_fraction=args.train_fraction,
        )
    )
