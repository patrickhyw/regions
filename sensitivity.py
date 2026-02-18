from __future__ import annotations

import argparse
from typing import Literal, NamedTuple

import numpy as np

from convexhull import Hull
from embedding import get_embeddings
from hyperellipsoid import Ellipsoid
from shape import Shape
from tree import build_named_tree


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

    shape_classes: dict[str, type[Shape]] = {
        "hyperellipsoid": Ellipsoid,
        "convexhull": Hull,
    }
    shape_cls = shape_classes[shape]

    _indices = np.random.permutation(len(spaceaug_concepts))
    _mid = int(len(spaceaug_concepts) * train_fraction)
    train_spaceaug = [spaceaug_concepts[i] for i in _indices[:_mid]]
    test_spaceaug = [spaceaug_concepts[i] for i in _indices[_mid:]]
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
        node_concepts = node.concepts()
        subtree_concepts = set(node_concepts)
        train_vecs: list[list[float]] = [original_embeddings[c] for c in node_concepts]
        for sa in train_spaceaug_set:
            if sa.strip() in subtree_concepts:
                train_vecs.append(spaceaug_embeddings[sa])
        train_embeddings = np.array(train_vecs)
        region = shape_cls.fit(train_embeddings)
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
