from __future__ import annotations

import argparse
from typing import Literal, NamedTuple

import numpy as np

from convexhull import ConvexHull
from embedding import get_embeddings
from hyperellipsoid import Hyperellipsoid
from shape import Shape
from tree import build_named_tree
from util import set_seed


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


class TrainTestSplit(NamedTuple):
    all_concepts: list[str]
    train_set: set[str]
    test_by_orig: dict[str, list[str]]


def create_train_test_split(
    concepts: list[str],
    train_fraction: float,
    use_spaceaug: bool,
) -> TrainTestSplit:
    """Split concepts into train and test sets.

    When use_spaceaug is True, spaceaug variants are generated and
    split by train_fraction; all originals go to train. When False,
    the originals themselves are split by train_fraction.
    """
    if not concepts:
        raise ValueError("concepts must not be empty")
    if train_fraction == 1.0:
        raise ValueError("train_fraction=1.0 leaves no test data")

    if use_spaceaug:
        spaceaug_concepts = [v for c in concepts for v in _spaceaug_concept(c)]
        all_concepts = concepts + spaceaug_concepts
        indices = np.random.permutation(len(spaceaug_concepts))
        mid = int(len(spaceaug_concepts) * train_fraction)
        train_spaceaug = {spaceaug_concepts[i] for i in indices[:mid]}
        train_set = set(concepts) | train_spaceaug
        test_by_orig: dict[str, list[str]] = {}
        for i in indices[mid:]:
            sa = spaceaug_concepts[i]
            orig = sa.strip()
            test_by_orig.setdefault(orig, []).append(sa)
        return TrainTestSplit(all_concepts, train_set, test_by_orig)
    else:
        if train_fraction == 0.0:
            raise ValueError(
                "train_fraction=0.0 with use_spaceaug=False leaves no training data"
            )
        all_concepts = list(concepts)
        indices = np.random.permutation(len(concepts))
        mid = int(len(concepts) * train_fraction)
        train_set = {concepts[i] for i in indices[:mid]}
        test_by_orig: dict[str, list[str]] = {}
        for i in indices[mid:]:
            c = concepts[i]
            test_by_orig[c] = [c]
        return TrainTestSplit(all_concepts, train_set, test_by_orig)


def print_node_results(results: list[NodeResult], top: int = 10) -> None:
    """Print per-node containment rate and overall summary."""

    def _pct(contained: int, total: int) -> str:
        return f"{contained / total * 100:.1f}%"

    # Only print the top N nodes by total; overall summary still uses all.
    ranked = sorted(results, key=lambda r: r.total, reverse=True)
    for r in ranked[:top]:
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
    tree_name: str = "primate",
    dimension: int = 128,
    train_fraction: float = 0.0,
    use_spaceaug: bool = True,
) -> list[NodeResult]:
    """Run sensitivity analysis.

    Builds the tree, splits concepts into train/test, fetches
    embeddings, fits shapes per node, and evaluates containment
    of held-out test vectors.
    """
    tree = build_named_tree(tree_name)
    concepts = tree.root.concepts()
    split = create_train_test_split(concepts, train_fraction, use_spaceaug)
    embeddings_list = get_embeddings(split.all_concepts, dimension=dimension)
    all_embeddings = dict(zip(split.all_concepts, embeddings_list))

    shape_classes: dict[str, type[Shape]] = {
        "hyperellipsoid": Hyperellipsoid,
        "convexhull": ConvexHull,
    }
    shape_cls = shape_classes[shape]

    # Fit shapes and evaluate per-node containment in a single DFS pass.
    results: list[NodeResult] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        subtree_concepts = set(node.concepts())
        train_vecs = [
            all_embeddings[c] for c in split.train_set if c.strip() in subtree_concepts
        ]
        train_embeddings = np.array(train_vecs)
        region = shape_cls.fit(train_embeddings)
        contained = 0
        total = 0
        for orig_concept in node.concepts():
            for test_concept in split.test_by_orig.get(orig_concept, []):
                if region.contains(all_embeddings[test_concept]):
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
        "--tree-name",
        default="primate",
        help="Name of the tree. Default: primate.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=128,
        help="Dimension of embeddings. Default: 128.",
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
        help="Fraction of concepts used for training (0.0 to 1.0). Default: 0.0.",
    )
    parser.add_argument(
        "--spaceaug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use spaceaug augmentation. Default: --spaceaug.",
    )
    args = parser.parse_args()

    set_seed()
    print_node_results(
        sensitivity(
            shape=args.shape,
            tree_name=args.tree_name,
            dimension=args.dimension,
            train_fraction=args.train_fraction,
            use_spaceaug=args.spaceaug,
        )
    )
