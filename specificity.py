from __future__ import annotations

import argparse
from itertools import combinations
from typing import Literal, NamedTuple

from tqdm import tqdm

from convexhull import ConvexHull
from embedding import get_embeddings
from hyperellipsoid import Hyperellipsoid
from hypersphere import Hypersphere
from shape import Shape
from tree import KnowledgeNode, build_named_tree
from util import set_seed

MAX_SIBLING_RATIO = 3.0
MIN_SUBTREE_SIZE = 5
SHAPE_CLASSES: dict[str, type[Shape]] = {
    "hyperellipsoid": Hyperellipsoid,
    "hypersphere": Hypersphere,
    "convexhull": ConvexHull,
}


class PairResult(NamedTuple):
    concepts: tuple[str, str]
    correct: int
    in_neither: int
    in_both: int
    total: int


def _fit_all(
    node: KnowledgeNode,
    representations: dict[str, list[float]],
    shape_cls: type[Shape],
    progress: bool = False,
) -> dict[str, Shape]:
    """Compute shapes for every node in the tree.

    Returns dict mapping each node's concept to its Shape.
    """
    # Collect all nodes via iterative DFS.
    stack = [node]
    nodes: list[KnowledgeNode] = []
    while stack:
        n = stack.pop()
        nodes.append(n)
        stack.extend(n.children)
    result: dict[str, Shape] = {}
    for n in tqdm(nodes, desc="Fitting shapes", disable=not progress):
        result[n.concept] = shape_cls.fit([representations[c] for c in n.concepts()])
    return result


def specificity(
    shape: Literal["hyperellipsoid", "hypersphere", "convexhull"],
    tree_name: str = "monkey",
    dimension: int = 128,
    progress: bool = False,
) -> list[PairResult]:
    """Run specificity analysis for a named tree and embedding dimension."""
    tree = build_named_tree(tree_name)
    concepts = tree.root.concepts()
    embeddings = get_embeddings(concepts, dimension=dimension)
    representations = dict(zip(concepts, embeddings))
    shapes = _fit_all(tree.root, representations, SHAPE_CLASSES[shape], progress)

    # Walk all nodes via iterative DFS, evaluating sibling pairs.
    results: list[PairResult] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        for a, b in combinations(node.children, 2):
            size_a, size_b = len(a.concepts()), len(b.concepts())
            if max(size_a, size_b) / min(size_a, size_b) > MAX_SIBLING_RATIO:
                continue
            if min(size_a, size_b) < MIN_SUBTREE_SIZE:
                continue
            correct = 0
            in_neither = 0
            in_both = 0
            total = 0
            # For each subtree, check whether its concepts fall
            # exclusively in its own shape and not in the sibling's
            # shape. The fourth case (in other but not own) is
            # implicitly wrong.
            for own, other in [(a, b), (b, a)]:
                own_shape = shapes[own.concept]
                other_shape = shapes[other.concept]
                for concept in own.concepts():
                    vec = representations[concept]
                    in_own = own_shape.contains(vec)
                    in_other = other_shape.contains(vec)
                    if in_own and not in_other:
                        correct += 1
                    elif not in_own and not in_other:
                        in_neither += 1
                    elif in_own and in_other:
                        in_both += 1
                    total += 1
            results.append(
                PairResult(
                    concepts=(a.concept, b.concept),
                    correct=correct,
                    in_neither=in_neither,
                    in_both=in_both,
                    total=total,
                )
            )
    return results


def print_results(results: list[PairResult], top: int = 10) -> None:
    """Print per-pair accuracy and overall accuracy summary."""

    def _pct(correct: int, total: int) -> str:
        return f"{correct / total * 100:.1f}%"

    ranked = sorted(results, key=lambda r: r.total, reverse=True)
    for result in ranked[:top]:
        a, b = result.concepts
        print(
            f"({a}, {b})  accuracy={result.correct}/{result.total}"
            f" ({_pct(result.correct, result.total)})"
            f"  in_neither={result.in_neither}"
            f"  in_both={result.in_both}"
        )
    overall_correct = sum(r.correct for r in results)
    overall_in_neither = sum(r.in_neither for r in results)
    overall_in_both = sum(r.in_both for r in results)
    overall_total = sum(r.total for r in results)
    print(
        f"overall accuracy={overall_correct}/{overall_total}"
        f" ({_pct(overall_correct, overall_total)})"
        f"  in_neither={overall_in_neither}"
        f"  in_both={overall_in_both}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate shape separation of sibling pairs."
    )
    parser.add_argument(
        "--tree-name",
        default="monkey",
        help="Name of the tree. Default: monkey.",
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
        choices=SHAPE_CLASSES,
        help="Shape type. Default: hyperellipsoid.",
    )
    args = parser.parse_args()

    set_seed()
    print_results(
        specificity(
            shape=args.shape,
            tree_name=args.tree_name,
            dimension=args.dimension,
            progress=True,
        )
    )
