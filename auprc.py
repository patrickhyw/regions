from __future__ import annotations

import argparse
from typing import Literal, NamedTuple

from tqdm import tqdm

from embedding import get_embeddings
from hyperellipsoid import Hyperellipsoid
from hypersphere import Hypersphere
from shape import Shape
from tree import KnowledgeNode, build_named_tree
from util import set_seed

MIN_SUBTREE_SIZE = 5
SHAPE_CLASSES: dict[str, type[Shape]] = {
    "hyperellipsoid": Hyperellipsoid,
    "hypersphere": Hypersphere,
}


class SubtreeResult(NamedTuple):
    concept: str
    tp: int
    fp: int
    fn: int


def auprc(
    shape: Literal["hyperellipsoid", "hypersphere"],
    tree_name: str = "monkey",
    dimension: int = 128,
    confidence: float = 0.95,
    progress: bool = False,
) -> list[SubtreeResult]:
    """Compute one-vs-rest precision/recall per subtree.

    For each subtree node with >= MIN_SUBTREE_SIZE concepts, fits
    a shape on the positive vectors (concepts in that subtree) and
    classifies all concepts as TP, FP, or FN.
    """
    tree = build_named_tree(tree_name)
    concepts = tree.root.concepts()
    embeddings = get_embeddings(concepts, dimension=dimension)
    representations = dict(zip(concepts, embeddings))
    all_concepts_set = set(concepts)
    shape_cls = SHAPE_CLASSES[shape]

    # Collect eligible nodes via iterative DFS.
    stack: list[KnowledgeNode] = [tree.root]
    nodes: list[KnowledgeNode] = []
    while stack:
        n = stack.pop()
        if len(n.concepts()) >= MIN_SUBTREE_SIZE:
            nodes.append(n)
        stack.extend(n.children)

    results: list[SubtreeResult] = []
    for node in tqdm(nodes, desc="Evaluating subtrees", disable=not progress):
        positive = set(node.concepts())
        negative = all_concepts_set - positive
        vecs = [representations[c] for c in positive]
        region = shape_cls.fit(vecs, confidence=confidence)
        tp = sum(1 for c in positive if region.contains(representations[c]))
        fp = sum(1 for c in negative if region.contains(representations[c]))
        fn = len(positive) - tp
        results.append(SubtreeResult(concept=node.concept, tp=tp, fp=fp, fn=fn))
    return results


def print_results(results: list[SubtreeResult], top: int = 10) -> None:
    """Print per-subtree precision/recall and weighted average."""

    def _pct(numerator: float, denominator: float) -> str:
        return f"{numerator / denominator * 100:.1f}%"

    def _safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else 0.0

    ranked = sorted(results, key=lambda r: r.tp + r.fn, reverse=True)
    for r in ranked[:top]:
        prec_denom = r.tp + r.fp
        rec_denom = r.tp + r.fn
        print(
            f"{r.concept}"
            f"  precision={r.tp}/{prec_denom}"
            f" ({_pct(r.tp, prec_denom)})"
            f"  recall={r.tp}/{rec_denom}"
            f" ({_pct(r.tp, rec_denom)})"
        )
    # Weighted average: weight each subtree by its size (tp + fn).
    total_weight = sum(r.tp + r.fn for r in results)
    if total_weight > 0:
        w_prec = sum(_safe_div(r.tp, r.tp + r.fp) * (r.tp + r.fn) for r in results)
        w_rec = sum(_safe_div(r.tp, r.tp + r.fn) * (r.tp + r.fn) for r in results)
        print(
            f"weighted avg"
            f"  precision={_pct(w_prec, total_weight)}"
            f"  recall={_pct(w_rec, total_weight)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate one-vs-rest precision/recall per subtree."
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
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for shape fitting. Default: 0.95.",
    )
    args = parser.parse_args()

    set_seed()
    print_results(
        auprc(
            shape=args.shape,
            tree_name=args.tree_name,
            dimension=args.dimension,
            confidence=args.confidence,
            progress=True,
        )
    )
