from __future__ import annotations

import argparse
from typing import Literal, NamedTuple

import plotly.graph_objects as go
from rich.progress import Progress

from embedding import get_embeddings
from hyperellipsoid import Hyperellipsoid
from hypersphere import Hypersphere
from shape import Shape
from tree import KnowledgeNode, build_named_tree
from util import set_seed

MIN_SUBTREE_SIZE = 10
SHAPE_CLASSES: dict[str, type[Shape]] = {
    "hyperellipsoid": Hyperellipsoid,
    "hypersphere": Hypersphere,
}


class SubtreeResult(NamedTuple):
    concept: str
    tp: int
    fp: int
    fn: int


class WeightedAverage(NamedTuple):
    precision: float
    recall: float


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _weighted_averages(results: list[SubtreeResult]) -> WeightedAverage:
    """Compute size-weighted average precision and recall."""
    total_weight = sum(r.tp + r.fn for r in results)
    w_prec = sum(_safe_div(r.tp, r.tp + r.fp) * (r.tp + r.fn) for r in results)
    w_rec = sum(_safe_div(r.tp, r.tp + r.fn) * (r.tp + r.fn) for r in results)
    return WeightedAverage(
        precision=_safe_div(w_prec, total_weight),
        recall=_safe_div(w_rec, total_weight),
    )


def pr_curve_area(recalls: list[float], precisions: list[float]) -> float:
    """Compute area under the precision-recall curve via trapezoidal rule.

    Points are sorted by recall, then endpoints (0, 1) and (1, 0) are
    interpolated before applying the trapezoidal rule.
    """
    if not recalls:
        raise ValueError("Inputs must not be empty")
    if len(recalls) != len(precisions):
        raise ValueError("Recalls and precisions must have same length")
    points = [(0.0, 1.0)] + sorted(zip(recalls, precisions)) + [(1.0, 0.0)]
    return sum(
        0.5 * (r2 - r1) * (p2 + p1) for (r1, p1), (r2, p2) in zip(points, points[1:])
    )


def auprc(
    shape: Literal["hyperellipsoid", "hypersphere"],
    tree_name: str = "mammalmin",
    dimension: int = 128,
    confidence: float = 0.95,
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
    for node in nodes:
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
    total_weight = sum(r.tp + r.fn for r in results)
    if total_weight > 0:
        avg = _weighted_averages(results)
        print(
            f"weighted avg"
            f"  precision={avg.precision * 100:.1f}%"
            f"  recall={avg.recall * 100:.1f}%"
        )


def graph(tree_name: str, dimension: int) -> go.Figure:
    """Plot precision-recall curve sweeping confidence for each shape."""
    confidence_levels = [
        0.5,
        0.9,
        0.99,
        0.999,
        0.9999,
        0.99999,
        0.999999,
    ]
    shapes_and_colors = [
        ("hyperellipsoid", "blue"),
        ("hypersphere", "red"),
    ]

    fig = go.Figure()
    with Progress() as progress:
        for shape_name, color in shapes_and_colors:
            task = progress.add_task(shape_name, total=len(confidence_levels))
            precisions: list[float] = []
            recalls: list[float] = []
            for conf in confidence_levels:
                results = auprc(
                    shape=shape_name,
                    tree_name=tree_name,
                    dimension=dimension,
                    confidence=conf,
                )
                avg = _weighted_averages(results)
                precisions.append(avg.precision)
                recalls.append(avg.recall)
                progress.advance(task)
            area = pr_curve_area(recalls, precisions)
            print(f"{shape_name} AUPRC={area:.4f}")
            points = sorted(zip(recalls, precisions))
            plot_r = [0.0] + [r for r, _ in points] + [1.0]
            plot_p = [1.0] + [p for _, p in points] + [0.0]
            fig.add_trace(
                go.Scatter(
                    x=plot_r,
                    y=plot_p,
                    mode="lines",
                    name=f"{shape_name} (AUPRC={area:.2f})",
                    line=dict(color=color),
                )
            )
    fig.update_layout(
        title=f"AUPRC: {tree_name} (dim={dimension})",
        xaxis=dict(title="Weighted Recall", range=[0, 1]),
        yaxis=dict(title="Weighted Precision", range=[0, 1]),
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate one-vs-rest precision/recall per subtree."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    print_parser = subparsers.add_parser("print")
    graph_parser = subparsers.add_parser("graph")

    for sub in [print_parser, graph_parser]:
        sub.add_argument(
            "--tree-name",
            default="mammalmin",
            help="Name of the tree. Default: mammalmin.",
        )
        sub.add_argument(
            "--dimension",
            type=int,
            default=128,
            help="Dimension of embeddings. Default: 128.",
        )

    print_parser.add_argument(
        "--shape",
        default="hyperellipsoid",
        choices=SHAPE_CLASSES,
        help="Shape type. Default: hyperellipsoid.",
    )
    print_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for shape fitting. Default: 0.95.",
    )

    args = parser.parse_args()
    set_seed()

    if args.command == "print":
        print_results(
            auprc(
                shape=args.shape,
                tree_name=args.tree_name,
                dimension=args.dimension,
                confidence=args.confidence,
            )
        )
    elif args.command == "graph":
        graph(args.tree_name, args.dimension).show()
