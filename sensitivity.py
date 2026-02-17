from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
from analytics.specificity import Shape
from pydmodels.knowledge import Concept, KnowledgeNode, KnowledgeTree
from pydmodels.representation import Vector


class NodeResult(NamedTuple):
    concept: Concept
    contained: int
    total: int


def _split_spaceaug(
    concepts: list[Concept],
    rng: np.random.Generator,
    train_fraction: float,
) -> tuple[list[Concept], list[Concept]]:
    """Split spaceaug concepts into train/test by the given fraction."""
    indices = rng.permutation(len(concepts))
    mid = int(len(concepts) * train_fraction)
    train = [concepts[i] for i in indices[:mid]]
    test = [concepts[i] for i in indices[mid:]]
    return train, test


def _collect_training_vectors(
    node: KnowledgeNode,
    original_reps: dict[Concept, Vector],
    spaceaug_reps: dict[Concept, Vector],
    train_spaceaug: set[Concept],
) -> np.ndarray:
    """Collect training vectors for a node.

    Includes all original subtree concepts plus train-split spaceaug
    concepts mapped to this subtree.
    """
    concepts = node.concepts()
    subtree_concepts = set(concepts)
    vecs: list[Vector] = [original_reps[c] for c in concepts]
    for sa_concept in train_spaceaug:
        orig = Concept(str(sa_concept).strip())
        if orig in subtree_concepts:
            vecs.append(spaceaug_reps[sa_concept])
    return np.array(vecs)


def _evaluate_sensitivity(
    tree: KnowledgeTree,
    original_reps: dict[Concept, Vector],
    spaceaug_reps: dict[Concept, Vector],
    fit_fn: Callable[[np.ndarray], Shape],
    rng: np.random.Generator,
    train_fraction: float = 0.0,
) -> list[NodeResult]:
    """Core sensitivity evaluation (no I/O)."""
    sa_concepts = list(spaceaug_reps)
    train_sa, test_sa = _split_spaceaug(sa_concepts, rng, train_fraction)
    train_sa_set = set(train_sa)

    # Build mapping: original concept -> list of test spaceaug concepts.
    test_sa_by_orig: dict[Concept, list[Concept]] = {}
    for sa_concept in test_sa:
        orig = Concept(str(sa_concept).strip())
        test_sa_by_orig.setdefault(orig, []).append(sa_concept)

    # Fit shapes and evaluate per-node containment in a single DFS pass.
    results: list[NodeResult] = []
    stack = [tree.root]
    while stack:
        node = stack.pop()
        vecs = _collect_training_vectors(
            node, original_reps, spaceaug_reps, train_sa_set
        )
        shape = fit_fn(vecs)
        contained = 0
        total = 0
        for orig_concept in node.concepts():
            for sa_concept in test_sa_by_orig.get(orig_concept, []):
                if shape.contains(spaceaug_reps[sa_concept]):
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
    shape: Literal["hyperellipsoid", "convex_hull"],
    tree_name: str,
    dimensionality: int,
    train_fraction: float = 0.0,
) -> list[NodeResult]:
    """Run sensitivity analysis.

    Loads data and delegates to _evaluate_sensitivity.
    """
    import json

    from analytics.convex_hull import fit_hull
    from analytics.hyperellipsoid import fit_ellipsoid
    from pydmodels.representation import RepresentationCollection
    from repgen.util import get_rep_path
    from treegen.util import TREES_DIR, get_tree_path

    rng = np.random.default_rng(seed=0)

    tree_path = get_tree_path(tree_name)
    tree = KnowledgeTree.model_validate_json(tree_path.read_text())

    def load_reps(rep_name: str) -> RepresentationCollection:
        raw = json.loads(get_rep_path(rep_name).read_text())
        raw["knowledge_tree"] = str(TREES_DIR / raw["knowledge_tree"])
        return RepresentationCollection.model_validate(raw)

    orig_repcol = load_reps(f"embeddings_{tree_name}_d{dimensionality}")
    sa_repcol = load_reps(f"embeddings_spaceaug_{tree_name}_d{dimensionality}")

    fit_fns = {"hyperellipsoid": fit_ellipsoid, "convex_hull": fit_hull}
    return _evaluate_sensitivity(
        tree,
        orig_repcol.representations,
        sa_repcol.representations,
        fit_fn=fit_fns[shape],
        rng=rng,
        train_fraction=train_fraction,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: fit shapes on training data"
        " and evaluate on held-out spaceaug data."
    )
    parser.add_argument(
        "tree_name",
        help="Name of the tree (e.g. 'manual_tiny').",
    )
    parser.add_argument(
        "dimensionality",
        type=int,
        help="Dimensionality of embeddings.",
    )
    parser.add_argument(
        "--shape",
        default="hyperellipsoid",
        choices=["hyperellipsoid", "convex_hull"],
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
            dimensionality=args.dimensionality,
            train_fraction=args.train_fraction,
        )
    )
