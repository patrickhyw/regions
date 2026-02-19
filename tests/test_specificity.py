from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from specificity import (
    MAX_SIBLING_RATIO,
    MIN_SUBTREE_SIZE,
    PairResult,
    evaluate_sibling_pairs,
    specificity,
)
from tree import KnowledgeNode, KnowledgeTree


class TestSpecificityConstants:
    @pytest.mark.parametrize(
        ("constant", "expected"),
        [
            pytest.param(MAX_SIBLING_RATIO, 3.0, id="max_sibling_ratio"),
            pytest.param(MIN_SUBTREE_SIZE, 5, id="min_subtree_size"),
        ],
    )
    def test_constant_values(self, constant: object, expected: object) -> None:
        assert constant == expected


class TestSpecificity:
    @pytest.fixture()
    def tree(self) -> KnowledgeTree:
        """Tree with two balanced siblings, each having 5 leaves.

        Each sibling subtree has 6 concepts (parent + 5 children),
        so each passes MIN_SUBTREE_SIZE (min(6, 6) = 6 >= 5) and
        MAX_SIBLING_RATIO (6/6 = 1.0 <= 3.0).
        """
        return KnowledgeTree(
            root=KnowledgeNode(
                concept="root",
                children=[
                    KnowledgeNode(
                        concept="child_a",
                        children=[KnowledgeNode(concept=f"a{i}") for i in range(5)],
                    ),
                    KnowledgeNode(
                        concept="child_b",
                        children=[KnowledgeNode(concept=f"b{i}") for i in range(5)],
                    ),
                ],
            )
        )

    @pytest.fixture()
    def embeddings(self) -> list[list[float]]:
        """3-D embeddings for the 13 concepts in the tree fixture.

        Concept order matches KnowledgeNode.concepts() DFS traversal:
        root, child_a, a0-a4, child_b, b0-b4.
        The a-cluster is near [1, 0, 0] and b-cluster near [0, 1, 0].
        """
        return [
            [0.5, 0.5, 0.0],  # root
            [1.0, 0.0, 0.0],  # child_a
            [0.9, 0.1, 0.0],  # a0
            [0.95, 0.05, 0.05],  # a1
            [1.0, 0.0, 0.1],  # a2
            [0.85, 0.1, 0.05],  # a3
            [0.9, 0.05, 0.0],  # a4
            [0.0, 1.0, 0.0],  # child_b
            [0.1, 0.9, 0.0],  # b0
            [0.05, 0.95, 0.05],  # b1
            [0.0, 1.0, 0.1],  # b2
            [0.1, 0.85, 0.05],  # b3
            [0.05, 0.9, 0.0],  # b4
        ]

    @pytest.fixture()
    def mock_build_named_tree(
        self, tree: KnowledgeTree
    ) -> Generator[MagicMock, None, None]:
        with patch("specificity.build_named_tree", return_value=tree) as mock:
            yield mock

    @pytest.fixture()
    def mock_get_embeddings(
        self, embeddings: list[list[float]]
    ) -> Generator[MagicMock, None, None]:
        with patch("specificity.get_embeddings", return_value=embeddings) as mock:
            yield mock

    def test_returns_pair_results(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
        tree: KnowledgeTree,
    ) -> None:
        """Returns a non-empty list of PairResult instances."""
        results = specificity("hyperellipsoid", "test_tree", 3)

        mock_build_named_tree.assert_called_once_with("test_tree")
        mock_get_embeddings.assert_called_once_with(tree.root.concepts(), dimension=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, PairResult) for r in results)

    def test_skips_pair_when_individual_subtree_too_small(self) -> None:
        """Skip pairs where either individual subtree is too small.

        child_a has 6 concepts (passes), child_b has only 4 (fails
        MIN_SUBTREE_SIZE=5). The pair is balanced enough for
        MAX_SIBLING_RATIO (6/4 = 1.5 <= 3.0) and the combined size
        of 10 would have passed the old check.
        """
        tree = KnowledgeNode(
            concept="root",
            children=[
                KnowledgeNode(
                    concept="child_a",
                    children=[KnowledgeNode(concept=f"a{i}") for i in range(5)],
                ),
                KnowledgeNode(
                    concept="child_b",
                    children=[KnowledgeNode(concept=f"b{i}") for i in range(3)],
                ),
            ],
        )
        # Shapes are unused because the pair should be skipped before
        # any containment checks.
        mock_shapes: dict[str, MagicMock] = {}
        reps: dict[str, list[float]] = {}
        results = evaluate_sibling_pairs(tree, reps, MagicMock, _shapes=mock_shapes)
        assert results == []
