from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from auprc import (
    MIN_SUBTREE_SIZE,
    SubtreeResult,
    auprc,
    graph,
    pr_curve_area,
    print_results,
)
from tree import KnowledgeNode, KnowledgeTree


class TestAuprcConstants:
    def test_min_subtree_size(self) -> None:
        assert MIN_SUBTREE_SIZE == 10


class TestAuprc:
    @pytest.fixture()
    def tree(self) -> KnowledgeTree:
        """Tree with two balanced siblings, each having 9 leaves.

        Each sibling subtree has 10 concepts (parent + 9 children),
        so each passes MIN_SUBTREE_SIZE. The root has 21 concepts.
        Eligible nodes: root (21), child_a (10), child_b (10).
        """
        return KnowledgeTree(
            root=KnowledgeNode(
                concept="root",
                children=[
                    KnowledgeNode(
                        concept="child_a",
                        children=[KnowledgeNode(concept=f"a{i}") for i in range(9)],
                    ),
                    KnowledgeNode(
                        concept="child_b",
                        children=[KnowledgeNode(concept=f"b{i}") for i in range(9)],
                    ),
                ],
            )
        )

    @pytest.fixture()
    def embeddings(self) -> list[list[float]]:
        """3-D embeddings for the 21 concepts in the tree fixture.

        Concept order matches KnowledgeNode.concepts() DFS traversal:
        root, child_a, a0-a8, child_b, b0-b8.
        The a-cluster is near [1, 0, 0] and b-cluster near
        [0, 1, 0].
        """
        return [
            [0.5, 0.5, 0.0],  # root
            [1.0, 0.0, 0.0],  # child_a
            [0.9, 0.1, 0.0],  # a0
            [0.95, 0.05, 0.05],  # a1
            [1.0, 0.0, 0.1],  # a2
            [0.85, 0.1, 0.05],  # a3
            [0.9, 0.05, 0.0],  # a4
            [0.92, 0.08, 0.0],  # a5
            [0.88, 0.12, 0.0],  # a6
            [0.93, 0.02, 0.05],  # a7
            [0.97, 0.03, 0.0],  # a8
            [0.0, 1.0, 0.0],  # child_b
            [0.1, 0.9, 0.0],  # b0
            [0.05, 0.95, 0.05],  # b1
            [0.0, 1.0, 0.1],  # b2
            [0.1, 0.85, 0.05],  # b3
            [0.05, 0.9, 0.0],  # b4
            [0.08, 0.92, 0.0],  # b5
            [0.12, 0.88, 0.0],  # b6
            [0.02, 0.93, 0.05],  # b7
            [0.03, 0.97, 0.0],  # b8
        ]

    @pytest.fixture()
    def mock_build_named_tree(
        self, tree: KnowledgeTree
    ) -> Generator[MagicMock, None, None]:
        with patch("auprc.build_named_tree", return_value=tree) as mock:
            yield mock

    @pytest.fixture()
    def mock_get_embeddings(
        self, embeddings: list[list[float]]
    ) -> Generator[MagicMock, None, None]:
        with patch("auprc.get_embeddings", return_value=embeddings) as mock:
            yield mock

    def test_returns_subtree_results(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
        tree: KnowledgeTree,
    ) -> None:
        """Returns a list of SubtreeResult with one per eligible node."""
        results = auprc("hyperellipsoid", "test_tree", 3)

        mock_build_named_tree.assert_called_once_with("test_tree")
        mock_get_embeddings.assert_called_once_with(tree.root.concepts(), dimension=3)
        assert len(results) == 3
        assert all(isinstance(r, SubtreeResult) for r in results)

    def test_defaults_to_monkey_128_095(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
        tree: KnowledgeTree,
    ) -> None:
        """Omitting tree_name/dimension/confidence uses defaults."""
        auprc("hyperellipsoid")

        mock_build_named_tree.assert_called_once_with("mammalmin")
        mock_get_embeddings.assert_called_once_with(tree.root.concepts(), dimension=128)

    def test_confidence_passed_to_fit(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """Confidence parameter is forwarded to shape_cls.fit."""
        mock_shape = MagicMock()
        mock_shape.contains.return_value = True
        mock_cls = MagicMock()
        mock_cls.fit.return_value = mock_shape
        with patch("auprc.SHAPE_CLASSES", {"hyperellipsoid": mock_cls}):
            auprc("hyperellipsoid", confidence=0.8)

        # Every fit call should pass confidence=0.8.
        assert mock_cls.fit.call_args_list
        assert all(c.kwargs["confidence"] == 0.8 for c in mock_cls.fit.call_args_list)

    def test_tp_plus_fn_equals_positive_count(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
        tree: KnowledgeTree,
    ) -> None:
        """TP + FN equals the number of positive concepts per subtree."""
        results = auprc("hyperellipsoid", "test_tree", 3)
        subtree_sizes = {
            "root": 21,
            "child_a": 10,
            "child_b": 10,
        }
        for r in results:
            assert r.tp + r.fn == subtree_sizes[r.concept]


class TestPrCurveArea:
    @pytest.mark.parametrize(
        "recalls, precisions, expected",
        [
            pytest.param([1.0], [1.0], 1.0, id="perfect_classifier"),
            pytest.param([0.5], [0.5], 0.5, id="single_midpoint"),
            pytest.param(
                [0.2, 0.5, 0.8],
                [0.8, 0.6, 0.4],
                0.58,
                id="three_points_sorted",
            ),
            pytest.param(
                [0.8, 0.2, 0.5],
                [0.4, 0.8, 0.6],
                0.58,
                id="three_points_unsorted",
            ),
            pytest.param(
                [0.5, 0.5, 0.5],
                [0.3, 0.6, 0.9],
                0.55,
                id="same_recall",
            ),
        ],
    )
    def test_area(
        self,
        recalls: list[float],
        precisions: list[float],
        expected: float,
    ) -> None:
        assert pr_curve_area(recalls, precisions) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "recalls, precisions, match",
        [
            pytest.param([], [], "empty", id="empty_input"),
            pytest.param([0.5], [], "same length", id="mismatched_lengths"),
        ],
    )
    def test_errors(
        self,
        recalls: list[float],
        precisions: list[float],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            pr_curve_area(recalls, precisions)


class TestPrintResults:
    @pytest.fixture()
    def results(self) -> list[SubtreeResult]:
        return [
            SubtreeResult(concept="alpha", tp=8, fp=2, fn=2),
            SubtreeResult(concept="beta", tp=4, fp=1, fn=2),
            SubtreeResult(concept="gamma", tp=1, fp=0, fn=1),
        ]

    def test_top_limits_printed_subtrees(
        self,
        results: list[SubtreeResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Only the top N subtrees by size (tp+fn) are printed."""
        print_results(results, top=2)
        output = capsys.readouterr().out
        assert "alpha" in output
        assert "beta" in output
        assert "gamma" not in output

    def test_prints_in_descending_size_order(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Subtrees are printed sorted by tp+fn descending."""
        results = [
            SubtreeResult(concept="small", tp=1, fp=0, fn=1),
            SubtreeResult(concept="big", tp=8, fp=2, fn=2),
        ]
        print_results(results, top=2)
        output = capsys.readouterr().out
        assert output.index("big") < output.index("small")

    def test_weighted_average_line(
        self,
        results: list[SubtreeResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Weighted average uses all subtrees regardless of top."""
        print_results(results, top=1)
        output = capsys.readouterr().out
        assert "weighted avg" in output
        # Weighted precision: (8/10*10 + 4/5*6 + 1/1*2) / (10+6+2)
        #   = (8 + 4.8 + 2) / 18 = 14.8/18 = 82.2%
        # Weighted recall: (8/10*10 + 4/6*6 + 1/2*2) / (10+6+2)
        #   = (8 + 4 + 1) / 18 = 13/18 = 72.2%
        assert "82.2%" in output
        assert "72.2%" in output

    def test_precision_recall_in_output(
        self,
        results: list[SubtreeResult],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Per-subtree lines show precision and recall values."""
        print_results(results, top=3)
        output = capsys.readouterr().out
        # Alpha: precision=8/10, recall=8/10.
        assert "precision=8/10" in output
        assert "recall=8/10" in output


class TestGraph:
    @pytest.fixture()
    def mock_auprc(self) -> Generator[MagicMock, None, None]:
        """Mock auprc to return fixed results for all calls."""
        results = [SubtreeResult(concept="root", tp=8, fp=2, fn=2)]
        with patch("auprc.auprc", return_value=results) as mock:
            yield mock

    def test_produces_two_traces(self, mock_auprc: MagicMock) -> None:
        """Graph has one line per shape with AUPRC in the name."""
        fig = graph("monkey", 128)
        assert len(fig.data) == 2
        assert [t.name for t in fig.data] == [
            "hyperellipsoid (AUPRC=0.80)",
            "hypersphere (AUPRC=0.80)",
        ]

    def test_sweeps_confidence_values(self, mock_auprc: MagicMock) -> None:
        """Each shape is evaluated at all nine confidence levels."""
        graph("monkey", 128)
        expected = [0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99]
        for shape_name in ["hyperellipsoid", "hypersphere"]:
            confidences = [
                c.kwargs["confidence"]
                for c in mock_auprc.call_args_list
                if c.kwargs["shape"] == shape_name
            ]
            assert confidences == expected

    def test_each_trace_has_eleven_data_points(self, mock_auprc: MagicMock) -> None:
        """Each trace has 9 swept points plus 2 interpolated endpoints."""
        fig = graph("monkey", 128)
        for trace in fig.data:
            assert len(trace.x) == 11
            assert len(trace.y) == 11

    def test_layout(self, mock_auprc: MagicMock) -> None:
        """Figure has correct title, axis labels, and axis ranges."""
        fig = graph("monkey", 128)
        assert fig.layout.title.text == "AUPRC: monkey (dim=128)"
        assert fig.layout.xaxis.title.text == "Weighted Recall"
        assert fig.layout.yaxis.title.text == "Weighted Precision"
        assert list(fig.layout.xaxis.range) == [0, 1]
        assert list(fig.layout.yaxis.range) == [0, 1]

    def test_prints_auprc(
        self,
        mock_auprc: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AUPRC scores are printed for each shape."""
        graph("monkey", 128)
        output = capsys.readouterr().out
        assert "hyperellipsoid AUPRC=" in output
        assert "hypersphere AUPRC=" in output
