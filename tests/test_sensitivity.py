from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import plotly.graph_objects as go
import pytest

from sensitivity import (
    MIN_SUBTREE_SIZE,
    NodeResult,
    TrainTestSplit,
    _spaceaug_concept,
    create_train_test_split,
    graph,
    print_node_results,
    sensitivity,
)
from tree import KnowledgeNode, KnowledgeTree


class TestSensitivity:
    def test_single_concept(self) -> None:
        """Generates three whitespace variants for one concept."""
        result = _spaceaug_concept("dog")
        assert result == [" dog", "dog ", " dog "]

    @pytest.fixture()
    def tree(self) -> KnowledgeTree:
        return KnowledgeTree(
            root=KnowledgeNode(
                concept="animal",
                children=[
                    KnowledgeNode(concept="dog"),
                    KnowledgeNode(concept="cat"),
                    KnowledgeNode(concept="fish"),
                    KnowledgeNode(concept="bird"),
                    KnowledgeNode(concept="frog"),
                ],
            )
        )

    @pytest.fixture()
    def mock_build_named_tree(
        self, tree: KnowledgeTree
    ) -> Generator[MagicMock, None, None]:
        with patch("sensitivity.build_named_tree", return_value=tree) as mock:
            yield mock

    @pytest.fixture()
    def mock_get_embeddings(self) -> Generator[MagicMock, None, None]:
        def _fake(texts: list[str], *, dimension: int) -> list[list[float]]:
            return [np.random.standard_normal(dimension).tolist() for _ in texts]

        with patch("sensitivity.get_embeddings", side_effect=_fake) as mock:
            yield mock

    @pytest.mark.parametrize("shape", ["hyperellipsoid", "convexhull"])
    def test_result_count_matches_tree_nodes(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
        shape: str,
    ) -> None:
        """Only nodes with subtree size >= MIN_SUBTREE_SIZE get results."""
        results = sensitivity(shape=shape, tree_name="test", dimension=3)
        assert len(results) == 1

    def test_small_subtrees_excluded(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """Nodes with subtree size < MIN_SUBTREE_SIZE are excluded."""
        results = sensitivity(shape="hyperellipsoid", tree_name="test", dimension=3)
        result_concepts = {r.concept for r in results}
        # Root has 6 concepts (animal + 5 leaves) >= MIN_SUBTREE_SIZE.
        assert "animal" in result_concepts
        # Each leaf has 1 concept < MIN_SUBTREE_SIZE.
        for leaf in ["dog", "cat", "fish", "bird", "frog"]:
            assert leaf not in result_concepts

    def test_train_fraction_zero_all_test(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """train_fraction=0.0 puts all spaceaug in test."""
        results = sensitivity(
            shape="hyperellipsoid",
            tree_name="test",
            dimension=3,
            train_fraction=0.0,
        )
        total = sum(r.test_total for r in results)
        assert total > 0

    def test_train_fraction_one_raises(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """train_fraction=1.0 raises ValueError (no test data)."""
        with pytest.raises(ValueError):
            sensitivity(
                shape="hyperellipsoid",
                tree_name="test",
                dimension=3,
                train_fraction=1.0,
            )

    def test_all_results_are_node_results(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """Every result is a NodeResult with the right fields."""
        results = sensitivity(shape="convexhull", tree_name="test", dimension=3)
        for r in results:
            assert isinstance(r, NodeResult)
            assert r.test_contained >= 0
            assert r.test_total >= 0
            assert r.test_contained <= r.test_total
            assert r.train_contained >= 0
            assert r.train_total >= 0
            assert r.train_contained <= r.train_total

    def test_defaults(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """tree_name defaults to 'monkey' and dimension defaults to 128."""
        sensitivity(shape="hyperellipsoid")
        mock_build_named_tree.assert_called_once_with("monkey")
        mock_get_embeddings.assert_called_once()
        _, kwargs = mock_get_embeddings.call_args
        assert kwargs["dimension"] == 128


class TestPrintNodeResults:
    @pytest.fixture()
    def results(self) -> list[NodeResult]:
        return [
            NodeResult(
                concept="alpha",
                test_contained=5,
                test_total=10,
                train_contained=8,
                train_total=10,
            ),
            NodeResult(
                concept="beta",
                test_contained=3,
                test_total=6,
                train_contained=5,
                train_total=6,
            ),
            NodeResult(
                concept="gamma",
                test_contained=1,
                test_total=2,
                train_contained=2,
                train_total=2,
            ),
        ]

    def test_top_limits_printed_nodes(
        self, results: list[NodeResult], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Only the top N nodes by total are printed."""
        print_node_results(results, top=2)
        output = capsys.readouterr().out
        assert "alpha" in output
        assert "beta" in output
        assert "gamma" not in output

    def test_prints_in_descending_total_order(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Nodes are printed sorted by test_total descending."""
        results = [
            NodeResult(
                concept="small",
                test_contained=1,
                test_total=2,
                train_contained=2,
                train_total=2,
            ),
            NodeResult(
                concept="big",
                test_contained=5,
                test_total=10,
                train_contained=8,
                train_total=10,
            ),
        ]
        print_node_results(results, top=2)
        output = capsys.readouterr().out
        assert output.index("big") < output.index("small")

    def test_overall_summary_includes_all_nodes(
        self, results: list[NodeResult], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Overall summary aggregates all nodes regardless of top."""
        print_node_results(results, top=1)
        output = capsys.readouterr().out
        assert "overall" in output
        assert "test=9/18" in output
        assert "train=15/18" in output

    def test_training_accuracy_in_output(
        self, results: list[NodeResult], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both training and test accuracy appear per node and overall."""
        print_node_results(results, top=3)
        output = capsys.readouterr().out
        # Per-node lines should have train= and test=.
        assert "train=8/10" in output
        assert "test=5/10" in output
        # Overall line should have train= and test=.
        assert "train=15/18" in output
        assert "test=9/18" in output


class TestGraph:
    @pytest.fixture()
    def tree(self) -> KnowledgeTree:
        return KnowledgeTree(
            root=KnowledgeNode(
                concept="animal",
                children=[
                    KnowledgeNode(concept="dog"),
                    KnowledgeNode(concept="cat"),
                    KnowledgeNode(concept="fish"),
                    KnowledgeNode(concept="bird"),
                    KnowledgeNode(concept="frog"),
                ],
            )
        )

    @pytest.fixture()
    def mock_build_named_tree(
        self, tree: KnowledgeTree
    ) -> Generator[MagicMock, None, None]:
        with patch("sensitivity.build_named_tree", return_value=tree) as mock:
            yield mock

    @pytest.fixture()
    def mock_get_embeddings(self) -> Generator[MagicMock, None, None]:
        def _fake(texts: list[str], *, dimension: int) -> list[list[float]]:
            return [np.random.standard_normal(dimension).tolist() for _ in texts]

        with patch("sensitivity.get_embeddings", side_effect=_fake) as mock:
            yield mock

    def test_returns_figure(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """graph() returns a plotly Figure."""
        fig = graph(tree_name="test", dimension=3)
        assert isinstance(fig, go.Figure)

    def test_has_four_traces(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """Figure has 4 traces (one per shape/spaceaug combo)."""
        fig = graph(tree_name="test", dimension=3)
        assert len(fig.data) == 4

    def test_spaceaug_traces_have_ten_x_values(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """Spaceaug traces have 10 x-values (0.0-0.9)."""
        fig = graph(tree_name="test", dimension=3)
        spaceaug_traces = [
            t for t in fig.data if "spaceaug" in t.name and "no-spaceaug" not in t.name
        ]
        for trace in spaceaug_traces:
            assert len(trace.x) == 10

    def test_no_spaceaug_traces_have_nine_x_values(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """No-spaceaug traces have 9 x-values (0.1-0.9)."""
        fig = graph(tree_name="test", dimension=3)
        no_spaceaug_traces = [t for t in fig.data if "no-spaceaug" in t.name]
        for trace in no_spaceaug_traces:
            assert len(trace.x) == 9

    def test_y_values_between_zero_and_one(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """All y-values are between 0 and 1."""
        fig = graph(tree_name="test", dimension=3)
        for trace in fig.data:
            for y in trace.y:
                assert 0.0 <= y <= 1.0

    def test_uses_progress_bar(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """graph() creates one progress task per series line."""
        with patch("sensitivity.Progress") as mock_progress_cls:
            mock_progress = MagicMock()
            mock_progress_cls.return_value.__enter__ = MagicMock(
                return_value=mock_progress
            )
            mock_progress_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_task = MagicMock()
            mock_progress.add_task.return_value = mock_task
            graph(tree_name="test", dimension=3)
            assert mock_progress.add_task.call_args_list == [
                (("hyperellipsoid spaceaug",), {"total": 10}),
                (("hyperellipsoid no-spaceaug",), {"total": 9}),
                (("convexhull spaceaug",), {"total": 10}),
                (("convexhull no-spaceaug",), {"total": 9}),
            ]
            assert mock_progress.advance.call_count == 38

    def test_title(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """Figure title includes tree name, dimension, and analysis type."""
        fig = graph(tree_name="test", dimension=3)
        assert fig.layout.title.text == "Sensitivity Analysis: test (dim=3)"

    def test_trace_styling(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """Traces are styled by shape (color) and spaceaug mode (dash)."""
        fig = graph(tree_name="test", dimension=3)
        traces = {t.name: t for t in fig.data}
        expected = {
            "hyperellipsoid spaceaug": ("blue", "solid"),
            "hyperellipsoid no-spaceaug": ("blue", "dash"),
            "convexhull spaceaug": ("green", "solid"),
            "convexhull no-spaceaug": ("green", "dash"),
        }
        for name, (color, dash) in expected.items():
            assert traces[name].line.color == color
            assert traces[name].line.dash == dash


class TestCreateTrainTestSplit:
    @pytest.fixture()
    def concepts(self) -> list[str]:
        return ["dog", "cat", "fish"]

    def test_no_spaceaug_splits_originals(self, concepts: list[str]) -> None:
        """use_spaceaug=False returns only originals, split into
        disjoint train/test sets where test_by_orig maps each test
        concept to [itself]."""
        split = create_train_test_split(
            concepts, train_fraction=0.5, use_spaceaug=False
        )
        assert split.all_concepts == concepts
        test_concepts = {c for cs in split.test_by_orig.values() for c in cs}
        assert split.train_set & test_concepts == set()
        assert split.train_set | test_concepts == set(concepts)
        mid = int(len(concepts) * 0.5)
        assert len(split.train_set) == mid
        assert len(test_concepts) == len(concepts) - mid
        for orig, tests in split.test_by_orig.items():
            assert tests == [orig]

    @pytest.mark.parametrize("train_fraction", [0.0, 0.5])
    def test_spaceaug_mode(self, concepts: list[str], train_fraction: float) -> None:
        """use_spaceaug=True generates 3 variants per concept, keeps
        all originals in train, and partitions spaceaug between
        train and test."""
        split = create_train_test_split(concepts, train_fraction, use_spaceaug=True)
        assert len(split.all_concepts) == len(concepts) * 4
        assert set(concepts) <= split.train_set
        spaceaug = set(split.all_concepts) - set(concepts)
        test_concepts = {c for cs in split.test_by_orig.values() for c in cs}
        train_spaceaug = split.train_set - set(concepts)
        assert train_spaceaug & test_concepts == set()
        assert train_spaceaug | test_concepts == spaceaug
        assert set(split.test_by_orig.keys()) <= set(concepts)

    @pytest.mark.parametrize(
        ("concepts_arg", "train_fraction", "use_spaceaug"),
        [
            pytest.param(["dog"], 1.0, True, id="spaceaug_fraction_one"),
            pytest.param(["dog"], 1.0, False, id="no_spaceaug_fraction_one"),
            pytest.param(["dog"], 0.0, False, id="no_spaceaug_fraction_zero"),
            pytest.param([], 0.5, True, id="empty_concepts_spaceaug"),
            pytest.param([], 0.5, False, id="empty_concepts_no_spaceaug"),
        ],
    )
    def test_validation_errors(
        self,
        concepts_arg: list[str],
        train_fraction: float,
        use_spaceaug: bool,
    ) -> None:
        """Invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            create_train_test_split(concepts_arg, train_fraction, use_spaceaug)
