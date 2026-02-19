from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sensitivity import (
    NodeResult,
    TrainTestSplit,
    _spaceaug_concept,
    create_train_test_split,
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
        """One NodeResult per tree node."""
        results = sensitivity(shape=shape, tree_name="test", dimension=3)
        assert len(results) == 3

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
        total = sum(r.total for r in results)
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
            assert r.contained >= 0
            assert r.total >= 0
            assert r.contained <= r.total

    def test_defaults(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
    ) -> None:
        """tree_name defaults to 'primate' and dimension defaults to 128."""
        sensitivity(shape="hyperellipsoid")
        mock_build_named_tree.assert_called_once_with("primate")
        mock_get_embeddings.assert_called_once()
        _, kwargs = mock_get_embeddings.call_args
        assert kwargs["dimension"] == 128


class TestPrintNodeResults:
    @pytest.fixture()
    def results(self) -> list[NodeResult]:
        return [
            NodeResult(concept="alpha", contained=5, total=10),
            NodeResult(concept="beta", contained=3, total=6),
            NodeResult(concept="gamma", contained=1, total=2),
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
        """Nodes are printed sorted by total descending."""
        results = [
            NodeResult(concept="small", contained=1, total=2),
            NodeResult(concept="big", contained=5, total=10),
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
        assert "overall contained=9/18" in output


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
