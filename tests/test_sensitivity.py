from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sensitivity import NodeResult, _spaceaug_concept, sensitivity
from tree import KnowledgeNode, KnowledgeTree


class TestSpaceaugConcepts:
    def test_single_concept(self) -> None:
        """Generates three whitespace variants for one concept."""
        result = _spaceaug_concept("dog")
        assert result == [" dog", "dog ", " dog "]


class TestSensitivity:
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

    @pytest.mark.parametrize(
        ("train_fraction", "expected_positive"),
        [
            pytest.param(0.0, True, id="zero_all_test"),
            pytest.param(1.0, False, id="one_no_test"),
        ],
    )
    def test_train_fraction_controls_test_data(
        self,
        mock_build_named_tree: MagicMock,
        mock_get_embeddings: MagicMock,
        train_fraction: float,
        expected_positive: bool,
    ) -> None:
        """train_fraction=0.0 puts all spaceaug in test;
        train_fraction=1.0 puts none in test."""
        results = sensitivity(
            shape="hyperellipsoid",
            tree_name="test",
            dimension=3,
            train_fraction=train_fraction,
        )
        total = sum(r.total for r in results)
        if expected_positive:
            assert total > 0
        else:
            assert total == 0

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
