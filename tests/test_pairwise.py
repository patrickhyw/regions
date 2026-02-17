from __future__ import annotations

import pytest

from pairwise import MAX_SIBLING_RATIO, MIN_SUBTREE_SIZE


class TestPairwiseConstants:
    @pytest.mark.parametrize(
        ("constant", "expected"),
        [
            pytest.param(MAX_SIBLING_RATIO, 3.0, id="max_sibling_ratio"),
            pytest.param(MIN_SUBTREE_SIZE, 10, id="min_subtree_size"),
        ],
    )
    def test_constant_values(self, constant: object, expected: object) -> None:
        assert constant == expected
