import numpy as np

from util import set_seed


class TestSetSeed:
    def test_produces_reproducible_output(self) -> None:
        """Calling set_seed with the same seed produces identical random
        sequences."""
        set_seed(0)
        a = np.random.standard_normal(5)
        set_seed(0)
        b = np.random.standard_normal(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different sequences."""
        set_seed(0)
        a = np.random.standard_normal(5)
        set_seed(1)
        b = np.random.standard_normal(5)
        assert not np.array_equal(a, b)
