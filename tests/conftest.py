import numpy as np
import pytest


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture()
def vecs() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 2.0],
    ]
