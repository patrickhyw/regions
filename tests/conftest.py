import pytest

from util import set_seed


@pytest.fixture(autouse=True)
def _seed() -> None:
    set_seed()


@pytest.fixture()
def vecs() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 2.0],
    ]
