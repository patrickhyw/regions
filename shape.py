from __future__ import annotations

from typing import Protocol

import numpy as np


class Shape(Protocol):
    def contains(self, vec: list[float]) -> bool: ...

    @classmethod
    def fit(cls, vecs: np.ndarray) -> Shape: ...
