from __future__ import annotations

from typing import Protocol


class Shape(Protocol):
    def contains(self, vec: list[float]) -> bool: ...

    @classmethod
    def fit(cls, vecs: list[list[float]]) -> Shape: ...
