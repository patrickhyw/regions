from __future__ import annotations

from typing import Protocol


class Shape(Protocol):
    def contains(self, vec: list[float]) -> bool: ...

    @classmethod
    # Confidence controls the tradeoff between precision and recall
    # of region containment.
    def fit(cls, vecs: list[list[float]], confidence: float = 0.95) -> Shape: ...
