from typing import Protocol


class Shape(Protocol):
    def contains(self, vec: list[float]) -> bool: ...
