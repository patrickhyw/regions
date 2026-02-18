import numpy as np

from shape import Shape


class _FakeShape:
    def contains(self, vec: list[float]) -> bool:
        return True

    @classmethod
    def fit(cls, vecs: np.ndarray) -> "_FakeShape":
        return cls()


class TestShape:
    def test_protocol_accepts_conforming_class(self) -> None:
        """A class with a contains method satisfies the Shape protocol."""
        shape: Shape = _FakeShape()
        assert shape.contains([1.0, 2.0])

    def test_fit_classmethod_satisfies_protocol(self) -> None:
        """A class with a fit classmethod satisfies the Shape protocol."""
        cls: type[Shape] = _FakeShape
        shape = cls.fit(np.array([[1.0, 2.0]]))
        assert shape.contains([1.0, 2.0])
