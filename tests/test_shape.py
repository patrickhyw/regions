from shape import Shape


class _FakeShape:
    def contains(self, vec: list[float]) -> bool:
        return True


class TestShape:
    def test_protocol_accepts_conforming_class(self) -> None:
        """A class with a contains method satisfies the Shape protocol."""
        shape: Shape = _FakeShape()
        assert shape.contains([1.0, 2.0])
