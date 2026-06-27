from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> T:
        if not name:
            raise ValueError("registry name must be non-empty")
        if name in self._items:
            raise KeyError(f"registry entry already exists: {name}")
        self._items[name] = item
        return item

    def decorator(self, name: str) -> Callable[[T], T]:
        def _register(item: T) -> T:
            return self.register(name, item)

        return _register

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown registry entry: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
