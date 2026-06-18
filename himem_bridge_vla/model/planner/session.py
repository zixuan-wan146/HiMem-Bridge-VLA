from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CoarsePlanCacheEntry:
    plan_tokens: Any
    age: int = 0


class CoarsePlanSessionCache:
    """Small runtime cache for transition-triggered plan reuse."""

    def __init__(self, *, max_age_steps: int) -> None:
        if max_age_steps <= 0:
            raise ValueError(f"max_age_steps must be positive, got {max_age_steps}")
        self.max_age_steps = int(max_age_steps)
        self._entries: dict[str, CoarsePlanCacheEntry] = {}

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._entries.clear()
            return
        self._entries.pop(str(key), None)

    def should_refresh(self, key: str, *, refresh_requested: bool | None = None) -> bool:
        entry = self._entries.get(str(key))
        if entry is None:
            return True
        if refresh_requested is True:
            return True
        return entry.age >= self.max_age_steps

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(str(key))
        if entry is None:
            return None
        entry.age += 1
        return entry.plan_tokens

    def put(self, key: str, plan_tokens: Any) -> Any:
        self._entries[str(key)] = CoarsePlanCacheEntry(plan_tokens=plan_tokens, age=0)
        return plan_tokens
