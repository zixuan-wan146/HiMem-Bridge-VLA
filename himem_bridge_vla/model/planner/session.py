from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PlanTokenQueueEntry:
    plan_tokens: Any
    executed_steps: int = 0


@dataclass(frozen=True)
class PlanTokenQueueState:
    executed_steps: int
    consumed_tokens: int
    residual_steps: int
    remaining_tokens: int


class PlanTokenQueue:
    """Per-session plan-token queue consumed by executed low-level steps."""

    def __init__(self, *, planning_horizon_steps: int, token_span_steps: int) -> None:
        if planning_horizon_steps <= 0:
            raise ValueError(f"planning_horizon_steps must be positive, got {planning_horizon_steps}")
        if token_span_steps <= 0:
            raise ValueError(f"token_span_steps must be positive, got {token_span_steps}")
        if planning_horizon_steps % token_span_steps != 0:
            raise ValueError("planning_horizon_steps must be divisible by token_span_steps")
        self.planning_horizon_steps = int(planning_horizon_steps)
        self.token_span_steps = int(token_span_steps)
        self.num_plan_steps = self.planning_horizon_steps // self.token_span_steps
        self._entries: dict[str, PlanTokenQueueEntry] = {}

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._entries.clear()
            return
        self._entries.pop(str(key), None)

    def record_executed_steps(self, key: str, executed_steps: int | None) -> PlanTokenQueueState | None:
        if executed_steps is None:
            return self.state(key)
        entry = self._entries.get(str(key))
        if entry is None:
            return None
        steps = int(executed_steps)
        if steps < 0:
            raise ValueError(f"executed_steps must be non-negative, got {executed_steps}")
        entry.executed_steps += steps
        return self.state(key)

    def should_refresh(
        self,
        key: str,
        *,
        refresh_requested: bool | None = None,
        requested_execute_steps: int | None = None,
    ) -> bool:
        entry = self._entries.get(str(key))
        if entry is None:
            return True
        if refresh_requested is True:
            return True
        if requested_execute_steps is None:
            requested_execute_steps = 0
        requested = int(requested_execute_steps)
        if requested < 0:
            raise ValueError(f"requested_execute_steps must be non-negative, got {requested_execute_steps}")
        return entry.executed_steps + requested > self.planning_horizon_steps

    def active_plan_tokens(self, key: str) -> Any | None:
        entry = self._entries.get(str(key))
        if entry is None:
            return None
        consumed_tokens, _ = divmod(max(0, entry.executed_steps), self.token_span_steps)
        consumed_tokens = min(consumed_tokens, self.num_plan_steps)
        return entry.plan_tokens[:, consumed_tokens:]

    def state(self, key: str) -> PlanTokenQueueState | None:
        entry = self._entries.get(str(key))
        if entry is None:
            return None
        consumed_tokens, residual_steps = divmod(max(0, entry.executed_steps), self.token_span_steps)
        consumed_tokens = min(consumed_tokens, self.num_plan_steps)
        return PlanTokenQueueState(
            executed_steps=int(entry.executed_steps),
            consumed_tokens=consumed_tokens,
            residual_steps=residual_steps,
            remaining_tokens=max(0, self.num_plan_steps - consumed_tokens),
        )

    def put(self, key: str, plan_tokens: Any) -> Any:
        self._entries[str(key)] = PlanTokenQueueEntry(plan_tokens=plan_tokens, executed_steps=0)
        return plan_tokens
