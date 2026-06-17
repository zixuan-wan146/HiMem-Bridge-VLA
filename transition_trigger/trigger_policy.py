from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TriggerDecision:
    score: float
    planner_threshold: float
    memory_write_threshold: float
    memory_write: bool
    soft_plan: bool
    hard_plan: bool

    @property
    def should_plan(self) -> bool:
        return self.soft_plan or self.hard_plan


def decide_transition_actions(
    score: float,
    *,
    planner_threshold: float,
    memory_write_threshold: float,
) -> TriggerDecision:
    """Map a transition score to memory and planning actions.

    A memory write is a high-confidence commit and must always trigger one hard
    plan/replan. Lower scores may still trigger soft planning without writing
    memory.
    """

    if planner_threshold >= memory_write_threshold:
        raise ValueError("planner_threshold must be lower than memory_write_threshold")
    score = float(score)
    planner_threshold = float(planner_threshold)
    memory_write_threshold = float(memory_write_threshold)
    memory_write = score >= memory_write_threshold
    soft_plan = (not memory_write) and score >= planner_threshold
    hard_plan = memory_write
    return TriggerDecision(
        score=score,
        planner_threshold=planner_threshold,
        memory_write_threshold=memory_write_threshold,
        memory_write=memory_write,
        soft_plan=soft_plan,
        hard_plan=hard_plan,
    )


def decide_transition_actions_from_config(score: float, config: dict[str, Any]) -> TriggerDecision:
    policy_config = config.get("trigger_policy", {})
    planner_threshold = policy_config.get("planner_threshold")
    memory_write_threshold = policy_config.get("memory_write_threshold")
    if planner_threshold is None:
        raise ValueError("trigger_policy.planner_threshold must be set before runtime use")
    if memory_write_threshold is None:
        raise ValueError("trigger_policy.memory_write_threshold must be set before runtime use")
    if policy_config.get("memory_write_implies_plan") is False:
        raise ValueError("transition_trigger requires trigger_policy.memory_write_implies_plan=true")
    return decide_transition_actions(
        score,
        planner_threshold=float(planner_threshold),
        memory_write_threshold=float(memory_write_threshold),
    )
