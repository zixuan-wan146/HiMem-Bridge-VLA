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


class StatefulTransitionPolicy:
    """Online policy with separate soft-plan and memory-write cooldowns.

    A recent soft plan must not suppress a later high-confidence memory write.
    Memory writes use their own cooldown and always trigger hard planning.
    """

    def __init__(
        self,
        *,
        planner_threshold: float,
        memory_write_threshold: float,
        replan_cooldown_frames: int = 0,
        memory_write_cooldown_frames: int = 0,
    ) -> None:
        if planner_threshold >= memory_write_threshold:
            raise ValueError("planner_threshold must be lower than memory_write_threshold")
        if replan_cooldown_frames < 0:
            raise ValueError("replan_cooldown_frames must be non-negative")
        if memory_write_cooldown_frames < 0:
            raise ValueError("memory_write_cooldown_frames must be non-negative")
        self.planner_threshold = float(planner_threshold)
        self.memory_write_threshold = float(memory_write_threshold)
        self.replan_cooldown_frames = int(replan_cooldown_frames)
        self.memory_write_cooldown_frames = int(memory_write_cooldown_frames)
        self._last_plan_frame: int | None = None
        self._last_memory_write_frame: int | None = None
        self._step = -1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StatefulTransitionPolicy":
        policy_config = config.get("trigger_policy", {})
        if policy_config.get("memory_write_implies_plan") is False:
            raise ValueError("transition_trigger requires trigger_policy.memory_write_implies_plan=true")
        return cls(
            planner_threshold=float(policy_config["planner_threshold"]),
            memory_write_threshold=float(policy_config["memory_write_threshold"]),
            replan_cooldown_frames=int(policy_config.get("replan_cooldown_frames", 0)),
            memory_write_cooldown_frames=int(policy_config.get("memory_write_cooldown_frames", 0)),
        )

    def reset(self) -> None:
        self._last_plan_frame = None
        self._last_memory_write_frame = None
        self._step = -1

    def decide(self, score: float, *, frame_index: int | None = None) -> TriggerDecision:
        if frame_index is None:
            self._step += 1
            frame = self._step
        else:
            frame = int(frame_index)
            self._step = max(self._step, frame)

        score = float(score)
        if score >= self.memory_write_threshold:
            if self._in_cooldown(frame, self._last_memory_write_frame, self.memory_write_cooldown_frames):
                return self._no_trigger(score)
            self._last_memory_write_frame = frame
            self._last_plan_frame = frame
            return TriggerDecision(
                score=score,
                planner_threshold=self.planner_threshold,
                memory_write_threshold=self.memory_write_threshold,
                memory_write=True,
                soft_plan=False,
                hard_plan=True,
            )

        if score >= self.planner_threshold:
            if self._in_cooldown(frame, self._last_plan_frame, self.replan_cooldown_frames):
                return self._no_trigger(score)
            self._last_plan_frame = frame
            return TriggerDecision(
                score=score,
                planner_threshold=self.planner_threshold,
                memory_write_threshold=self.memory_write_threshold,
                memory_write=False,
                soft_plan=True,
                hard_plan=False,
            )

        return self._no_trigger(score)

    def _no_trigger(self, score: float) -> TriggerDecision:
        return TriggerDecision(
            score=float(score),
            planner_threshold=self.planner_threshold,
            memory_write_threshold=self.memory_write_threshold,
            memory_write=False,
            soft_plan=False,
            hard_plan=False,
        )

    @staticmethod
    def _in_cooldown(frame: int, last_frame: int | None, cooldown_frames: int) -> bool:
        if last_frame is None:
            return False
        return frame - last_frame <= cooldown_frames


class CausalPeakTransitionPolicy:
    """Causal one-step delayed peak confirmation policy.

    At time ``t`` the policy can confirm whether the score at ``t-1`` was a
    local peak because both its left and right neighbors are known. The action
    is emitted at the current frame, so the policy remains causal while avoiding
    repeated threshold firing on rising score plateaus.
    """

    def __init__(self, threshold_policy: StatefulTransitionPolicy) -> None:
        self.threshold_policy = threshold_policy
        self._left_score: float | None = None
        self._previous_score: float | None = None
        self._step = -1

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CausalPeakTransitionPolicy":
        return cls(StatefulTransitionPolicy.from_config(config))

    def reset(self) -> None:
        self.threshold_policy.reset()
        self._left_score = None
        self._previous_score = None
        self._step = -1

    def decide(self, score: float, *, frame_index: int | None = None) -> TriggerDecision:
        if frame_index is None:
            self._step += 1
            frame = self._step
        else:
            frame = int(frame_index)
            self._step = max(self._step, frame)

        current_score = float(score)
        decision = self._no_trigger(current_score)
        if self._previous_score is not None:
            left_score = -float("inf") if self._left_score is None else self._left_score
            if self._previous_score >= left_score and self._previous_score >= current_score:
                decision = self.threshold_policy.decide(self._previous_score, frame_index=frame)

        self._left_score = self._previous_score
        self._previous_score = current_score
        return decision

    def _no_trigger(self, score: float) -> TriggerDecision:
        return TriggerDecision(
            score=float(score),
            planner_threshold=self.threshold_policy.planner_threshold,
            memory_write_threshold=self.threshold_policy.memory_write_threshold,
            memory_write=False,
            soft_plan=False,
            hard_plan=False,
        )


def build_transition_policy_from_config(config: dict[str, Any]) -> StatefulTransitionPolicy | CausalPeakTransitionPolicy:
    mode = str(config.get("trigger_policy", {}).get("score_mode", "threshold"))
    if mode == "threshold":
        return StatefulTransitionPolicy.from_config(config)
    if mode == "causal_peak":
        return CausalPeakTransitionPolicy.from_config(config)
    raise ValueError("trigger_policy.score_mode must be 'threshold' or 'causal_peak'")
