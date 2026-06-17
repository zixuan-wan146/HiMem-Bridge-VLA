from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TransitionTriggerServerResult:
    ready: bool
    dataset_name: str | None
    score: float | None
    memory_write: bool
    soft_plan: bool
    hard_plan: bool
    should_plan: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ServerTransitionTriggerManager:
    """Maintain transition-trigger online sessions for websocket inference."""

    def __init__(self, runtime: Any, *, default_dataset_name: str | None = None) -> None:
        self.runtime = runtime
        self.default_dataset_name = default_dataset_name
        self._sessions: dict[tuple[str, str | None], Any] = {}

    @classmethod
    def from_package(
        cls,
        package_dir: str | Path,
        *,
        device: str,
        default_dataset_name: str | None = None,
    ) -> "ServerTransitionTriggerManager":
        from transition_trigger.runtime import TransitionTriggerRuntime

        runtime = TransitionTriggerRuntime.from_package(package_dir, device=device)
        return cls(runtime, default_dataset_name=default_dataset_name)

    def reset(self, episode_key: str | None = None) -> None:
        if episode_key is None:
            self._sessions.clear()
            return
        for key in [key for key in self._sessions if key[0] == str(episode_key)]:
            self._sessions.pop(key, None)

    def update(
        self,
        *,
        episode_key: str | None,
        frame: Mapping[str, Any],
        dataset_name: str | None = None,
        frame_index: int | None = None,
        reset: bool = False,
    ) -> TransitionTriggerServerResult:
        if not episode_key:
            raise ValueError("transition trigger requires episode_id or session_id")
        resolved_dataset = dataset_name or self.default_dataset_name
        session_key = (str(episode_key), None if resolved_dataset is None else str(resolved_dataset))
        if reset:
            self._sessions.pop(session_key, None)
        session = self._sessions.get(session_key)
        if session is None:
            session = self.runtime.new_online_session(dataset_name=resolved_dataset)
            self._sessions[session_key] = session
        output = session.append(frame, frame_index=frame_index)
        if output is None:
            return TransitionTriggerServerResult(
                ready=False,
                dataset_name=session.spec.dataset_name,
                score=None,
                memory_write=False,
                soft_plan=False,
                hard_plan=False,
                should_plan=False,
            )
        decision = output.decision
        return TransitionTriggerServerResult(
            ready=True,
            dataset_name=session.spec.dataset_name,
            score=float(output.score),
            memory_write=bool(decision.memory_write),
            soft_plan=bool(decision.soft_plan),
            hard_plan=bool(decision.hard_plan),
            should_plan=bool(decision.should_plan),
        )
