from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from transition_trigger.config import load_config
from transition_trigger.model import TransitionTriggerHead
from transition_trigger.online_features import OnlineFeatureSpec, OnlineTransitionFeatureBuffer
from transition_trigger.trigger_policy import (
    CausalPeakTransitionPolicy,
    StatefulTransitionPolicy,
    TriggerDecision,
    build_transition_policy_from_config,
    decide_transition_actions_from_config,
)


@dataclass(frozen=True)
class TriggerRuntimeOutput:
    score: float
    decision: TriggerDecision


class TransitionTriggerSession:
    """Online scoring session for one trajectory/rollout.

    The trained head itself is stateless, but the deployed trigger policy is
    not: cooldowns and causal-peak confirmation both need prior scores. Create
    one session per environment episode or robot rollout.
    """

    def __init__(
        self,
        runtime: "TransitionTriggerRuntime",
        *,
        policy: StatefulTransitionPolicy | CausalPeakTransitionPolicy | None = None,
    ) -> None:
        self.runtime = runtime
        self.policy = policy if policy is not None else runtime.new_policy()

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def decide_window(self, features: torch.Tensor, *, frame_index: int | None = None) -> TriggerRuntimeOutput:
        scores = self.runtime.score_window(features)
        if scores.numel() != 1:
            raise ValueError("TransitionTriggerSession.decide_window expects one window")
        score = float(scores.item())
        return TriggerRuntimeOutput(score=score, decision=self.policy.decide(score, frame_index=frame_index))


class TransitionTriggerOnlineSession:
    """End-to-end online session with config-driven feature construction."""

    def __init__(
        self,
        runtime: "TransitionTriggerRuntime",
        *,
        dataset_name: str | None,
        policy: StatefulTransitionPolicy | CausalPeakTransitionPolicy | None = None,
    ) -> None:
        self.runtime = runtime
        self.feature_buffer = OnlineTransitionFeatureBuffer(runtime.config, dataset_name=dataset_name)
        if self.feature_buffer.spec.input_dim != runtime.input_dim:
            raise ValueError(
                f"online feature input_dim={self.feature_buffer.spec.input_dim} "
                f"does not match runtime input_dim={runtime.input_dim}"
            )
        self.session = TransitionTriggerSession(runtime, policy=policy)

    @property
    def spec(self) -> OnlineFeatureSpec:
        return self.feature_buffer.spec

    @property
    def ready(self) -> bool:
        return self.feature_buffer.ready

    def reset(self) -> None:
        self.feature_buffer.reset()
        self.session.reset()

    @torch.no_grad()
    def append(self, frame: Mapping[str, Any], *, frame_index: int | None = None) -> TriggerRuntimeOutput | None:
        window = self.feature_buffer.append_and_build(frame)
        if window is None:
            return None
        return self.session.decide_window(window, frame_index=frame_index)


class TransitionTriggerRuntime:
    """Runtime wrapper for a trained TransitionTrigger checkpoint.

    The wrapper keeps loading, feature shape validation, score computation, and
    policy decisions in one place so memory-write/planner integrations do not
    need to duplicate checkpoint-specific details.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.device = torch.device(device)
        self.input_dim = self._resolve_input_dim()
        self.window_size = int(self.config["data"]["window_size"])
        self.model = TransitionTriggerHead(input_dim=self.input_dim, **self.config["model"]).to(self.device)
        self._load_checkpoint(self.checkpoint_path)
        self.model.eval()

    @classmethod
    def from_package(cls, package_dir: str | Path, *, device: str | torch.device = "cpu") -> "TransitionTriggerRuntime":
        package_path = Path(package_dir).expanduser()
        config_path = package_path / "config.yaml"
        checkpoint_path = package_path / "checkpoint.pt"
        if not config_path.exists():
            raise FileNotFoundError(f"selected trigger config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"selected trigger checkpoint not found: {checkpoint_path}")
        return cls(config=load_config(config_path), checkpoint_path=checkpoint_path, device=device)

    @torch.no_grad()
    def score_window(self, features: torch.Tensor) -> torch.Tensor:
        """Return transition probabilities for one or more causal windows.

        Accepts either ``[W, D]`` for one window or ``[B, W, D]`` for a batch.
        The output is always a 1-D tensor with one score per input window.
        """

        batch = self._prepare_features(features)
        logits = self.model(batch)
        return torch.sigmoid(logits).reshape(-1).detach().cpu()

    @torch.no_grad()
    def decide_window(self, features: torch.Tensor) -> TriggerRuntimeOutput:
        score_mode = str(self.config.get("trigger_policy", {}).get("score_mode", "threshold"))
        if score_mode != "threshold":
            raise ValueError(
                "TransitionTriggerRuntime.decide_window is stateless and only supports "
                "trigger_policy.score_mode='threshold'; use runtime.new_session().decide_window() "
                f"for score_mode='{score_mode}'"
            )
        scores = self.score_window(features)
        if scores.numel() != 1:
            raise ValueError("decide_window expects one window; use score_window for batched scoring")
        score = float(scores.item())
        return TriggerRuntimeOutput(
            score=score,
            decision=decide_transition_actions_from_config(score, self.config),
        )

    def new_policy(self) -> StatefulTransitionPolicy | CausalPeakTransitionPolicy:
        return build_transition_policy_from_config(self.config)

    def new_session(self) -> TransitionTriggerSession:
        return TransitionTriggerSession(self)

    def new_online_session(self, *, dataset_name: str | None = None) -> TransitionTriggerOnlineSession:
        return TransitionTriggerOnlineSession(self, dataset_name=dataset_name)

    def _resolve_input_dim(self) -> int:
        expected_input_dim = self.config.get("features", {}).get("expected_input_dim")
        if expected_input_dim is None:
            raise ValueError("features.expected_input_dim must be set for runtime loading")
        input_dim = int(expected_input_dim)
        if input_dim <= 0:
            raise ValueError(f"features.expected_input_dim must be positive, got {input_dim}")
        return input_dim

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint must be a dict: {checkpoint_path}")
        checkpoint_input_dim = checkpoint.get("input_dim")
        if checkpoint_input_dim is not None and int(checkpoint_input_dim) != self.input_dim:
            raise ValueError(f"checkpoint input_dim={checkpoint_input_dim} != config input_dim={self.input_dim}")
        state = checkpoint.get("model")
        if state is None:
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state is None:
            raise ValueError(f"checkpoint does not contain model weights: {checkpoint_path}")
        self.model.load_state_dict(state, strict=True)

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(features):
            features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"features must have shape [W, D] or [B, W, D], got {tuple(features.shape)}")
        if int(features.shape[1]) != self.window_size:
            raise ValueError(f"feature window {features.shape[1]} != configured window_size={self.window_size}")
        if int(features.shape[2]) != self.input_dim:
            raise ValueError(f"feature dim {features.shape[2]} != configured input_dim={self.input_dim}")
        return features.to(device=self.device, dtype=torch.float32)


def load_selected_trigger(
    package_dir: str | Path = "/root/autodl-tmp/runs/transition_trigger/selected/robomme_rmbench_w32_value_delta_transformer_d512",
    *,
    device: str | torch.device = "cpu",
) -> TransitionTriggerRuntime:
    return TransitionTriggerRuntime.from_package(package_dir, device=device)
