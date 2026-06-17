"""Causal transition trigger experiments."""

__all__ = [
    "CausalPeakTransitionPolicy",
    "StatefulTransitionPolicy",
    "TransitionTriggerHead",
    "TransitionTriggerOnlineSession",
    "TransitionTriggerRuntime",
    "TransitionTriggerSession",
    "build_transition_policy_from_config",
    "decide_transition_actions",
    "load_selected_trigger",
]


def __getattr__(name: str):
    if name == "TransitionTriggerHead":
        from .model import TransitionTriggerHead

        return TransitionTriggerHead
    if name == "decide_transition_actions":
        from .trigger_policy import decide_transition_actions

        return decide_transition_actions
    if name == "StatefulTransitionPolicy":
        from .trigger_policy import StatefulTransitionPolicy

        return StatefulTransitionPolicy
    if name == "CausalPeakTransitionPolicy":
        from .trigger_policy import CausalPeakTransitionPolicy

        return CausalPeakTransitionPolicy
    if name == "build_transition_policy_from_config":
        from .trigger_policy import build_transition_policy_from_config

        return build_transition_policy_from_config
    if name == "TransitionTriggerRuntime":
        from .runtime import TransitionTriggerRuntime

        return TransitionTriggerRuntime
    if name == "TransitionTriggerOnlineSession":
        from .runtime import TransitionTriggerOnlineSession

        return TransitionTriggerOnlineSession
    if name == "TransitionTriggerSession":
        from .runtime import TransitionTriggerSession

        return TransitionTriggerSession
    if name == "load_selected_trigger":
        from .runtime import load_selected_trigger

        return load_selected_trigger
    raise AttributeError(name)
