"""Causal transition trigger experiments."""

__all__ = ["TransitionTriggerHead", "decide_transition_actions"]


def __getattr__(name: str):
    if name == "TransitionTriggerHead":
        from .model import TransitionTriggerHead

        return TransitionTriggerHead
    if name == "decide_transition_actions":
        from .trigger_policy import decide_transition_actions

        return decide_transition_actions
    raise AttributeError(name)
