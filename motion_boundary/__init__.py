"""Motion-state skill boundary detector experiments."""

__all__ = ["MotionStateBoundaryHead"]


def __getattr__(name: str):
    if name == "MotionStateBoundaryHead":
        from .model import MotionStateBoundaryHead

        return MotionStateBoundaryHead
    raise AttributeError(name)
