__all__ = [
    "build_arg_parser",
    "build_stage1_config",
    "enforce_stage1_contract",
    "train_stage1",
]


def __getattr__(name: str):
    if name == "build_arg_parser":
        from .cli import build_arg_parser

        return build_arg_parser
    if name == "build_stage1_config":
        from .config import build_stage1_config

        return build_stage1_config
    if name == "enforce_stage1_contract":
        from .validators import enforce_stage1_contract

        return enforce_stage1_contract
    if name == "train_stage1":
        from .loop import train_stage1

        return train_stage1
    raise AttributeError(name)
