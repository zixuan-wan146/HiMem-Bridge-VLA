__all__ = [
    "ProgressWarmupTrainingConfig",
    "ProgressWarmupTrainingResult",
    "progress_warmup_batch_loss",
    "run_progress_warmup_training",
]


def __getattr__(name: str):
    if name in __all__:
        from himem_bridge_vla.training import progress_warmup

        return getattr(progress_warmup, name)
    raise AttributeError(name)
