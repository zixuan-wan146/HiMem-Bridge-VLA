from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from himem_bridge_vla.runtime import websocket_server


class FakeRuntimeModel:
    captured_config: dict | None = None

    def __init__(self, config):
        self.config = dict(config)
        self.per_action_dim = int(self.config.get("per_action_dim", 7))
        FakeRuntimeModel.captured_config = self.config

    def eval(self):
        return self

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def to(self, device):
        self.loaded_device = device
        return self


def test_runtime_server_overrides_checkpoint_vlm_with_local_only_config(tmp_path, monkeypatch):
    ckpt_dir = _checkpoint_dir(tmp_path)
    monkeypatch.setattr(websocket_server, "HiMemBridgeVLA", FakeRuntimeModel)
    monkeypatch.setattr(websocket_server, "load_checkpoint_payload", lambda *args, **kwargs: {})

    websocket_server.load_model_and_normalizer(
        ckpt_dir,
        device="cpu",
        vlm_name="/models/InternVL3-1B",
        vlm_local_files_only=True,
    )

    config = FakeRuntimeModel.captured_config
    assert config is not None
    assert config["vlm_name"] == "/models/InternVL3-1B"
    assert config["vlm_local_files_only"] is True
    assert config["load_vlm"] is True


def _checkpoint_dir(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "config.json").write_text(
        json.dumps(
            {
                "vlm_name": "OpenGVLab/InternVL3-1B",
                "load_vlm": False,
                "horizon": 14,
                "per_action_dim": 7,
                "state_dim": 7,
                "action_dim": 98,
            }
        )
    )
    (ckpt_dir / "norm_stats.json").write_text(
        json.dumps(
            {
                "libero": {
                    "observation.state": {"min": [0.0] * 7, "max": [1.0] * 7},
                    "action": {"min": [-1.0] * 7, "max": [1.0] * 7},
                }
            }
        )
    )
    (ckpt_dir / "model.pt").write_bytes(b"checkpoint")
    return ckpt_dir
