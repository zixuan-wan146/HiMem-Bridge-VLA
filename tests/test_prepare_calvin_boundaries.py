from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_calvin_boundaries.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_calvin_boundaries", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_calvin_boundaries_exports_segments(tmp_path: Path):
    module = load_module()
    auto_lang_ann = tmp_path / "auto_lang_ann.npy"
    output = tmp_path / "boundaries.jsonl"
    payload = {
        "language": {
            "ann": np.asarray(["open drawer", "close drawer"], dtype=object),
            "task": np.asarray(["open_drawer", "close_drawer"], dtype=object),
        },
        "info": {
            "indx": np.asarray([[2, 5], [10, 12]], dtype=np.int64),
        },
    }
    np.save(auto_lang_ann, payload, allow_pickle=True)

    exit_code = module.main(["--auto-lang-ann", str(auto_lang_ann), "--output", str(output)])

    assert exit_code == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["start"] == 2
    assert rows[0]["end"] == 5
    assert rows[0]["language"] == "open drawer"
    assert rows[0]["skill_id"] != rows[1]["skill_id"]


def test_prepare_calvin_boundaries_exports_lerobot_episode_segments(tmp_path: Path):
    module = load_module()
    root = tmp_path / "calvin_lerobot"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 4, "task": "move the door to the left side"}) + "\n"
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 12, "tasks": ["move the door to the left side"], "length": 65}) + "\n"
    )
    output = tmp_path / "boundaries.jsonl"

    exit_code = module.main(["--lerobot-root", str(root), "--output", str(output)])

    assert exit_code == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {
            "segment_id": 0,
            "episode_id": "12",
            "start": 0,
            "end": 64,
            "task": "move the door to the left side",
            "skill_id": 4,
            "language": "move the door to the left side",
        }
    ]
