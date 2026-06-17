from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transition_trigger.config import load_config, write_resolved_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transition trigger ablation configs sequentially.")
    parser.add_argument("--index", default="transition_trigger/configs/robomme_rmbench_ablations/index.tsv")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = read_index(Path(args.index))
    if args.pattern:
        entries = [entry for entry in entries if args.pattern in entry["run_name"]]
    entries = entries[args.start :]
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        raise ValueError("no ablation configs selected")

    status_path = Path("transition_trigger/outputs/ablation_queue_status.jsonl")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    for ordinal, entry in enumerate(entries, start=1):
        run_one(entry, ordinal, len(entries), args.device, args.rerun, status_path)
    return 0


def read_index(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"empty index: {path}")
    header = lines[0].split("\t")
    entries = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        entries.append(dict(zip(header, values, strict=True)))
    return entries


def run_one(
    entry: dict[str, str],
    ordinal: int,
    total: int,
    device: str,
    rerun: bool,
    status_path: Path,
) -> None:
    config_path = Path(entry["config"])
    config = load_config(config_path)
    run_dir = Path(config["outputs"]["run_dir"]).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"

    write_status(status_path, entry, "start", {"ordinal": ordinal, "total": total})
    if rerun or not checkpoint_path.exists():
        train_log = run_dir / "train.log"
        command = [
            sys.executable,
            "-m",
            "transition_trigger.train",
            "--config",
            str(config_path),
            "--device",
            device,
        ]
        run_command(command, train_log)
    else:
        write_status(status_path, entry, "skip_train_existing_checkpoint", {"checkpoint": str(checkpoint_path)})

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"training did not produce checkpoint: {checkpoint_path}")

    evaluate_checkpoints(config, run_dir, device)
    write_status(status_path, entry, "done", {"run_dir": str(run_dir)})


def evaluate_checkpoints(config: dict[str, Any], run_dir: Path, device: str) -> None:
    eval_config_path = write_split_eval_config(config, run_dir, "eval")
    test_config_path = write_split_eval_config(config, run_dir, "test")
    checkpoints = {
        "best": run_dir / "best.pt",
        "best_auprc": run_dir / "best_auprc.pt",
        "best_event_f1": run_dir / "best_event_f1.pt",
        "best_memory_write": run_dir / "best_memory_write.pt",
    }
    for checkpoint_name, checkpoint_path in checkpoints.items():
        if not checkpoint_path.exists():
            continue
        for split_name, eval_config_path_for_split in (
            ("eval", eval_config_path),
            ("test", test_config_path),
        ):
            output_path = run_dir / f"{split_name}_metrics_{checkpoint_name}.json"
            if output_path.exists():
                continue
            run_command(
                [
                    sys.executable,
                    "-m",
                    "transition_trigger.evaluate",
                    "--config",
                    str(eval_config_path_for_split),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output",
                    str(output_path),
                    "--device",
                    device,
                ],
                run_dir / f"{split_name}_{checkpoint_name}.log",
            )


def write_split_eval_config(config: dict[str, Any], run_dir: Path, split_name: str) -> Path:
    eval_config = deepcopy(config)
    eval_config.setdefault("evaluation", {})["dataset_split"] = split_name
    eval_config.setdefault("outputs", {})["run_dir"] = str(run_dir)
    path = run_dir / f"{split_name}_eval_config.yaml"
    write_resolved_config(eval_config, path)
    return path


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    with log_path.open("a") as log_file:
        log_file.write(f"\n$ {' '.join(command)}\n")
        log_file.flush()
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
        log_file.write(f"\nexit_code={result.returncode} seconds={time.time() - started_at:.2f}\n")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def write_status(path: Path, entry: dict[str, str], status: str, extra: dict[str, Any]) -> None:
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name": entry["run_name"],
        "config": entry["config"],
        "status": status,
    }
    row.update(extra)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
