from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export standalone CoarsePlanner checkpoint for main-model loading.")
    parser.add_argument("--checkpoint", required=True, help="Standalone train checkpoint, usually coarse_planner run best.pt.")
    parser.add_argument("--output", required=True, help="Output .pt file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=False)
    exported = {
        "coarse_planner_state_dict": checkpoint["model"],
        "coarse_planner_config": checkpoint["planner_config"],
        "source_config": checkpoint.get("config"),
        "val_metrics": checkpoint.get("val_metrics"),
    }
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
