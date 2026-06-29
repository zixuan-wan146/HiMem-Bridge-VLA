#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

DEFAULT_STAGE2_CONFIG = "configs/training/stage2/libero_10_full_e2e_from_stage1_best.yaml"
DEFAULT_STAGE2_SAVE_DIR = "local_data/runs/stage2/libero_10_full_e2e_from_stage1_best"
DEFAULT_RUN_ROOT = "local_data/runs/stage2/orchestration"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for an idle GPU, run LIBERO-10 Stage2 training, and evaluate 10 tasks x 10 episodes."
    )
    parser.add_argument("--config", default=DEFAULT_STAGE2_CONFIG, help="Project-relative Stage2 training YAML.")
    parser.add_argument("--python", default=os.environ.get("HIMEM_PYTHON", sys.executable), help="Python executable.")
    parser.add_argument("--stage2-save-dir", default=DEFAULT_STAGE2_SAVE_DIR, help="Project-relative training save_dir.")
    parser.add_argument("--eval-checkpoint-tag", default="step_best", help="Checkpoint tag under --stage2-save-dir to evaluate.")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="Project-relative orchestration root.")
    parser.add_argument("--run-name", default=None, help="Optional orchestration run directory name.")
    parser.add_argument("--gpu-threshold-mb", type=int, default=100)
    parser.add_argument("--stable-seconds", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9000)
    parser.add_argument("--server-ready-timeout", type=int, default=1800)
    parser.add_argument("--eval-task-suite", default="libero_10")
    parser.add_argument("--eval-task-limit", type=int, default=10)
    parser.add_argument("--eval-task-offset", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-episode-offset", type=int, default=0)
    parser.add_argument("--eval-max-steps", type=int, default=95)
    parser.add_argument("--eval-horizon", type=int, default=32)
    parser.add_argument("--eval-ckpt-name", default="libero_10_stage2_full_e2e_joint_step_best")
    parser.add_argument("--mujoco-gl", default="osmesa", choices=("osmesa", "egl", "glfw"))
    parser.add_argument("--shutdown", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true", help="Validate commands and write reports without running train/eval/shutdown.")
    return parser.parse_args(argv)


class OrchestrationState:
    def __init__(self, *, repo_root: Path, run_dir: Path, args: argparse.Namespace) -> None:
        self.repo_root = repo_root
        self.run_dir = run_dir
        self.args = args
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.status_path = self.run_dir / "STATUS.md"
        self.gpu_history_path = self.run_dir / "gpu_history.csv"
        with self.gpu_history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("timestamp", "max_memory_mb", "raw"))
            writer.writeheader()
        (self.run_dir / "args.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record(self, level: str, phase: str, message: str, data: Mapping[str, Any] | None = None) -> None:
        event = {
            "timestamp": now_iso(),
            "level": level,
            "phase": phase,
            "message": message,
            "data": dict(data or {}),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"[{event['timestamp']}] [{level.upper()}] [{phase}] {message}", flush=True)

    def write_status(self, status: str, message: str) -> None:
        content = [
            "# Stage2 LIBERO-10 Orchestration Status",
            "",
            f"- status: `{status}`",
            f"- updated_at: `{now_iso()}`",
            f"- message: {message}",
            f"- run_dir: `{self.run_dir.as_posix()}`",
            f"- config: `{self.args.config}`",
            f"- training_save_dir: `{self.args.stage2_save_dir}`",
            "",
        ]
        self.status_path.write_text("\n".join(content), encoding="utf-8")

    def append_gpu_history(self, max_memory_mb: int, raw: str) -> None:
        with self.gpu_history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("timestamp", "max_memory_mb", "raw"))
            writer.writerow({"timestamp": now_iso(), "max_memory_mb": max_memory_mb, "raw": raw})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = find_repo_root(Path(__file__))
    os.chdir(repo_root)
    validate_args(args)
    state = OrchestrationState(repo_root=repo_root, run_dir=make_run_dir(Path(args.run_root), args.run_name), args=args)
    state.write_status("initialized", "orchestration initialized")
    state.record("info", "init", "orchestration initialized", {"run_dir": state.run_dir.as_posix(), "dry_run": args.dry_run})

    try:
        if args.dry_run:
            write_dry_run_plan(state)
            state.write_status("dry_run_complete", "dry-run completed; no training, eval, or shutdown was started")
            return 0

        wait_for_gpu_idle(state)
        success = False
        last_error = ""
        for attempt in range(1, args.max_attempts + 1):
            state.record("info", "attempt", f"starting attempt {attempt}/{args.max_attempts}")
            try:
                run_training(state, attempt=attempt)
                result_path, eval_checkpoint_dir = run_evaluation(state, attempt=attempt)
                summary = load_eval_summary(result_path)
                validate_eval_summary(summary, expected_episodes=args.eval_task_limit * args.eval_episodes)
                write_success_report(
                    state,
                    result_path=result_path,
                    eval_checkpoint_dir=eval_checkpoint_dir,
                    summary=summary,
                    attempt=attempt,
                )
                success = True
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                state.record("error", "attempt", f"attempt {attempt} failed", {"error": last_error})
                write_failure_note(state, attempt=attempt, error=last_error)
                if attempt < args.max_attempts:
                    state.record("info", "attempt", "will retry after failure", {"next_attempt": attempt + 1})

        if not success:
            write_final_failure_report(state, error=last_error)
            state.write_status("failed", f"pipeline failed after {args.max_attempts} attempts: {last_error}")
            if args.shutdown:
                shutdown(state, reason="failed after retry budget")
            return 1

        state.write_status("complete", "training and LIBERO-10 evaluation completed")
        if args.shutdown:
            shutdown(state, reason="successful completion")
        return 0
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        state.record("error", "fatal", "fatal orchestration error", {"error": error})
        write_final_failure_report(state, error=error)
        state.write_status("fatal", error)
        if args.shutdown and not args.dry_run:
            shutdown(state, reason="fatal orchestration error")
        return 1


def wait_for_gpu_idle(state: OrchestrationState) -> None:
    args = state.args
    idle_started_at: float | None = None
    state.write_status("waiting_gpu", f"waiting for GPU memory < {args.gpu_threshold_mb} MB for {args.stable_seconds}s")
    state.record("info", "gpu_wait", "waiting for GPU idle window", {"threshold_mb": args.gpu_threshold_mb, "stable_seconds": args.stable_seconds})
    while True:
        max_memory_mb, raw = query_gpu_memory_mb()
        state.append_gpu_history(max_memory_mb, raw)
        now = time.monotonic()
        if max_memory_mb < args.gpu_threshold_mb:
            if idle_started_at is None:
                idle_started_at = now
                state.record("info", "gpu_wait", "GPU entered idle threshold", {"max_memory_mb": max_memory_mb})
            idle_seconds = now - idle_started_at
            if idle_seconds >= args.stable_seconds:
                state.record("info", "gpu_wait", "GPU idle window satisfied", {"idle_seconds": round(idle_seconds, 1)})
                return
        else:
            if idle_started_at is not None:
                state.record("warning", "gpu_wait", "GPU idle window reset", {"max_memory_mb": max_memory_mb})
            idle_started_at = None
        time.sleep(args.poll_seconds)


def run_training(state: OrchestrationState, *, attempt: int) -> None:
    args = state.args
    log_path = state.logs_dir / f"train_attempt_{attempt}.log"
    command = [args.python, "scripts/train/stage2/libero.py", "--config", args.config]
    state.write_status("training", f"training attempt {attempt} running")
    state.record("info", "training", "starting Stage2 training", {"attempt": attempt, "command": command, "log": log_path.as_posix()})
    result = run_logged_command(command, log_path=log_path, env=base_env())
    if result != 0:
        raise RuntimeError(f"Stage2 training failed with exit code {result}; see {log_path.as_posix()}")
    checkpoint_dir = checkpoint_dir_for_eval(args)
    checkpoint_model = checkpoint_dir / "model.pt"
    if not checkpoint_model.exists():
        raise FileNotFoundError(
            "training finished but required best checkpoint was not found: "
            f"{checkpoint_model.as_posix()}"
        )
    state.record("info", "training", "Stage2 training completed", {"attempt": attempt})


def run_evaluation(state: OrchestrationState, *, attempt: int) -> tuple[Path, Path]:
    args = state.args
    checkpoint_dir = resolve_eval_checkpoint_dir(args)

    server_log = state.logs_dir / f"server_attempt_{attempt}.log"
    eval_log = state.logs_dir / f"eval_attempt_{attempt}.log"
    eval_run_dir = state.run_dir / f"eval_attempt_{attempt}"
    eval_ckpt_name = f"{args.eval_ckpt_name}_attempt{attempt}"
    result_path = eval_run_dir / "results" / f"{eval_ckpt_name}_results.json"

    server_env = base_env()
    server_env.update(
        {
            "HIMEM_PYTHON": args.python,
            "HIMEM_CKPT_DIR": checkpoint_dir.as_posix(),
            "HIMEM_HOST": args.server_host,
            "HIMEM_PORT": str(args.server_port),
            "HIMEM_DEVICE": "cuda:0",
            "HIMEM_INFERENCE_STEPS": "15",
            "HIMEM_SKIP_PREFLIGHT": "1",
            "HIMEM_ALLOW_UNSAFE_CHECKPOINT_LOAD": "1",
        }
    )
    state.write_status("serving", f"starting server for eval attempt {attempt}")
    state.record("info", "server", "starting policy server", {"checkpoint_dir": checkpoint_dir.as_posix(), "log": server_log.as_posix()})
    with server_log.open("ab") as handle:
        server_proc = subprocess.Popen(["bash", "scripts/serve/start_himem_server.sh"], cwd=state.repo_root, env=server_env, stdout=handle, stderr=subprocess.STDOUT)
    try:
        wait_for_server(state, server_proc=server_proc, log_path=server_log)
        eval_env = base_env()
        eval_env.update(
            {
                "LIBERO_PYTHON": args.python,
                "HIMEM_SERVER_URI": f"ws://{args.server_host}:{args.server_port}",
                "HIMEM_MUJOCO_GL": args.mujoco_gl,
                "HIMEM_LIBERO_RUN_DIR": eval_run_dir.as_posix(),
                "HIMEM_LIBERO_EPISODES": str(args.eval_episodes),
                "HIMEM_LIBERO_TASK_SUITES": args.eval_task_suite,
                "HIMEM_LIBERO_TASK_LIMIT": str(args.eval_task_limit),
                "HIMEM_LIBERO_TASK_OFFSET": str(args.eval_task_offset),
                "HIMEM_LIBERO_EPISODE_OFFSET": str(args.eval_episode_offset),
                "HIMEM_LIBERO_MAX_STEPS": str(args.eval_max_steps),
                "HIMEM_LIBERO_HORIZON": str(args.eval_horizon),
                "HIMEM_LIBERO_CKPT_NAME": eval_ckpt_name,
            }
        )
        state.write_status("evaluating", f"LIBERO-10 eval attempt {attempt} running")
        state.record("info", "eval", "starting LIBERO evaluation", {"attempt": attempt, "task_suite": args.eval_task_suite, "task_limit": args.eval_task_limit, "episodes": args.eval_episodes, "result_path": result_path.as_posix(), "log": eval_log.as_posix()})
        result = run_logged_command(["bash", "scripts/eval/run_libero_eval.sh"], log_path=eval_log, env=eval_env)
        if result != 0:
            raise RuntimeError(f"LIBERO eval failed with exit code {result}; see {eval_log.as_posix()}")
        if not result_path.exists():
            raise FileNotFoundError(f"LIBERO eval did not produce result file: {result_path.as_posix()}")
        state.record("info", "eval", "LIBERO evaluation completed", {"result_path": result_path.as_posix()})
        return result_path, checkpoint_dir
    finally:
        terminate_process(state, server_proc, phase="server", name="policy server")


def wait_for_server(state: OrchestrationState, *, server_proc: subprocess.Popen[Any], log_path: Path) -> None:
    deadline = time.monotonic() + state.args.server_ready_timeout
    while time.monotonic() < deadline:
        exit_code = server_proc.poll()
        if exit_code is not None:
            raise RuntimeError(f"policy server exited early with code {exit_code}; see {log_path.as_posix()}")
        if can_connect(state.args.server_host, state.args.server_port):
            state.record("info", "server", "policy server is accepting TCP connections")
            return
        time.sleep(5)
    raise TimeoutError(f"policy server did not become ready within {state.args.server_ready_timeout}s")


def run_logged_command(command: list[str], *, log_path: Path, env: Mapping[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write(("\n===== COMMAND: " + json.dumps(command) + " =====\n").encode("utf-8"))
        handle.flush()
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=dict(env))
        return proc.wait()


def load_eval_summary(result_path: Path) -> dict[str, Any]:
    with result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"result file has no summary object: {result_path.as_posix()}")
    return summary


def validate_eval_summary(summary: Mapping[str, Any], *, expected_episodes: int) -> None:
    total_episodes = int(summary.get("total_episodes", 0))
    if total_episodes != expected_episodes:
        raise ValueError(f"expected {expected_episodes} eval episodes, got {total_episodes}")
    success_rate = float(summary.get("success_rate", 0.0))
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError(f"success_rate is out of range: {success_rate}")


def write_success_report(
    state: OrchestrationState,
    *,
    result_path: Path,
    eval_checkpoint_dir: Path,
    summary: Mapping[str, Any],
    attempt: int,
) -> None:
    report_path = state.run_dir / "RESULT_ANALYSIS.md"
    suites = summary.get("suites", {}) if isinstance(summary.get("suites"), dict) else {}
    lines = [
        "# Stage2 LIBERO-10 Training And Evaluation Result",
        "",
        f"- completed_at: `{now_iso()}`",
        f"- successful_attempt: `{attempt}`",
        f"- training_config: `{state.args.config}`",
        f"- training_save_dir: `{state.args.stage2_save_dir}`",
        f"- evaluated_checkpoint: `{eval_checkpoint_dir.as_posix()}`",
        f"- result_file: `{result_path.as_posix()}`",
        "",
        "## Overall",
        "",
        f"- total_episodes: `{summary.get('total_episodes')}`",
        f"- successful_episodes: `{summary.get('successful_episodes')}`",
        f"- failed_episodes: `{summary.get('failed_episodes')}`",
        f"- success_rate: `{float(summary.get('success_rate', 0.0)):.4f}`",
        f"- average_decision_steps: `{float(summary.get('average_decision_steps', 0.0)):.2f}`",
        f"- average_control_steps: `{float(summary.get('average_control_steps', 0.0)):.2f}`",
        "",
        "## Suites",
        "",
    ]
    for name, suite_summary in sorted(suites.items()):
        lines.append(
            f"- `{name}`: {suite_summary.get('successful_episodes')}/{suite_summary.get('total_episodes')} "
            f"success_rate={float(suite_summary.get('success_rate', 0.0)):.4f}"
        )
    lines.extend(
        [
            "",
            "## Logs",
            "",
            f"- events: `{state.events_path.as_posix()}`",
            f"- gpu_history: `{state.gpu_history_path.as_posix()}`",
            f"- logs_dir: `{state.logs_dir.as_posix()}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    state.record("info", "report", "success report written", {"report": report_path.as_posix()})


def write_failure_note(state: OrchestrationState, *, attempt: int, error: str) -> None:
    path = state.run_dir / f"FAILURE_ATTEMPT_{attempt}.md"
    lines = [
        f"# Failure Attempt {attempt}",
        "",
        f"- timestamp: `{now_iso()}`",
        f"- error: `{error}`",
        f"- logs_dir: `{state.logs_dir.as_posix()}`",
        "",
        "Next step: inspect the corresponding training, server, and eval logs to classify environment, resource, checkpoint, or code-path failure.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_failure_report(state: OrchestrationState, *, error: str) -> None:
    path = state.run_dir / "FAILURE_REPORT.md"
    lines = [
        "# Stage2 LIBERO-10 Orchestration Failure",
        "",
        f"- failed_at: `{now_iso()}`",
        f"- error: `{error}`",
        f"- max_attempts: `{state.args.max_attempts}`",
        f"- config: `{state.args.config}`",
        f"- training_save_dir: `{state.args.stage2_save_dir}`",
        f"- events: `{state.events_path.as_posix()}`",
        f"- gpu_history: `{state.gpu_history_path.as_posix()}`",
        f"- logs_dir: `{state.logs_dir.as_posix()}`",
        "",
        "## Required Follow-Up",
        "",
        "Inspect the failed attempt logs. If this is a code defect, patch and restart the orchestration script.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    state.record("error", "report", "failure report written", {"report": path.as_posix()})


def write_dry_run_plan(state: OrchestrationState) -> None:
    args = state.args
    plan = {
        "wait_gpu": {"threshold_mb": args.gpu_threshold_mb, "stable_seconds": args.stable_seconds, "poll_seconds": args.poll_seconds},
        "training_command": [args.python, "scripts/train/stage2/libero.py", "--config", args.config],
        "server_checkpoint": checkpoint_dir_for_eval(args).as_posix(),
        "eval": {
            "task_suite": args.eval_task_suite,
            "task_limit": args.eval_task_limit,
            "episodes_per_task": args.eval_episodes,
            "expected_total_episodes": args.eval_task_limit * args.eval_episodes,
            "horizon": args.eval_horizon,
            "max_steps": args.eval_max_steps,
        },
        "shutdown": bool(args.shutdown),
    }
    path = state.run_dir / "DRY_RUN_PLAN.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.record("info", "dry_run", "dry-run plan written", {"plan": path.as_posix()})


def shutdown(state: OrchestrationState, *, reason: str) -> None:
    state.record("warning", "shutdown", "shutting down host", {"reason": reason})
    subprocess.Popen(["bash", "/usr/bin/shutdown"], cwd=state.repo_root)


def terminate_process(state: OrchestrationState, proc: subprocess.Popen[Any], *, phase: str, name: str) -> None:
    if proc.poll() is not None:
        state.record("info", phase, f"{name} already exited", {"exit_code": proc.returncode})
        return
    state.record("info", phase, f"terminating {name}")
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        state.record("warning", phase, f"killing {name} after graceful timeout")
        proc.kill()
        proc.wait(timeout=30)


def query_gpu_memory_mb() -> tuple[int, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    values = [int(line.strip().split()[0]) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory rows")
    raw = ";".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    return max(values), raw


def can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def base_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = ".:src" + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    return env


def checkpoint_dir_for_eval(args: argparse.Namespace) -> Path:
    return Path(args.stage2_save_dir) / args.eval_checkpoint_tag


def resolve_eval_checkpoint_dir(args: argparse.Namespace) -> Path:
    checkpoint_dir = checkpoint_dir_for_eval(args)
    if (checkpoint_dir / "model.pt").exists():
        return checkpoint_dir
    raise FileNotFoundError(
        "required best checkpoint for evaluation is missing: "
        f"{(checkpoint_dir / 'model.pt').as_posix()}"
    )


def make_run_dir(run_root: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = "libero_10_stage2_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    return run_root / run_name


def find_repo_root(start: Path) -> Path:
    resolved = start.resolve()
    candidates = [resolved if resolved.is_dir() else resolved.parent]
    candidates.extend(candidates[0].parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "himem_bridge_vla").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start}")


def validate_args(args: argparse.Namespace) -> None:
    for value, label in ((args.config, "--config"), (args.stage2_save_dir, "--stage2-save-dir"), (args.run_root, "--run-root")):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{label} must be project-relative, got {value!r}")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.stable_seconds <= 0:
        raise ValueError("--stable-seconds must be positive")
    if args.eval_task_suite != "libero_10":
        raise ValueError("This orchestration is intended for the trained LIBERO-10 suite; keep --eval-task-suite=libero_10")
    if args.eval_task_limit != 10 or args.eval_episodes != 10:
        raise ValueError("Expected LIBERO-10 eval shape is 10 tasks x 10 episodes")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
