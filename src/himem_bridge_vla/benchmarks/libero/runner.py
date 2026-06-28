from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import random
from typing import Any

import imageio
import numpy as np
import websockets

from himem_bridge_vla.benchmarks.libero.action_protocol import parse_action_response
from himem_bridge_vla.benchmarks.libero.action_protocol import to_libero_action
from himem_bridge_vla.benchmarks.libero.config import LiberoClientConfig
from himem_bridge_vla.benchmarks.libero.config import configure_mujoco_environment
from himem_bridge_vla.benchmarks.libero.eval_summary import EpisodeResult
from himem_bridge_vla.benchmarks.libero.eval_summary import write_result_summary
from himem_bridge_vla.benchmarks.libero.history import LiberoObservationHistory
from himem_bridge_vla.benchmarks.libero.request_builder import build_request_from_observation
from himem_bridge_vla.benchmarks.libero.spec import LIBERO_SPEC


LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]
LOG = logging.getLogger(__name__)


def obs_to_json_dict(
    obs: Any,
    prompt: str,
    resize_size: int = 448,
    history: LiberoObservationHistory | None = None,
    current_step: int | None = None,
    reset_memory: bool = False,
    executed_actions: list[list[float]] | None = None,
    executed_action_mask: list[bool] | None = None,
) -> dict[str, Any]:
    _ = resize_size
    return build_request_from_observation(
        obs,
        prompt,
        history=history,
        current_step=current_step,
        reset_memory=reset_memory,
        executed_actions=executed_actions,
        executed_action_mask=executed_action_mask,
    )


def configure_logging(config: LiberoClientConfig) -> None:
    os.makedirs(os.path.dirname(config.log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(config.log_file, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def get_libero_env(task: Any, config: LiberoClientConfig, resolution: int = 448, seed: int | None = None):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    seed = config.seed if seed is None else seed
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def save_video(frames: list[np.ndarray], filename: str, fps: int = 20, save_dir: str = "videos_2") -> str:
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if not frames:
        LOG.warning("No frames to save. File not created: %s", filepath)
        return ""

    imageio.mimsave(filepath, frames, fps=fps)
    LOG.info("Video saved: %s (%s frames)", filepath, len(frames))
    return filepath


async def run(
    server_url: str,
    *,
    config: LiberoClientConfig,
    max_steps: int,
    num_episodes: int | None = None,
    horizon: int | None = None,
    task_suite_name: str,
) -> list[EpisodeResult]:
    from libero.libero import benchmark

    horizon = config.horizon if horizon is None else horizon
    num_episodes = config.num_episodes if num_episodes is None else num_episodes
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_start = min(config.task_offset, num_tasks_in_suite)
    task_stop = num_tasks_in_suite
    if config.task_limit > 0:
        task_stop = min(task_start + config.task_limit, num_tasks_in_suite)
    task_ids = range(task_start, task_stop)

    LOG.info("Number of tasks: %s", num_tasks_in_suite)

    total_success = 0
    total_episodes = 0
    total_decision_steps = 0
    total_success_decision_steps = 0
    suite_results: list[EpisodeResult] = []

    async with websockets.connect(server_url, ping_interval=None, ping_timeout=None) as ws:
        LOG.info("===========================Start task suite %s========================", task_suite_name)

        for task_id in task_ids:
            LOG.info("task_id=%s", task_id)

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env = None
            try:
                env, task_description = get_libero_env(task, config, resolution=448, seed=config.seed)

                LOG.info("\n========= Start task%s: %s =========", task_id + 1, task_description)

                task_success = 0
                episode_start = min(config.episode_offset, len(initial_states))
                episode_stop = min(episode_start + num_episodes, len(initial_states))
                episode_indices = range(episode_start, episode_stop)
                task_episodes = len(episode_indices)

                for ep in episode_indices:
                    LOG.info("===== Task %s | Episode %s =====", task_id, ep + 1)

                    env.reset()

                    obs = env.set_init_state(initial_states[ep])
                    for _ in range(10):
                        obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)

                    prompt = str(task_description)
                    LOG.info(prompt)
                    episode_done = False
                    episode_failed = False
                    failure_reason = ""
                    decision_steps = 0
                    control_steps = 0
                    frames: list[np.ndarray] = []
                    history = LiberoObservationHistory(max_offset=max(LIBERO_SPEC.short_memory_offsets))
                    history.record(control_steps, obs)
                    last_executed_actions: list[list[float]] = []
                    last_executed_action_mask: list[bool] = []
                    episode_gripper_values: list[float] = []

                    for step in range(max_steps):
                        decision_steps += 1

                        send_data = obs_to_json_dict(
                            obs,
                            prompt,
                            history=history,
                            current_step=control_steps,
                            reset_memory=(step == 0),
                            executed_actions=last_executed_actions or None,
                            executed_action_mask=last_executed_action_mask or None,
                        )
                        await ws.send(json.dumps(send_data))
                        LOG.debug("[Step %s] Send observation", step)

                        result = await ws.recv()
                        try:
                            actions = parse_action_response(result, horizon=horizon)
                            LOG.debug("[Step %s] received actions (gripper=%s)", step, actions[0][6])
                        except Exception as exc:
                            failure_reason = f"action_parse_error: {exc}"
                            LOG.error("Action parsing failed: %s, content: %s", exc, result)
                            break

                        current_executed_actions: list[list[float]] = []
                        current_executed_action_mask: list[bool] = []
                        for action_values in actions:
                            action = to_libero_action(action_values)
                            episode_gripper_values.append(float(action[6]))
                            LOG.debug(action[:7])
                            LOG.debug("gripper action %s", action[6])
                            try:
                                obs, reward, done, info = env.step(action)
                                control_steps += 1
                                history.record(control_steps, obs)
                                current_executed_actions.append(action)
                                current_executed_action_mask.append(True)
                            except ValueError as exc:
                                failure_reason = f"invalid_action: {exc}"
                                LOG.error("Action is not valid: %s", exc)
                                episode_failed = True
                                break

                            frame = np.hstack(
                                [
                                    np.rot90(obs["agentview_image"], 2),
                                    np.rot90(obs["robot0_eye_in_hand_image"], 2),
                                ]
                            )
                            frames.append(frame)

                            LOG.debug("[Step %s] reward=%.2f, done=%s, info=%s", step, reward, done, info)
                            if done:
                                LOG.info("Task completed")
                                episode_done = True
                                task_success += 1
                                total_success += 1
                                total_success_decision_steps += decision_steps
                                break
                        last_executed_actions = current_executed_actions
                        last_executed_action_mask = current_executed_action_mask
                        if episode_done or episode_failed:
                            break

                    if not episode_done and not failure_reason:
                        failure_reason = "max_steps_exhausted"
                    if episode_gripper_values:
                        positive = sum(1 for value in episode_gripper_values if value >= 0.0)
                        negative = len(episode_gripper_values) - positive
                        LOG.info(
                            "Episode gripper sign distribution: close_ratio(raw>=0,+1)=%.4f negative_ratio=%.4f count=%s",
                            positive / len(episode_gripper_values),
                            negative / len(episode_gripper_values),
                            len(episode_gripper_values),
                        )

                    video_path = save_video(
                        frames,
                        f"task{task_id + 1}_episode{ep + 1}.mp4",
                        fps=30,
                        save_dir=os.path.join(config.video_dir, task_suite_name),
                    )

                    total_decision_steps += decision_steps
                    suite_results.append(
                        EpisodeResult(
                            task_suite=task_suite_name,
                            task_id=task_id,
                            episode_id=ep,
                            task_description=prompt,
                            success=episode_done,
                            decision_steps=decision_steps,
                            control_steps=control_steps,
                            failure_reason="" if episode_done else failure_reason,
                            video_path=video_path,
                        )
                    )

                    if episode_done:
                        LOG.info("Task %s | Episode %s: Success", task_id, ep + 1)
                    else:
                        LOG.info("Task %s | Episode %s: Fail (%s)", task_id, ep + 1, failure_reason)

                LOG.info("========= Task %s Summary: %s/%s Successful =========", task_id + 1, task_success, task_episodes)
                total_episodes += task_episodes
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        LOG.warning("Failed to close LIBERO env for task %s: %s", task_id, exc)

        LOG.info("\n========= Overall Task Summary =========")
        LOG.info("Total Successful Episodes: %s/%s", total_success, total_episodes)
        if total_episodes > 0:
            LOG.info("Success Rate: %.4f", total_success / total_episodes)
            LOG.info("Average Decision Steps: %.2f", total_decision_steps / total_episodes)
        if total_success > 0:
            LOG.info("Average Successful Decision Steps: %.2f", total_success_decision_steps / total_success)

    return suite_results


def main() -> int:
    config = LiberoClientConfig.from_env()
    configure_mujoco_environment(config)
    configure_logging(config)
    np.random.seed(config.seed)
    random.seed(config.seed)

    all_results: list[EpisodeResult] = []
    for name, max_steps in zip(config.task_suites, config.max_steps):
        suite_results = asyncio.run(
            run(
                config.server_url,
                config=config,
                max_steps=max_steps,
                num_episodes=config.num_episodes,
                horizon=config.horizon,
                task_suite_name=name,
            )
        )
        all_results.extend(suite_results)
        result_path = write_result_summary(config.result_file, config=config, results=all_results)
        LOG.info("LIBERO result summary saved: %s", result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
