from __future__ import annotations

import asyncio
import websockets
import numpy as np
import json
import pathlib
import os
import logging
import math
import imageio
import random

from libero_action_protocol import parse_action_response, to_libero_action
from libero_client_config import LiberoClientConfig, configure_mujoco_environment
from libero_eval_summary import EpisodeResult, write_result_summary

args = LiberoClientConfig.from_env()
configure_mujoco_environment(args)

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]

########################################

os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(args.log_file, mode='a'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ========= Photos to list[list[list[int]]] =========
def encode_image_array(img_array: np.ndarray):
    return img_array.astype(np.uint8).tolist()

# ========= Quaternion to Axis-Angle =========
def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# ========= Observation to JSON-compatible dict =========
def obs_to_json_dict(obs, prompt, resize_size=448):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

    data = {
        "image": [
            encode_image_array(img),
            encode_image_array(wrist_img),
            encode_image_array(dummy_proc)
        ],
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }
    return data

# ========= Get the environment of LIBERO =========
def get_libero_env(task, resolution=448, seed=None):
    seed = args.SEED if seed is None else seed
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

# ========= Save the video log =========
def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        log.info(f"Video saved: {filepath} ({len(frames)} frames)")
        return filepath
    else:
        log.warning(f"No frames to save. File not created: {filepath}")
        return ""

# ========= Main Function =========
async def run(SERVER_URL: str, max_steps: int = None, num_episodes: int = None, horizon = None, task_suite_name = None):
    if horizon is None:
        horizon = args.horizon
    if max_steps is None:
        raise ValueError("max_steps is required")
    if num_episodes is None:
        num_episodes = args.num_episodes
    if task_suite_name is None:
        raise ValueError("task_suite_name is required")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = range(num_tasks_in_suite)
    if args.task_limit > 0:
        task_ids = range(min(args.task_limit, num_tasks_in_suite))

    log.info(f"Number of tasks: {num_tasks_in_suite}")

    total_success = 0
    total_episodes = 0
    total_decision_steps = 0
    total_success_decision_steps = 0
    suite_results = []

    async with websockets.connect(SERVER_URL) as ws:
        log.info(f"===========================Start task suite {task_suite_name}========================")

        for task_id in task_ids:

            log.info(f"task_id={task_id}")

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env = None
            try:
                env, task_description = get_libero_env(task, resolution=448, seed=args.SEED)

                log.info(f"\n========= Start task{task_id+1}: {task_description} =========")

                task_success = 0
                task_episodes = min(num_episodes, len(initial_states))

                for ep in range(task_episodes):
                    log.info(f"===== Task {task_id} | Episode {ep+1} =====")

                    env.reset()

                    obs = env.set_init_state(initial_states[ep])
                    t = 0
                    while t < 10:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1

                    prompt = str(task_description)
                    log.info(prompt)
                    episode_done = False
                    episode_failed = False
                    failure_reason = ""
                    decision_steps = 0
                    control_steps = 0
                    frames = []

                    for step in range(max_steps):
                        decision_steps += 1

                        send_data = obs_to_json_dict(obs, prompt)
                        await ws.send(json.dumps(send_data))
                        log.debug(f"[Step {step}] Send observation")

                        result = await ws.recv()
                        try:
                            actions = parse_action_response(result, horizon=horizon)
                            log.debug(f"[Step {step}] received actions (gripper={actions[0][6]})")
                        except Exception as e:
                            failure_reason = f"action_parse_error: {e}"
                            log.error(f"Action parsing failed: {e}, content: {result}")
                            break

                        for action_values in actions:
                            action = to_libero_action(action_values)
                            log.debug(action[:7])
                            log.debug(f"gripper action {action[6]}")
                            try:
                                obs, reward, done, info = env.step(action)
                                control_steps += 1
                            except ValueError as ve:
                                failure_reason = f"invalid_action: {ve}"
                                log.error(f"Action is not valid: {ve}")
                                episode_failed = True
                                break

                            frame = np.hstack([
                                np.rot90(obs["agentview_image"], 2),
                                np.rot90(obs["robot0_eye_in_hand_image"], 2)
                            ])
                            frames.append(frame)

                            log.debug(f"[Step {step}] reward={reward:.2f}, done={done}")
                            if done:
                                log.info("Task completed")
                                episode_done = True
                                task_success += 1
                                total_success += 1
                                total_success_decision_steps += decision_steps
                                break
                        if episode_done or episode_failed:
                            break

                    if not episode_done and not failure_reason:
                        failure_reason = "max_steps_exhausted"

                    video_path = save_video(
                        frames,
                        f"task{task_id+1}_episode{ep+1}.mp4",
                        fps=30,
                        save_dir=os.path.join(args.video_dir, task_suite_name),
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
                        log.info(f"Task {task_id} | Episode {ep+1}: Success")
                    else:
                        log.info(f"Task {task_id} | Episode {ep+1}: Fail ({failure_reason})")

                log.info(f"========= Task {task_id + 1} Summary: {task_success}/{task_episodes} Successful =========")
                total_episodes += task_episodes
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as e:
                        log.warning(f"Failed to close LIBERO env for task {task_id}: {e}")

        # ======= Overall Summary =======
        log.info("\n========= Overall Task Summary =========")
        log.info(f"Total Successful Episodes: {total_success}/{total_episodes}")
        if total_episodes > 0:
            log.info(f"Success Rate: {total_success / total_episodes:.4f}")
            log.info(f"Average Decision Steps: {total_decision_steps / total_episodes:.2f}")
        if total_success > 0:
            log.info(f"Average Successful Decision Steps: {total_success_decision_steps / total_success:.2f}")

    return suite_results




if __name__ == "__main__":
    np.random.seed(args.SEED)
    random.seed(args.SEED)

    all_results = []
    for name, max_steps in zip(args.task_suites, args.max_steps):
        suite_results = asyncio.run(
            run(
                SERVER_URL=args.SERVER_URL,
                max_steps=max_steps,
                num_episodes=args.num_episodes,
                horizon=args.horizon,
                task_suite_name=name,
            )
        )
        all_results.extend(suite_results)
        result_path = write_result_summary(args.result_file, config=args, results=all_results)
        log.info(f"LIBERO result summary saved: {result_path}")
