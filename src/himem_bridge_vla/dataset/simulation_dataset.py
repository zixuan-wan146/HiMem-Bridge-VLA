# Multithreaded dataset loader for simulation training data.
import os
import torch
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm  
from typing import List, Union, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import multiprocessing as mp
import logging
import pickle

from himem_bridge_vla.dataset.cache_utils import dataset_cache_namespace
from himem_bridge_vla.dataset.cache_utils import default_dataset_cache_dir
from himem_bridge_vla.dataset.action_segments import build_action_segment_target
from himem_bridge_vla.utils.normalization import minmax_normalize


class DatasetInputAdapter:
    """Configurable reader for LeRobot-style parquet datasets."""

    def __init__(self, dataset_config: Dict[str, Any], dataset_path: Path) -> None:
        self.dataset_config = dataset_config
        self.dataset_path = Path(dataset_path)
        self.view_map = dataset_config.get("view_map", {})
        self.state_keys = dataset_config.get("state_keys", ("observation.state", "state", "robot_obs"))
        self.action_keys = dataset_config.get("action_keys", ("action", "actions", "rel_actions"))
        self.task_index_keys = dataset_config.get("task_index_keys", ("task_index",))
        self.frame_keys = dataset_config.get("frame_keys", ("frame_index", "frame_idx"))
        self.episode_keys = dataset_config.get("episode_keys", ("episode_index", "episode_id"))

    def resolve_video_paths(self, base_video_path: Path, parquet_path: Path) -> Dict[str, str]:
        resolved = {}
        for view_key, aliases in self.view_map.items():
            for alias in _as_tuple(aliases):
                candidates = [
                    base_video_path / alias / f"{parquet_path.stem}.mp4",
                    base_video_path / alias.replace(".", "/") / f"{parquet_path.stem}.mp4",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        resolved[view_key] = str(candidate)
                        break
                if view_key in resolved:
                    break
        return resolved

    def sample_metadata(self, row: pd.Series, parquet_path: Path, row_index: int) -> Dict[str, Any]:
        metadata = {
            "row_index": int(row_index),
            "source_parquet": str(parquet_path),
        }
        task_index = _first_row_value(row, self.task_index_keys)
        frame_index = _first_row_value(row, self.frame_keys)
        episode_id = _first_row_value(row, self.episode_keys)
        if task_index is not None:
            metadata["task_index"] = int(task_index)
        if frame_index is not None:
            metadata["frame_index"] = int(frame_index)
        if episode_id is not None:
            metadata["episode_id"] = str(episode_id)
        return metadata

    def prompt(self, row: pd.Series, task_mapping: Dict[int, str], metadata: Dict[str, Any]) -> str:
        task_index = metadata.get("task_index")
        if task_index is not None and int(task_index) in task_mapping:
            return task_mapping[int(task_index)]
        for key in ("task", "language", "instruction", "prompt"):
            if key in row and row[key] is not None:
                return str(row[key])
        return ""

    def state(self, row: pd.Series) -> list[float]:
        return _row_array(row, self.state_keys).astype(float).tolist()

    def action(self, row: pd.Series) -> list[float]:
        return _row_array(row, self.action_keys).astype(float).tolist()

    def timestamp(self, row: pd.Series, row_index: int) -> float:
        if "timestamp" in row and row["timestamp"] is not None:
            return float(np.asarray(row["timestamp"]).reshape(-1)[0])
        return float(row_index)


def build_dataset_input_adapter(dataset_config: Dict[str, Any], dataset_path: Path) -> DatasetInputAdapter:
    adapter = dataset_config.get("adapter", "generic")
    if adapter not in {"generic", "lerobot"}:
        raise ValueError(f"unsupported dataset adapter: {adapter!r}")
    return DatasetInputAdapter(dataset_config, dataset_path)


def _as_tuple(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _first_row_value(row: pd.Series, keys) -> Any:
    for key in _as_tuple(keys):
        if key in row and row[key] is not None:
            return row[key]
    return None


def _row_array(row: pd.Series, keys) -> np.ndarray:
    for key in _as_tuple(keys):
        if key in row and row[key] is not None:
            return np.asarray(row[key], dtype=np.float32).reshape(-1)
    raise KeyError(f"none of the configured row keys are present: {tuple(_as_tuple(keys))}")

def compute_normalization_stats_from_minmax(jsonl_path, dataset_config=None):
    state_mins, state_maxs = [], []
    action_mins, action_maxs = [], []

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Extracting min/max"):
            obj = json.loads(line)
            stats = obj.get("stats", {})
            try:
                normalized_stats = normalize_stats_payload(stats, dataset_config)
                state_mins.append(normalized_stats["observation.state"]["min"])
                state_maxs.append(normalized_stats["observation.state"]["max"])
                action_mins.append(normalized_stats["action"]["min"])
                action_maxs.append(normalized_stats["action"]["max"])
            except Exception as e:
                logging.warning(f"Skipping abnormal stats line: {e}")


    state_min_global = np.min(np.array(state_mins), axis=0).tolist()
    state_max_global = np.max(np.array(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.array(action_mins), axis=0).tolist()
    action_max_global = np.max(np.array(action_maxs), axis=0).tolist()

    return {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global}
    }

def merge_normalization_stats(stats_list: List[Dict[str, Dict[str, List[float]]]]) -> Dict:
    state_mins = [np.array(d["observation.state"]["min"]) for d in stats_list]
    state_maxs = [np.array(d["observation.state"]["max"]) for d in stats_list]
    action_mins = [np.array(d["action"]["min"]) for d in stats_list]
    action_maxs = [np.array(d["action"]["max"]) for d in stats_list]
    state_min_global = np.min(np.stack(state_mins), axis=0).tolist()
    state_max_global = np.max(np.stack(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.stack(action_mins), axis=0).tolist()
    action_max_global = np.max(np.stack(action_maxs), axis=0).tolist()

    return {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global}
    }


def normalize_stats_payload(stats: Dict, dataset_config=None) -> Dict:
    dataset_config = dataset_config or {}
    state_keys = dataset_config.get("state_stat_keys", ("observation.state", "state"))
    action_keys = dataset_config.get("action_stat_keys", ("action", "actions"))
    state_stats = _first_stats_entry(stats, state_keys)
    action_stats = _first_stats_entry(stats, action_keys)
    return {
        "observation.state": {"min": state_stats["min"], "max": state_stats["max"]},
        "action": {"min": action_stats["min"], "max": action_stats["max"]},
    }


def _first_stats_entry(stats: Dict, keys) -> Dict:
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        value = stats.get(key)
        if isinstance(value, dict) and "min" in value and "max" in value:
            return value
    raise KeyError(f"none of the configured stat keys are present: {tuple(keys)}")


def _process_parquet_file_worker(args):
    (
        parquet_path,
        arm_name,
        dataset_name,
        dataset_config,
        dataset_path,
        task_mapping,
        action_horizon,
        max_samples_per_file,
        cache_dir,
        action_segment_config,
    ) = args
    
    try:
        df = pd.read_parquet(parquet_path)
        adapter = build_dataset_input_adapter(dataset_config, dataset_path)
        segments_enabled = bool((action_segment_config or {}).get("enabled", False))
        planning_horizon = int((action_segment_config or {}).get("planning_horizon", action_horizon))
        num_plan_steps = int((action_segment_config or {}).get("num_plan_steps", 1))
        max_future_horizon = max(action_horizon, planning_horizon if segments_enabled else action_horizon)
        real_row_count = len(df)
        if max_samples_per_file is not None:
            df = df.head(max_samples_per_file)
            real_row_count = min(real_row_count, max_samples_per_file)

        cache_namespace = dataset_cache_namespace(
            dataset_config,
            dataset_path,
            action_horizon=action_horizon,
            max_samples_per_file=max_samples_per_file,
            action_segment_config=action_segment_config if segments_enabled else None,
        )

        last_row = df.iloc[-1:]  
        padding_rows = pd.concat([last_row] * max_future_horizon, ignore_index=True)
        df = pd.concat([df, padding_rows], ignore_index=True)

        base_video_path = dataset_path / "videos" / parquet_path.parent.name
        video_paths = adapter.resolve_video_paths(base_video_path, parquet_path)
        missing_views = sorted(set(adapter.view_map) - set(video_paths))
        for view_key in missing_views:
            logging.warning(
                "missing video file for %s/%s view %s; tried %s",
                arm_name,
                dataset_name,
                view_key,
                ", ".join(adapter.view_map[view_key]),
            )
        if not video_paths:
            logging.warning(
                "skipping %s/%s file %s because no configured video views exist",
                arm_name,
                dataset_name,
                parquet_path,
            )
            return [], None

        episode_files = []
        for anchor_idx in range(real_row_count):
            start_idx = anchor_idx
            end_idx = start_idx + action_horizon

            cache_subdir = cache_dir / cache_namespace / arm_name / dataset_name / parquet_path.parent.name / parquet_path.stem
            cache_filename = f"{anchor_idx}_{start_idx}_{end_idx}.pkl"
            cache_filepath = cache_subdir / cache_filename

            if cache_filepath.exists():
                episode_files.append(str(cache_filepath))
                continue

            logging.info(f"build {cache_filename}")
            sub_df = df.iloc[start_idx: end_idx]
            first_row = sub_df.iloc[0]
            metadata = adapter.sample_metadata(first_row, parquet_path, start_idx)

            prompt = adapter.prompt(first_row, task_mapping, metadata)
            if not prompt:
                task_index = first_row.get("task_index", None)
                logging.info(f"cannot find task description from task_index={task_index}")

            episode = {
                "arm_key": arm_name,
                "dataset_key": dataset_name,
                "prompt": prompt,
                "state": adapter.state(first_row),
                "action": [adapter.action(row) for _, row in sub_df.iterrows()],
                "video_paths": video_paths,
                "timestamp": adapter.timestamp(first_row, start_idx),
                "planner_prompt": prompt,
                "planner_state": adapter.state(first_row),
                "planner_timestamp": adapter.timestamp(first_row, start_idx),
                **metadata,
            }
            if segments_enabled:
                future_df = df.iloc[anchor_idx: anchor_idx + planning_horizon]
                future_actions = [adapter.action(row) for _, row in future_df.iterrows()]
                action_segments, action_segment_mask = build_action_segment_target(
                    future_actions,
                    num_plan_steps=num_plan_steps,
                    planning_horizon=planning_horizon,
                    valid_action_count=max(0, real_row_count - anchor_idx),
                )
                episode["action_segments"] = action_segments
                episode["action_segment_mask"] = action_segment_mask

            cache_subdir.mkdir(parents=True, exist_ok=True)
            with open(cache_filepath, 'wb') as f:
                pickle.dump(episode, f)

            episode_files.append(str(cache_filepath))
        return episode_files, None 
        
    except Exception as e:
        error_msg = f"Error processing file {parquet_path}: {str(e)}"
        logging.error(error_msg)
        return [], error_msg

class SimulationDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        image_size: int = 448,
        max_samples_per_file: Union[int, None] = None,
        video_backend: str = "av",
        action_horizon: int = 50,
        video_backend_kwargs: Dict[str, Any] = None,
        binarize_gripper: bool = False,
        cache_dir: Union[str, Path] = None,  
        use_augmentation: bool = False,
        action_segment_config: Dict[str, Any] | None = None,
    ):
        self.config = config

        sorted_datasets = sorted(self.config['data_groups'].keys())
        self.arm_to_embodiment_id = {key: i for i, key in enumerate(sorted_datasets)}

        self.max_action_dim = config['max_action_dim']
        self.max_state_dim = config['max_state_dim']
        self.max_views = config['max_views']

        self.image_size = image_size
        self.max_samples_per_file = max_samples_per_file
        self.binarize_gripper = binarize_gripper
        self.use_augmentation = use_augmentation
        self.action_segment_config = dict(action_segment_config or {})
        self.action_segment_enabled = bool(self.action_segment_config.get("enabled", False))
        self.action_segment_dim = int(self.action_segment_config.get("action_dim", self.max_action_dim))


        cache_dir_value = cache_dir if cache_dir is not None else os.getenv("HIMEM_CACHE_DIR")
        self.cache_dir = Path(cache_dir_value).expanduser() if cache_dir_value else default_dataset_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []  
        self.arm2stats_dict = {}
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs or {}  

        if self.video_backend == "decord" and not self.video_backend_kwargs:
            self.video_backend_kwargs = {"ctx": "cpu"}  

        self._load_metadata()
        self._load_trajectories()

        self.basic_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        self.aug_transform = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.95, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BICUBIC), 
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            T.ToTensor()
        ])
        self.max_sample_retries = 10

    def _load_metadata(self):
     
        self.episodes = []
        self.tasks = {}

        # for arms
        for arm_name, arm_config in self.config['data_groups'].items():
            logging.info(f"Processing arm group: '{arm_name}'")

            norm_arm_list = []
            self.tasks[arm_name] = {}
            for dataset_name, dataset_config in arm_config.items():
                logging.info(f"Processing dataset: '{dataset_name}'")
                logging.info(f"Dataset config: {dataset_config}")
                path_str = dataset_config['path']
                dataset_path = Path(path_str)
                tasks_path = dataset_path / "meta" / "tasks.jsonl"
                if tasks_path.exists():
                    dataset_tasks = pd.read_json(tasks_path, lines=True).to_dict("records")
                    task_index_to_task = {
                        task_obj["task_index"]: task_obj["task"]
                        for task_obj in dataset_tasks
                        if "task_index" in task_obj and "task" in task_obj
                    }
                    self.tasks[arm_name][dataset_name] = task_index_to_task
                else:
                    raise FileNotFoundError(f"tasks file not found: {tasks_path}")
                
                episodes_path = dataset_path / "meta" / "episodes.jsonl"
                if episodes_path.exists():
                    self.episodes += pd.read_json(episodes_path, lines=True).to_dict("records")

     
                stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
                stats_path_after_compute = dataset_path / "meta" / "stats.json"
                if stats_path_after_compute.exists():
                    logging.info(f"Using existing stats file: {stats_path_after_compute}")
                    with open(stats_path_after_compute, "r") as f:
                        stats = json.load(f)
                    norm_arm_list.append(normalize_stats_payload(stats, dataset_config))
                elif stats_path.exists():
                    stats = compute_normalization_stats_from_minmax(stats_path, dataset_config)
                   
                    with open(stats_path_after_compute, "w") as f:
                        json.dump(stats, f, indent=4)
               
                    logging.info(f"Computed stats and saved to: {stats_path_after_compute}")
                    norm_arm_list.append(stats)
                else:
                    raise FileNotFoundError(f"normalization stats file not found: {stats_path}")
            

            self.arm2stats_dict[arm_name] = merge_normalization_stats(norm_arm_list)


    def _load_trajectories(self):

        

        parquet_process_units = []
        for arm_name, arm_config in self.config['data_groups'].items():
            for dataset_name, dataset_config in arm_config.items():
                dataset_path = dataset_config.get('path', None)
                if dataset_path is None:
                    raise ValueError(f"Dataset path for '{arm_name}-{dataset_name}' is not configured, please check the config")
                dataset_path = Path(dataset_path)
                parquet_files = list(dataset_path.glob("data/*/*.parquet"))
                
                task_mapping = self.tasks[arm_name][dataset_name]
                
                for parquet_path in parquet_files:
                    parquet_process_units.append((
                        parquet_path, 
                        arm_name, 
                        dataset_name, 
                        dataset_config, 
                        dataset_path,
                        task_mapping,  
                        self.action_horizon,
                        self.max_samples_per_file,
                        self.cache_dir,
                        self.action_segment_config,
                    ))

       
        logging.info(f"Found {len(parquet_process_units)} parquet files to process")
        if not parquet_process_units:
            logging.warning("No parquet files found. Check dataset paths in the dataset config.")
            return
        
   
        num_processes = min(16, len(parquet_process_units))

        logging.info(f"Using {num_processes} processes for concurrent processing")
        
 
        with mp.Pool(processes=num_processes) as pool:
            
            total_episodes = 0
            with tqdm(total=len(parquet_process_units), desc="Processing Parquet files to cache") as pbar:
                for episode_files, error in pool.imap_unordered(_process_parquet_file_worker, parquet_process_units):
                    if error:
                        logging.error(error)
                    else:
                        self.data.extend(episode_files)  
                        total_episodes += len(episode_files)
                    
                    pbar.set_postfix({
                        'episodes_this_file': len(episode_files),
                        'total_episodes': total_episodes
                    })
                    pbar.update(1)
        
        logging.info(f"Data processing completed, total {len(self.data)} files generated")


    def _pad_tensor(
        self, 
        source_tensor: torch.Tensor, 
        max_dim: int
    ) -> (torch.Tensor, torch.Tensor):

        source_dim = source_tensor.shape[-1]
        if source_dim > max_dim:
            raise ValueError(f"source tensor dimension {source_dim} exceeds configured max_dim {max_dim}")
        
        if source_tensor.dim() > 1:
            padded_shape = (*source_tensor.shape[:-1], max_dim)
        else:
            padded_shape = (max_dim,)

        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype, device=source_tensor.device)
        mask = torch.zeros(padded_shape, dtype=torch.bool, device=source_tensor.device)

        data_slice = (..., slice(0, source_dim))
        
        padded_tensor[data_slice] = source_tensor
        mask[data_slice] = True
            
        return padded_tensor, mask


    def _load_video_frame(self, video_paths: dict, timestamp: float) -> List[Image.Image]:
    
        frames = []
        for view, path in video_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"video file not found: {path}")
            
            if self.video_backend == "decord":
                import decord

                try:
                    ctx = self.video_backend_kwargs.get("ctx", "cpu")
                    if ctx == "cpu":
                        ctx = decord.cpu(0)
                    elif ctx == "gpu":
                        ctx = decord.gpu(0)
                    logging.info(f"Using video backend {self.video_backend}, context: {ctx}")
                    vr = decord.VideoReader(path, ctx=ctx)
                    logging.info(f"Successfully opened video file: {path}")
                    fps = vr.get_avg_fps()
                    logging.info(f"Video {path} FPS: {fps}")
                    if fps is None or np.isnan(fps):
                        raise ValueError(f"Unable to read FPS, video may be corrupted: {path}")

                    frame_idx = int(timestamp * fps)
                    logging.info(f"Reading video {path} frame index: {frame_idx} (timestamp: {timestamp}, fps: {fps})")
                    if frame_idx >= len(vr):
                        logging.info(f"the requested frame index exceeds video length: frame_idx={frame_idx}, len={len(vr)}. Using last frame instead.")
                        
                        frame_idx = len(vr) - 1

                    frame = vr[frame_idx].asnumpy()
                    frames.append(Image.fromarray(frame))
                    logging.info(f"Successfully read video frame: {path}, frame index: {frame_idx}")

                except Exception as e:
                    logging.info(f"Failed to read video file: {path}")
                    logging.info(f"Error message: {str(e)}")
                    raise

            elif self.video_backend == "av":
                import av
                try:
                    with av.open(path) as container:
                        for frame in container.decode(video=0):
                            if frame.time >= timestamp:
                                frames.append(Image.fromarray(frame.to_ndarray(format='rgb24')))
                                break

                except Exception as e:
                    logging.info(f"Failed to read video file: {path}")
                    logging.info(f"Error message: {str(e)}")
                    raise
            else:
                raise NotImplementedError(f"Video backend {self.video_backend} not implemented")
        
        return frames

    def _transform_and_pad_images(self, frames: List[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        images = frames
        if self.use_augmentation:
            images = [
                self.aug_transform(img) if random.random() < 0.5 else self.basic_transform(img)
                for img in images
            ]
        else:
            images = [self.basic_transform(img) for img in images]

        num_real_views = len(images)
        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:num_real_views] = True
        if image_mask.sum().item() == 0:
            raise ValueError("sample has no valid image views")

        while len(images) < self.max_views:
            dummy_image = torch.zeros_like(images[0]) if images else torch.zeros(3, self.image_size, self.image_size)
            images.append(dummy_image)
        return torch.stack(images), image_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if len(self.data) == 0:
            raise IndexError("SimulationDataset is empty")
        last_error = None
        for attempt in range(self.max_sample_retries):
            sample_idx = idx if attempt == 0 else random.randint(0, len(self.data) - 1)
            try:
                return self._load_sample(sample_idx)
            except (OSError, EOFError, pickle.PickleError, RuntimeError, ValueError, KeyError, TypeError) as e:
                last_error = e
                logging.info(f"Skipping sample {sample_idx}: {e}")
        raise RuntimeError(f"Failed to load a valid sample after {self.max_sample_retries} attempts") from last_error

    def _load_sample(self, idx):
        cache_filepath = self.data[idx]
        
        with open(cache_filepath, 'rb') as f:
            item = pickle.load(f)
 
        
        arm_key = item["arm_key"]
        dataset_key = item["dataset_key"]
        embodiment_id = self.arm_to_embodiment_id[arm_key]

 
        frames = self._load_video_frame(item["video_paths"], item["timestamp"])
        images, image_mask = self._transform_and_pad_images(frames)
        planner_images = None
        planner_image_mask = None
        if "planner_timestamp" in item:
            planner_frames = self._load_video_frame(item["video_paths"], item["planner_timestamp"])
            planner_images, planner_image_mask = self._transform_and_pad_images(planner_frames)


        if item["state"] is None:
            raise ValueError("missing observation.state, please check data integrity")
        
    

        try:
            norm_stats = self.arm2stats_dict[arm_key]
        except KeyError:
        
            raise KeyError(f"Normalization stats not found for arm_key={arm_key} and dataset_key={dataset_key}")

        

        state = torch.tensor(item["state"], dtype=torch.float32)
        device = state.device
        state_min = torch.tensor(norm_stats["observation.state"]["min"], dtype=torch.float32, device=device)
        state_max = torch.tensor(norm_stats["observation.state"]["max"], dtype=torch.float32, device=device)
        
        state = minmax_normalize(state, state_min, state_max)

        state_padded, _ = self._pad_tensor(
            state, self.max_state_dim
        )

        planner_state_value = item.get("planner_state", item["state"])
        planner_state = torch.tensor(planner_state_value, dtype=torch.float32)
        planner_state = minmax_normalize(
            planner_state,
            state_min.to(device=planner_state.device),
            state_max.to(device=planner_state.device),
        )
        planner_state_padded, _ = self._pad_tensor(
            planner_state, self.max_state_dim
        )


        if item["action"] is None:
            raise ValueError("missing action, please check data integrity")

  
        action = torch.from_numpy(np.stack(item["action"])).float()
        device = action.device
        action_min = torch.tensor(norm_stats["action"]["min"], dtype=torch.float32, device=device)
        action_max = torch.tensor(norm_stats["action"]["max"], dtype=torch.float32, device=device)
        action = minmax_normalize(action, action_min.unsqueeze(0), action_max.unsqueeze(0))
        if self.binarize_gripper and action.shape[-1] > 0:
            action[..., -1] = (action[..., -1] > 0.5).to(action.dtype)

        action_padded, action_mask = self._pad_tensor(
            action, self.max_action_dim
        )

        prompt = item["prompt"] if item["prompt"] is not None else ""
        
        sample = {
            "images": images,
            "image_mask": image_mask,
            "prompt": prompt,
            "state": state_padded.to(dtype=torch.bfloat16),
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "embodiment_id": torch.tensor(embodiment_id, dtype=torch.long),
            "planner_prompt": item.get("planner_prompt", prompt),
            "planner_state": planner_state_padded.to(dtype=torch.bfloat16),
        }
        if planner_images is not None and planner_image_mask is not None:
            sample["planner_images"] = planner_images
            sample["planner_image_mask"] = planner_image_mask
        if "action_segments" in item and "action_segment_mask" in item:
            action_segments = torch.as_tensor(item["action_segments"], dtype=torch.float32)
            if self.binarize_gripper and action_segments.shape[-1] > 0:
                action_segments[..., -1] = (action_segments[..., -1] > 0.5).to(action_segments.dtype)
            action_segments_padded, _ = self._pad_tensor(action_segments, self.action_segment_dim)
            sample["action_segments"] = action_segments_padded.to(dtype=torch.bfloat16)
            sample["action_segment_mask"] = torch.as_tensor(item["action_segment_mask"], dtype=torch.bool)
        if "boundary" in item:
            sample["boundary"] = torch.tensor(float(item["boundary"]), dtype=torch.float32)
        if "progress" in item:
            sample["progress"] = torch.tensor(float(item["progress"]), dtype=torch.float32)
        if item.get("skill_id") is not None:
            sample["skill_id"] = torch.tensor(int(item["skill_id"]), dtype=torch.long)
        for metadata_key in ("episode_id", "frame_index", "global_frame_index", "segment_id", "segment_start", "segment_end"):
            if metadata_key in item and item[metadata_key] is not None:
                sample[metadata_key] = item[metadata_key]
        return sample
