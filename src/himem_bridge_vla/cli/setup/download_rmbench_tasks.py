from __future__ import annotations

import argparse
from email.message import Message
import json
from pathlib import Path
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse, parse_qs
from urllib.request import Request, urlopen

from huggingface_hub import hf_hub_download


DEFAULT_RMBENCH_TASKS = [
    "observe_and_pickup",
    "rearrange_blocks",
    "put_back_block",
    "swap_blocks",
    "swap_T",
    "blocks_ranking_try",
    "press_button",
    "cover_blocks",
    "battery_try",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download selected RMBench demo_clean task folders from Hugging Face."
    )
    parser.add_argument("--repo-id", default="TianxingChen/RMBench")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", default=".", help="Destination root for the dataset files.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_RMBENCH_TASKS,
        help="RMBench task names under data/<task>/demo_clean.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip demo_clean/video files. HDF5 files already contain RGB byte streams.",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest path. Defaults to <local-dir>/data/rmbench_9tasks_manifest.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest).expanduser()
        if args.manifest
        else local_dir / "data" / "rmbench_9tasks_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_files: list[str] = []
    task_summaries = {}
    for task in args.tasks:
        files = list(
            list_demo_clean_files(
                args.repo_id,
                args.revision,
                task,
                skip_video=bool(args.skip_video),
                retries=args.retries,
            )
        )
        if args.skip_video:
            files = [path for path in files if "/video/" not in path]
        prefix = f"data/{task}/demo_clean"
        task_summaries[task] = {"files": len(files), "prefix": prefix}
        all_files.extend(files)
        print(json.dumps({"task": task, "files": len(files), "prefix": prefix}, sort_keys=True), flush=True)

    downloaded = []
    started = time.time()
    for index, filename in enumerate(all_files, start=1):
        path = download_with_retry(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            filename=filename,
            local_dir=str(local_dir),
            retries=args.retries,
        )
        downloaded.append(filename)
        if index == 1 or index % 25 == 0 or index == len(all_files):
            print(
                json.dumps(
                    {
                        "downloaded": index,
                        "total": len(all_files),
                        "last": filename,
                        "local_path": str(path),
                        "elapsed_sec": round(time.time() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "local_dir": str(local_dir),
        "tasks": list(args.tasks),
        "skip_video": bool(args.skip_video),
        "task_summaries": task_summaries,
        "files": downloaded,
        "file_count": len(downloaded),
        "created_at_unix": int(time.time()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"manifest": str(manifest_path), "file_count": len(downloaded)}, sort_keys=True))
    return 0


def list_demo_clean_files(
    repo_id: str,
    revision: str,
    task: str,
    *,
    skip_video: bool,
    retries: int,
) -> Iterable[str]:
    prefix = f"data/{task}/demo_clean"
    yield f"{prefix}/language_annotation.json"
    subdirs = ["_traj_data", "data", "instructions"]
    if not skip_video:
        subdirs.append("video")
    seen = {f"{prefix}/language_annotation.json"}
    for subdir in subdirs:
        for path in list_repo_tree_files(
            repo_id,
            revision,
            f"{prefix}/{subdir}",
            retries=retries,
            recursive=True,
        ):
            if path in seen:
                continue
            seen.add(path)
            yield path


def list_repo_tree_files(
    repo_id: str,
    revision: str,
    path_in_repo: str,
    *,
    retries: int,
    recursive: bool,
) -> Iterable[str]:
    cursor: str | None = None
    while True:
        query = {
            "recursive": "true" if recursive else "false",
            "expand": "true",
            "limit": "50",
        }
        if cursor:
            query["cursor"] = cursor
        quoted_path = quote(path_in_repo, safe="")
        url = (
            f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}/"
            f"{quoted_path}?{urlencode(query)}"
        )
        data, headers = request_json(url, retries=retries)
        for item in data:
            if item.get("type") == "file":
                yield str(item["path"])
        cursor = next_cursor(headers)
        if not cursor:
            break


def request_json(url: str, *, retries: int) -> tuple[list[dict], Message]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"Accept": "application/json"})
            with urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8")), response.headers
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed to query {url}: {last_error}") from last_error


def next_cursor(headers: Message) -> str | None:
    link = headers.get("Link")
    if not link:
        return None
    for part in link.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">")
        if start == -1 or end == -1:
            continue
        parsed = urlparse(part[start + 1 : end])
        values = parse_qs(parsed.query).get("cursor")
        if values:
            return values[0]
    return None


def download_with_retry(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    filename: str,
    local_dir: str,
    retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                filename=filename,
                local_dir=local_dir,
                resume_download=True,
            )
        except Exception as exc:  # noqa: BLE001 - hub errors vary across versions.
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed to download {filename}: {last_error}") from last_error


if __name__ == "__main__":
    raise SystemExit(main())
