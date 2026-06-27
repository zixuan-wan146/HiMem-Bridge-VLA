from __future__ import annotations

import logging
import os
from datetime import datetime


def setup_file_logging(log_dir: str, *, is_main_process: bool, filename_prefix: str = "train_log") -> None:
    if not is_main_process:
        return
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Logging to: %s", log_path)
