from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def rgb_array_to_pil(image: Any, image_size: int) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) and not np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"RGB image must contain numeric pixel values, got dtype={array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("RGB image must contain only finite pixel values")
    if array.min() < 0 or array.max() > 255:
        raise ValueError("RGB image pixel values must be in the 0..255 range")
    pil_image = Image.fromarray(array.astype(np.uint8, copy=False))
    return pil_image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
