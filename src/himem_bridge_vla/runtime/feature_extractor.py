from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torchvision import transforms

from himem_bridge_vla.image_preprocessing import rgb_array_to_pil
from himem_bridge_vla.runtime_config import IMAGE_SIZE


def decode_images_by_view(images_by_view: Mapping[str, np.ndarray], device: torch.device) -> list[torch.Tensor]:
    images = []
    for view_name, image in images_by_view.items():
        tensor = decode_image_array(image, device)
        expected_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{view_name} image_size must be {expected_shape}, got {tuple(tensor.shape)}")
        images.append(tensor)
    return images


def decode_image_array(image: np.ndarray, device: torch.device) -> torch.Tensor:
    pil = rgb_array_to_pil(image, IMAGE_SIZE)
    return transforms.ToTensor()(pil).to(device)
