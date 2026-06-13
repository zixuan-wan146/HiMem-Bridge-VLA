import numpy as np
import pytest

from himem_bridge_vla.image_preprocessing import rgb_array_to_pil


def test_rgb_array_to_pil_preserves_rgb_channel_order():
    image = [
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 255, 255]],
    ]

    pil_image = rgb_array_to_pil(image, image_size=2)

    array = np.asarray(pil_image)
    assert array[0, 0].tolist() == [255, 0, 0]
    assert array[0, 1].tolist() == [0, 255, 0]
    assert array[1, 0].tolist() == [0, 0, 255]


def test_rgb_array_to_pil_resizes_to_square_image_size():
    image = np.zeros((2, 4, 3), dtype=np.uint8)

    pil_image = rgb_array_to_pil(image, image_size=8)

    assert pil_image.size == (8, 8)


def test_rgb_array_to_pil_rejects_invalid_pixel_ranges():
    with pytest.raises(ValueError, match="0..255"):
        rgb_array_to_pil([[[-1, 0, 0]]], image_size=2)

    with pytest.raises(ValueError, match="0..255"):
        rgb_array_to_pil([[[256, 0, 0]]], image_size=2)


def test_rgb_array_to_pil_rejects_nonfinite_values():
    image = np.array([[[np.nan, 0, 0]]])

    with pytest.raises(ValueError, match="finite"):
        rgb_array_to_pil(image, image_size=2)
