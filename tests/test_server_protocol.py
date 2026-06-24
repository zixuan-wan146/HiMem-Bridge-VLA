from __future__ import annotations

import math

import pytest

from himem_bridge_vla.server_protocol import (
    checkpoint_normalizer_dim,
    normalize_action_mask,
    normalize_binary_mask,
    validate_inference_request,
)


def tiny_rgb_image(value: int = 0):
    return [
        [[value, value, value], [value, value, value]],
        [[value, value, value], [value, value, value]],
    ]


def valid_request() -> dict:
    return {
        "image": [tiny_rgb_image(1), tiny_rgb_image(2), tiny_rgb_image(3)],
        "state": [0.1, 0.2, 0.3],
        "prompt": "pick up the object",
        "image_mask": [1, 1, 0],
        "action_mask": [1, 1, 1, 0],
    }


def test_validate_inference_request_accepts_and_pads_valid_payload():
    request = validate_inference_request(valid_request(), target_state_dim=6)

    assert request["prompt"] == "pick up the object"
    assert request["state"] == pytest.approx([0.1, 0.2, 0.3])
    assert request["image_mask"] == [1, 1, 0]
    assert request["action_mask"] == [1, 1, 1, 0, 0, 0]
    assert request["return_debug"] is False


def test_validate_inference_request_accepts_optional_runtime_fields():
    payload = valid_request()
    payload["robot_key"] = "libero"
    payload["return_debug"] = True

    request = validate_inference_request(payload, target_state_dim=6)

    assert request["robot_key"] == "libero"
    assert request["return_debug"] is True


def test_validate_inference_request_accepts_zero_padded_action_mask_for_smaller_model_dim():
    payload = valid_request()
    payload["state"] = [0.1, 0.2, 0.3]
    payload["action_mask"] = [1, 1, 1, 0, 0, 0]

    request = validate_inference_request(payload, target_state_dim=3, target_action_dim=3, max_action_mask_dim=6)

    assert request["state"] == pytest.approx([0.1, 0.2, 0.3])
    assert request["action_mask"] == [1, 1, 1]


def test_validate_inference_request_rejects_missing_required_fields():
    payload = valid_request()
    del payload["action_mask"]

    with pytest.raises(ValueError, match="Missing required request fields"):
        validate_inference_request(payload)


def test_validate_inference_request_rejects_wrong_image_count():
    payload = valid_request()
    payload["image"] = [tiny_rgb_image()]

    with pytest.raises(ValueError, match="exactly 3 images"):
        validate_inference_request(payload)


def test_validate_inference_request_rejects_non_rgb_image_shape():
    payload = valid_request()
    payload["image"][0] = [[[1, 2], [3, 4]]]

    with pytest.raises(ValueError, match="3 channels"):
        validate_inference_request(payload)


def test_validate_inference_request_rejects_out_of_range_pixels():
    payload = valid_request()
    payload["image"][0] = [[[256, 0, 0]]]

    with pytest.raises(ValueError, match="0..255"):
        validate_inference_request(payload)


def test_validate_inference_request_rejects_nonfinite_state():
    payload = valid_request()
    payload["state"] = [0.0, math.inf]

    with pytest.raises(ValueError, match="finite"):
        validate_inference_request(payload)


def test_validate_inference_request_rejects_overlong_state():
    payload = valid_request()
    payload["state"] = [0.0] * 25

    with pytest.raises(ValueError, match="state length"):
        validate_inference_request(payload, target_state_dim=24)


def test_validate_inference_request_rejects_all_zero_masks():
    payload = valid_request()
    payload["action_mask"] = [0, 0, 0]

    with pytest.raises(ValueError, match="action_mask must activate"):
        validate_inference_request(payload)


def test_normalize_binary_mask_rejects_nonbinary_values():
    with pytest.raises(ValueError, match="only 0/1"):
        normalize_binary_mask([1, 2, 0], 4, "action_mask")


def test_normalize_binary_mask_rejects_overlong_masks():
    with pytest.raises(ValueError, match="exceeds target dimension"):
        normalize_binary_mask([1, 0, 1], 2, "image_mask")


def test_normalize_action_mask_rejects_active_dims_beyond_model_dim():
    with pytest.raises(ValueError, match="beyond model action dimension"):
        normalize_action_mask([1, 1, 1, 0, 1], target_action_dim=3, max_action_mask_dim=6)


def test_checkpoint_normalizer_dim_tracks_checkpoint_state_and_action_dims():
    assert checkpoint_normalizer_dim({"state_dim": 7, "per_action_dim": 7}) == 7
    assert checkpoint_normalizer_dim({"state_dim": 8, "per_action_dim": 7}) == 8
    assert checkpoint_normalizer_dim({"state_dim": 7, "per_action_dim": 9}) == 9


def test_checkpoint_normalizer_dim_falls_back_for_missing_or_invalid_values():
    assert checkpoint_normalizer_dim({}) == 24
    assert checkpoint_normalizer_dim({"state_dim": 0, "per_action_dim": "bad"}) == 24
