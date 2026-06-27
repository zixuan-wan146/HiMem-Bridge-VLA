"""Compatibility shim for older imports.

New runtime code should use :mod:`himem_bridge_vla.runtime.contract`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from himem_bridge_vla.runtime.contract import PolicyRequest
from himem_bridge_vla.runtime.contract import checkpoint_normalizer_dim
from himem_bridge_vla.runtime.contract import policy_request_from_json


def validate_inference_request(data: Mapping[str, Any], **_kwargs) -> PolicyRequest:
    return policy_request_from_json(data)


__all__ = ["PolicyRequest", "checkpoint_normalizer_dim", "policy_request_from_json", "validate_inference_request"]
