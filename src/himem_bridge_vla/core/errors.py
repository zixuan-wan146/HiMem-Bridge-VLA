from __future__ import annotations


class HiMemBridgeVLAError(Exception):
    """Base exception for project-level errors."""


class ContractError(HiMemBridgeVLAError):
    """Raised when a data, runtime, or benchmark contract is violated."""


class ConfigurationError(HiMemBridgeVLAError):
    """Raised when configuration parsing or validation fails."""
