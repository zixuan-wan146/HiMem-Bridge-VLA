from __future__ import annotations

from typing import Protocol

from himem_bridge_vla.core import BenchmarkSpec
from himem_bridge_vla.runtime.contract import PolicyRequest


class BenchmarkAdapter(Protocol):
    spec: BenchmarkSpec

    def build_request(
        self,
        obs,
        prompt: str,
        history,
        *,
        reset_memory: bool,
    ) -> PolicyRequest:
        ...

    def parse_model_action(self, action_values):
        ...
