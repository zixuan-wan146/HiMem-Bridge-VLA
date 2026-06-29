from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any


BYTES_PER_GIB = 1024**3


@dataclass
class CudaMemoryFloor:
    """Keep CUDA memory usage above a requested floor for long-running jobs."""

    torch: Any
    target_gb: float
    device: Any = "cuda"
    chunk_mb: int = 256
    _chunks: list[Any] = field(default_factory=list, init=False)

    def start(self) -> "CudaMemoryFloor":
        if self.target_gb <= 0:
            raise ValueError("target_gb must be positive")
        if self.chunk_mb <= 0:
            raise ValueError("chunk_mb must be positive")
        stats = self.refill_to_target()
        logging.info(
            "CUDA memory floor active: target=%.2f GiB used=%.2f GiB reserved_by_floor=%.2f GiB",
            self.target_gb,
            stats["used_gb"],
            self.reserved_gb,
        )
        return self

    def refill_to_target(self) -> dict[str, float | int | str]:
        device = _normalize_cuda_device(self.torch, self.device)
        if device.type != "cuda":
            raise ValueError(f"CUDA memory floor requires a CUDA device, got {device}")
        self.torch.cuda.set_device(device)

        target_bytes = int(float(self.target_gb) * BYTES_PER_GIB)
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(device)
        if target_bytes >= total_bytes:
            raise ValueError(
                f"CUDA memory floor {self.target_gb:.2f} GiB is not below total GPU memory "
                f"{total_bytes / BYTES_PER_GIB:.2f} GiB"
            )

        chunk_bytes = int(self.chunk_mb * 1024 * 1024)
        while True:
            stats = cuda_memory_stats(self.torch, device)
            used_bytes = int(stats["used_bytes"])
            if used_bytes >= target_bytes:
                break
            bytes_to_allocate = min(chunk_bytes, target_bytes - used_bytes)
            self._chunks.append(_allocate_cuda_chunk(self.torch, device, bytes_to_allocate))

        stats = cuda_memory_stats(self.torch, device)
        if int(stats["used_bytes"]) < target_bytes:
            raise RuntimeError(
                f"CUDA memory floor target was not reached: target={self.target_gb:.2f} GiB, "
                f"used={stats['used_gb']:.2f} GiB"
            )
        return stats

    @property
    def reserved_gb(self) -> float:
        return sum(int(chunk.numel() * chunk.element_size()) for chunk in self._chunks) / BYTES_PER_GIB

    def trim_to_target(self) -> dict[str, float | int | str]:
        device = _normalize_cuda_device(self.torch, self.device)
        target_bytes = int(float(self.target_gb) * BYTES_PER_GIB)
        while self._chunks:
            stats = cuda_memory_stats(self.torch, device)
            used_bytes = int(stats["used_bytes"])
            chunk = self._chunks[-1]
            chunk_bytes = int(chunk.numel() * chunk.element_size())
            if used_bytes - chunk_bytes < target_bytes:
                break
            self._chunks.pop()
            del chunk
            self.torch.cuda.empty_cache()
        return cuda_memory_stats(self.torch, device)

    def release(self) -> None:
        self._chunks.clear()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def close(self) -> None:
        self.release()


def reserve_cuda_memory_floor(
    torch: Any,
    *,
    target_gb: float | None,
    device: Any = "cuda",
    chunk_mb: int = 256,
) -> CudaMemoryFloor | None:
    if target_gb is None:
        return None
    return CudaMemoryFloor(torch=torch, target_gb=float(target_gb), device=device, chunk_mb=int(chunk_mb)).start()


def cuda_memory_stats(torch: Any, device: Any = "cuda") -> dict[str, float | int | str]:
    device = _normalize_cuda_device(torch, device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "device": str(device),
            "total_bytes": 0,
            "free_bytes": 0,
            "used_bytes": 0,
            "total_gb": 0.0,
            "free_gb": 0.0,
            "used_gb": 0.0,
        }
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = int(total_bytes - free_bytes)
    return {
        "device": str(device),
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "used_bytes": used_bytes,
        "total_gb": float(total_bytes / BYTES_PER_GIB),
        "free_gb": float(free_bytes / BYTES_PER_GIB),
        "used_gb": float(used_bytes / BYTES_PER_GIB),
    }


def _normalize_cuda_device(torch: Any, device: Any) -> Any:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _allocate_cuda_chunk(torch: Any, device: Any, bytes_to_allocate: int) -> Any:
    bytes_to_allocate = max(1, int(bytes_to_allocate))
    chunk_bytes = bytes_to_allocate
    while chunk_bytes > 0:
        try:
            return torch.empty((chunk_bytes,), dtype=torch.uint8, device=device)
        except RuntimeError:
            if chunk_bytes <= 1024 * 1024:
                raise
            chunk_bytes //= 2
            torch.cuda.empty_cache()
    raise RuntimeError("failed to allocate CUDA memory floor chunk")
