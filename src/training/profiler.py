"""CPU-safe timing, memory sampling, and parameter inspection utilities."""

import time
from typing import Any, Dict, Optional

import torch


class VRAMProfiler:
    """Measure CUDA peak memory when available, with a CPU-safe timing fallback."""

    def __init__(self, tokens: int = 0, device: Optional[Any] = None):
        self.tokens = int(tokens)
        self.device = device
        self.samples = []
        self.cuda = False
        self.start_mem = 0
        self.peak_mem = 0
        self.reserved_mem = 0
        self.allocated_gb = 0.0
        self.reserved_gb = 0.0
        self.elapsed_time = 0.0
        self.tokens_per_second = 0.0

    def __enter__(self):
        requested = torch.device(self.device) if self.device is not None else None
        self.cuda = torch.cuda.is_available() and (requested is None or requested.type == "cuda")
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_mem = torch.cuda.memory_allocated()
        self.start_time = time.perf_counter()
        return self

    def sample(self, label: str = "", tokens: int = 0) -> Dict[str, Any]:
        """Record one real-time sample using absolute bytes on CUDA."""
        sample = {
            "label": label,
            "elapsed_seconds": time.perf_counter() - self.start_time,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "tokens": int(tokens),
        }
        if self.cuda:
            torch.cuda.synchronize()
            sample["allocated_bytes"] = torch.cuda.memory_allocated()
            sample["reserved_bytes"] = torch.cuda.memory_reserved()
        self.samples.append(sample)
        return sample

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cuda:
            torch.cuda.synchronize()
            self.peak_mem = torch.cuda.max_memory_allocated()
            self.reserved_mem = torch.cuda.max_memory_reserved()
            self.allocated_gb = self.peak_mem / 1024**3
            self.reserved_gb = self.reserved_mem / 1024**3
        self.elapsed_time = time.perf_counter() - self.start_time
        self.tokens_per_second = self.tokens / self.elapsed_time if self.elapsed_time else 0.0
        return False

    def as_dict(self) -> Dict[str, Any]:
        """Return a stable JSON-friendly profile record for Member C's logs."""
        return {
            "peak_allocated_bytes": int(self.peak_mem),
            "peak_reserved_bytes": int(self.reserved_mem),
            "peak_allocated_gb": float(self.allocated_gb),
            "peak_reserved_gb": float(self.reserved_gb),
            "elapsed_seconds": float(self.elapsed_time),
            "tokens": self.tokens,
            "tokens_per_second": float(self.tokens_per_second),
            "samples": list(self.samples),
        }


def parameter_counts(model) -> Dict[str, Any]:
    """Match Member A's count schema while retaining a short ``percent`` alias."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = 100.0 * trainable / total if total else 0.0
    return {
        "trainable": trainable,
        "frozen": total - trainable,
        "total": total,
        "percent_trainable": round(percent, 4),
        "percent": percent,
    }


def print_trainable_parameters(model):
    stats = parameter_counts(model)
    print(
        f"Trainable: {stats['trainable']:,} || Total: {stats['total']:,} || "
        f"Trainable %: {stats['percent_trainable']:.4f}%"
    )
    return stats
