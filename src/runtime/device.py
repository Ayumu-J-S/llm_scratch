from __future__ import annotations

import torch


def select_device(requested: str) -> torch.device:
    if requested not in {"cuda", "cpu"}:
        raise ValueError("runtime.device must be exactly 'cuda' or 'cpu'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "runtime.device=cuda was requested, but CUDA is unavailable. "
            "Use the pinned DGX Spark container or explicitly set runtime.device=cpu for tests."
        )
    if requested == "cuda":
        try:
            device_count = torch.cuda.device_count()
            if device_count < 1:
                raise RuntimeError("CUDA reported zero visible devices")
            torch.cuda.init()
            device_name = torch.cuda.get_device_name(0).strip()
            if not device_name:
                raise RuntimeError("CUDA device 0 has no readable name")
        except Exception as error:
            raise RuntimeError(f"CUDA runtime preflight failed: {error}") from error
    return torch.device(requested)
