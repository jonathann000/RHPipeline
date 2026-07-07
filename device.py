"""
Device resolution — picks the right accelerate device_map across
CUDA clusters and local Apple Silicon.
"""

import torch


def resolve_device_map() -> str | dict:
    """
    CUDA: "auto" lets accelerate shard/offload across (multi-)GPU.
    MPS (Apple Silicon): must be pinned explicitly — accelerate's "auto"
                         targets CUDA/CPU and won't reliably place weights
                         on the "mps" device.
    Otherwise: CPU.
    """
    if torch.cuda.is_available():
        return "auto"
    if torch.backends.mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}
