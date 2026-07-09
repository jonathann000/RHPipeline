"""
Device resolution — picks the right accelerate device_map across
CUDA clusters and local Apple Silicon.
"""

import logging
import torch

logger = logging.getLogger(__name__)


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


def resolve_quantization_config(approx_params_b: float, safety_factor: float = 1.2):
    """
    Decide whether a model needs 8-bit quantization to safely fit the GPU
    actually detected at runtime — checked against real hardware via
    torch.cuda, not a guess about which variant (e.g. A100 40GB vs 80GB)
    happens to be connected. Returns a BitsAndBytesConfig if quantization
    is needed, else None (load in plain bf16).

    approx_params_b: rough parameter count in billions for the model being
        loaded (see LLM_BACKENDS in run.py) — used to estimate its bf16
        memory footprint (2 bytes/param).
    safety_factor: raw weight size alone isn't the whole story — activations,
        KV-cache, and CUDA overhead need headroom too. Requiring the GPU to
        have 1.2x the raw bf16 weight size before trusting bf16 is what
        correctly separates "fits on an 80GB card" from "needs quantizing on
        a 40GB card" for a ~32B model (64GB raw -> ~77GB threshold).
    """
    if not torch.cuda.is_available():
        return None  # quantization is a CUDA/bitsandbytes-only concern

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    estimated_bf16_gb = approx_params_b * 2
    threshold_gb = estimated_bf16_gb * safety_factor

    if total_vram_gb < threshold_gb:
        from transformers import BitsAndBytesConfig
        logger.info(
            f"Detected {total_vram_gb:.0f}GB VRAM, ~{estimated_bf16_gb:.0f}GB "
            f"needed in bf16 for a ~{approx_params_b:.0f}B model — loading in 8-bit instead"
        )
        return BitsAndBytesConfig(load_in_8bit=True)

    logger.info(
        f"Detected {total_vram_gb:.0f}GB VRAM, ~{estimated_bf16_gb:.0f}GB "
        f"needed in bf16 for a ~{approx_params_b:.0f}B model — loading in bf16"
    )
    return None
