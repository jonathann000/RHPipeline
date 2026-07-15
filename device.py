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
    Decide whether a model needs quantization to safely fit the GPU actually
    detected at runtime — checked against real hardware via torch.cuda, not a
    guess about which variant (e.g. A100 40GB vs 80GB) happens to be
    connected. Returns a BitsAndBytesConfig if quantization is needed, else
    None (load in plain bf16).

    Three tiers, checked from least to most aggressive: bf16 (no
    quantization) -> 8-bit -> 4-bit.

    approx_params_b: rough parameter count in billions for the model being
        loaded (see LLM_BACKENDS in run.py) — used to estimate memory
        footprint at each tier (2 bytes/param bf16, 1 byte/param 8-bit,
        0.5 bytes/param 4-bit).
    safety_factor: raw weight size alone isn't the whole story — activations,
        KV-cache, and CUDA overhead need headroom too.

    The 8-bit tier uses a wider margin (1.6x by default vs bf16/4-bit's
    1.2x) based on two real failures, not just one:
    - A ~32B model's 8-bit weights alone (~32GB) fit a 40GB card under a
      1.2x margin (38.4GB threshold), but loading it that way actually
      failed — accelerate's device_map wanted to offload a small piece to
      CPU for KV-cache/activation headroom, which bitsandbytes' 8-bit path
      refuses without extra config.
    - A 1.4x margin looked sufficient after that first fix, but a ~27B
      model on a 39GB card (27 * 1.4 = 37.8 < 39, so 8-bit was still
      selected) OOM'd anyway — this time not even at load, but partway
      into the first generation call, when the attention softmax needed a
      few more GB than the sliver left free. 1.4x wasn't conservative
      enough; the real requirement here was >1.44x, so 1.6x is used to
      leave real margin above the second observed failure, not just
      clear it by a hair.
    """
    if not torch.cuda.is_available():
        return None  # quantization is a CUDA/bitsandbytes-only concern

    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    estimated_bf16_gb = approx_params_b * 2
    estimated_8bit_gb = approx_params_b * 1
    estimated_4bit_gb = approx_params_b * 0.5
    int8_safety_factor = safety_factor + 0.4

    if total_vram_gb >= estimated_bf16_gb * safety_factor:
        logger.info(
            f"Detected {total_vram_gb:.0f}GB VRAM, ~{estimated_bf16_gb:.0f}GB "
            f"needed in bf16 for a ~{approx_params_b:.0f}B model — loading in bf16"
        )
        return None

    from transformers import BitsAndBytesConfig

    if total_vram_gb >= estimated_8bit_gb * int8_safety_factor:
        logger.info(
            f"Detected {total_vram_gb:.0f}GB VRAM, ~{estimated_bf16_gb:.0f}GB "
            f"needed in bf16 for a ~{approx_params_b:.0f}B model — loading in 8-bit instead"
        )
        return BitsAndBytesConfig(load_in_8bit=True)

    logger.info(
        f"Detected {total_vram_gb:.0f}GB VRAM, ~{estimated_8bit_gb:.0f}GB needed "
        f"in 8-bit for a ~{approx_params_b:.0f}B model — not enough headroom for "
        f"activations/KV-cache on top of that, loading in 4-bit instead"
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
