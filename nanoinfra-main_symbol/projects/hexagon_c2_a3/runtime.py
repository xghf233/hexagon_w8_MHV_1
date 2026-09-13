"""Linux/single-visible-GPU/bf16 gate. No CPU or MPS model fallback."""

import os
import sys
import torch


def require_cuda_server():
    if sys.platform != "linux":
        raise RuntimeError("M2 execution is restricted to the approved Linux GPU server")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "RANK" in os.environ:
        raise RuntimeError("Use plain python, not torchrun")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES; no CPU fallback")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible CUDA GPU must support bf16")
    return torch.device("cuda:0")
