"""No CPU/MPS execution fallback. Host file I/O and preprocessing remain necessary."""

import os
import sys

import torch


def require_cuda_server():
    if sys.platform != "linux":
        raise RuntimeError("Hexagon execution is restricted to the approved Linux GPU server")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "RANK" in os.environ:
        raise RuntimeError("Launch with plain python on one visible CUDA device, not torchrun")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A bf16-capable CUDA GPU is required; no CPU fallback")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")
