"""Backend selection and memory accounting for CUDA/ROCm and Apple Metal."""
import sys
from typing import TypedDict

import torch

IS_MLX = sys.platform == "darwin"

if IS_MLX:
    from mlx_talker import Talker
    from mlx_thinker import Thinker
else:
    from engine import Talker, Thinker


class Memory(TypedDict):
    gpu_mem_gib: float
    gpu_reserved_gib: float
    gpu_free_gib: float


def memory() -> Memory:
    if IS_MLX:
        import mlx.core as mx
        active = mx.get_active_memory()
        cached = mx.get_cache_memory()
        budget = int(mx.device_info()["max_recommended_working_set_size"])
        return Memory(gpu_mem_gib=round(active / 2**30, 2),
                      gpu_reserved_gib=round((active + cached) / 2**30, 2),
                      gpu_free_gib=round(max(0, budget - active - cached) / 2**30, 2))
    return Memory(gpu_mem_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
                  gpu_reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 2),
                  gpu_free_gib=round(torch.cuda.mem_get_info()[0] / 2**30, 2))


def empty_cache() -> None:
    if IS_MLX:
        import mlx.core as mx
        mx.clear_cache()
    else:
        torch.cuda.empty_cache()
