"""CUDA backend: namespace assembly.

``core`` wraps torch with the MLX/numpy calling convention, ``nn`` is a
minimal module factory, ``io`` and ``quant`` are edge0 wrappers.  Selected
with ``EDGE0_BACKEND=cuda`` (see ``edge0.backends``).
"""

from __future__ import annotations

import torch

from edge0.backends.cuda import core, io, nn, quant  # noqa: F401


class BackendImpl:
    """Handle to the active backend implementation."""

    name = "cuda"
    description = "PyTorch / CUDA on NVIDIA GPUs (work in progress)"

    @property
    def version(self) -> str:
        return f"torch {torch.__version__}, cuda {torch.version.cuda}"

    @property
    def device(self) -> torch.device:
        return core.DEVICE

    @property
    def device_name(self) -> str:
        if core.DEVICE.type == "cuda":
            return torch.cuda.get_device_name(core.DEVICE)
        return str(core.DEVICE)
