"""CUDA backend: ``nn`` namespace — a minimal module factory.

Deliberately NOT ``torch.nn``.  The framework relies on MLX module
semantics that ``torch.nn.Module`` rejects:

* parameters are plain array attributes (``self.lora_A = array`` in
  ``adapters/lora.py``) — no ``Parameter`` wrapping, no registration;
* weights are overwritten by assignment (``head.fc1.weight = w`` in
  ``prerouter/install.py``) — ``torch.nn`` raises ``TypeError`` when a raw
  tensor is assigned over a ``Parameter``;
* modules are invoked through ``__call__``, not ``forward``.

Only what the framework exercises is provided: ``Module``, ``Linear``,
``RMSNorm``, ``silu``, ``gelu``.  The vendored MLX models use more of
``mlx.nn``; a torch model implementation is a separate piece of work.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from edge0.backends.cuda import core


class Module:
    """Attribute-bag module: any ``torch.Tensor`` attribute is a parameter,
    any ``Module`` attribute is a submodule."""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement __call__")

    def parameters(self) -> dict:
        """Nested dict of tensors, keyed by attribute name (MLX layout)."""
        out = {}
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                out[name] = value
            elif isinstance(value, Module):
                out[name] = value.parameters()
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Linear(Module):
    """``y = x @ weight.T + bias``; ``weight`` is ``[out, in]`` as in MLX
    and in the safetensors adapters the framework loads into it."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = core.random.uniform(-scale, scale, (output_dims, input_dims))
        if bias:
            self.bias = core.random.uniform(-scale, scale, (output_dims,))

    def __call__(self, x):
        # MLX promotes mixed dtypes in matmul (fp16 x @ fp32 weight -> fp32);
        # torch's F.linear requires equal dtypes, so promote explicitly.
        bias = getattr(self, "bias", None)
        x, w, bias = core.promote(x, self.weight, bias)
        return F.linear(x, w, bias)


class RMSNorm(Module):
    """RMS normalization computed in fp32 and cast back to the input dtype,
    matching ``mx.fast.rms_norm``."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = core.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def silu(x):
    return F.silu(x)


def gelu(x):
    """Exact (erf) GELU, as ``mlx.nn.gelu``."""
    return F.gelu(x)


__all__ = ["Module", "Linear", "RMSNorm", "silu", "gelu"]
