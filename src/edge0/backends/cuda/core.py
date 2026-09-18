"""CUDA backend: ``core`` namespace — array ops on top of PyTorch.

The MLX backend re-exports ``mlx.core`` as ``core`` unchanged.  Here the
namespace is assembled by hand so every entry keeps the numpy / MLX
calling convention the framework was written against: ``axis=`` rather
than ``dim=``, ``expand_dims`` with a tuple of axes, ``split`` by number
of sections, ``topk`` returning ascending values only, ``argpartition``
and ``put_along_axis`` with numpy semantics, and so on.  Where torch
differs, the wrapper adapts; where a construct has no eager equivalent
(``compile``, ``eval``, ``set_cache_limit``) the wrapper is an explicit,
documented no-op rather than a silent one.

Two things differ from MLX by nature and are NOT papered over:

* Tensors live on ``DEVICE`` (``cuda:0`` when available; ``EDGE0_CUDA_DEVICE``
  overrides, e.g. ``cpu`` for tests).  There is no unified memory, so
  ``core.array(numpy_view)`` is a real host-to-device copy.
* torch is eager.  ``eval`` and ``compile`` cannot reorder or fuse work;
  read/compute overlap has to come from CUDA streams instead.

Two array methods the framework uses are added to ``torch.Tensor`` at
import time because torch does not have them: ``astype`` and ``at`` (see
``_install_tensor_shims``).  ``.size`` cannot be added — torch already
defines it as a method — so framework code that reads ``x.size`` as an
element count does not work on this backend yet (CUDA_BACKEND.md).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

# ---- device ---------------------------------------------------------------

_DEVICE_ENV = os.environ.get("EDGE0_CUDA_DEVICE")
if _DEVICE_ENV:
    DEVICE = torch.device(_DEVICE_ENV)
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda", 0)
else:
    raise RuntimeError(
        "EDGE0_BACKEND=cuda but torch reports no CUDA device.  Install a "
        "CUDA build of torch, or set EDGE0_CUDA_DEVICE=cpu to run the torch "
        "backend on the CPU (tests / debugging only).")

# ---- dtypes ---------------------------------------------------------------

float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
bfloat16 = torch.bfloat16
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
uint8 = torch.uint8
uint16 = torch.uint16
uint32 = torch.uint32
bool_ = torch.bool

# ---- array construction ---------------------------------------------------


def _shape(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)


def _from_numpy(a: np.ndarray) -> torch.Tensor:
    """numpy -> torch on DEVICE.

    mmap-backed views are read-only and torch warns about wrapping them;
    the warning is about writes we never do, so it is silenced here.
    ``.to(DEVICE)`` is a copy on CUDA; on CPU we force a copy when the
    source is read-only so the result never aliases the mmap.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable")
        t = torch.from_numpy(a)
    return t.to(DEVICE, copy=not a.flags.writeable)


def _to_tensor(data, dtype=None) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        t = data.to(DEVICE)
    elif isinstance(data, np.ndarray):
        t = _from_numpy(data)
    else:
        # python scalars / lists; torch infers int64 for ints where MLX
        # would pick int32 — harmless for indexing, pass dtype= to pin it.
        t = torch.tensor(data, device=DEVICE)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t


class _ArrayMeta(type):
    """``core.array`` must be BOTH a constructor (``core.array(data, dtype)``)
    and a class usable in ``isinstance(x, core.array)`` and annotations,
    as ``mx.array`` is.  The metaclass makes the class object callable as
    a factory and makes every ``torch.Tensor`` an instance of it."""

    def __call__(cls, data, dtype=None):
        return _to_tensor(data, dtype)

    def __instancecheck__(cls, obj):
        return isinstance(obj, torch.Tensor)


class array(metaclass=_ArrayMeta):
    """Backend array type: ``torch.Tensor`` behind an MLX-shaped facade."""


def zeros(shape, dtype=float32):
    return torch.zeros(_shape(shape), dtype=dtype, device=DEVICE)


def ones(shape, dtype=float32):
    return torch.ones(_shape(shape), dtype=dtype, device=DEVICE)


def zeros_like(a):
    return torch.zeros_like(a)


def ones_like(a):
    return torch.ones_like(a)


def full(shape, vals, dtype=None):
    if isinstance(vals, torch.Tensor):
        out = vals.to(DEVICE).expand(_shape(shape)).clone()
        return out.to(dtype) if dtype is not None else out
    return torch.full(_shape(shape), vals, dtype=dtype, device=DEVICE)


def eye(n, m=None, dtype=float32):
    return torch.eye(n, m if m is not None else n, dtype=dtype, device=DEVICE)


def arange(start, stop=None, step=1, dtype=None):
    if stop is None:
        start, stop = 0, start
    return torch.arange(start, stop, step, dtype=dtype, device=DEVICE)


# ---- shape ops ------------------------------------------------------------


def reshape(a, shape):
    return a.reshape(_shape(shape))


def squeeze(a, axis=None):
    return a.squeeze() if axis is None else a.squeeze(axis)


def expand_dims(a, axis):
    """numpy semantics: ``axis`` may be a tuple and refers to positions in
    the OUTPUT shape, so ``expand_dims(x[T,H], (-2,-3))`` is ``[T,1,1,H]``."""
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    out_ndim = a.ndim + len(axes)
    for ax in sorted(ax + out_ndim if ax < 0 else ax for ax in axes):
        a = a.unsqueeze(ax)
    return a


def transpose(a, axes=None):
    if axes is None:
        axes = tuple(reversed(range(a.ndim)))
    return a.permute(tuple(axes))


def concatenate(arrays, axis=0):
    return torch.cat(list(arrays), dim=axis)


def stack(arrays, axis=0):
    return torch.stack(list(arrays), dim=axis)


def split(a, indices_or_sections, axis=0):
    """numpy semantics: an int is the NUMBER of equal sections (torch's
    ``split`` takes a chunk size instead); a list is split points."""
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if a.shape[axis] % n:
            raise ValueError(
                f"array split does not result in an equal division: "
                f"axis {axis} has size {a.shape[axis]}, sections={n}")
        return list(torch.tensor_split(a, n, dim=axis))
    return list(torch.tensor_split(a, list(indices_or_sections), dim=axis))


# ---- math -----------------------------------------------------------------


def promote(*tensors):
    """Cast every tensor operand to their common promoted dtype (MLX
    matmul semantics: ``fp16 @ fp32 -> fp32``); ``None`` passes through.
    torch's matmul / einsum / linear insist on equal dtypes instead."""
    present = [t for t in tensors if isinstance(t, torch.Tensor)]
    if len(present) < 2:
        return tensors
    dtype = present[0].dtype
    for t in present[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    return tuple(t.to(dtype) if isinstance(t, torch.Tensor) else t
                 for t in tensors)


def matmul(a, b):
    a, b = promote(a, b)
    return torch.matmul(a, b)


def einsum(subscripts, *operands):
    return torch.einsum(subscripts, *promote(*operands))


def softmax(a, axis=-1, precise=False):
    """``precise=True`` (MLX) computes in fp32 and casts back."""
    if precise and a.dtype in (torch.float16, torch.bfloat16):
        return torch.softmax(a.float(), dim=axis).to(a.dtype)
    return torch.softmax(a, dim=axis)


def sigmoid(a):
    return torch.sigmoid(a)


def erf(a):
    return torch.erf(a)


def exp(a):
    return torch.exp(a)


def log(a):
    return torch.log(a)


def sqrt(a):
    return torch.sqrt(a)


def rsqrt(a):
    return torch.rsqrt(a)


def abs(a):  # noqa: A001 — mirrors mx.abs
    return torch.abs(a)


def _scalar_like(x, ref):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def where(condition, x, y):
    return torch.where(condition, x, y)


def maximum(a, b):
    if not isinstance(a, torch.Tensor):
        a = _scalar_like(a, b)
    if not isinstance(b, torch.Tensor):
        b = _scalar_like(b, a)
    return torch.maximum(a, b)


def minimum(a, b):
    if not isinstance(a, torch.Tensor):
        a = _scalar_like(a, b)
    if not isinstance(b, torch.Tensor):
        b = _scalar_like(b, a)
    return torch.minimum(a, b)


# ---- reductions -----------------------------------------------------------


def _all_axes(a):
    return tuple(range(a.ndim))


def sum(a, axis=None, keepdims=False):  # noqa: A001 — mirrors mx.sum
    if axis is None:
        return a.sum(dim=_all_axes(a), keepdim=True) if keepdims else a.sum()
    return a.sum(dim=axis, keepdim=keepdims)


def mean(a, axis=None, keepdims=False):
    if axis is None:
        return a.mean(dim=_all_axes(a), keepdim=True) if keepdims else a.mean()
    return a.mean(dim=axis, keepdim=keepdims)


def max(a, axis=None, keepdims=False):  # noqa: A001 — mirrors mx.max
    if axis is None:
        return a.amax(dim=_all_axes(a), keepdim=True) if keepdims else a.amax()
    return a.amax(dim=axis, keepdim=keepdims)


def min(a, axis=None, keepdims=False):  # noqa: A001 — mirrors mx.min
    if axis is None:
        return a.amin(dim=_all_axes(a), keepdim=True) if keepdims else a.amin()
    return a.amin(dim=axis, keepdim=keepdims)


def argmax(a, axis=None, keepdims=False):
    return torch.argmax(a, dim=axis, keepdim=keepdims)


def cumsum(a, axis=None):
    if axis is None:
        return torch.cumsum(a.reshape(-1), dim=0)
    return torch.cumsum(a, dim=axis)


def any(a, axis=None, keepdims=False):  # noqa: A001 — mirrors mx.any
    if axis is None:
        return a.any()
    return a.any(dim=axis, keepdim=keepdims)


def all(a, axis=None, keepdims=False):  # noqa: A001 — mirrors mx.all
    if axis is None:
        return a.all()
    return a.all(dim=axis, keepdim=keepdims)


def isfinite(a):
    return torch.isfinite(a)


def allclose(a, b, rtol=1e-5, atol=1e-8):
    return torch.allclose(a, _scalar_like(b, a), rtol=rtol, atol=atol)


# ---- sorting / selection --------------------------------------------------


def sort(a, axis=-1):
    return torch.sort(a, dim=axis).values


def argsort(a, axis=-1):
    return torch.argsort(a, dim=axis, stable=True)


def topk(a, k, axis=-1):
    """MLX contract (relied on by ``sampling.py``): the k LARGEST values
    only — no indices — in ASCENDING order, so ``[..., 0]`` is the
    threshold."""
    return torch.topk(a, k, dim=axis, largest=True, sorted=True).values.flip(axis)


def argpartition(a, kth, axis=-1):
    """numpy contract: indices that put the ``kth`` element in its sorted
    place with smaller elements before and larger after.  A full stable
    argsort satisfies that contract, and the framework only ever slices
    the head or tail of the result, so this is exact — but the order
    INSIDE each partition differs from MLX's (which is unspecified anyway),
    and that is one reason routing is numerically equivalent to MLX rather
    than bit-identical."""
    del kth  # every valid kth is satisfied by a full sort
    return torch.argsort(a, dim=axis, stable=True)


def _index(idx) -> torch.Tensor:
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(np.asarray(idx), device=DEVICE)
    return idx.to(device=DEVICE, dtype=torch.long)


def take(a, indices, axis=None):
    """``axis=None`` gathers from the flattened array (numpy / MLX).  With
    ``axis``, the index dims replace that axis in the output."""
    idx = _index(indices)
    if axis is None:
        return a.reshape(-1)[idx]
    axis = axis % a.ndim
    out = torch.index_select(a, axis, idx.reshape(-1))
    return out.reshape(tuple(a.shape[:axis]) + tuple(idx.shape)
                       + tuple(a.shape[axis + 1:]))


def _broadcast_index(a, idx, axis):
    """numpy's take/put_along_axis broadcast size-1 index dims against the
    array on every axis but ``axis``; torch's gather/scatter do not."""
    shape = list(idx.shape)
    for d in range(a.ndim):
        if d != axis and idx.shape[d] == 1 and a.shape[d] != 1:
            shape[d] = a.shape[d]
    return idx.expand(shape)


def take_along_axis(a, indices, axis):
    axis = axis % a.ndim
    idx = _broadcast_index(a, _index(indices), axis)
    return torch.gather(a, axis, idx)


def put_along_axis(a, indices, values, axis):
    """Out-of-place (MLX returns a new array).  A 0-d ``values`` is
    broadcast to every indexed element, like numpy."""
    axis = axis % a.ndim
    idx = _broadcast_index(a, _index(indices), axis)
    if isinstance(values, torch.Tensor) and values.ndim == 0:
        values = values.item()
    if isinstance(values, torch.Tensor):
        return a.scatter(axis, idx, src=values.to(a.dtype).expand(idx.shape))
    return a.scatter(axis, idx, value=values)


# ---- conversion -----------------------------------------------------------


def astype(a, dtype):
    return a.to(dtype)


def item(a):
    return a.item()


def tolist(a):
    return a.tolist()


# ---- evaluation / compilation / memory ------------------------------------


def eval(*args):  # noqa: A001 — mirrors mx.eval
    """No-op.  torch is eager: by the time ``eval`` is called the work has
    been issued.  MLX code uses ``eval`` both to force lazy graphs AND to
    flush in-place writes before a lazy consumer reads them
    (``streaming/layer.py`` ``FIX(2026-09-02)``); neither concern exists
    here.  Host reads (``item``/``tolist``) synchronize on their own."""
    del args


def compile(fun=None, *args, **kwargs):  # noqa: A001 — mirrors mx.compile
    """Identity decorator.  ``streaming/layer.py`` applies ``@core.compile``
    unconditionally in ``__init__``; the un-compiled closures run eagerly
    and are correct.  ``torch.compile`` is a later, separate step — one of
    the compiled closures takes a python tuple argument, which would need
    reshaping first."""
    del args, kwargs
    if fun is None:
        return lambda f: f
    return fun


def set_cache_limit(limit: int) -> int:
    """No-op, returns 0.  MLX caps its allocator cache with this; torch's
    caching allocator is configured through ``PYTORCH_CUDA_ALLOC_CONF``
    and has no per-process cap call.  ``engine/base.py`` already treats
    this call as best-effort."""
    del limit
    return 0


def get_peak_memory() -> int:
    if DEVICE.type != "cuda":
        return 0
    return torch.cuda.max_memory_allocated(DEVICE)


def reset_peak_memory() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)


def get_active_memory() -> int:
    if DEVICE.type != "cuda":
        return 0
    return torch.cuda.memory_allocated(DEVICE)


# ---- random ---------------------------------------------------------------


class _Random:
    """``core.random`` — global torch RNG.  Streams differ from MLX's, so
    seeded sequences match run-to-run on this backend, not across
    backends."""

    @staticmethod
    def seed(seed: int) -> None:
        torch.manual_seed(seed)

    @staticmethod
    def categorical(logits, axis=-1, num_samples=None):
        """One draw per row from UNNORMALIZED logits along ``axis``."""
        if axis not in (-1, logits.ndim - 1):
            logits = logits.movedim(axis, -1)
        probs = torch.softmax(logits.float(), dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        n = 1 if num_samples is None else num_samples
        draws = torch.multinomial(flat, n)
        tail = (n,) if num_samples is not None else ()
        return draws.reshape(tuple(probs.shape[:-1]) + tail).to(torch.int32)

    @staticmethod
    def normal(shape, dtype=float32, loc=0.0, scale=1.0):
        return torch.normal(loc, scale, size=_shape(shape), device=DEVICE).to(dtype)

    @staticmethod
    def uniform(low=0.0, high=1.0, shape=(), dtype=float32):
        u = torch.rand(_shape(shape), device=DEVICE, dtype=torch.float32)
        return (low + (high - low) * u).to(dtype)


random = _Random()

# ---- torch.Tensor shims ---------------------------------------------------


class _At:
    """``x.at[idx].add(v)`` — MLX's out-of-place scatter-add."""

    def __init__(self, tensor, idx):
        self._t = tensor
        self._idx = idx if isinstance(idx, tuple) else (idx,)

    def add(self, v):
        out = self._t.clone()
        idx = tuple(
            _index(i) if isinstance(i, (torch.Tensor, list, np.ndarray)) else i
            for i in self._idx)
        v = _scalar_like(v, out).to(out.dtype)
        return out.index_put_(idx, v, accumulate=True)


class _AtIndexer:
    def __init__(self, tensor):
        self._t = tensor

    def __getitem__(self, idx):
        return _At(self._t, idx)


def _install_tensor_shims() -> None:
    """Add the two MLX array methods the framework uses that torch lacks.
    Only names torch does not define are added, so this cannot shadow
    torch behaviour."""
    if not hasattr(torch.Tensor, "astype"):
        torch.Tensor.astype = lambda self, dtype: self.to(dtype)
    if not hasattr(torch.Tensor, "at"):
        torch.Tensor.at = property(_AtIndexer)


_install_tensor_shims()
