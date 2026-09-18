"""CUDA backend: quantized gather kernels.

``gather_qmm`` is the one kernel the framework needs from this namespace
and the one piece of the backend that has no torch equivalent.  It is
not implemented yet; the function below documents and checks the
contract so that a wrong call fails as a ``ValueError`` and a correct
call fails as ``NotImplementedError`` — nothing falls back silently.

Layout (from ``docs/streaming.md`` and the MLX call sites in
``streaming/layer.py``), for ``transpose=True`` which is the only mode
the framework uses:

* ``w``       packed ``uint32`` ``[E, out, in/8]`` — 8 four-bit values per
              word (``bits=4``)
* ``scales``  ``bfloat16`` ``[E, out, in/group_size]``
* ``biases``  ``bfloat16`` ``[E, out, in/group_size]`` (affine offset)
* ``x``       ``[..., 1, in]`` (the framework calls ``expand_dims(x, (-2,-3))``
              before every call)
* ``rhs_indices`` ``[...]`` integer, one expert row of ``w`` per ``x`` row
* result      ``[..., 1, out]`` = ``x @ dequant(w[e]).T`` per row

``E`` is the number of SLOTS, not experts: the staged path passes
``staged_n + 1`` rows where the last row is an all-zero overflow slot and
must contribute exactly zero.  ``sorted_indices=True`` promises that equal
indices are contiguous, so a kernel may dequantize each expert once per
run instead of once per row.
"""

from __future__ import annotations

import torch


def _describe(x, w, scales, biases, rhs_indices) -> str:
    def d(t):
        return f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')}"
    return (f"x={d(x)}, w={d(w)}, scales={d(scales)}, biases={d(biases)}, "
            f"rhs_indices={d(rhs_indices)}")


def _check_layout(x, w, scales, biases, rhs_indices, transpose, group_size,
                  bits, mode):
    """Validate the packed layout so shape bugs surface as ValueError."""
    problems = []
    if mode != "affine":
        problems.append(f"mode={mode!r}: only 'affine' is supported")
    if bits != 4:
        problems.append(f"bits={bits}: only 4-bit packing is supported")
    if not transpose:
        problems.append("transpose=False is not used by the framework")
    if w.dtype != torch.uint32:
        problems.append(f"w must be packed uint32, got {w.dtype}")
    if w.ndim < 3:
        problems.append(f"w must be [E, out, in/8], got {tuple(w.shape)}")
    else:
        e, out, packed_in = w.shape[-3], w.shape[-2], w.shape[-1]
        in_dim = packed_in * (32 // bits)
        if x.shape[-1] != in_dim:
            problems.append(
                f"x last dim {x.shape[-1]} != in={in_dim} implied by w "
                f"[..., in/8={packed_in}]")
        groups = in_dim // group_size
        for name, t in (("scales", scales), ("biases", biases)):
            if t.dtype != torch.bfloat16:
                problems.append(f"{name} must be bfloat16, got {t.dtype}")
            if tuple(t.shape[-3:]) != (e, out, groups):
                problems.append(
                    f"{name} must be [E={e}, out={out}, in/group_size={groups}], "
                    f"got {tuple(t.shape)}")
        if rhs_indices.dtype not in (torch.int32, torch.int64, torch.uint32):
            problems.append(
                f"rhs_indices must be an integer tensor, got {rhs_indices.dtype}")
    if problems:
        raise ValueError(
            "gather_qmm layout mismatch:\n  - " + "\n  - ".join(problems)
            + "\n  call: " + _describe(x, w, scales, biases, rhs_indices))


def gather_qmm(x, w, scales, biases, rhs_indices, transpose=True,
               group_size=64, bits=4, mode="affine",
               sorted_indices=False):
    """Quantized matmul over a gathered subset of expert rows.

    Same signature and semantics as ``mx.gather_qmm`` as used by
    ``streaming/layer.py``.  NOT IMPLEMENTED YET — see module docstring
    and CUDA_BACKEND.md for the plan (reference unpack + dequant + bmm
    first, validated against the MLX CPU backend; fused Triton int4 qmv
    for decode after that).
    """
    _check_layout(x, w, scales, biases, rhs_indices, transpose, group_size,
                  bits, mode)
    raise NotImplementedError(
        "edge0 CUDA backend: quant.gather_qmm is not implemented yet.\n"
        f"  layout OK: {_describe(x, w, scales, biases, rhs_indices)}, "
        f"transpose={transpose}, group_size={group_size}, bits={bits}, "
        f"mode={mode!r}, sorted_indices={sorted_indices}\n"
        "  expected result: [..., 1, out] = x @ dequant(w[rhs_indices]).T "
        "with dequant(q) = scales * q + biases per group of "
        f"{group_size} along 'in'.\n"
        "  This is the kernel this backend exists to provide; nothing "
        "earlier in the stack is missing.  See CUDA_BACKEND.md.")


__all__ = ["gather_qmm"]
