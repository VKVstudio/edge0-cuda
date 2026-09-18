"""CUDA backend: model / tokenizer / tensor-store loading.

``load_tokenizer`` and ``open_shards`` are backend-independent and are
carried over from the MLX backend unchanged.  ``SafeTensorsStore`` and
``load_safetensors`` are the MLX versions with the final ``mx.array``
swapped for a torch tensor on ``core.DEVICE``.  ``load_model`` is the one
entry that cannot be ported: it wraps ``mlx_lm.utils.load_model`` and the
vendored MLX model classes, and this backend carries no model
implementation yet (CUDA_BACKEND.md).
"""

from __future__ import annotations

import json
import mmap
import struct

import numpy as np
import torch

from edge0.backends.base import TensorStore
from edge0.backends.cuda import core


def load_model(model_path, lazy=True, strict=False, model_config=None,
               get_model_classes=None):
    """Not available on the CUDA backend yet.

    On MLX this calls ``mlx_lm.utils.load_model`` with the vendored model
    classes from ``edge0.backends.mlx._impl`` (``get_model_classes``).
    A torch equivalent needs (a) torch implementations of the vendored
    models — ``bailing_hybrid`` for edge0-8b, ``qwen3_5`` for edge0-35b —
    with the edge0 hooks (``before_layer_cb``, ``prerouter_cache``), and
    (b) a loader that dequantizes the non-expert 4-bit weights into
    fp16/bf16 at load time.  Neither exists yet.
    """
    raise NotImplementedError(
        "edge0 CUDA backend: io.load_model is not implemented — the backend "
        "has no torch model implementation yet (the MLX models under "
        f"edge0/backends/mlx/_impl/ are MLX-only).  model_path={model_path!r}, "
        f"model_config={model_config!r}.  See CUDA_BACKEND.md, 'Status'.")


def load_tokenizer(model_path):
    """Load a HuggingFace tokenizer from a LOCAL directory only (same
    rationale as the MLX backend: no hub round trip, no
    ``trust_remote_code`` prompt)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True)


def open_shards(model_dir: str) -> list:
    """Open every ``model*.safetensors`` shard as a byte-range mmap
    (qwen-style multi-shard checkpoints; a single-file checkpoint yields
    a one-element list)."""
    import glob
    import os
    from edge0.streaming.mmap import SafetensorsMmap
    shards = []
    for path in sorted(glob.glob(os.path.join(
            os.fspath(model_dir), "model*.safetensors"))):
        shards.append(SafetensorsMmap(path))
    if not shards:
        raise FileNotFoundError(
            f"no model*.safetensors shards under {model_dir}")
    return shards


def load_safetensors(path: str, dtype=None) -> dict[str, torch.Tensor]:
    """Load every tensor of a (small) safetensors file as torch tensors on
    ``core.DEVICE``.  Used for adapter / prerouter weight files; ``dtype``
    converts (e.g. ``core.float16``)."""
    store = SafeTensorsStore(path)
    try:
        out = {}
        for name in store.keys():
            arr = store.get(name)
            if dtype is not None and arr.dtype != dtype:
                arr = arr.to(dtype)
            out[name] = arr
        return out
    finally:
        store.close()


class SafeTensorsStore(TensorStore):
    """mmap-backed safetensors store with per-tensor lazy access.

    ``get`` reads one tensor's bytes from the mmap and returns a torch
    tensor on ``core.DEVICE`` — a host-to-device copy on CUDA (there is no
    zero-copy path into device memory).  The whole file is never
    materialized.
    """

    def __init__(self, path: str):
        self._path = path
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        self._metadata = header.pop("__metadata__", {})
        self._entries = {
            name: {
                "offset": meta["data_offsets"][0] + 8 + header_len,
                "size": meta["data_offsets"][1] - meta["data_offsets"][0],
                "dtype": meta["dtype"],
                "shape": tuple(meta["shape"]),
            }
            for name, meta in header.items()
        }
        self._keys = sorted(self._entries)
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def path(self) -> str:
        return self._path

    def keys(self) -> list[str]:
        return list(self._keys)

    def metadata(self) -> dict:
        return dict(self._metadata)

    def get(self, name: str):
        e = self._entries.get(name)
        if e is None:
            raise KeyError(f"{self.path}: no tensor {name!r}")
        buf = np.frombuffer(self._mm, dtype=np.uint8, count=e["size"],
                            offset=e["offset"])
        np_dtype, bitcast = _DTYPES[e["dtype"]]
        arr = np.frombuffer(buf, dtype=np_dtype,
                            count=int(np.prod(e["shape"])) if e["shape"] else 1)
        t = core.array(arr.reshape(e["shape"]))
        if bitcast is not None:
            # numpy has no bfloat16: the bytes come in as int16 and are
            # reinterpreted (not converted) as bf16 on the torch side.
            t = t.view(bitcast)
        return t

    def close(self):
        self._mm.close()
        self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 — best-effort on interpreter exit
            pass


# safetensors dtype tag -> (numpy dtype to read the bytes as, torch dtype to
# bitcast to afterwards or None).  BF16 goes through int16 because numpy
# has no bfloat16; int16 -> bfloat16 is a same-width reinterpret.
_DTYPES = {
    "F64": (np.float64, None), "F32": (np.float32, None),
    "F16": (np.float16, None), "BF16": (np.int16, torch.bfloat16),
    "I64": (np.int64, None), "I32": (np.int32, None),
    "I16": (np.int16, None), "I8": (np.int8, None),
    "U8": (np.uint8, None), "BOOL": (np.bool_, None),
}
