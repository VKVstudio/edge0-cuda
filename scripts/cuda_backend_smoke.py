#!/usr/bin/env python
"""CUDA backend smoke: exercise the backend contract up to ``gather_qmm``.

    EDGE0_BACKEND=cuda python scripts/cuda_backend_smoke.py

(the script sets ``EDGE0_BACKEND=cuda`` itself when unset).

Every reachable check must pass.  The last step calls ``quant.gather_qmm``
with the exact shapes ``streaming/layer.py`` uses and is EXPECTED to raise
``NotImplementedError`` — that traceback, ending in
``backends/cuda/quant.py``, is the proof the scaffold is complete up to
the kernel and nothing earlier is missing.

Framework modules that cannot be imported on this backend today are
reported as BLOCKED together with the import that leaks MLX; they are
framework-side issues, tracked in CUDA_BACKEND.md, not backend gaps.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import tempfile

os.environ.setdefault("EDGE0_BACKEND", "cuda")
if os.environ["EDGE0_BACKEND"] != "cuda":
    sys.exit("this smoke targets the CUDA backend: set EDGE0_BACKEND=cuda")

import numpy as np  # noqa: E402

from edge0.backends import backend, core, io, nn, quant  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEG_INF = float("-inf")
_blocked: list[str] = []


def step(name: str) -> None:
    print(f"\n[smoke] {name}", flush=True)


def ok(msg: str) -> None:
    print(f"    ok       {msg}", flush=True)


def blocked(msg: str) -> None:
    _blocked.append(msg)
    print(f"    BLOCKED  {msg}", flush=True)


def close(a, b, rtol=1e-3) -> bool:
    return bool(core.allclose(a.astype(core.float32), b, rtol=rtol, atol=1e-3))


# ---- 1. backend identity ---------------------------------------------------

step("backend")
assert backend.name == "cuda", backend.name
ok(f"{backend.name}: {backend.version}; device {backend.device} "
   f"({backend.device_name})")

# ---- 2. core semantics the framework relies on ----------------------------

step("core: numpy/MLX calling conventions")
x = core.zeros((3, 8), dtype=core.float16)
assert isinstance(x, core.array)
assert core.expand_dims(x, (-2, -3)).shape == (3, 1, 1, 8)
ok("expand_dims with a tuple of axes -> [T,1,1,H]")

parts = core.split(core.arange(8), 2, axis=-1)
assert len(parts) == 2 and parts[0].tolist() == [0, 1, 2, 3]
ok("split(int) = number of equal sections")

v = core.array([3.0, 9.0, 1.0, 7.0])
assert core.topk(v, 2, axis=-1).tolist() == [7.0, 9.0]
ok("topk -> values only, ascending (sampling.py threshold = [..., 0])")

ap = core.argpartition(v, kth=-2, axis=-1)[..., -2:]
assert sorted(ap.tolist()) == [1, 3]
ok("argpartition(kth=-k)[..., -k:] selects the top-k set")

grouped = core.zeros((2, 4, 3))
drop = core.array([[0], [2]], dtype=core.int32)            # [B, k_drop]
masked = core.put_along_axis(
    grouped, core.expand_dims(drop, -1),
    core.array(NEG_INF, grouped.dtype), axis=-2)
assert masked[0, 0].tolist() == [NEG_INF] * 3
assert masked[0, 1].tolist() == [0.0] * 3
assert masked[1, 2].tolist() == [NEG_INF] * 3
ok("put_along_axis broadcasts a [.., k, 1] index across the group row")

assert core.take(core.array([10, 20, 30, 40]),
                 core.array([3, 0], dtype=core.int32)).tolist() == [40, 10]
oh = core.take(core.eye(4), core.array([[1, 3], [0, 2]], dtype=core.int32), axis=0)
assert oh.shape == (2, 2, 4) and oh.sum(axis=-2)[0].tolist() == [0, 1, 0, 1]
ok("take: flat gather without axis; multi-dim index with axis=0 (one-hot)")

bits = np.array([0x3F80, 0x4000, 0xC040], dtype="<u2")     # bf16 1.0, 2.0, -3.0
bf = core.array(bits).view(core.bfloat16)
assert bf.dtype == core.bfloat16
assert bf.astype(core.float32).tolist() == [1.0, 2.0, -3.0]
ok("core.array(uint16 view).view(bfloat16) bitcast (layer.py scales/biases path)")

packed = core.array(np.arange(16, dtype=np.uint8).view("<u4").reshape(2, 2))
assert packed.dtype == core.uint32 and packed.shape == (2, 2)
ok("core.array(uint8 view -> '<u4') packed weight path")

s = core.softmax(core.array([1.0, 2.0], dtype=core.float16), axis=-1, precise=True)
assert s.dtype == core.float16 and close(s.sum(), core.array(1.0))
w_ = core.where(core.array([True, False]), NEG_INF, core.array([1.0, 2.0]))
assert w_.tolist() == [NEG_INF, 2.0]
assert core.array([[1.0, 2.0], [3.0, 4.0]]).sum(axis=-1, keepdims=True).shape == (2, 1)
ok("softmax(precise=True) keeps dtype; where() with python scalars; .sum(axis=, keepdims=)")

acc = core.zeros((4,)).at[core.array([1, 1, 3], dtype=core.int32)].add(
    core.array([1.0, 1.0, 1.0]))
assert acc.tolist() == [0.0, 2.0, 0.0, 1.0]
ok(".at[idx].add(v) accumulates duplicates, out of place")

lg = core.zeros((5,))
lg[core.array([1, 3], dtype=core.int32)] = core.array([7.0, 8.0])
assert lg.tolist() == [0.0, 7.0, 0.0, 8.0, 0.0]
ok("in-place indexed assignment (sampling.py repetition penalty)")


@core.compile
def _twice(z):
    return z * 2


assert _twice(core.array([1.0])).tolist() == [2.0]
core.eval(x)
ok("compile is an identity decorator; eval is a no-op")

# ---- 3. framework: routing math -------------------------------------------

step("edge0.moe.routing")
from edge0.moe.routing import group_select_from_logits, select_from_logits  # noqa: E402

core.random.seed(0)
logits = core.random.normal((2, 3, 16), dtype=core.float16)
inds, scores = select_from_logits(logits, 4)
assert inds.shape == (2, 3, 4) and close(scores.sum(axis=-1), core.array(1.0))
ok("select_from_logits: [2,3,16] -> inds [2,3,4], scores renormalized to 1")

logits = core.random.normal((2, 3, 128), dtype=core.float16)
inds, w = group_select_from_logits(
    logits, top_k=8, n_group=8, topk_group=4, routed_scaling=2.5)
assert inds.shape == (2, 3, 8)
assert close(w.sum(axis=-1), core.array(2.5))
rows = inds.reshape(-1, 8).tolist()
assert all(len(set(r)) == 8 for r in rows), "duplicate experts in a row"
assert all(len({e // 16 for e in r}) <= 4 for r in rows), "more than topk_group groups"
ok("group_select_from_logits: 8 of 128 from <=4 of 8 groups, weights sum to 2.5")

# ---- 4. framework: sampler ------------------------------------------------

step("edge0.sampling")
from edge0.sampling import sample  # noqa: E402

logits = core.random.normal((1000,))
t1 = sample(logits, temperature=0.7, top_k=64, repetition_penalty=1.1,
            history=[1, 2, 3], seed=123)
t2 = sample(logits, temperature=0.7, top_k=64, repetition_penalty=1.1,
            history=[1, 2, 3], seed=123)
assert t1 == t2 and 0 <= t1 < 1000
ok(f"top-k + repetition penalty, seeded -> deterministic (token {t1})")
assert sample(logits, temperature=0.0) == int(core.argmax(logits).item())
ok("temperature=0 -> greedy")
try:
    sample(logits, temperature=0.7, top_p=0.95, seed=1)
    ok("top-p path")
except ValueError as e:
    blocked("top-p path: sampling.py:24 `core.sort(...)[..., ::-1]` -- torch "
            f"has no negative-stride slicing ({e})")

# ---- 5. framework: prerouter head (nn.Module / nn.Linear semantics) -------

step("edge0.prerouter.heads")
try:
    import edge0.prerouter  # noqa: F401
except ImportError as e:
    blocked(f"`import edge0.prerouter` -> {e.__class__.__name__}: {e} "
            "(prerouter/install.py imports edge0.backends.mlx directly)")
# The heads module itself only uses core/nn; load it by path to get past
# the package __init__ above and exercise the nn contract on real code.
_spec = importlib.util.spec_from_file_location(
    "edge0_prerouter_heads_smoke", ROOT / "src" / "edge0" / "prerouter" / "heads.py")
heads = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heads)

head = heads.PrerouterHead(hidden=32, num_experts=16, prerouter_hidden=8)
inds = core.array([[[1, 5, 9, 13]]], dtype=core.int32)          # [B=1,T=1,k=4]
oh = heads.topk_onehot(inds, 16)
assert oh.shape == (1, 1, 16) and oh.sum().item() == 4
h = core.random.normal((1, 1, 32), dtype=core.float16)
# Fresh nn.Linear weights are fp32 (as in MLX); fp16 features @ fp32
# weights promote to fp32 — MLX matmul semantics, reproduced by the
# backend rather than raising as torch would.
out = head(h, oh, oh)
assert out.shape == (1, 1, 16) and out.dtype == core.float32
# prerouter/install.py loads fp16 weights by plain attribute assignment.
for lin in (head.fc1, head.fc2, head.linear_init):
    lin.weight = lin.weight.astype(core.float16)
out2 = head(h, oh, oh)
assert out2.shape == (1, 1, 16) and out2.dtype == core.float16
assert bool(core.isfinite(out2).all())
assert set(head.parameters()) == {"fc1", "fc2", "linear_init"}
ok("PrerouterHead: fp16 x fp32 promotes (MLX semantics); fp16 after weight "
   "overwrite by assignment; parameters()")

for pkg, why in (("edge0.adapters", "adapters/lora.py imports edge0.backends.mlx.io"),
                 ("edge0.engine", "engine/{qwen,ling}.py bind the vendored MLX models")):
    try:
        __import__(pkg)
        ok(f"import {pkg}")
    except ImportError as e:
        blocked(f"`import {pkg}` -> {e.__class__.__name__}: {e} ({why})")

# ---- 6. io: tensor store round trip ---------------------------------------

step("edge0.backends.io")
import torch  # noqa: E402  — only to author a bf16 safetensors file
from safetensors.torch import save_file  # noqa: E402

tmp = tempfile.mkdtemp(prefix="edge0-cuda-smoke-")
path = os.path.join(tmp, "adapter.safetensors")
fc1 = np.random.default_rng(0).standard_normal((8, 32)).astype(np.float16)
save_file({
    "layers.7.fc1.weight": torch.from_numpy(fc1),
    "layers.7.scale": torch.tensor([1.0, 2.0, -3.0], dtype=torch.bfloat16),
    "layers.7.ids": torch.tensor([4, 5, 6], dtype=torch.int32),
}, path, metadata={"edge0": "smoke"})

store = io.SafeTensorsStore(path)
assert store.keys() == ["layers.7.fc1.weight", "layers.7.ids", "layers.7.scale"]
assert store.metadata() == {"edge0": "smoke"}
got = store.get("layers.7.fc1.weight")
assert got.dtype == core.float16 and got.shape == (8, 32)
assert np.array_equal(got.cpu().numpy(), fc1)
assert store.get("layers.7.scale").dtype == core.bfloat16
assert store.get("layers.7.scale").astype(core.float32).tolist() == [1.0, 2.0, -3.0]
assert store.get("layers.7.ids").tolist() == [4, 5, 6]
try:
    store.get("nope")
    raise AssertionError("KeyError expected")
except KeyError:
    pass
store.close()
ok("SafeTensorsStore: keys/metadata/get for F16, BF16 (bitcast), I32; KeyError on miss")

from edge0.backends.base import open_tensor_store  # noqa: E402
st = open_tensor_store(path)
assert isinstance(st, io.SafeTensorsStore) and st.path == path
st.close()
loaded = io.load_safetensors(path, dtype=core.float16)
assert {t.dtype for t in loaded.values()} == {core.float16}
ok("open_tensor_store() dispatches to the CUDA store; load_safetensors(dtype=) converts")

try:
    io.load_model(tmp)
    raise AssertionError("NotImplementedError expected")
except NotImplementedError as e:
    ok(f"load_model raises NotImplementedError: {str(e).splitlines()[0][:60]}...")

try:
    io.open_shards(tmp)
except ImportError as e:
    blocked(f"open_shards -> `import edge0.streaming` -> {e.__class__.__name__}: {e} "
            "(streaming/__init__.py imports layer.py, which imports mlx_lm at :49)")

# ---- 7. the kernel ---------------------------------------------------------

step("quant.gather_qmm — expected to raise NotImplementedError")
B, T, H, INTER, K, N_SLOTS = 1, 3, 64, 128, 4, 4
rng = np.random.default_rng(0)
xe = core.expand_dims(core.random.normal((B, T, H), dtype=core.float16), (-2, -3))
w_u = core.array(rng.integers(0, 2**32, size=(N_SLOTS + 1, INTER, H // 8), dtype=np.uint32))
s_u = core.array(rng.integers(0, 2**16, size=(N_SLOTS + 1, INTER, H // 64),
                              dtype=np.uint16)).view(core.bfloat16)
b_u = core.array(rng.integers(0, 2**16, size=(N_SLOTS + 1, INTER, H // 64),
                              dtype=np.uint16)).view(core.bfloat16)
local = core.array(rng.integers(0, N_SLOTS + 1, size=(B, T, K)), dtype=core.int32)

print(f"\n[smoke] reachable checks passed; {len(_blocked)} framework-side blocker(s):")
for b in _blocked:
    print(f"    - {b}")
print("\n[smoke] calling quant.gather_qmm with layer.py shapes "
      f"(x{tuple(xe.shape)}, w{tuple(w_u.shape)}, rhs_indices{tuple(local.shape)}) ...",
      flush=True)

quant.gather_qmm(xe, w_u, s_u, b_u, rhs_indices=local, transpose=True,
                 group_size=64, bits=4, mode="affine", sorted_indices=False)
