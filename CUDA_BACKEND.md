# edge0 — CUDA/PyTorch backend (work in progress)

edge0 runs on Apple Silicon only, through MLX. `backends/cuda/` is
reserved in upstream and empty. This branch is an attempt to fill it.

Target hardware: RTX 4080 Laptop, 12 GB VRAM, 32 GB RAM, NVMe.
Development environment so far: Windows 11, Python 3.12, torch 2.11.0+cu128.

## Why this is not trivial

MLX's own CUDA backend does not implement affine-quantized matmul. In
`mlx/backend/cuda/quantized/quantized.cpp` (v0.30.6):

    if (M > 8 || !transpose_ || mode_ == QuantizationMode::Affine) {
        throw std::runtime_error("QMM NYI");
    }

edge0 checkpoints are affine 4-bit (group size 64), so "just run MLX on
CUDA" is not a path. The gather-matmul kernel has to be written.

Two more things do not port and have to be rebuilt rather than translated:

- MLX is lazy; torch is eager. The streaming layer's pool threads build
  lazy arrays and never evaluate them (`streaming/layer.py:543-546`), and
  `core.eval` is used for correctness, not speed (`layer.py:694-699`). In
  torch a thread that moves an expert to the GPU does a synchronous copy
  on the critical path. Read/compute overlap has to come from CUDA
  streams, pinned buffers and events.
- Apple unified memory makes `mx.array(numpy_view)` almost free. On CUDA
  every such call is a host-to-device copy over PCIe. `materialize_hot`,
  `load_full_layer` and the memory arithmetic around them assume the
  former.

## Status

Verified by `scripts/cuda_backend_smoke.py` (see "Running" below).

Working:

- `backends/cuda/{core,nn,io}.py` — torch-backed implementation of the
  backend contract documented in `backends/__init__.py`, with the
  numpy/MLX calling conventions the framework relies on (details in the
  module docstrings and `git log`).
- `EDGE0_BACKEND=cuda` selects it.
- Real framework code runs on it: `moe/routing.py` (both routers),
  `sampling.py` (top-k, repetition penalty, greedy), `prerouter/heads.py`
  (`PrerouterHead` forward, fp16), `backends/base.py` tensor stores.
- `pip install -e .` works on Windows (mlx gated to non-Windows in
  `pyproject.toml`).

Not working yet:

- `quant.gather_qmm` — raises `NotImplementedError`. The stub validates
  the packed layout, so a wrong call is a `ValueError` and a correct one
  is the expected `NotImplementedError` with the shapes it received.
- `io.load_model` — raises `NotImplementedError`. There is no torch
  implementation of the vendored models (`bailing_hybrid` for edge0-8b,
  `qwen3_5` for edge0-35b); that is a separate piece of work, see "Next".

Blocked on the framework side (not backend gaps; each is a small change
outside `backends/`, left untouched on this branch on purpose):

| What | Where | Why |
|---|---|---|
| `import edge0.streaming` | `streaming/layer.py:49` | imports `_gather_sort` / `_scatter_unsort` from `mlx_lm` |
| `import edge0.prerouter` | `prerouter/install.py:23-24` | imports `edge0.backends.mlx.*` directly; `load_safetensors` is on the facade |
| `import edge0.adapters` | `adapters/lora.py:31` | same, `edge0.backends.mlx.io` for `load_safetensors` |
| `import edge0.engine` | `engine/{qwen,ling}.py` | bind the vendored MLX model classes (documented exception in `docs/architecture.md`) |
| top-p sampling | `sampling.py:24` | `core.sort(...)[..., ::-1]` — torch has no negative-stride slicing |
| `x.size` as element count | `streaming/layer.py:1121,1231,1320,1384`, `sampling.py:52` | torch defines `.size` as a method; cannot be shimmed |

## Running

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install -e ".[dev,cuda]"
    $env:EDGE0_BACKEND = "cuda"
    python scripts\cuda_backend_smoke.py

Expected end state: every reachable check prints `ok`, the five blockers
above print `BLOCKED` with the leaking import, and the run ends with

    NotImplementedError: edge0 CUDA backend: quant.gather_qmm is not implemented yet.
      layout OK: x=(1, 3, 1, 1, 64) float16, w=(5, 128, 8) uint32, ...

raised from `backends/cuda/quant.py`. That traceback is the point: the
scaffold is complete up to the kernel and nothing earlier is missing.

`tests/test_repo_hygiene.py` passes on this branch. The rest of the test
suite is written directly against `mlx.core` and does not run here yet.

## Design notes

- `core` adapts semantics instead of aliasing torch: `expand_dims` takes a
  tuple of output-shape axes, `split(int)` is a number of sections,
  `topk` returns ascending values only, `argpartition` is a stable
  argsort, `take`/`put_along_axis` broadcast size-1 index dims,
  `softmax(precise=True)` computes in fp32, `matmul`/`einsum`/`Linear`
  promote mixed dtypes as MLX does.
- `core.array` is a class whose metaclass makes it both a constructor and
  the target of `isinstance(x, core.array)` (`streaming/install.py:103`).
- `nn` is not `torch.nn`: parameters are plain tensor attributes and are
  overwritten by assignment (`prerouter/install.py:128-136`), which
  `torch.nn.Module` refuses.
- `compile`, `eval` and `set_cache_limit` are explicit, documented
  no-ops. `torch.compile` is a later step — one of the compiled closures
  in `streaming/layer.py` takes a python tuple argument.
- `.astype` and `.at[idx].add(v)` are added to `torch.Tensor` at import
  because the framework uses them and torch lacks them.
- Routing on this backend is numerically equivalent to MLX, not
  bit-identical: the order inside an `argpartition` partition differs,
  which changes the fp16 accumulation order downstream. The project's own
  tests gate on relative L2 < 1 %, not on equality.

## Next

1. Reference `gather_qmm`: unpack 4-bit affine (group 64), dequant
   `scales * q + biases`, batched matmul over gathered experts.
   Correctness first, speed later. Oracle: `mx.gather_qmm` on the MLX CPU
   backend (mlx ships Linux x86_64 wheels) — to be confirmed on a Linux
   box before relying on it; fallback oracle is `mx.dequantize` plus a
   plain matmul.
2. Speed, split by regime: fused Triton int4 qmv for decode (memory
   bound, `sorted_indices` grouping), dequant-once plus cuBLAS for prefill
   (compute bound).
3. Move `_gather_sort` / `_scatter_unsort` into the framework so
   `streaming/layer.py` imports on this backend, then port the layer's
   eager path (`layer.py:1397-1425` is the branch that does not depend on
   `compile`).
4. Rebuild the prefetch path on CUDA streams. Metal exposes no explicit
   copy engines and CUDA does, so this should end up cleaner than the
   reference rather than merely equivalent.
5. Port `tests/test_streaming_math.py` to `edge0.backends`; it writes its
   own synthetic 4-bit shards and covers all four streaming paths without
   a checkpoint.

A torch implementation of the base models is deliberately after all of
the above. The HF remote code for Ling 3.0 depends on `fla`/triton — the
reason it had to be vendored for MLX (`bailing_hybrid.py:16-17`) — and on
CUDA that dependency is available, so it may be reusable; unverified.

## Notes on upstream

- `streaming/layer.py:49` imports `_gather_sort` / `_scatter_unsort` from
  `mlx_lm`, which violates the backend boundary documented in
  `backends/__init__.py`. The CI guard in `tests/test_repo_hygiene.py`
  misses it: the regex requires a space or dot after `mlx`, and `mlx_lm`
  has an underscore. Fix submitted as a separate PR (branch
  `fix/hygiene-mlx-lm-regex`).
- `adapters/lora.py:31` and `prerouter/install.py:24` import
  `edge0.backends.mlx.io` for `load_safetensors`, which the
  `edge0.backends.io` facade already provides.
- `pyproject.toml` declares `mlx` and `mlx-lm` unconditionally; only
  `mlx-metal` is platform-gated, so installation fails on Windows.
  `transformers` and `jinja2` are used directly but not declared. Fixed
  on this branch (first commit).
- `backends/mlx/quant.py:24` `swiglu` is not imported anywhere.
- Docstring/doc drift: `models/edge0_8b/__init__.py:11-13` says 22 heads
  from layer 1, the code (`:56-59`) and `tests/test_registry.py:80-81` say
  16 heads from layer 7; `docs/models/edge0-8b.md:35` says staged decode
  is off, `LayerOptions.prod_k8()` and `tests/test_registry.py:36` say on.
