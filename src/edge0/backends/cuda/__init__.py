"""CUDA / PyTorch backend (work in progress).

Implements the backend contract documented in ``edge0.backends`` on top
of torch: ``core`` (array ops), ``nn`` (module factory), ``io`` (tensor
stores, tokenizer, shards), ``quant`` (``gather_qmm`` — not implemented
yet).  Selected with ``EDGE0_BACKEND=cuda``.  Status and plan:
``CUDA_BACKEND.md`` at the repository root.
"""
