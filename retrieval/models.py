"""Pinned model loaders — single source of truth for model identity.

CUDA is preferred whenever available; CPU is used as a loud (never
silent) fallback so developer laptops without a GPU still run. Model
identity (IDs, revisions, precision) is unchanged by the device choice.
"""

from __future__ import annotations

import os
import warnings

BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_MODEL = "BAAI/bge-reranker-base"
QUERY_BATCH_SIZE = 1

#: Explicit override, e.g. ``BIS_DEVICE=cpu``. Never a GPU name.
DEVICE_ENV_VAR = "BIS_DEVICE"


def resolve_device(device: str | None = None) -> str:
    """Resolve the execution device: CUDA when available, else CPU.

    Args:
        device: ``None`` (prefer CUDA), ``"cuda"``, or ``"cpu"``.
            ``BIS_DEVICE`` overrides ``None``. Requesting ``"cuda"``
            without CUDA falls back to ``"cpu"`` with a loud warning
            (graceful degradation, never a fatal error, never silent).

    Returns:
        ``"cuda"`` or ``"cpu"`` (sentence-transformers maps ``"cuda"``
        to the default GPU; no GPU name is ever hardcoded).
    """
    import torch

    want = (device or os.environ.get(DEVICE_ENV_VAR) or "cuda").strip().lower()
    if want.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        warnings.warn(
            "CUDA requested but torch.cuda.is_available() is False; "
            "falling back to CPU. Retrieval will be slower but identical "
            "in behavior (same models, same fp32 precision).",
            RuntimeWarning,
            stacklevel=3,
        )
        return "cpu"
    return "cpu"


def device_info() -> dict:
    """Safe runtime diagnostics (no keys, no secrets).

    GPU-name lookup is guarded by ``cuda.is_available()`` so this call
    never raises on CPU-only machines.
    """
    import torch

    available = bool(torch.cuda.is_available())
    info = {
        "cuda_available": available,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device_count": torch.cuda.device_count() if available else 0,
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "resolved_device": resolve_device(),
    }
    return info


def load_query_encoder(device: str | None = None, dtype: str = "float32"):
    """Pinned BGE-M3 sentence model for query encoding.

    Default dtype is float32 to match the validated benchmark runs
    (`reranker_benchmark.py` loads with library defaults); fp16 is
    available explicitly for offline bulk embedding. Query-side fp16/fp32
    differences are ~1e-4 cosine but can flip close rankings, so the
    serving default stays fp32. Device resolves via :func:`resolve_device`
    (CUDA preferred, loud CPU fallback).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    resolved = resolve_device(device)
    kwargs = {"torch_dtype": torch.float16} if dtype == "float16" else {}
    return SentenceTransformer(
        BGE_MODEL,
        revision=BGE_REVISION,
        device=resolved,
        model_kwargs=kwargs,
    )


def load_reranker(device: str | None = None):
    """Cross-encoder reranker over the SAME enriched representation.

    Loaded exactly as the frozen benchmark does (default precision —
    deliberately NOT fp16, so rerank scores match the validated runs).
    Device resolves via :func:`resolve_device` (CUDA preferred, loud
    CPU fallback).
    """
    from sentence_transformers import CrossEncoder

    resolved = resolve_device(device)
    return CrossEncoder(RERANKER_MODEL, device=resolved)
