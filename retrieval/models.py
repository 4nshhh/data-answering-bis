"""Pinned model loaders — single source of truth for model identity.

CUDA/FP16 with explicit device (never silent CPU fallback); the caller
decides CPU operation explicitly if ever needed.
"""

from __future__ import annotations

BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_MODEL = "BAAI/bge-reranker-base"
QUERY_BATCH_SIZE = 1


def load_query_encoder(device: str = "cuda", dtype: str = "float32"):
    """Pinned BGE-M3 sentence model for query encoding.

    Default dtype is float32 to match the validated benchmark runs
    (`reranker_benchmark.py` loads with library defaults); fp16 is
    available explicitly for offline bulk embedding. Query-side fp16/fp32
    differences are ~1e-4 cosine but can flip close rankings, so the
    serving default stays fp32.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available; refusing silent CPU fallback.")
    kwargs = {"torch_dtype": torch.float16} if dtype == "float16" else {}
    return SentenceTransformer(
        BGE_MODEL,
        revision=BGE_REVISION,
        device=device,
        model_kwargs=kwargs,
    )


def load_reranker(device: str = "cuda"):
    """Cross-encoder reranker over the SAME enriched representation.

    Loaded exactly as the frozen benchmark does (default precision —
    deliberately NOT fp16, so rerank scores match the validated runs).
    """
    import torch
    from sentence_transformers import CrossEncoder

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available; refusing silent CPU fallback.")
    return CrossEncoder(RERANKER_MODEL, device=device)
