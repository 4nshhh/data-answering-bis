"""Canonical programmatic API for the BIS answering pipeline.

Repo 2 exposes ``retrieve(query, top_k=10)``; this package exposes the
Repo 3 counterpart::

    from app.generator import answer

    result = answer(
        "What is the minimum pH value of water for mixing concrete in IS 456?",
        top_k=3,
    )

``answer()`` executes the complete answering pipeline (masking,
retrieval, reranking, context assembly, prompting, generation,
citation verification, correction retry) by delegating to the single
canonical implementation, ``pipeline.run_query``. There is exactly one
pipeline; FastAPI (``app.main``) is a thin HTTP adapter around this
function.

This package is a standalone library: it never imports FastAPI or
Uvicorn. Expensive local retrieval resources (chunk index, BGE-M3
encoder, reranker, vector store) load lazily on first use, or eagerly
via :func:`warmup`::

    from app.generator import answer, warmup

    warmup()  # optional; no LLM call, no API key required

    result = answer(query, mode="ask")

``warmup()`` never builds an LLM provider and never touches the
network beyond loading local model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from app.generator.adapters import to_ask_response, to_match_response
from app.generator.context_builder import ChunkIndex, load_chunk_index
from app.generator.llm_client import LLMProvider, build_provider
from app.generator.pipeline import (
    DEFAULT_CANDIDATES_K,
    CitationOut,
    QueryResult,
    RetrievalMeta,
    Telemetry,
    run_query,
)
from app.generator.prompts import validate_mode
from app.generator.refusal import DEFAULT_THRESHOLD

__all__ = [
    "answer",
    "warmup",
    "WARMUP_QUERY",
    "reset_singletons",
    "to_ask_response",
    "to_match_response",
    "CitationOut",
    "QueryResult",
    "RetrievalMeta",
    "Telemetry",
    "DEFAULT_THRESHOLD",
    "DEFAULT_CANDIDATES_K",
]

#: Repository root (this file lives at ``<root>/app/generator/__init__.py``).
#: Default chunk paths anchor here — not to the process working
#: directory — so direct-library backends keep working no matter which
#: directory the host process starts from. Identical to the old
#: CWD-relative default when run from the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "chunks"

_shared_chunk_index: ChunkIndex | None = None
_shared_provider: LLMProvider | None = None


def reset_singletons() -> None:
    """Drop cached singletons (tests / explicit lifecycle control)."""
    global _shared_chunk_index, _shared_provider
    _shared_chunk_index = None
    _shared_provider = None


def _shared_index(chunks_dir: str | Path = DEFAULT_CHUNKS_DIR) -> ChunkIndex:
    global _shared_chunk_index
    if _shared_chunk_index is None:
        _shared_chunk_index = load_chunk_index(chunks_dir)
    return _shared_chunk_index


def _shared_llm_provider() -> LLMProvider:
    global _shared_provider
    if _shared_provider is None:
        _shared_provider = build_provider()
    return _shared_provider


#: Dummy query used by :func:`warmup` to force construction of the
#: shared ``retrieval.Retriever`` (BGE-M3 encoder + reranker + vector
#: store). Plain corpus-domain prose: no IS number (so no candidate
#: mask), no LLM involvement.
WARMUP_QUERY = "Bureau of Indian Standards specification"


def warmup(chunks_dir: str | Path = DEFAULT_CHUNKS_DIR) -> dict:
    """Preload expensive local retrieval resources (no LLM call).

    Loads the shared chunk index and constructs the shared
    ``retrieval`` retriever (BGE-M3 query encoder, CrossEncoder
    reranker, pre-computed vector store) so the first real query pays
    ~0.5s instead of ~40s model load. Idempotent: repeat calls reuse
    the already-loaded singletons.

    Never builds an LLM provider (no API key required) and never
    generates text. ``answer()`` stays safe to call without ``warmup()``
    — lazy initialization remains the fallback.

    Returns:
        ``{"chunks": <indexed chunk count>, "retriever_loaded": True}``.
    """
    index = _shared_index(chunks_dir)
    # Imported here (not at module top) to preserve this package's
    # dependency-free footprint; looked up at call time so tests can
    # substitute ``retrieval.retrieve`` without loading real models.
    from retrieval import loaded_retriever
    from retrieval import retrieve as _retrieve

    _retrieve(WARMUP_QUERY)
    return {"chunks": len(index.by_id), "retriever_loaded": loaded_retriever() is not None}


def answer(
    query: str,
    top_k: int = 3,
    *,
    expand_neighbors: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    candidates_k: int = DEFAULT_CANDIDATES_K,
    retrieve_fn: Callable[..., list] | None = None,
    chunk_index: Optional[ChunkIndex] = None,
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    telemetry: Optional[Telemetry] = None,
    mode: str = "ask",
) -> QueryResult:
    """Answer one BIS question with the complete RAG pipeline.

    Args mirror :func:`pipeline.run_query`; omitted ``chunk_index`` /
    ``provider`` fall back to process-wide lazy singletons (chunk index
    from ``data/chunks``, provider from ``LLM_PROVIDER``), while
    ``retrieve_fn`` falls back to a fresh ``retrieval.retrieve`` lookup
    per call (matching the FastAPI adapter's monkeypatch-friendly
    behavior). ``model`` defaults to the provider's ``default_model``.
    ``mode`` (``"ask"`` or ``"product_match"``) selects task-specific
    prompt instructions only; retrieval, verification, refusal, and
    retries are identical across modes.

    Returns:
        :class:`QueryResult` (answer, citations, refusal info,
        retrieval metadata, request-scoped telemetry).
    """
    validate_mode(mode)
    if retrieve_fn is None:
        from retrieval import retrieve as _retrieve

        retrieve_fn = _retrieve
    if chunk_index is None:
        chunk_index = _shared_index()
    if provider is None:
        provider = _shared_llm_provider()
    return run_query(
        query,
        top_k=top_k,
        expand_neighbors=expand_neighbors,
        threshold=threshold,
        candidates_k=candidates_k,
        retrieve_fn=retrieve_fn,
        chunk_index=chunk_index,
        provider=provider,
        model=model,
        telemetry=telemetry,
        mode=mode,
    )
