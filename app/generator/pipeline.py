"""Phase 8: end-to-end RAG pipeline orchestration (transport-agnostic).

Composes the frozen retrieval package with the Phase 3-7 generation
layer into a single ``run_query()`` call shared by the FastAPI service
(``app/main.py``) and the benchmark harness
(``evaluation/run_benchmark.py``):

```text
retrieve (top-10) -> pre-generation refusal check -> build_context
  -> build_prompt -> generate_answer -> verify_answer
  -> post-generation refusal check (citation grounding) -> QueryResult
```

The pre-generation check saves the LLM call when evidence is already
insufficient; the post-generation check enforces the negative
grounding rule (AGENTS.md section 11.3). Both refusal paths return the
canonical ``REFUSAL_TEXT`` with empty citations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from retrieval import retrieve
from retrieval.types import RetrievedEvidence

from app.generator.citations import verify_answer
from app.generator.context_builder import ChunkIndex, ContextBlock, build_context
from app.generator.llm_client import GeneratedAnswer, GroqProvider, LLMProvider, generate_answer
from app.generator.prompts import build_prompt
from app.generator.refusal import DEFAULT_THRESHOLD, REFUSAL_TEXT, evaluate_refusal

__all__ = [
    "DEFAULT_CANDIDATES_K",
    "CitationOut",
    "RetrievalMeta",
    "QueryResult",
    "run_query",
]

#: Dense+rerank candidate window passed to the reranker (frozen retrieval spec).
DEFAULT_CANDIDATES_K = 10


@dataclass
class CitationOut:
    """One citation for the API response (AGENTS.md section 12 schema)."""

    standard_no: str  # e.g. "IS 456"
    year: Optional[str]  # e.g. "2000"; None when not on record
    clause: str
    page: int
    chunk_id: str
    verified: bool  # Phase 6 verdict; extension over the base schema


@dataclass
class RetrievalMeta:
    """Retrieval diagnostics for the API response."""

    filtered_standard: Optional[str]  # top standard when the IS mask fired
    is_mask_restricted: bool
    candidates_retrieved: int
    top_reranker_score: Optional[float]
    execution_time_ms: float


@dataclass
class QueryResult:
    """Complete pipeline outcome for one user query."""

    query: str
    answer: str
    citations: list[CitationOut] = field(default_factory=list)
    retrieval_meta: Optional[RetrievalMeta] = None
    refused: bool = False
    refusal_reason: Optional[str] = None


def _record_year(
    block: Optional[ContextBlock],
    chunk_index: Optional[ChunkIndex],
) -> Optional[str]:
    """Record year for a linked block, else None (caller falls back)."""
    if block is None:
        return None
    if chunk_index is not None:
        chunk_dict = chunk_index.by_id.get(block.evidence.chunk_id)
        if chunk_dict is not None:
            year = chunk_dict.get("metadata", {}).get("year")
            if year:
                return str(year)
    extra = block.evidence.extra or {}
    if extra.get("year"):
        return str(extra["year"])
    return None


def _refused_result(
    query: str,
    reason: str,
    evidence: list[RetrievedEvidence],
    elapsed_ms: float,
) -> QueryResult:
    mask = bool(evidence) and bool(evidence[0].is_mask_restricted)
    top = evidence[0] if evidence else None
    return QueryResult(
        query=query,
        answer=REFUSAL_TEXT,
        citations=[],
        retrieval_meta=RetrievalMeta(
            filtered_standard=top.standard_no if (mask and top) else None,
            is_mask_restricted=mask,
            candidates_retrieved=len(evidence),
            top_reranker_score=top.rerank_score if top else None,
            execution_time_ms=elapsed_ms,
        ),
        refused=True,
        refusal_reason=reason,
    )


def run_query(
    query: str,
    *,
    top_k: int = 3,
    expand_neighbors: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    candidates_k: int = DEFAULT_CANDIDATES_K,
    retrieve_fn: Callable[..., list[RetrievedEvidence]] = retrieve,
    chunk_index: Optional[ChunkIndex] = None,
    provider: Optional[LLMProvider] = None,
) -> QueryResult:
    """Execute the full retrieval-to-answer pipeline for one query.

    Args:
        query: user question (non-blank).
        top_k: context blocks assembled (1-5 recommended).
        expand_neighbors: stitch ``chunk_index +/- 1`` context.
        threshold: refusal confidence threshold (tau).
        candidates_k: retrieval candidate window (default 10, frozen).
        retrieve_fn: injectable retrieval entry point (tests pass fakes).
        chunk_index: read-only chunk lookup for enrichment/expansion.
        provider: LLM backend; a ``GroqProvider`` is built when omitted
            (requires ``GROQ_API_KEY`` only on the answerable path).

    Raises:
        ValueError: blank query or invalid ``top_k``.
        RuntimeError: provider failures (Phase 8 maps these to HTTP 502).
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-blank string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")

    started = time.perf_counter()
    elapsed_ms = lambda: (time.perf_counter() - started) * 1000.0

    evidence = list(retrieve_fn(query.strip(), top_k=candidates_k))

    pre = evaluate_refusal(evidence, threshold=threshold)
    if pre.should_refuse:
        return _refused_result(query.strip(), pre.reason, evidence, elapsed_ms())

    if provider is None:
        provider = GroqProvider()
    context = build_context(evidence, chunk_index=chunk_index, top_k=top_k,
                            expand_neighbors=expand_neighbors)
    bundle = build_prompt(query.strip(), context, chunk_index=chunk_index)
    generated: GeneratedAnswer = generate_answer(query.strip(), bundle, provider)
    verified = verify_answer(generated, context, chunk_index)

    post = evaluate_refusal(evidence, threshold=threshold, verified=verified)
    if post.should_refuse:
        return _refused_result(query.strip(), post.reason, evidence, elapsed_ms())

    mask = bool(evidence[0].is_mask_restricted)
    blocks_by_id = {b.evidence.chunk_id: b for b in context.blocks}
    citations = [
        CitationOut(
            standard_no=f"IS {v.citation.standard_no}",
            year=_record_year(blocks_by_id.get(v.chunk_id or ""), chunk_index)
            or v.citation.year or None,
            clause=v.citation.clause,
            page=v.citation.page,
            chunk_id=v.chunk_id or "",
            verified=(v.verdict == "verified"),
        )
        for v in verified.citations
    ]
    return QueryResult(
        query=query.strip(),
        answer=verified.text,
        citations=citations,
        retrieval_meta=RetrievalMeta(
            filtered_standard=evidence[0].standard_no if mask else None,
            is_mask_restricted=mask,
            candidates_retrieved=len(evidence),
            top_reranker_score=evidence[0].rerank_score,
            execution_time_ms=elapsed_ms(),
        ),
        refused=False,
        refusal_reason=None,
    )
