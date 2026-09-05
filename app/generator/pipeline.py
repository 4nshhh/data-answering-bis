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
from app.generator.context_builder import BuiltContext, ChunkIndex, ContextBlock, build_context
from app.generator.llm_client import GeneratedAnswer, GroqProvider, LLMProvider, generate_answer
from app.generator.prompts import DEFAULT_MODEL, build_prompt, validate_mode
from app.generator.refusal import DEFAULT_THRESHOLD, REFUSAL_TEXT, evaluate_refusal
from app.generator.telemetry import Telemetry

__all__ = [
    "DEFAULT_CANDIDATES_K",
    "CitationOut",
    "RetrievalMeta",
    "QueryResult",
    "Telemetry",
    "run_query",
]

#: Dense+rerank candidate window passed to the reranker (frozen retrieval spec).
DEFAULT_CANDIDATES_K = 10

#: Fallback context widening on abstention: when the first generation pass
#: yields zero parseable citations (the model itself reports the evidence
#: as insufficient, as in table-heavy standards where the table chunk
#: ranks just below top-K), retry once with a wider window. Strictly
#: monotonic — the wider pass replaces the first only when it fully
#: verifies; otherwise the first-pass result stands byte-identical.
WIDEN_STEP = 3

#: Generic retrieval-expansion suffix for unmasked low-confidence queries
#: (product-applicability questions naming no IS number). Same confidence
#: threshold applies to the expanded retrieval; it only changes which
#: evidence is considered, never the bar for answering. Contains no
#: standard names or query-specific terms.
EXPANSION_SUFFIX = " Indian Standard specification requirements scope"

#: Lower bound of the near-miss band licensing the expansion above.
#: The suffix matches scope-defining chunks on its own strength, so an
#: unrestricted expansion could launder any query (even clearly
#: out-of-corpus ones) into an answerable one. The expansion therefore
#: fires only when the ORIGINAL query already shows borderline affinity
#: to the corpus (top score in [EXPANSION_MIN_SCORE, threshold)):
#: product phrasing that nearly matches earns a
#: second opinion; rock-bottom scores (out-of-corpus probes at ~0.03)
#: stay refused on the first verdict. General and query-agnostic.
EXPANSION_MIN_SCORE = 0.20

#: Phrases marking an honest abstention (the model itself reports the
#: blocks lack the answer). A correction retry is never pressured onto
#: these: the widen path already handles coverage, and pressuring an
#: abstention risks a verified-but-unfounded claim. Non-abstention
#: answers with zero parseable citations (e.g. non-bracket markers)
#: are retry-eligible instead.
_ABSTENTION_MARKERS = frozenset(
    {
        "do not contain",
        "does not contain",
        "cannot be determined",
        "cannot be provided",
        "not given in",
        "not included in",
        "insufficient technical information",
        "no specification",
        "not specify",
        "not state",
    }
)


def _looks_like_abstention(text: str) -> bool:
    """True when the answer reads as an honest insufficient-evidence report."""
    lowered = text.lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


def _correction_bundle(
    query: str,
    context: BuiltContext,
    chunk_index: Optional[ChunkIndex],
    feedback: str,
    mode: str = "ask",
):
    """Rebuild the prompt with verifier feedback appended (one retry only).

    The honest exit stays open: when the blocks lack the answer the
    model must still abstain plainly without citing. Verification is
    re-applied unchanged, so a failed retry can never leak through.
    """
    from dataclasses import replace

    retry_bundle = build_prompt(query.strip(), context, chunk_index=chunk_index, mode=mode)
    retry_bundle = replace(
        retry_bundle,
        user=retry_bundle.user
        + "\n\nCorrection required before answering: "
        + feedback
        + " If the evidence blocks above do not contain the answer, "
        "reply in one plain sentence WITHOUT any citation.",
    )
    return retry_bundle


@dataclass
class CitationOut:
    """One citation for the API response (AGENTS.md section 12 schema)."""

    standard_no: str  # e.g. "IS 456"
    year: Optional[str]  # e.g. "2000"; None when not on record
    clause: str
    page: int
    chunk_id: str
    verified: bool  # Phase 6 verdict; extension over the base schema
    rerank_score: Optional[float] = None  # linked evidence score, for ranking only


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
    telemetry: Optional[Telemetry] = None  # observability only; never affects behavior


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
    telemetry: Optional[Telemetry] = None,
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
        telemetry=telemetry,
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
    model: Optional[str] = None,
    telemetry: Optional[Telemetry] = None,
    mode: str = "ask",
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
        model: generation model identifier; defaults to the provider's
            ``default_model`` (Groq historical default preserved).
        telemetry: request-scoped counters; a fresh instance is used
            when omitted. Observability only — never affects behavior.
        mode: task mode (``"ask"`` or ``"product_match"``); selects
            mode-specific prompt instructions only. Retrieval,
            verification, refusal, and retries are identical.

    Raises:
        ValueError: blank query, invalid ``top_k``, or unknown mode.
        RuntimeError: provider failures (Phase 8 maps these to HTTP 502).
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-blank string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")
    validate_mode(mode)
    if telemetry is None:
        telemetry = Telemetry()
    telemetry.mode = mode

    started = time.perf_counter()
    elapsed_ms = lambda: (time.perf_counter() - started) * 1000.0

    _t0 = time.perf_counter()
    evidence = list(retrieve_fn(query.strip(), top_k=candidates_k))
    telemetry.add_stage("retrieval_ms", (time.perf_counter() - _t0) * 1000.0)

    pre = evaluate_refusal(evidence, threshold=threshold)
    if pre.should_refuse and pre.reason == "below_threshold" and not pre.is_mask_restricted:
        # Product-applicability fallback: the query names no IS number and
        # generic phrasing scores low against technical chunks even when
        # the right standard was retrieved. Re-retrieve once with a
        # generic scope-seeking expansion (no standard names, no
        # query-specific terms) and apply the SAME threshold — a
        # different selection strategy, not a lowered bar. Gated to the
        # near-miss band (see EXPANSION_MIN_SCORE): clearly
        # out-of-corpus queries stay refused on the first verdict.
        # Out-of-corpus queries still score ~0.00-0.03 and stay refused.
        top1 = pre.top_score if pre.top_score is not None else float("-inf")
        if EXPANSION_MIN_SCORE <= top1 < threshold:
            telemetry.retrieval_expansion = True
            _t0 = time.perf_counter()
            expanded = list(retrieve_fn(query.strip() + EXPANSION_SUFFIX, top_k=candidates_k))
            telemetry.add_stage("retrieval_expansion_ms", (time.perf_counter() - _t0) * 1000.0)
            if evaluate_refusal(expanded, threshold=threshold).reason == "ok":
                evidence = expanded
                pre = evaluate_refusal(evidence, threshold=threshold)
    if pre.should_refuse:
        telemetry.latency_ms = elapsed_ms()
        return _refused_result(query.strip(), pre.reason, evidence, elapsed_ms(), telemetry)

    if provider is None:
        provider = GroqProvider()
    resolved_model = model or getattr(provider, "default_model", None) or DEFAULT_MODEL
    _t0 = time.perf_counter()
    context = build_context(evidence, chunk_index=chunk_index, top_k=top_k,
                            expand_neighbors=expand_neighbors)
    telemetry.add_stage("context_ms", (time.perf_counter() - _t0) * 1000.0)
    _t0 = time.perf_counter()
    bundle = build_prompt(query.strip(), context, chunk_index=chunk_index, model=resolved_model, mode=mode)
    telemetry.add_stage("prompt_ms", (time.perf_counter() - _t0) * 1000.0)
    telemetry.prompt_chars = len(bundle.user)
    _t0 = time.perf_counter()
    generated: GeneratedAnswer = generate_answer(query.strip(), bundle, provider, telemetry=telemetry)
    telemetry.add_stage("gen_initial_ms", (time.perf_counter() - _t0) * 1000.0)
    _t0 = time.perf_counter()
    verified = verify_answer(generated, context, chunk_index)
    telemetry.add_stage("verify_ms", (time.perf_counter() - _t0) * 1000.0)

    if not verified.has_citations and len(evidence) > top_k:
        # Abstention recovery: the model found nothing citable in top-K
        # (typical when a table chunk ranks just below the cutoff).
        # One wider retry over already-retrieved evidence — no new
        # retrieval, no threshold change, verification still enforced.
        telemetry.widen_retry = True
        wider_k = min(len(evidence), top_k + WIDEN_STEP)
        _t0 = time.perf_counter()
        wider_context = build_context(evidence, chunk_index=chunk_index,
                                      top_k=wider_k, expand_neighbors=expand_neighbors)
        wider_bundle = build_prompt(query.strip(), wider_context, chunk_index=chunk_index, mode=mode)
        telemetry.add_stage("context_widen_ms", (time.perf_counter() - _t0) * 1000.0)
        _t0 = time.perf_counter()
        wider_generated: GeneratedAnswer = generate_answer(query.strip(), wider_bundle, provider, telemetry=telemetry)
        telemetry.add_stage("gen_widen_ms", (time.perf_counter() - _t0) * 1000.0)
        _t0 = time.perf_counter()
        wider_verified = verify_answer(wider_generated, wider_context, chunk_index)
        telemetry.add_stage("verify_ms", (time.perf_counter() - _t0) * 1000.0)
        if wider_verified.all_verified:
            context, bundle, generated, verified = (
                wider_context, wider_bundle, wider_generated, wider_verified,
            )

    post = evaluate_refusal(evidence, threshold=threshold, verified=verified)
    if post.should_refuse and post.reason == "citation_mismatch":
        # Correction retry: the model attempted citations but linked them
        # wrongly (variance-induced refusal despite strong evidence).
        # One regeneration with verifier feedback on the SAME context;
        # adoption requires full verification, else the refusal stands.
        mismatches = [v for v in verified.citations if v.verdict == "mismatch"]
        feedback = (
            "your previous answer failed citation verification (" +
            "; ".join(v.detail for v in mismatches) +
            "). Rewrite using ONLY the exact canonical citation format "
            "[IS <standard_no>:<year>, Clause <clause>, Page <page>] copied "
            "from the Standard/Clause/Location headers above, citing only "
            "sub-clause numbers actually shown in the blocks."
        )
        retry_bundle = _correction_bundle(query.strip(), context, chunk_index, feedback, mode)
        telemetry.correction_retry = True
        _t0 = time.perf_counter()
        retry_generated: GeneratedAnswer = generate_answer(query.strip(), retry_bundle, provider, telemetry=telemetry)
        telemetry.add_stage("gen_correction_ms", (time.perf_counter() - _t0) * 1000.0)
        _t0 = time.perf_counter()
        retry_verified = verify_answer(retry_generated, context, chunk_index)
        telemetry.add_stage("verify_ms", (time.perf_counter() - _t0) * 1000.0)
        if retry_verified.all_verified:
            generated, verified = retry_generated, retry_verified
            post = evaluate_refusal(evidence, threshold=threshold, verified=verified)
    elif (
        not post.should_refuse
        and not verified.has_citations
        and not _looks_like_abstention(generated.text)
        and (evidence[0].rerank_score or 0.0) >= threshold
    ):
        # Non-abstention answer with zero parseable citations (e.g.
        # non-bracket markers): the model made claims it did not ground
        # in the canonical format. Same one-shot correction pattern.
        retry_bundle = _correction_bundle(
            query.strip(),
            context,
            chunk_index,
            "your previous answer contained no citations in the required "
            "canonical format [IS <standard_no>:<year>, Clause <clause>, "
            "Page <page>]. Restate the same facts with one such citation "
            "per technical assertion, copied from the block headers above.",
            mode,
        )
        telemetry.correction_retry = True
        _t0 = time.perf_counter()
        retry_generated = generate_answer(query.strip(), retry_bundle, provider, telemetry=telemetry)
        telemetry.add_stage("gen_correction_ms", (time.perf_counter() - _t0) * 1000.0)
        _t0 = time.perf_counter()
        retry_verified = verify_answer(retry_generated, context, chunk_index)
        telemetry.add_stage("verify_ms", (time.perf_counter() - _t0) * 1000.0)
        if retry_verified.all_verified:
            generated, verified = retry_generated, retry_verified
            post = evaluate_refusal(evidence, threshold=threshold, verified=verified)
    if post.should_refuse:
        telemetry.latency_ms = elapsed_ms()
        return _refused_result(query.strip(), post.reason, evidence, elapsed_ms(), telemetry)

    mask = bool(evidence[0].is_mask_restricted)
    blocks_by_id = {b.evidence.chunk_id: b for b in context.blocks}

    def _block_rerank(chunk_id: str | None) -> Optional[float]:
        block = blocks_by_id.get(chunk_id or "")
        return block.evidence.rerank_score if block is not None else None

    citations = [
        CitationOut(
            standard_no=f"IS {v.citation.standard_no}",
            year=_record_year(blocks_by_id.get(v.chunk_id or ""), chunk_index)
            or v.citation.year or None,
            clause=v.citation.clause,
            page=v.citation.page,
            chunk_id=v.chunk_id or "",
            verified=(v.verdict == "verified"),
            rerank_score=_block_rerank(v.chunk_id),
        )
        for v in verified.citations
    ]
    telemetry.latency_ms = elapsed_ms()
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
        telemetry=telemetry,
    )
