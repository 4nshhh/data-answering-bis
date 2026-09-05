"""Frontend-contract adapters (pramanak API shapes).

Converts the internal :class:`QueryResult` into the response contracts
defined by the frontend/backend plan (``POST /api/ask``,
``POST /api/match-product``) without touching the RAG pipeline.

Decoupling guarantees (do not break):
  * This module never imports ``retrieval`` and never calls ``retrieve()``.
    Evidence flows in exclusively through ``answer()`` → ``QueryResult``.
  * No new LLM calls, no re-verification, no threshold changes.
  * Every mapped value derives from ``QueryResult`` fields; anything the
    corpus cannot supply (official standard titles, QCO legal status) is
    either resolved through an explicit caller-supplied catalog lookup or
    surfaced with a documented fallback — never invented.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from app.generator.pipeline import CitationOut, QueryResult

__all__ = [
    "standard_id_for",
    "clause_id_for",
    "confidence_band",
    "to_ask_response",
    "to_match_response",
]


def _digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


def standard_id_for(standard_no: str | None, year: str | None) -> str:
    """Stable viewer id, e.g. ``IS-456-2000`` (``IS-456`` without a year)."""
    core = _digits(standard_no)
    base = f"IS-{core}" if core else "IS-unknown"
    return f"{base}-{year}" if year and year.isdigit() else base


def clause_id_for(standard_no: str | None, clause: str | None) -> str:
    """Stable clause id, e.g. ``clause-456-8242`` (``na`` when unnumbered)."""
    core = _digits(standard_no) or "unknown"
    label = _digits(clause) or "na"
    return f"clause-{core}-{label}"


def confidence_band(top_reranker_score: float | None) -> str:
    """Display confidence from the top evidence score (already 0-1).

    Bands are a presentation mapping only, not a RAG verdict: refused
    is decided by the pipeline before this adapter ever runs.
    """
    if top_reranker_score is None:
        return "low"
    if top_reranker_score >= 0.7:
        return "high"
    if top_reranker_score >= 0.5:
        return "medium"
    return "low"


def _citation_label(citation: CitationOut) -> str:
    number = citation.standard_no + (f":{citation.year}" if citation.year else "")
    return f"{number}, Cl. {citation.clause}"


def _snippet_for(
    chunk_id: str,
    chunk_index: Any | None,
    limit: int = 160,
) -> str:
    """Best-effort clause preview; empty when no chunk index is supplied."""
    if chunk_index is None:
        return ""
    try:
        chunk_dict = chunk_index.by_id.get(chunk_id)
    except AttributeError:
        return ""
    if not chunk_dict:
        return ""
    text = str(chunk_dict.get("text", "")).strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def to_ask_response(
    result: QueryResult,
    *,
    language: str = "en",
    chunk_index: Any | None = None,
) -> dict[str, Any]:
    """Adapt a (mode=``"ask"``) result to the ``POST /api/ask`` contract.

    The answer text is carried through byte-identical (canonical
    ``[IS ...]`` citations intact); the ``citations`` array exposes
    viewer ids derived deterministically from the verified citations.
    ``refusal_message`` is the pipeline refusal text when refused, else
    ``None`` — no contact channels are invented. ``suggested_action``
    is set only when the refusal names a standard (certification
    checklist deep-link), else ``None``.
    """
    meta = result.retrieval_meta
    top = meta.top_reranker_score if meta else None
    if result.refused:
        confidence = "refused"
        refusal_message: Optional[str] = result.answer
    else:
        confidence = confidence_band(top)
        refusal_message = None

    citations = []
    for citation in result.citations:
        sid = standard_id_for(citation.standard_no, citation.year)
        citations.append(
            {
                "id": clause_id_for(citation.standard_no, citation.clause),
                "label": _citation_label(citation),
                "standard_id": sid,
                "clause_number": citation.clause,
                "snippet": _snippet_for(citation.chunk_id, chunk_index),
            }
        )

    suggested_action: Optional[dict[str, Any]] = None
    filtered = meta.filtered_standard if meta else None
    if result.refused and filtered:
        sid = standard_id_for(filtered, next(
            (c.year for c in result.citations if c.year), None,
        ) if result.citations else None)
        suggested_action = {
            "label": f"Start certification checklist for {filtered}",
            "page": "/certification",
            "params": {"standard_id": sid},
        }

    return {
        "answer": result.answer,
        "citations": citations,
        "confidence": confidence,
        "refusal_message": refusal_message,
        "suggested_action": suggested_action,
        "language": language,
    }


def to_match_response(
    result: QueryResult,
    description: str,
    *,
    catalog_lookup: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
    max_matches: int = 5,
) -> dict[str, Any]:
    """Adapt a (mode=``"product_match"``) result to ``POST /api/match-product``.

    Standards are aggregated from **verified** citations only, ranked by
    the best linked rerank score (already 0-1). ``title`` and
    ``is_qco_mandatory`` come from ``catalog_lookup`` when supplied;
    otherwise the title falls back to the standard number (the corpus
    carries headings, not official titles) and ``is_qco_mandatory`` to
    ``False`` — QCO legal status is catalog data the RAG layer cannot
    determine, and the fallback is documented, not asserted as fact.
    ``reason`` describes the evidence linkage only. ``input_interpreted_as``
    echoes the normalized description (no separate parsing stage exists).
    """
    interpreted = (description or "").strip()
    groups: dict[tuple[str, Optional[str]], dict[str, Any]] = {}
    for citation in result.citations:
        if not citation.verified:
            continue
        key = (citation.standard_no, citation.year)
        group = groups.get(key)
        score = citation.rerank_score if citation.rerank_score is not None else 0.0
        if group is None:
            groups[key] = {
                "standard_no": citation.standard_no,
                "year": citation.year,
                "confidence": score,
                "clauses": [(citation.clause, citation.page)],
            }
        else:
            group["confidence"] = max(group["confidence"], score)
            group["clauses"].append((citation.clause, citation.page))

    ranked = sorted(groups.values(),
                    key=lambda g: (-g["confidence"], g["standard_no"] or ""))
    matches = []
    for group in ranked[:max(0, max_matches)]:
        number = group["standard_no"] + (f":{group['year']}" if group["year"] else "")
        sid = standard_id_for(group["standard_no"], group["year"])
        catalog = catalog_lookup(sid) if catalog_lookup is not None else None
        catalog = catalog or {}
        clauses = "; ".join(
            f"Clause {clause} (Page {page})" for clause, page in group["clauses"]
        )
        matches.append(
            {
                "standard_id": sid,
                "number": number,
                "title": catalog.get("title") or number,
                "confidence": round(float(group["confidence"]), 4),
                "is_qco_mandatory": bool(catalog.get("is_qco_mandatory", False)),
                "reason": f"Matched in retrieved evidence: {clauses} of {number}.",
            }
        )
    return {"matches": matches, "input_interpreted_as": interpreted}
