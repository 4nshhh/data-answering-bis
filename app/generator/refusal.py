"""Phase 7: refusal and abstention logic.

Decides whether a query is answerable from the corpus *before* (or
after) generation, per AGENTS.md section 11. ``retrieve()`` returns
candidates unconditionally, so this generation-layer module owns the
"not in corpus" decision by inspecting:

  * the top ``rerank_score`` against threshold ``tau``
    (default 0.5, calibrated — see ``DEFAULT_THRESHOLD``),
  * the top score margin (top1 minus top2) against ``min_margin``,
  * ``is_mask_restricted`` (query named an IS number with no support),
  * Phase 6 citation verdicts (negative grounding check).

Scope: decision only. This module returns ``RefusalDecision`` values
and the canonical ``REFUSAL_TEXT``; composing the final user-facing
response (HTTP mapping, logging) belongs to Phase 8.

Refusal triggers (any one fires):
  * no evidence retrieved at all,
  * ``insufficient_hook`` set (Phase 4 fit zero blocks),
  * top ``rerank_score`` missing (confidence cannot be established),
  * top ``rerank_score`` below ``threshold``,
  * top margin below ``min_margin`` (opt-in; default 0.0 is dormant),
  * any Phase 6 citation verdict of ``mismatch``.

Deliberately *not* refusal triggers:
  * ``unverifiable`` citations (absent metadata is not contradiction)
    — reported in ``warnings`` instead,
  * ``is_mask_restricted`` alone — it contextualizes the reason and
    is echoed for Phase 8 metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval.types import RetrievedEvidence

from app.generator.citations import VerifiedAnswer

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_MIN_MARGIN",
    "REFUSAL_TEXT",
    "RefusalDecision",
    "evaluate_refusal",
]

#: Default confidence threshold tau: top rerank score below this refuses.
#:
#: Calibrated against live bge-reranker-base outputs, which are sigmoid
#: probabilities in [0, 1] (NOT raw logits — the AGENTS.md "-2.0" figure
#: was a logit-scale example that can never fire on this scale).
#: Measured: answerable probes top 0.66-1.0 (n=7: clause lookup,
#: paraphrase, natural, definition, adversarial, plus the canonical IS 456
#: pH query at 0.99); unanswerable probes top 0.025-0.04 (n=3: France
#: capital, BIS full form, Paris rainfall). Default 0.5 separates both
#: clusters with headroom. Always overridable per call / per request.
DEFAULT_THRESHOLD = 0.5

#: Default margin floor: top1-top2 below this refuses (0.0 keeps the
#: rule dormant unless an operator configures a positive floor).
DEFAULT_MIN_MARGIN = 0.0

#: Canonical refusal sentence (AGENTS.md section 11). Defined once,
#: here. Phase 8 renders it; no other module may duplicate it.
REFUSAL_TEXT = (
    "The provided Indian Standards documents do not contain "
    "sufficient technical information to answer this query."
)


@dataclass
class RefusalDecision:
    """Abstention verdict for one query."""

    should_refuse: bool
    reason: str  # machine-readable code; "ok" when answerable
    detail: str  # human-readable explanation for logs / metadata
    top_score: float | None = None
    margin: float | None = None  # top1 minus top2 logit gap
    threshold: float = DEFAULT_THRESHOLD
    is_mask_restricted: bool = False
    warnings: list[str] = field(default_factory=list)


def _top_scores(evidence: list[RetrievedEvidence]) -> tuple[float | None, float | None]:
    """Return (top1, top2) rerank scores; evidence is rerank-ordered."""
    if not evidence:
        return None, None
    top1 = evidence[0].rerank_score
    top2 = evidence[1].rerank_score if len(evidence) > 1 else None
    return top1, top2


def evaluate_refusal(
    evidence: list[RetrievedEvidence],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_margin: float = DEFAULT_MIN_MARGIN,
    insufficient_hook: bool = False,
    verified: VerifiedAnswer | None = None,
) -> RefusalDecision:
    """Decide whether to abstain for one query's evidence set.

    Args:
        evidence: rerank-ordered candidates from ``retrieve()``.
        threshold: minimum acceptable top rerank score (tau).
        min_margin: minimum acceptable top1-top2 gap.
        insufficient_hook: Phase 4/5 flag that zero blocks fit the budget.
        verified: optional Phase 6 verdicts for the negative grounding check.

    Returns:
        ``RefusalDecision``; ``reason == "ok"`` means answerable.
    """
    mask = bool(evidence) and bool(evidence[0].is_mask_restricted)

    if not evidence:
        return RefusalDecision(
            should_refuse=True,
            reason="no_candidates",
            detail="retrieval returned no candidates to ground an answer in",
            threshold=threshold,
            is_mask_restricted=mask,
        )

    if insufficient_hook:
        return RefusalDecision(
            should_refuse=True,
            reason="budget_hook",
            detail="no retrieved evidence fit the model context budget",
            threshold=threshold,
            is_mask_restricted=mask,
        )

    top1, top2 = _top_scores(evidence)
    if top1 is None:
        return RefusalDecision(
            should_refuse=True,
            reason="no_scores",
            detail="top candidate has no rerank score; confidence cannot be established",
            threshold=threshold,
            is_mask_restricted=mask,
        )

    margin = (top1 - top2) if top2 is not None else None

    if top1 < threshold:
        scope = "the named standard has " if mask else ""
        return RefusalDecision(
            should_refuse=True,
            reason="below_threshold",
            detail=f"top rerank score {top1:.3f} below threshold {threshold} "
            f"({scope}no supporting evidence in corpus)",
            top_score=top1,
            margin=margin,
            threshold=threshold,
            is_mask_restricted=mask,
        )

    if margin is not None and margin < min_margin:
        return RefusalDecision(
            should_refuse=True,
            reason="ambiguous_margin",
            detail=f"top margin {margin:.3f} below floor {min_margin}: "
            f"leading candidates are indistinguishable",
            top_score=top1,
            margin=margin,
            threshold=threshold,
            is_mask_restricted=mask,
        )

    warnings: list[str] = []
    if verified is not None:
        mismatches = [v for v in verified.citations if v.verdict == "mismatch"]
        if mismatches:
            where = "; ".join(v.detail for v in mismatches)
            return RefusalDecision(
                should_refuse=True,
                reason="citation_mismatch",
                detail=f"answer cites facts contradicting retrieved evidence: {where}",
                top_score=top1,
                margin=margin,
                threshold=threshold,
                is_mask_restricted=mask,
            )
        for v in verified.citations:
            if v.verdict == "unverifiable":
                warnings.append(f"unverifiable citation kept (not refused): {v.detail}")

    return RefusalDecision(
        should_refuse=False,
        reason="ok",
        detail="top evidence clears the confidence threshold",
        top_score=top1,
        margin=margin,
        threshold=threshold,
        is_mask_restricted=mask,
        warnings=warnings,
    )
