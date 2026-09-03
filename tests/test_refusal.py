"""Offline unit tests for Phase 7 refusal / abstention logic.

No models, API keys, database, or network required.
"""

from __future__ import annotations

import pytest

from retrieval.types import RetrievedEvidence

from app.generator.citations import Citation, VerifiedAnswer, VerifiedCitation
from app.generator.refusal import (
    DEFAULT_MIN_MARGIN,
    DEFAULT_THRESHOLD,
    REFUSAL_TEXT,
    RefusalDecision,
    evaluate_refusal,
)


def make_evidence(
    chunk_id: str,
    rerank_score: float | None = 5.0,
    dense_score: float | None = 0.9,
    is_mask_restricted: bool = False,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        text="Evidence text.",
        source="doc.md",
        clause="5.4",
        heading="5.4 Water",
        standard_no="IS 456",
        page_start=15,
        page_end=16,
        dense_score=dense_score,
        rerank_score=rerank_score,
        is_mask_restricted=is_mask_restricted,
    )


def verified_answer(*verdicts: str) -> VerifiedAnswer:
    citations = [
        VerifiedCitation(
            citation=Citation("456", "2000", "5.4", 15, 0, 10),
            verdict=v,
            chunk_id="doc_0000",
            detail=f"{v} detail",
            checked_fields=["standard_no", "clause"],
        )
        for v in verdicts
    ]
    return VerifiedAnswer(text="answer", citations=citations,
                          all_verified=bool(citations) and all(v == "verified" for v in verdicts),
                          has_citations=bool(citations))


# --- canonical text ------------------------------------------------------------

def test_refusal_text_is_canonical_sentence():
    assert REFUSAL_TEXT == (
        "The provided Indian Standards documents do not contain "
        "sufficient technical information to answer this query."
    )


# --- answerable baseline ---------------------------------------------------------

def test_strong_evidence_is_ok():
    decision = evaluate_refusal([make_evidence("a", 5.0), make_evidence("b", 1.0)])
    assert isinstance(decision, RefusalDecision)
    assert decision.should_refuse is False
    assert decision.reason == "ok"
    assert decision.top_score == 5.0
    assert decision.margin == pytest.approx(4.0)
    assert decision.threshold == DEFAULT_THRESHOLD == 0.5
    assert decision.warnings == []


def test_single_candidate_has_no_margin_but_ok():
    decision = evaluate_refusal([make_evidence("a", 3.0)])
    assert decision.should_refuse is False
    assert decision.margin is None


# --- threshold ---------------------------------------------------------------------

def test_below_threshold_refuses():
    # 0.0344: live-measured top score for the out-of-corpus France probe.
    decision = evaluate_refusal([make_evidence("a", 0.0344), make_evidence("b", 0.0044)])
    assert decision.should_refuse is True
    assert decision.reason == "below_threshold"
    assert "0.034" in decision.detail


def test_threshold_boundary_is_inclusive():
    decision = evaluate_refusal([make_evidence("a", 0.5), make_evidence("b", 0.01)])
    assert decision.should_refuse is False
    assert decision.reason == "ok"


def test_custom_threshold_honored():
    evidence = [make_evidence("a", 0.7)]
    assert evaluate_refusal(evidence, threshold=0.9).should_refuse is True
    assert evaluate_refusal(evidence, threshold=0.5).should_refuse is False


def test_mask_restricted_contextualizes_reason():
    decision = evaluate_refusal(
        [make_evidence("a", -5.0, is_mask_restricted=True)]
    )
    assert decision.should_refuse is True
    assert decision.is_mask_restricted is True
    assert "named standard" in decision.detail


def test_mask_alone_does_not_refuse():
    decision = evaluate_refusal([make_evidence("a", 5.0, is_mask_restricted=True)])
    assert decision.should_refuse is False
    assert decision.is_mask_restricted is True


# --- structural guards ---------------------------------------------------------------

def test_empty_evidence_refuses():
    decision = evaluate_refusal([])
    assert decision.should_refuse is True
    assert decision.reason == "no_candidates"
    assert decision.top_score is None


def test_budget_hook_refuses_despite_scores():
    decision = evaluate_refusal([make_evidence("a", 9.0)], insufficient_hook=True)
    assert decision.should_refuse is True
    assert decision.reason == "budget_hook"


def test_missing_scores_refuse():
    decision = evaluate_refusal([make_evidence("a", None)])
    assert decision.should_refuse is True
    assert decision.reason == "no_scores"


# --- margin ----------------------------------------------------------------------------

def test_exact_tie_passes_on_default_floor():
    # Default floor is dormant: ties pass unless the operator opts into
    # margin-based refusal by configuring min_margin > 0.
    decision = evaluate_refusal([make_evidence("a", 4.0), make_evidence("b", 4.0)])
    assert decision.should_refuse is False
    assert decision.margin == pytest.approx(0.0)
    assert DEFAULT_MIN_MARGIN == 0.0
    tuned = evaluate_refusal(
        [make_evidence("a", 4.0), make_evidence("b", 4.0)], min_margin=0.1
    )
    assert tuned.should_refuse is True
    assert tuned.reason == "ambiguous_margin"


def test_clear_margin_passes_and_tunable_floor():
    evidence = [make_evidence("a", 4.0), make_evidence("b", 3.9)]
    assert evaluate_refusal(evidence).should_refuse is False
    tuned = evaluate_refusal(evidence, min_margin=0.5)
    assert tuned.should_refuse is True
    assert tuned.reason == "ambiguous_margin"


# --- citation grounding ------------------------------------------------------------------

def test_citation_mismatch_refuses():
    decision = evaluate_refusal(
        [make_evidence("a", 6.0)],
        verified=verified_answer("verified", "mismatch"),
    )
    assert decision.should_refuse is True
    assert decision.reason == "citation_mismatch"


def test_verified_citations_pass():
    decision = evaluate_refusal(
        [make_evidence("a", 6.0)], verified=verified_answer("verified", "verified")
    )
    assert decision.should_refuse is False
    assert decision.reason == "ok"


def test_unverifiable_citations_warn_but_pass():
    decision = evaluate_refusal(
        [make_evidence("a", 6.0)], verified=verified_answer("unverifiable")
    )
    assert decision.should_refuse is False
    assert len(decision.warnings) == 1
    assert "unverifiable" in decision.warnings[0]


def test_no_verified_arg_skips_grounding_check():
    decision = evaluate_refusal([make_evidence("a", 6.0)], verified=None)
    assert decision.should_refuse is False
