"""Offline unit tests for Phase 8 pipeline orchestration.

Retrieval and LLM calls are faked; no models, keys, or network needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval.types import RetrievedEvidence

from app.generator.context_builder import load_chunk_index
from app.generator.llm_client import GeneratedAnswer, LLMResponse
from app.generator.pipeline import QueryResult, run_query
from app.generator.refusal import REFUSAL_TEXT


def make_evidence(
    chunk_id: str = "456_2000_amd5_reff2021_0014",
    rerank_score: float | None = 5.0,
    is_mask_restricted: bool = False,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        text="Water shall have pH not less than 6.",
        source="456_2000_amd5_reff2021.md",
        clause="5.4",
        heading="5.4 Water",
        standard_no="IS 456",
        page_start=15,
        page_end=16,
        dense_score=0.9,
        rerank_score=rerank_score,
        is_mask_restricted=is_mask_restricted,
    )


GOOD_TEXT = "Water pH shall be not less than 6 [IS 456:2000, Clause 5.4, Page 15]."
BAD_TEXT = "Water pH shall be not less than 6 [IS 9999:2020, Clause 1, Page 1]."


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = GOOD_TEXT):
        self.text = text
        self.calls = 0

    def generate(self, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.text, model="fake-model",
                           prompt_tokens=10, completion_tokens=5)


def retrieve_ok(query: str, top_k: int = 10) -> list[RetrievedEvidence]:
    return [make_evidence("456_2000_amd5_reff2021_0014", 5.0),
            make_evidence("456_2000_amd5_reff2021_0015", 3.0)]


def retrieve_weak(query: str, top_k: int = 10) -> list[RetrievedEvidence]:
    return [make_evidence("a", -5.0), make_evidence("b", -6.0)]


@pytest.fixture()
def chunk_index(tmp_path: Path):
    chunk = {
        "id": "456_2000_amd5_reff2021_0014",
        "text": "Water shall have pH not less than 6.",
        "metadata": {
            "source": "456_2000_amd5_reff2021.md",
            "chunk_index": 14,
            "clause": "5.4",
            "heading": "5.4 Water",
            "heading_path": ["5 MATERIALS", "5.4 Water"],
            "standard_no": "IS 456",
            "year": "2000",
            "page_start": 15,
            "page_end": 16,
            "tail_truncated": False,
            "table_repaired": False,
        },
    }
    (tmp_path / "c_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    return load_chunk_index(tmp_path)


# --- success path ------------------------------------------------------------------

def test_success_path_returns_answer_and_verified_citation(chunk_index):
    provider = FakeProvider()
    result = run_query("What is the pH?", retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index, provider=provider)
    assert isinstance(result, QueryResult)
    assert result.refused is False
    assert result.refusal_reason is None
    assert result.answer == GOOD_TEXT
    assert provider.calls == 1
    (c,) = result.citations
    assert (c.standard_no, c.year, c.clause, c.page) == ("IS 456", "2000", "5.4", 15)
    assert c.chunk_id == "456_2000_amd5_reff2021_0014"
    assert c.verified is True
    meta = result.retrieval_meta
    assert meta.candidates_retrieved == 2
    assert meta.top_reranker_score == 5.0
    assert meta.is_mask_restricted is False
    assert meta.filtered_standard is None
    assert meta.execution_time_ms >= 0.0


def test_success_without_index_reports_cited_year_unverified():
    provider = FakeProvider()
    result = run_query("What is the pH?", retrieve_fn=retrieve_ok, provider=provider)
    assert result.refused is False
    (c,) = result.citations
    assert c.year == "2000"  # falls back to the as-written year
    assert c.verified is False  # year could not be checked


def test_mask_restricted_success_reports_filtered_standard(chunk_index):
    def retrieve_masked(query: str, top_k: int = 10):
        return [make_evidence("456_2000_amd5_reff2021_0014", 5.0, is_mask_restricted=True)]

    result = run_query("IS 456 pH?", retrieve_fn=retrieve_masked,
                       chunk_index=chunk_index, provider=FakeProvider())
    assert result.refused is False
    assert result.retrieval_meta.is_mask_restricted is True
    assert result.retrieval_meta.filtered_standard == "IS 456"


# --- refusal paths -------------------------------------------------------------------

def test_pre_generation_refusal_skips_llm(chunk_index):
    provider = FakeProvider()
    result = run_query("Out of corpus?", retrieve_fn=retrieve_weak,
                       chunk_index=chunk_index, provider=provider)
    assert result.refused is True
    assert result.refusal_reason == "below_threshold"
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert provider.calls == 0  # LLM call saved
    assert result.retrieval_meta.candidates_retrieved == 2


def test_post_generation_citation_mismatch_refuses(chunk_index):
    provider = FakeProvider(text=BAD_TEXT)
    result = run_query("What is the pH?", retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index, provider=provider)
    assert result.refused is True
    assert result.refusal_reason == "citation_mismatch"
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert provider.calls == 1


def test_empty_retrieval_refuses():
    result = run_query("Anything?", retrieve_fn=lambda q, top_k=10: [],
                       provider=FakeProvider())
    assert result.refused is True
    assert result.refusal_reason == "no_candidates"
    assert result.retrieval_meta.candidates_retrieved == 0


# --- validation / errors ----------------------------------------------------------------

def test_blank_query_raises():
    with pytest.raises(ValueError):
        run_query("   ", retrieve_fn=retrieve_ok, provider=FakeProvider())


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        run_query("q?", top_k=0, retrieve_fn=retrieve_ok, provider=FakeProvider())


def test_provider_error_propagates():
    class ExplodingProvider:
        name = "boom"

        def generate(self, **kwargs):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        run_query("q?", retrieve_fn=retrieve_ok, provider=ExplodingProvider())


# --- abstention-triggered context widening -------------------------------------------
#
# When the first pass yields zero parseable citations, run_query retries
# once over already-retrieved evidence with a wider window. The wider
# pass wins only on full verification; otherwise the first result stands.


class ScriptedProvider:
    name = "scripted"

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls = 0

    def generate(self, **kwargs) -> LLMResponse:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return LLMResponse(text=text, model="fake-model")


def test_widening_recovers_table_answer(chunk_index):
    provider = ScriptedProvider([
        "The excerpts do not contain the value.",  # abstention, no citations
        GOOD_TEXT,  # wider pass cites canonically
    ])
    result = run_query("What is the pH?", top_k=1, retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index, provider=provider)
    assert provider.calls == 2
    assert result.refused is False
    assert result.answer == GOOD_TEXT
    (c,) = result.citations
    assert c.verified is True


def test_widening_keeps_first_pass_when_retry_fails(chunk_index):
    provider = ScriptedProvider([
        "The excerpts do not contain the value.",
        "Still nothing citable here.",
    ])
    result = run_query("What is the pH?", top_k=1, retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index, provider=provider)
    assert provider.calls == 2
    assert result.refused is False  # byte-identical to pre-widening behavior
    assert result.citations == []
    assert result.answer == "The excerpts do not contain the value."


def test_no_widening_when_first_pass_cites(chunk_index):
    provider = ScriptedProvider([GOOD_TEXT, GOOD_TEXT])
    result = run_query("What is the pH?", retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index, provider=provider)
    assert provider.calls == 1
    assert result.refused is False


def test_no_widening_when_no_reserve_evidence(chunk_index):
    def retrieve_single(query: str, top_k: int = 10) -> list[RetrievedEvidence]:
        return [make_evidence("456_2000_amd5_reff2021_0014", 5.0)]

    provider = ScriptedProvider(["No citations at all."])
    result = run_query("What is the pH?", retrieve_fn=retrieve_single,
                       chunk_index=chunk_index, provider=provider)
    assert provider.calls == 1
    assert result.refused is False


# --- Tier 3 abstention regressions (live-measured rerank scores) ----------------------
#
# The bge-reranker emits sigmoid probabilities in [0, 1]. Unanswerable
# probes cluster at ~0.03, answerable ones at 0.66+, so the calibrated
# default threshold (0.5) must refuse the former and pass the latter.

def test_france_query_refuses_with_empty_citations():
    # Live: "What is the capital of France?" -> top 0.0344.
    def retrieve_france(query: str, top_k: int = 10):
        return [make_evidence("18000_2020_0002", rerank_score=0.0344),
                make_evidence("18000_2020_0003", rerank_score=0.0044)]

    provider = FakeProvider()
    result = run_query("What is the capital of France?", retrieve_fn=retrieve_france,
                       provider=provider)
    assert result.refused is True
    assert result.refusal_reason == "below_threshold"
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert provider.calls == 0  # LLM never sees insufficient evidence


def test_low_confidence_bis_query_refuses():
    # Live: "What is the full form of BIS?" -> top 0.0254.
    def retrieve_bis(query: str, top_k: int = 10):
        return [make_evidence("19778_2026_0022", rerank_score=0.0254),
                make_evidence("19778_2026_0023", rerank_score=0.0238)]

    provider = FakeProvider()
    result = run_query("What is the full form of BIS?", retrieve_fn=retrieve_bis,
                       provider=provider)
    assert result.refused is True
    assert result.refusal_reason == "below_threshold"
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert provider.calls == 0


def test_canonical_is456_query_answers(chunk_index):
    # Live: IS 456 pH query -> top 0.9941, must stay answerable.
    def retrieve_is456(query: str, top_k: int = 10):
        return [make_evidence("456_2000_amd5_reff2021_0014", rerank_score=0.9941),
                make_evidence("456_2000_amd5_reff2021_0024", rerank_score=0.9515)]

    provider = FakeProvider()
    result = run_query(
        "What is the minimum pH value of water for mixing concrete in IS 456?",
        retrieve_fn=retrieve_is456, chunk_index=chunk_index, provider=provider,
    )
    assert result.refused is False
    assert result.refusal_reason is None
    assert result.answer == GOOD_TEXT
    assert provider.calls == 1
    (c,) = result.citations
    assert c.verified is True
    assert c.chunk_id == "456_2000_amd5_reff2021_0014"


def test_unicode_answer_preserved_byte_identical(chunk_index):
    live = (
        "Water pH \u2265 6 "
        "【IS 456:2000, Clause 5.4, Page 15‑16】."
    )
    provider = FakeProvider(text=live)
    result = run_query("What is the pH?", retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index, provider=provider)
    assert result.refused is False
    assert result.answer == live  # ≥, CJK brackets, NBSP/NB-hyphen intact
    (c,) = result.citations
    assert (c.standard_no, c.year, c.clause, c.page) == ("IS 456", "2000", "5.4", 15)
    assert c.verified is True
    assert c.chunk_id == "456_2000_amd5_reff2021_0014"


def test_subclause_answer_passes_end_to_end(tmp_path: Path):
    # Tier 3 nominal-cover mirror: chunk metadata says 26.4 while the
    # answer cites text-anchored sub-clauses 26.4.1/26.4.2.1 (page 47).
    chunk_text = (
        "#### 26.4.1 Nominal Cover\n\nCover defined here.\n\n"
        "26.4.2.1 However for a column cover shall not be less than 40 mm.\n"
    )
    chunk = {
        "id": "cov_0074",
        "text": chunk_text,
        "metadata": {
            "source": "cov.md",
            "chunk_index": 74,
            "clause": "26.4",
            "heading": "26.4 Nominal Cover to Reinforcement",
            "heading_path": [],
            "standard_no": "IS 456",
            "year": "2000",
            "page_start": 47,
            "page_end": 47,
            "tail_truncated": False,
            "table_repaired": False,
        },
    }
    (tmp_path / "cov_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    evidence = [RetrievedEvidence(
        chunk_id="cov_0074", text=chunk_text, source="cov.md", clause="26.4",
        heading="26.4 Nominal Cover to Reinforcement", standard_no="IS 456",
        page_start=47, page_end=47, dense_score=0.99, rerank_score=0.9991,
        is_mask_restricted=True,
    )]
    answer = ("Cover defined [IS 456:2000, Clause 26.4.1, Page 47]; columns "
              "[IS 456:2000, Clause 26.4.2.1, Page 47].")
    result = run_query(
        "What is the nominal cover?",
        retrieve_fn=lambda q, top_k=10: evidence,
        chunk_index=index, provider=FakeProvider(text=answer),
    )
    assert result.refused is False
    assert [c.verified for c in result.citations] == [True, True]
    assert {c.chunk_id for c in result.citations} == {"cov_0074"}
