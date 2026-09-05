"""Offline tests for task modes and frontend-contract adapters.

No models, keys, GPU, database, network, or quota required: providers
are stubbed and adapters are exercised with hand-built QueryResults
(never touching retrieval, proving API-layer decoupling).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval.types import RetrievedEvidence

import app.generator.adapters as adapters
from app.generator import answer, to_ask_response, to_match_response
from app.generator.adapters import (
    clause_id_for,
    confidence_band,
    standard_id_for,
)
from app.generator.context_builder import build_context, load_chunk_index
from app.generator.llm_client import LLMResponse
from app.generator.pipeline import CitationOut, QueryResult, RetrievalMeta, run_query
from app.generator.prompts import (
    MODE_INSTRUCTIONS,
    SYSTEM_PROMPT,
    build_prompt,
    validate_mode,
)
from app.generator.refusal import REFUSAL_TEXT
from app.generator.telemetry import Telemetry

GOOD_TEXT = "Water pH shall be not less than 6 [IS 456:2000, Clause 5.4, Page 15]."


def make_evidence(chunk_id="c14", rerank=5.0, **kwargs) -> RetrievedEvidence:
    params = dict(
        chunk_id=chunk_id,
        text="Water shall have pH not less than 6.",
        source="c.md",
        clause="5.4",
        heading="5.4 Water",
        standard_no="IS 456",
        page_start=15,
        page_end=16,
        dense_score=0.9,
        rerank_score=rerank,
        is_mask_restricted=False,
    )
    params.update(kwargs)
    return RetrievedEvidence(**params)


def retrieve_ok(query: str, top_k: int = 10):
    return [make_evidence("c14", 5.0), make_evidence("c15", 3.0)]


class FakeProvider:
    name = "fake"
    default_model = "fake-model"

    def __init__(self, text: str = GOOD_TEXT):
        self.text = text

    def generate(self, **kwargs) -> LLMResponse:
        return LLMResponse(text=self.text, model="fake-model")


@pytest.fixture()
def chunk_index(tmp_path: Path):
    chunk = {
        "id": "c14",
        "text": "Water shall have pH not less than 6.",
        "metadata": {
            "source": "c.md",
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


def _result(**kwargs) -> QueryResult:
    base = dict(
        query="q?",
        answer=GOOD_TEXT,
        citations=[CitationOut(standard_no="IS 456", year="2000", clause="5.4",
                               page=15, chunk_id="c14", verified=True,
                               rerank_score=0.99)],
        retrieval_meta=RetrievalMeta(filtered_standard=None,
                                     is_mask_restricted=False,
                                     candidates_retrieved=2,
                                     top_reranker_score=0.99,
                                     execution_time_ms=1.0),
        refused=False,
        refusal_reason=None,
    )
    base.update(kwargs)
    return QueryResult(**base)


# --- mode validation ------------------------------------------------------

def test_validate_mode_accepts_both_modes():
    assert validate_mode("ask") == "ask"
    assert validate_mode("product_match") == "product_match"


def test_validate_mode_rejects_arbitrary_strings():
    with pytest.raises(ValueError, match="mode must be one of"):
        validate_mode("bullet")
    with pytest.raises(ValueError, match="mode must be one of"):
        validate_mode("")


def test_answer_defaults_to_ask(chunk_index):
    tele = Telemetry()
    result = answer("What is the pH?", retrieve_fn=retrieve_ok,
                    chunk_index=chunk_index, provider=FakeProvider(),
                    telemetry=tele)
    assert tele.mode == "ask"


def test_answer_rejects_unknown_mode_before_any_work():
    with pytest.raises(ValueError, match="mode must be one of"):
        answer("q?", retrieve_fn=retrieve_ok, chunk_index=None,
               provider=FakeProvider(), mode="nope")


def test_run_query_records_product_match_mode(chunk_index):
    tele = Telemetry()
    result = run_query("9W B22 LED bulb", retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index, provider=FakeProvider(),
                       telemetry=tele, mode="product_match")
    assert result.refused is False
    assert tele.mode == "product_match"


def test_run_query_rejects_unknown_mode(chunk_index):
    with pytest.raises(ValueError, match="mode must be one of"):
        run_query("q?", retrieve_fn=retrieve_ok, chunk_index=chunk_index,
                  provider=FakeProvider(), mode="whatever")


# --- prompt composition ---------------------------------------------------

def test_universal_rules_identical_across_modes(chunk_index):
    context = build_context(retrieve_ok("q"), chunk_index=chunk_index, top_k=2)
    ask = build_prompt("q?", context, chunk_index=chunk_index, mode="ask")
    match = build_prompt("q?", context, chunk_index=chunk_index, mode="product_match")
    assert ask.user == match.user  # evidence section untouched by mode
    assert ask.system.startswith(SYSTEM_PROMPT)
    assert match.system.startswith(SYSTEM_PROMPT)
    assert ask.system != match.system
    assert "product_match mode" in match.system
    assert "ask mode" in ask.system
    assert MODE_INSTRUCTIONS["ask"] in ask.system
    assert MODE_INSTRUCTIONS["product_match"] in match.system
    assert ask.mode == "ask"
    assert match.mode == "product_match"


def test_build_prompt_rejects_unknown_mode(chunk_index):
    context = build_context(retrieve_ok("q"), chunk_index=chunk_index, top_k=1)
    with pytest.raises(ValueError, match="mode must be one of"):
        build_prompt("q?", context, mode="nope")


def test_mode_survives_correction_retry(chunk_index):
    from app.generator.pipeline import run_query as _run

    tele = Telemetry()
    result = _run("q?", retrieve_fn=retrieve_ok, chunk_index=chunk_index,
                  provider=FakeProvider(text="x [IS 9999:2020, Clause 1, Page 1]"),
                  telemetry=tele, mode="product_match")
    # Mismatch triggers the one-shot correction with the always-bad text.
    assert result.refused is True
    assert tele.mode == "product_match"
    assert tele.correction_retry is True


# --- slug + band helpers --------------------------------------------------

def test_standard_and_clause_slugs():
    assert standard_id_for("IS 456", "2000") == "IS-456-2000"
    assert standard_id_for("IS 456", None) == "IS-456"
    assert clause_id_for("IS 456", "8.2.4.2") == "clause-456-8242"
    assert clause_id_for("IS 456", "N/A") == "clause-456-na"


def test_confidence_bands():
    assert confidence_band(0.99) == "high"
    assert confidence_band(0.7) == "high"
    assert confidence_band(0.6) == "medium"
    assert confidence_band(0.5) == "medium"
    assert confidence_band(0.1) == "low"
    assert confidence_band(None) == "low"


# --- /api/ask adapter -----------------------------------------------------

def test_ask_adapter_maps_answer_shape(chunk_index):
    body = to_ask_response(_result(), language="en", chunk_index=chunk_index)
    assert body["answer"] == GOOD_TEXT  # byte-identical text
    assert body["confidence"] == "high"
    assert body["refusal_message"] is None
    assert body["suggested_action"] is None
    assert body["language"] == "en"
    (cite,) = body["citations"]
    assert cite["id"] == "clause-456-54"
    assert cite["label"] == "IS 456:2000, Cl. 5.4"
    assert cite["standard_id"] == "IS-456-2000"
    assert cite["clause_number"] == "5.4"
    assert "pH not less than 6" in cite["snippet"]


def test_ask_adapter_refusal_shape():
    refused = _result(answer=REFUSAL_TEXT, citations=[], refused=True,
                      refusal_reason="below_threshold",
                      retrieval_meta=RetrievalMeta(
                          filtered_standard="IS 456", is_mask_restricted=True,
                          candidates_retrieved=7, top_reranker_score=0.03,
                          execution_time_ms=1.0))
    body = to_ask_response(refused, language="hi-IN")
    assert body["confidence"] == "refused"
    assert body["refusal_message"] == REFUSAL_TEXT
    assert body["citations"] == []
    assert body["suggested_action"]["page"] == "/certification"
    assert body["suggested_action"]["params"] == {"standard_id": "IS-456"}
    assert body["language"] == "hi-IN"


def test_ask_adapter_no_snippet_without_index():
    body = to_ask_response(_result())
    assert body["citations"][0]["snippet"] == ""


# --- /api/match-product adapter -------------------------------------------

def _match_result() -> QueryResult:
    return QueryResult(
        query="9W B22 LED bulb",
        answer="IS 2415 stuff [IS 2415:2025, Clause 5.2, Page 5].",
        citations=[
            CitationOut(standard_no="IS 2415", year="2025", clause="5.2",
                        page=5, chunk_id="r3", verified=True, rerank_score=0.92),
            CitationOut(standard_no="IS 2414", year="2004", clause="3.1",
                        page=2, chunk_id="r9", verified=True, rerank_score=0.61),
            CitationOut(standard_no="IS 9999", year="2020", clause="1",
                        page=1, chunk_id="rx", verified=False, rerank_score=0.99),
        ],
        retrieval_meta=RetrievalMeta(
            filtered_standard=None, is_mask_restricted=False,
            candidates_retrieved=10, top_reranker_score=0.92,
            execution_time_ms=1.0),
        refused=False,
        refusal_reason=None,
    )


def test_match_adapter_ranks_verified_only():
    body = to_match_response(_match_result(), "9W B22 LED bulb")
    assert body["input_interpreted_as"] == "9W B22 LED bulb"
    assert [m["standard_id"] for m in body["matches"]] == ["IS-2415-2025", "IS-2414-2004"]
    first, second = body["matches"]
    assert first["number"] == "IS 2415:2025"
    assert first["confidence"] == 0.92
    assert second["confidence"] == 0.61
    assert "Clause 5.2 (Page 5)" in first["reason"]
    # No catalog: title falls back to the number, QCO defaults False.
    assert first["title"] == "IS 2415:2025"
    assert first["is_qco_mandatory"] is False


def test_match_adapter_catalog_override():
    body = to_match_response(
        _match_result(), "bulb",
        catalog_lookup=lambda sid: {"title": "Cycle tubes", "is_qco_mandatory": True}
        if sid == "IS-2415-2025" else None,
    )
    first = body["matches"][0]
    assert first["title"] == "Cycle tubes"
    assert first["is_qco_mandatory"] is True
    assert body["matches"][1]["is_qco_mandatory"] is False


def test_match_adapter_empty_on_refusal():
    refused = _result(answer=REFUSAL_TEXT, citations=[], refused=True,
                      refusal_reason="below_threshold")
    body = to_match_response(refused, "mystery product")
    assert body == {"matches": [], "input_interpreted_as": "mystery product"}


def test_match_adapter_respects_max_matches():
    body = to_match_response(_match_result(), "bulb", max_matches=1)
    assert len(body["matches"]) == 1


def test_adapters_never_touch_retrieval():
    import ast

    tree = ast.parse(open(adapters.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    assert "retrieval" not in imported
    assert adapters.__name__ == "app.generator.adapters"


# --- prose-abstention consistency (product_match) ----------------------

ABSTAIN_TEXT = (
    "The evidence blocks mention standards inside reference lists "
    "(such as IS 15658 and IS 17452) [IS 875:2026, Clause ANNEX B, Page 41], "
    "but do not provide their own scope or requirements under a `Standard:` "
    "header. Therefore, based on the provided evidence blocks, there is not "
    "enough basis to identify and present an applicable standard."
)


def _abstain_result() -> QueryResult:
    base = _match_result()
    return QueryResult(
        query=base.query,
        answer=ABSTAIN_TEXT,
        citations=list(base.citations),
        retrieval_meta=base.retrieval_meta,
        refused=False,
        refusal_reason=None,
    )


def test_match_adapter_prose_abstention_empties_matches():
    body = to_match_response(_abstain_result(), "concrete product")
    assert body == {"matches": [], "input_interpreted_as": "concrete product"}


def test_match_adapter_abstention_variants_empty_matches():
    for text in (
        "I abstain: the blocks cannot identify an applicable standard.",
        "Unable to identify any applicable standard from the evidence.",
        "There is no applicable standard established by the blocks.",
    ):
        result = _match_result()
        result = QueryResult(
            query=result.query, answer=text, citations=list(result.citations),
            retrieval_meta=result.retrieval_meta, refused=False,
            refusal_reason=None,
        )
        body = to_match_response(result, "widget")
        assert body["matches"] == [], text
        assert body["input_interpreted_as"] == "widget"


def test_match_adapter_normal_answer_preserves_matches():
    # Normal presenting prose — including an "Applicable" header and a
    # "does not specify" limitation note — must never be misread as abstention.
    result = _match_result()
    result = QueryResult(
        query=result.query,
        answer="**Applicable Standard:** IS 2415:2025. The blocks do not "
        "specify packaging details. Scope per [IS 2415:2025, Clause 5.2, Page 5].",
        citations=list(result.citations),
        retrieval_meta=result.retrieval_meta, refused=False,
        refusal_reason=None,
    )
    body = to_match_response(result, "9W B22 LED bulb")
    assert [m["standard_id"] for m in body["matches"]] == ["IS-2415-2025", "IS-2414-2004"]


# --- O2: product_match must not promote mentioned-only standards -----

def _bundles_both_modes(chunk_index):
    context = build_context(retrieve_ok("q"), chunk_index=chunk_index, top_k=2)
    return (
        build_prompt("q?", context, chunk_index=chunk_index, mode="ask"),
        build_prompt("q?", context, chunk_index=chunk_index, mode="product_match"),
    )


def test_product_match_forbids_mentioned_only_standards(chunk_index):
    ask_bundle, pm_bundle = _bundles_both_modes(chunk_index)
    assert "merely mentioned inside another standard" in pm_bundle.system
    assert "merely mentioned inside another standard" not in ask_bundle.system


def test_universal_grounding_present_in_both_modes(chunk_index):
    ask_bundle, pm_bundle = _bundles_both_modes(chunk_index)
    for bundle in (ask_bundle, pm_bundle):
        assert SYSTEM_PROMPT in bundle.system
        assert "citation" in bundle.system.lower()
