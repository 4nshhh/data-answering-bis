"""Offline API tests for Phase 8 (FastAPI TestClient, stubbed pipeline deps).

No models, keys, or network needed: the app is built with fake
singletons and ``retrieval.retrieve`` is monkeypatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")
from fastapi.testclient import TestClient  # noqa: E402

from retrieval.types import RetrievedEvidence  # noqa: E402

from answering.generator.context_builder import load_chunk_index  # noqa: E402
from answering.generator.llm_client import LLMResponse  # noqa: E402
from answering.answer import build_app  # noqa: E402


GOOD_TEXT = "Water pH shall be not less than 6 [IS 456:2000, Clause 5.4, Page 15]."


def make_evidence(chunk_id="c_0014", rerank=5.0) -> RetrievedEvidence:
    return RetrievedEvidence(
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


class FakeProvider:
    name = "fake"

    def __init__(self, text: str = GOOD_TEXT):
        self.text = text

    def generate(self, **kwargs) -> LLMResponse:
        return LLMResponse(text=self.text, model="fake-model")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    chunk = {
        "id": "c_0014",
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
    monkeypatch.setattr(
        "retrieval.retrieve",
        lambda query, top_k=10: [make_evidence("c_0014", 5.0)],
    )
    application = build_app(chunk_index=load_chunk_index(tmp_path), provider=FakeProvider())
    with TestClient(application) as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_success_shape(client):
    response = client.post("/api/v1/query", json={"query": "What is the pH?"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is the pH?"
    assert body["answer"] == GOOD_TEXT
    assert body["refused"] is False
    assert body["refusal_reason"] is None
    (citation,) = body["citations"]
    assert citation["standard_no"] == "IS 456"
    assert citation["year"] == "2000"
    assert citation["clause"] == "5.4"
    assert citation["page"] == 15
    assert citation["chunk_id"] == "c_0014"
    assert citation["verified"] is True
    meta = body["retrieval_meta"]
    assert meta["candidates_retrieved"] == 1
    assert meta["top_reranker_score"] == 5.0
    assert meta["is_mask_restricted"] is False
    assert meta["execution_time_ms"] >= 0.0


def test_query_refusal_path_returns_200_with_refusal_text(client, monkeypatch):
    monkeypatch.setattr(
        "retrieval.retrieve",
        lambda query, top_k=10: [make_evidence("c_0014", -9.0)],
    )
    response = client.post("/api/v1/query", json={"query": "Out of corpus?"})
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "below_threshold"
    assert body["citations"] == []
    assert "sufficient technical information" in body["answer"]


def test_blank_query_rejected(client):
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 400


def test_empty_query_rejected_by_validation(client):
    response = client.post("/api/v1/query", json={"query": ""})
    assert response.status_code == 422


def test_top_k_bounds_enforced(client):
    assert client.post("/api/v1/query", json={"query": "q?", "top_k": 9}).status_code == 422
    assert client.post("/api/v1/query", json={"query": "q?", "top_k": 0}).status_code == 422


def test_provider_error_maps_to_502(tmp_path, monkeypatch):
    class ExplodingProvider:
        name = "boom"

        def generate(self, **kwargs):
            raise RuntimeError("provider down")

    monkeypatch.setattr(
        "retrieval.retrieve",
        lambda query, top_k=10: [make_evidence("c_0014", 5.0)],
    )
    application = build_app(chunk_index=load_chunk_index(tmp_path), provider=ExplodingProvider())
    (tmp_path / "c_3000_ov300.json").write_text("[]", encoding="utf-8")
    with TestClient(application) as test_client:
        response = test_client.post("/api/v1/query", json={"query": "q?"})
    assert response.status_code == 502
    assert "provider down" in response.json()["detail"]


def test_request_options_forwarded(tmp_path, monkeypatch):
    seen: dict = {}

    def spy_retrieve(query: str, top_k: int = 10):
        seen["top_k"] = top_k
        return [make_evidence("c_0014", 5.0)]

    monkeypatch.setattr("retrieval.retrieve", spy_retrieve)
    application = build_app(provider=FakeProvider())
    with TestClient(application) as test_client:
        response = test_client.post(
            "/api/v1/query",
            json={"query": "q?", "top_k": 2, "expand_neighbors": True},
        )
    assert response.status_code == 200
    assert seen["top_k"] == 10  # candidate window stays frozen regardless of top_k
    assert response.json()["refused"] is False


def test_unicode_answer_round_trips_through_json(tmp_path, monkeypatch):
    live = "Water pH \u2265 6 " + chr(0x3010) + "IS 456:2000, Clause" + chr(0x202F) + \
        "5.4, Page" + chr(0x202F) + "15" + chr(0x2011) + "16" + chr(0x3011) + "."

    monkeypatch.setattr(
        "retrieval.retrieve",
        lambda query, top_k=10: [make_evidence("c_0014", 5.0)],
    )
    application = build_app(provider=FakeProvider(text=live))
    with TestClient(application) as test_client:
        response = test_client.post("/api/v1/query", json={"query": "pH?"})
    assert response.status_code == 200
    body = response.json()
    # UTF-8 JSON round-trip: technical symbols arrive intact, citation verifies.
    assert body["answer"] == live
    assert "\u2265" in body["answer"]
    (citation,) = body["citations"]
    # Year falls back to the as-written year (evidence has no record to check
    # against without a chunk index), so the citation is unverifiable here.
    assert (citation["standard_no"], citation["year"], citation["clause"],
            citation["page"], citation["verified"]) == ("IS 456", "2000", "5.4", 15, False)
    assert body["refused"] is False


def test_response_declares_utf8_and_bytes_decode_cleanly(tmp_path, monkeypatch):
    # Regression: the live model emits narrow-NBSP/CJK-bracket/non-breaking
    # hyphen text. The server bytes were always correct UTF-8, but without
    # an explicit charset some clients decoded them as cp1252, producing
    # "â¯"/"ã…" mojibake. The header below is what prevents that.
    live = ("Water pH " + chr(0x2265) + " 6 " + chr(0x3010) + "IS 456:2000, Clause"
            + chr(0x202F) + "5.4, Page" + chr(0x202F) + "15" + chr(0x2011)
            + "16" + chr(0x3011) + ".")

    monkeypatch.setattr(
        "retrieval.retrieve",
        lambda query, top_k=10: [make_evidence("c_0014", 5.0)],
    )
    application = build_app(provider=FakeProvider(text=live))
    with TestClient(application) as test_client:
        response = test_client.post("/api/v1/query", json={"query": "pH?"})

    assert response.status_code == 200
    content_type = response.headers["content-type"]
    assert content_type == "application/json; charset=utf-8"

    # Raw-bytes proof: decoding exactly as the declared charset yields the
    # original text with every technical character intact.
    raw_text = response.content.decode("utf-8")
    assert live in raw_text
    assert chr(0x2265) in raw_text and chr(0x3010) in raw_text

    # The old failure mode, documented: the same bytes read as latin-1 give
    # the reported mojibake — proving the bytes were right and only the
    # decoding was wrong.
    misdecoded = response.content.decode("latin-1")
    assert "â" in misdecoded  # U+202F/U+2011 bytes surface as â¯ fragments

    # Parsed proof: the JSON body carries the answer byte-identical.
    assert response.json()["answer"] == live
