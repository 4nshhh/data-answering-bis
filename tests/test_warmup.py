"""Offline tests for standalone library usage (no FastAPI/Uvicorn).

Covers the production-backend contract::

    from app.generator import answer, warmup

    warmup()  # optional; must make zero LLM calls
    result = answer(query, mode="ask")

No models, API keys, GPU, database, or network required: the real
``retrieval.retrieve`` is substituted (it would load BGE-M3 weights),
while ``answer()`` runs the full pipeline with stubbed retrieval and
a fake LLM provider. Frozen RAG behavior is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.generator as generator_api
import retrieval
from app.generator import answer, warmup
from app.generator.context_builder import load_chunk_index
from app.generator.pipeline import QueryResult
from app.generator.telemetry import Telemetry
from retrieval.types import RetrievedEvidence

GOOD_TEXT = "Water pH shall be not less than 6 [IS 456:2000, Clause 5.4, Page 15]."


def make_evidence(chunk_id="c14", rerank_score=5.0) -> RetrievedEvidence:
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
        rerank_score=rerank_score,
        is_mask_restricted=False,
    )


def retrieve_ok(query: str, top_k: int = 10):
    return [make_evidence("c14", 5.0), make_evidence("c15", 3.0)]


class FakeProvider:
    name = "fake"
    default_model = "fake-model"

    def __init__(self, text: str = GOOD_TEXT):
        self.text = text
        self.calls: list = []

    def generate(self, **kwargs):
        from app.generator.llm_client import LLMResponse

        self.calls.append(kwargs)
        return LLMResponse(text=self.text, model="fake-model")


def write_tiny_index(tmp_path: Path):
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


@pytest.fixture()
def tiny_index(tmp_path: Path):
    return write_tiny_index(tmp_path)


@pytest.fixture(autouse=True)
def _clean_singletons():
    generator_api.reset_singletons()
    yield
    generator_api.reset_singletons()


@pytest.fixture()
def stub_retrieval(monkeypatch):
    """Substitute the model-loading retrieval entry points (no weights)."""
    calls: list = []

    def fake_retrieve(query: str, top_k: int = 10):
        calls.append((query, top_k))
        return []

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    monkeypatch.setattr(retrieval, "loaded_retriever", lambda: object())
    return calls


def _strip_llm_keys(monkeypatch):
    for var in ("LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY",
                "OPENAI_API_KEY", "GROQ_MODEL", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_warmup_importable_from_package():
    assert generator_api.warmup is warmup
    assert callable(warmup)


def test_shared_index_default_survives_foreign_cwd(tmp_path, monkeypatch):
    """The library chunk index must load from any working directory
    (JSON reads only — no models, keys, or network)."""
    monkeypatch.chdir(tmp_path)
    generator_api.reset_singletons()
    try:
        index = generator_api._shared_index()
    finally:
        generator_api.reset_singletons()
    assert len(index.by_id) == 2081


def test_warmup_preloads_without_llm_or_keys(monkeypatch, tmp_path, stub_retrieval):
    _strip_llm_keys(monkeypatch)
    result = warmup(tmp_path)
    assert stub_retrieval == [(generator_api.WARMUP_QUERY, 10)]
    assert result == {"chunks": 0, "retriever_loaded": True}
    # No provider singleton built: warmup must work with zero API keys
    # and must never touch an LLM SDK.
    assert generator_api._shared_provider is None


def test_warmup_uses_shared_chunk_index_singleton(monkeypatch, tiny_index, stub_retrieval):
    monkeypatch.setattr(generator_api, "load_chunk_index", lambda _d: tiny_index)
    result = warmup()
    assert result["chunks"] == len(tiny_index.by_id) == 1
    assert generator_api._shared_chunk_index is tiny_index


def test_warmup_is_idempotent(monkeypatch, tiny_index, stub_retrieval):
    monkeypatch.setattr(generator_api, "load_chunk_index", lambda _d: tiny_index)
    first = warmup()
    second = warmup()
    assert first == second
    assert len(stub_retrieval) == 2  # repeat warmup reuses loaded models
    assert generator_api._shared_provider is None


def test_answer_works_without_warmup_lazy_fallback(tiny_index):
    tele = Telemetry()
    provider = FakeProvider()
    result = answer("What is the pH?", top_k=3, retrieve_fn=retrieve_ok,
                    chunk_index=tiny_index, provider=provider,
                    telemetry=tele, mode="ask")
    assert isinstance(result, QueryResult)
    assert result.refused is False
    assert result.answer == GOOD_TEXT
    assert provider.calls, "answer() must still reach the LLM without warmup()"


def test_warmup_then_answer_shares_index(monkeypatch, tiny_index, stub_retrieval):
    monkeypatch.setattr(generator_api, "load_chunk_index", lambda _d: tiny_index)
    warmup()
    # answer() with no explicit chunk_index must reuse the warmed one.
    result = answer("What is the pH?", retrieve_fn=retrieve_ok,
                    provider=FakeProvider(), mode="ask")
    assert result.refused is False
    assert generator_api._shared_chunk_index is tiny_index


def test_fastapi_lifespan_delegates_warmup(monkeypatch, tmp_path):
    httpx = pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")
    del httpx
    import app.main as main_module
    from fastapi.testclient import TestClient

    calls: list = []

    def spy(chunks_dir=None):
        calls.append(chunks_dir)
        return {"chunks": 0, "retriever_loaded": True}

    monkeypatch.setattr(main_module, "warmup", spy)
    monkeypatch.setenv("BIS_WARMUP", "1")
    application = main_module.build_app(
        chunk_index=load_chunk_index(tmp_path), provider=FakeProvider())
    with TestClient(application):
        pass
    assert len(calls) == 1, "BIS_WARMUP=1 lifespan must delegate to generator.warmup()"


def test_fastapi_lifespan_passes_chunks_dir_to_warmup(monkeypatch, tmp_path):
    """P1: a custom chunks_dir must reach warmup(), not just the app index."""
    httpx = pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")
    del httpx
    import app.main as main_module
    from fastapi.testclient import TestClient

    calls: list = []

    def spy(chunks_dir=None):
        calls.append(chunks_dir)
        return {"chunks": 0, "retriever_loaded": True}

    monkeypatch.setattr(main_module, "warmup", spy)
    monkeypatch.setenv("BIS_WARMUP", "1")
    custom = tmp_path / "custom_chunks"
    application = main_module.build_app(
        chunk_index=load_chunk_index(tmp_path), provider=FakeProvider(),
        chunks_dir=custom)
    with TestClient(application):
        pass
    assert calls == [custom]
