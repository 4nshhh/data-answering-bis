"""Offline tests for LLM provider selection (Gemini default, Groq fallback).

No models, API keys, GPU, database, or network required: the `.env`
seam (``llm_client._read_env_file``) is stubbed, credentials are fake,
and the LLM network edge (``<Provider>.generate``) is faked. The real
``build_provider()`` selection chain and the real ``answer()``
orchestration — including ``Telemetry.llm_provider`` — run unmodified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval.types import RetrievedEvidence

import answering.generator as generator_api
import answering.generator.llm_client as llm_client
from answering.generator import answer, warmup
from answering.generator.context_builder import load_chunk_index
from answering.generator.llm_client import (
    GeminiProvider,
    GroqProvider,
    LLMResponse,
    build_provider,
)
from answering.generator.telemetry import Telemetry

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


@pytest.fixture()
def tiny_index(tmp_path: Path):
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


@pytest.fixture(autouse=True)
def _isolated_provider_config(monkeypatch):
    """Reset singletons; clear provider env; stub the `.env` seam to empty.

    Each test then opts into exactly the configuration it exercises, so
    the developer's real `.env` (and CWD) can never leak into assertions.
    """
    generator_api.reset_singletons()
    for var in ("LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY",
                "GROQ_MODEL", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_client, "_read_env_file", lambda _key, _f=".env": None)
    yield
    generator_api.reset_singletons()


def _stub_env_file(monkeypatch, values: dict):
    """Serve `.env` lookups from an in-memory dict (per-key)."""
    monkeypatch.setattr(
        llm_client, "_read_env_file", lambda key, _f=".env": values.get(key)
    )


def _fake_generate(monkeypatch, provider_cls):
    """Fake the LLM network edge; return the list of recorded calls."""
    calls: list = []

    def fake(self, **kwargs):
        calls.append(kwargs)
        return LLMResponse(text=GOOD_TEXT, model="fake-model")

    monkeypatch.setattr(provider_cls, "generate", fake)
    return calls


# --- build_provider() selection chain -------------------------------------

def test_default_is_gemini_when_nothing_configured(monkeypatch):
    """No explicit name, no env, no `.env` provider → Gemini default."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    provider = build_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"


def test_dotenv_gemini_is_selected(monkeypatch):
    """`.env` LLM_PROVIDER=gemini → Gemini (key also resolved from `.env`)."""
    _stub_env_file(monkeypatch, {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "file-gemini-key",
    })
    provider = build_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"
    assert provider._api_key == "file-gemini-key"


def test_dotenv_groq_is_selected(monkeypatch):
    """`.env` LLM_PROVIDER=groq → Groq fallback path."""
    _stub_env_file(monkeypatch, {
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "file-groq-key",
    })
    provider = build_provider()
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"
    assert provider._api_key == "file-groq-key"


def test_process_env_overrides_dotenv(monkeypatch):
    """Env LLM_PROVIDER=groq beats `.env` LLM_PROVIDER=gemini → Groq."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    _stub_env_file(monkeypatch, {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "file-gemini-key",
    })
    provider = build_provider()
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"


def test_explicit_gemini_overrides_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    provider = build_provider("gemini", api_key="explicit-key")
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"


def test_explicit_groq_overrides_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    provider = build_provider("groq", api_key="explicit-key")
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"


def test_blank_env_value_falls_through_to_dotenv(monkeypatch):
    """A blank LLM_PROVIDER in process env is treated as unset."""
    monkeypatch.setenv("LLM_PROVIDER", "   ")
    _stub_env_file(monkeypatch, {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "file-gemini-key",
    })
    provider = build_provider()
    assert isinstance(provider, GeminiProvider)


def test_gemini_model_and_key_resolution(monkeypatch):
    """Existing config mechanism: GEMINI_MODEL env override + key from `.env`."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-test")
    _stub_env_file(monkeypatch, {"GEMINI_API_KEY": "file-gemini-key"})
    provider = build_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.default_model == "gemini-custom-test"
    assert provider._api_key == "file-gemini-key"


def test_groq_remains_usable_as_explicit_fallback(monkeypatch):
    """Groq is fully usable when explicitly selected (name or env)."""
    explicit = build_provider("groq", api_key="explicit-key")
    assert isinstance(explicit, GroqProvider)
    assert explicit.default_model == "openai/gpt-oss-120b"

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    via_env = build_provider()
    assert isinstance(via_env, GroqProvider)
    assert via_env.name == "groq"


# --- public answer() path (backend integration contract) -------------------

def test_answer_resolves_to_gemini_by_default(monkeypatch, tiny_index):
    """Backend contract: plain answer(mode="ask") uses Gemini by default."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    calls = _fake_generate(monkeypatch, GeminiProvider)
    tele = Telemetry()
    result = answer("What is the minimum pH of water per IS 456?", top_k=3,
                    retrieve_fn=retrieve_ok, chunk_index=tiny_index,
                    telemetry=tele, mode="ask")
    assert result.refused is False
    assert result.answer == GOOD_TEXT
    assert tele.llm_provider == "gemini"
    assert isinstance(generator_api._shared_provider, GeminiProvider)
    assert len(calls) >= 1


def test_answer_resolves_to_gemini_when_selected(monkeypatch, tiny_index):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    _fake_generate(monkeypatch, GeminiProvider)
    tele = Telemetry()
    result = answer("What is the minimum pH of water per IS 456?",
                    retrieve_fn=retrieve_ok, chunk_index=tiny_index,
                    telemetry=tele, mode="ask")
    assert result.refused is False
    assert tele.llm_provider == "gemini"
    assert isinstance(generator_api._shared_provider, GeminiProvider)


def test_answer_resolves_to_groq_when_selected(monkeypatch, tiny_index):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    _fake_generate(monkeypatch, GroqProvider)
    tele = Telemetry()
    result = answer("What is the minimum pH of water per IS 456?",
                    retrieve_fn=retrieve_ok, chunk_index=tiny_index,
                    telemetry=tele, mode="ask")
    assert result.refused is False
    assert tele.llm_provider == "groq"
    assert isinstance(generator_api._shared_provider, GroqProvider)


def test_warmup_ignores_provider_selection(monkeypatch, tmp_path):
    """warmup() stays provider-independent: no keys, no LLM call, no
    provider singleton — even with LLM_PROVIDER=groq and zero credentials."""
    import retrieval

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    calls: list = []

    def fake_retrieve(query: str, top_k: int = 10):
        calls.append((query, top_k))
        return []

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    monkeypatch.setattr(retrieval, "loaded_retriever", lambda: object())
    result = warmup(tmp_path)
    assert calls == [(generator_api.WARMUP_QUERY, 10)]
    assert result == {"chunks": 0, "retriever_loaded": True}
    assert generator_api._shared_provider is None


def test_run_query_provider_fallback_uses_build_provider_selection(monkeypatch, tiny_index):
    """Regression: run_query(provider=None) must follow the configured
    selection (Gemini default) instead of hardcoding Groq."""
    from answering.generator.pipeline import run_query

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    _fake_generate(monkeypatch, GeminiProvider)
    tele = Telemetry()
    result = run_query("What is the minimum pH of water per IS 456?",
                       retrieve_fn=retrieve_ok, chunk_index=tiny_index,
                       telemetry=tele, mode="ask")
    assert result.refused is False
    assert result.answer == GOOD_TEXT
    assert tele.llm_provider == "gemini"


def test_run_query_provider_fallback_honors_explicit_env(monkeypatch, tiny_index):
    """Regression: run_query(provider=None) with LLM_PROVIDER=groq uses Groq."""
    from answering.generator.pipeline import run_query

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    _fake_generate(monkeypatch, GroqProvider)
    tele = Telemetry()
    result = run_query("What is the minimum pH of water per IS 456?",
                       retrieve_fn=retrieve_ok, chunk_index=tiny_index,
                       telemetry=tele, mode="ask")
    assert result.refused is False
    assert tele.llm_provider == "groq"
