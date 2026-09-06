"""Offline tests for provider abstraction + canonical answer() API.

No models, API keys, GPU, database, or network required: Gemini/Groq
SDKs are faked, providers are stubbed, and the FastAPI app is built
with injected singletons.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval.types import RetrievedEvidence

import answering.generator as generator_api
import answering.answer as main_module
from answering.generator import answer
from answering.generator.context_builder import load_chunk_index
from answering.generator.llm_client import (
    GeminiProvider,
    GroqProvider,
    build_provider,
)
from answering.generator.pipeline import QueryResult
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


class FakeProvider:
    name = "fake"
    default_model = "fake-model"

    def __init__(self, text: str = GOOD_TEXT):
        self.text = text

    def generate(self, **kwargs):
        from answering.generator.llm_client import LLMResponse

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


@pytest.fixture(autouse=True)
def _clean_singletons():
    generator_api.reset_singletons()
    yield
    generator_api.reset_singletons()


def test_answer_importable_from_package():
    assert generator_api.answer is answer
    assert callable(answer)


def test_answer_runs_full_pipeline_with_explicit_deps(chunk_index):
    tele = Telemetry()
    result = answer("What is the pH?", top_k=3, retrieve_fn=retrieve_ok,
                    chunk_index=chunk_index, provider=FakeProvider(),
                    telemetry=tele)
    assert isinstance(result, QueryResult)
    assert result.refused is False
    assert result.answer == GOOD_TEXT
    assert result.telemetry is tele
    assert tele.llm_generation_attempts == 1
    assert tele.llm_provider == "fake"


def test_answer_uses_lazy_singletons(monkeypatch, chunk_index):
    monkeypatch.setattr(generator_api, "load_chunk_index", lambda _d: chunk_index)
    monkeypatch.setattr(generator_api, "build_provider", lambda: FakeProvider())
    result = answer("What is the pH?", retrieve_fn=retrieve_ok)
    assert result.refused is False
    assert generator_api._shared_chunk_index is chunk_index
    assert isinstance(generator_api._shared_provider, FakeProvider)


def test_fastapi_delegates_to_answer(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    seen: dict = {}
    stub = QueryResult(query="q?", answer=GOOD_TEXT)

    def spy(query: str, top_k: int = 3, **kwargs):
        seen["query"] = query
        seen["top_k"] = top_k
        return stub

    monkeypatch.setattr(main_module, "answer", spy)
    application = main_module.build_app(
        chunk_index=load_chunk_index(tmp_path), provider=FakeProvider())
    with TestClient(application) as client:
        response = client.post("/api/v1/query", json={"query": "q?"})
    assert response.status_code == 200
    assert seen == {"query": "q?", "top_k": 3}
    assert response.json()["answer"] == GOOD_TEXT


def test_build_provider_defaults_to_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    provider = build_provider()
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"


def test_build_provider_selects_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    provider = build_provider(api_key="test-gemini-key")
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"
    assert provider.default_model == "gemini-3.5-flash-lite"


def test_build_provider_rejects_unknown():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_provider("anthropic")


def test_gemini_missing_key_raises(monkeypatch):
    import answering.generator.llm_client as llm_client

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "_read_env_file", lambda _key, _f=".env": None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key=None)


def test_gemini_model_override(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-x-test")
    provider = GeminiProvider(api_key="test-key")
    assert provider.default_model == "gemini-x-test"


def test_groq_default_model_unchanged(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert GroqProvider(api_key="test-key").default_model == "openai/gpt-oss-120b"


class _FakeGeminiResponse:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = None


class _FakeGeminiModels:
    def __init__(self, owner):
        self._owner = owner

    def generate_content(self, **kwargs):
        owner = self._owner
        owner.calls.append(kwargs)
        action = owner.actions[min(len(owner.calls) - 1, len(owner.actions) - 1)]
        if isinstance(action, BaseException):
            raise action
        return _FakeGeminiResponse(action)


class _FakeGeminiClient:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls: list = []
        self.models = _FakeGeminiModels(self)


def _patch_genai(monkeypatch, actions: list):
    import google.genai as genai_module

    made: list = []

    def factory(api_key=None):
        client = _FakeGeminiClient(actions)
        made.append(client)
        return client

    monkeypatch.setattr(genai_module, "Client", factory)
    return made


def test_gemini_generates_and_reuses_client(monkeypatch, chunk_index):
    from answering.generator.llm_client import generate_answer
    from answering.generator.prompts import build_prompt
    from answering.generator.context_builder import build_context

    made = _patch_genai(monkeypatch, [GOOD_TEXT, GOOD_TEXT])
    provider = GeminiProvider(api_key="test-key")
    tele = Telemetry()
    context = build_context(retrieve_ok("q"), chunk_index=chunk_index, top_k=2)
    for _ in range(2):
        bundle = build_prompt("What is the pH?", context, chunk_index=chunk_index,
                              model=provider.default_model)
        answer_out = generate_answer("What is the pH?", bundle, provider, telemetry=tele)
        assert GOOD_TEXT in answer_out.text
    assert len(made) == 1  # one SDK client reused across generations
    assert tele.llm_generation_attempts == 2
    assert tele.llm_api_calls == 2
    assert tele.llm_provider == "gemini"
    assert tele.llm_model == "gemini-3.5-flash-lite"


def test_gemini_transient_error_retried(monkeypatch, chunk_index):
    from answering.generator.llm_client import generate_answer
    from answering.generator.prompts import build_prompt
    from answering.generator.context_builder import build_context

    class Rate429(Exception):
        def __init__(self):
            super().__init__("rate limited")
            self.status_code = 429

    made = _patch_genai(monkeypatch, [Rate429(), GOOD_TEXT])
    provider = GeminiProvider(api_key="test-key", max_retries=2)
    tele = Telemetry()
    context = build_context(retrieve_ok("q"), chunk_index=chunk_index, top_k=2)
    bundle = build_prompt("What is the pH?", context, chunk_index=chunk_index,
                          model=provider.default_model)
    answer_out = generate_answer("What is the pH?", bundle, provider, telemetry=tele,
                                 timeout_s=5.0)
    assert GOOD_TEXT in answer_out.text
    assert tele.llm_api_calls == 2
    assert tele.llm_generation_attempts == 1
    assert len(made) == 1


def test_gemini_fatal_error_maps_to_runtime_error(monkeypatch, chunk_index):
    from answering.generator.llm_client import generate_answer
    from answering.generator.prompts import build_prompt
    from answering.generator.context_builder import build_context

    class Bad400(Exception):
        def __init__(self):
            super().__init__("bad request")
            self.status_code = 400

    _patch_genai(monkeypatch, [Bad400()])
    provider = GeminiProvider(api_key="test-key", max_retries=0)
    context = build_context(retrieve_ok("q"), chunk_index=chunk_index, top_k=2)
    bundle = build_prompt("What is the pH?", context, chunk_index=chunk_index,
                          model=provider.default_model)
    with pytest.raises(RuntimeError, match="Gemini API error"):
        generate_answer("What is the pH?", bundle, provider, timeout_s=5.0)


def test_refusal_semantics_provider_agnostic(chunk_index):
    def retrieve_weak(query: str, top_k: int = 10):
        return [make_evidence("x", 0.03)]

    result = answer("What is the capital of France?", retrieve_fn=retrieve_weak,
                    chunk_index=chunk_index, provider=FakeProvider())
    assert result.refused is True
    assert result.refusal_reason == "below_threshold"
    assert result.citations == []
