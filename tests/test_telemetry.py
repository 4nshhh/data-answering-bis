"""Offline tests for request-scoped generation telemetry.

No models, API keys, GPU, database, or network required: a stub
``_ChatCompletionsProvider`` serves scripted completions (and scripted
transient failures) without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval.types import RetrievedEvidence

from app.generator.context_builder import load_chunk_index
from app.generator.llm_client import (
    _ChatCompletionsProvider,
    generate_answer,
)
from app.generator.pipeline import run_query
from app.generator.prompts import build_prompt
from app.generator.telemetry import Telemetry
from app.main import _result_to_response
from app.generator.pipeline import QueryResult


GOOD_TEXT = "Water pH shall be not less than 6 [IS 456:2000, Clause 5.4, Page 15]."
BAD_TEXT = "Water pH shall be not less than 6 [IS 9999:2020, Clause 1, Page 1]."
ABSTAIN_TEXT = "The excerpts do not contain the value."
NO_CITE_TEXT = "No citations at all."


class RateLimitError(Exception):
    """Name-matched transient error (see _TRANSIENT_ERROR_NAMES)."""


class _StubCompletion:
    def __init__(self, text: str):
        message = type("Message", (), {"content": text})()
        self.choices = [type("Choice", (), {"message": message})()]
        self.model = "stub-model"
        self.usage = None

class _StubCompletions:
    def __init__(self, owner: "_StubClient"):
        self._owner = owner

    def create(self, **kwargs):
        owner = self._owner
        action = owner.actions[min(owner.calls, len(owner.actions) - 1)]
        owner.calls += 1
        if isinstance(action, BaseException):
            raise action
        return _StubCompletion(action)


class _StubClient:
    def __init__(self, actions: list):
        self.actions = list(actions)
        self.calls = 0
        self.chat = type("Chat", (), {"completions": _StubCompletions(self)})()


class StubProvider(_ChatCompletionsProvider):
    """Offline provider: scripted texts / errors, zero network."""

    name = "stub"
    env_var = "TELEMETRY_STUB_KEY"

    def __init__(self, actions: list, **kwargs):
        super().__init__(api_key="test-key", **kwargs)
        self._stub = _StubClient(actions)

    def _client(self):
        return self._stub


def make_evidence(chunk_id="c14", rerank_score=5.0, **kwargs) -> RetrievedEvidence:
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
        rerank_score=rerank_score,
        is_mask_restricted=False,
    )
    params.update(kwargs)
    return RetrievedEvidence(**params)


def retrieve_ok(query: str, top_k: int = 10):
    return [make_evidence("c14", 5.0), make_evidence("c15", 3.0)]


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


def test_single_generation_single_call(chunk_index):
    tele = Telemetry()
    result = run_query("What is the pH?", retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index,
                       provider=StubProvider([GOOD_TEXT]), telemetry=tele)
    assert result.refused is False
    assert tele.llm_generation_attempts == 1
    assert tele.groq_api_calls == 1
    assert tele.correction_retry is False
    assert tele.widen_retry is False
    assert tele.retrieval_expansion is False
    assert tele.latency_ms >= 0.0
    assert result.telemetry is tele


def test_stage_timings_recorded_per_pass(chunk_index):
    tele = Telemetry()
    run_query("What is the pH?", retrieve_fn=retrieve_ok,
              chunk_index=chunk_index,
              provider=StubProvider([GOOD_TEXT]), telemetry=tele)
    for key in ("retrieval_ms", "context_ms", "prompt_ms",
                "gen_initial_ms", "verify_ms"):
        assert key in tele.stages, f"missing stage {key}"
        assert tele.stages[key] >= 0.0
    assert "gen_widen_ms" not in tele.stages
    assert "gen_correction_ms" not in tele.stages
    assert tele.prompt_chars > 0
    snapshot = tele.to_dict()
    assert snapshot["stages"]["gen_initial_ms"] >= 0.0
    assert snapshot["prompt_chars"] == tele.prompt_chars


def test_widen_and_correction_stages_recorded(chunk_index):
    tele = Telemetry()
    run_query("What is the pH?", top_k=1, retrieve_fn=retrieve_ok,
              chunk_index=chunk_index,
              provider=StubProvider([NO_CITE_TEXT, BAD_TEXT, GOOD_TEXT]),
              telemetry=tele)
    assert tele.stages.get("gen_widen_ms", -1.0) >= 0.0
    assert tele.stages.get("gen_correction_ms", -1.0) >= 0.0
    assert tele.stages.get("context_widen_ms", -1.0) >= 0.0


def test_initial_plus_correction(chunk_index):
    tele = Telemetry()
    result = run_query("What is the pH?", retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index,
                       provider=StubProvider([BAD_TEXT, GOOD_TEXT]), telemetry=tele)
    assert result.refused is False
    assert tele.llm_generation_attempts == 2
    assert tele.groq_api_calls == 2
    assert tele.correction_retry is True
    assert tele.widen_retry is False


def test_initial_plus_widen(chunk_index):
    tele = Telemetry()
    result = run_query("What is the pH?", top_k=1, retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index,
                       provider=StubProvider([ABSTAIN_TEXT, GOOD_TEXT]), telemetry=tele)
    assert result.refused is False
    assert result.answer == GOOD_TEXT
    assert tele.llm_generation_attempts == 2
    assert tele.groq_api_calls == 2
    assert tele.widen_retry is True
    assert tele.correction_retry is False


def test_initial_plus_widen_plus_correction(chunk_index):
    tele = Telemetry()
    result = run_query("What is the pH?", top_k=1, retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index,
                       provider=StubProvider([NO_CITE_TEXT, BAD_TEXT, GOOD_TEXT]),
                       telemetry=tele)
    assert result.refused is False
    assert result.answer == GOOD_TEXT
    assert tele.llm_generation_attempts == 3
    assert tele.groq_api_calls == 3
    assert tele.widen_retry is True
    assert tele.correction_retry is True


def test_transient_retry_counts_every_provider_attempt(chunk_index):
    tele = Telemetry()
    provider = StubProvider([RateLimitError("slow down"), GOOD_TEXT])
    context_evidence = retrieve_ok("q")
    from app.generator.context_builder import build_context
    context = build_context(context_evidence, chunk_index=chunk_index, top_k=2)
    bundle = build_prompt("What is the pH?", context, chunk_index=chunk_index)
    answer = generate_answer("What is the pH?", bundle, provider, telemetry=tele)
    assert answer.text == GOOD_TEXT
    assert tele.llm_generation_attempts == 1
    assert tele.groq_api_calls == 2
    assert provider._stub.calls == 2


def test_telemetry_isolation_between_queries(chunk_index):
    tele_a = Telemetry()
    tele_b = Telemetry()
    run_query("What is the pH?", retrieve_fn=retrieve_ok, chunk_index=chunk_index,
              provider=StubProvider([GOOD_TEXT]), telemetry=tele_a)
    run_query("What is the pH?", top_k=1, retrieve_fn=retrieve_ok, chunk_index=chunk_index,
              provider=StubProvider([ABSTAIN_TEXT, GOOD_TEXT]), telemetry=tele_b)
    assert (tele_a.llm_generation_attempts, tele_a.groq_api_calls) == (1, 1)
    assert (tele_b.llm_generation_attempts, tele_b.groq_api_calls) == (2, 2)
    assert tele_a.widen_retry is False
    assert tele_b.widen_retry is True


def test_refusal_reports_zero_calls_with_latency():
    def retrieve_weak(query: str, top_k: int = 10):
        return [make_evidence("x", 0.03), make_evidence("y", 0.02)]

    tele = Telemetry()
    result = run_query("What is the capital of France?",
                       retrieve_fn=retrieve_weak,
                       provider=StubProvider([GOOD_TEXT]), telemetry=tele)
    assert result.refused is True
    assert tele.llm_generation_attempts == 0
    assert tele.groq_api_calls == 0
    assert tele.latency_ms >= 0.0


def test_token_totals_accumulate_across_attempts(chunk_index):
    from app.generator.llm_client import LLMResponse

    class UsageProvider:
        name = "usage"

        def __init__(self, texts):
            self.texts = list(texts)
            self.calls = 0

        def generate(self, **kwargs):
            text = self.texts[min(self.calls, len(self.texts) - 1)]
            self.calls += 1
            return LLMResponse(text=text, model="m",
                               prompt_tokens=100, completion_tokens=25)

    tele = Telemetry()
    result = run_query("What is the pH?", retrieve_fn=retrieve_ok,
                       chunk_index=chunk_index,
                       provider=UsageProvider([GOOD_TEXT]), telemetry=tele)
    assert result.refused is False
    assert tele.prompt_tokens_total == 100
    assert tele.completion_tokens_total == 25
    assert tele.to_dict()["completion_tokens_total"] == 25


def test_sdk_client_reused_across_generations(monkeypatch, chunk_index):
    import sys
    import types

    from app.generator.llm_client import GroqProvider

    built: list = []

    class FakeGroq:
        def __init__(self, api_key=None, timeout=None):
            built.append((api_key, timeout))
            stub_owner = _StubClient([GOOD_TEXT])
            self.chat = type("Chat", (), {"completions": _StubCompletions(stub_owner)})()

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    provider = GroqProvider(api_key="test-key")
    tele = Telemetry()
    for _ in range(2):
        result = run_query("What is the pH?", retrieve_fn=retrieve_ok,
                           chunk_index=chunk_index, provider=provider,
                           telemetry=tele)
        assert result.refused is False
    assert len(built) == 1  # one SDK client for both generations
    assert built[0][0] == "test-key"
    assert tele.llm_generation_attempts == 2
    assert tele.groq_api_calls == 2


def test_response_model_carries_telemetry():
    tele = Telemetry()
    tele.llm_generation_attempts = 2
    tele.groq_api_calls = 2
    tele.correction_retry = True
    tele.widen_retry = False
    tele.retrieval_expansion = False
    tele.latency_ms = 18000.0
    result = QueryResult(query="q", answer="a", telemetry=tele)
    response = _result_to_response(result)
    assert response.telemetry.llm_generation_attempts == 2
    assert response.telemetry.groq_api_calls == 2
    assert response.telemetry.correction_retry is True
    assert response.telemetry.widen_retry is False
    assert response.telemetry.retrieval_expansion is False
    assert response.telemetry.latency_ms == 18000.0
    dumped = response.model_dump()
    assert dumped["telemetry"]["groq_api_calls"] == 2


def test_runner_records_telemetry(monkeypatch):
    import evaluation.run_queries as runner

    body = {
        "query": "q",
        "answer": "a [IS 456:2000, Clause 5.4, Page 15].",
        "citations": [],
        "retrieval_meta": {},
        "refused": False,
        "refusal_reason": None,
        "telemetry": {
            "llm_generation_attempts": 2,
            "groq_api_calls": 2,
            "correction_retry": True,
            "widen_retry": False,
            "retrieval_expansion": False,
            "latency_ms": 18000.0,
        },
    }
    monkeypatch.setattr(runner, "post_query", lambda url, query, top_k, timeout_s: (200, body, None))
    records = [{"id": "C002", "category": "supported_clause", "query": "q?", "expected": "answer"}]
    (record,) = runner.run_suite(records, url="http://x", top_k=3, timeout_s=5.0)
    assert record["telemetry"]["groq_api_calls"] == 2
    assert record["telemetry"]["correction_retry"] is True
