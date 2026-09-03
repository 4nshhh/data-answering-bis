"""Offline unit tests for Phase 5 LLM client / orchestration.

No network, API keys, or model access required. Provider calls are
exercised through recording fakes and stub SDK objects; the one test
needing the real ``groq`` package only constructs the client (no call).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from app.generator.llm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE,
    GeneratedAnswer,
    GroqProvider,
    LLMResponse,
    OpenAIProvider,
    _extract_text,
    _is_transient_error,
    _read_env_file,
    _resolve_api_key,
    generate_answer,
)
from app.generator.prompts import PromptBundle


def make_bundle(
    system: str = "SYSTEM RULES",
    user: str = "Question:\nq?\n\nEvidence:\nblock",
    model: str = "openai/gpt-oss-120b",
    hook: bool = False,
) -> PromptBundle:
    return PromptBundle(
        system=system,
        user=user,
        model=model,
        used_block_ids=["doc_0000"],
        reserve_block_ids=[],
        insufficient_evidence_hook=hook,
    )


class RecordingProvider:
    """Fake LLMProvider capturing call kwargs and returning canned text."""

    name = "fake"

    def __init__(self, text: str = "Canned grounded answer.", model: str = "fake-model"):
        self.calls: list[dict] = []
        self.text = text
        self.model = model

    def generate(self, *, system, user, model, temperature, timeout_s) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "temperature": temperature,
                "timeout_s": timeout_s,
            }
        )
        return LLMResponse(text=self.text, model=self.model,
                           prompt_tokens=10, completion_tokens=5)


# --- orchestration --------------------------------------------------------------

def test_generate_answer_passes_bundle_through_unchanged():
    provider = RecordingProvider()
    bundle = make_bundle()
    answer = generate_answer("What is the pH?", bundle, provider)
    assert isinstance(answer, GeneratedAnswer)
    assert answer.text == "Canned grounded answer."
    assert answer.model == "fake-model"
    assert (answer.prompt_tokens, answer.completion_tokens) == (10, 5)
    call = provider.calls[0]
    assert call["system"] == "SYSTEM RULES"
    assert call["user"].startswith("Question:\nq?")
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["temperature"] == DEFAULT_TEMPERATURE == 0.0


def test_generate_answer_carries_hook_for_phase7():
    provider = RecordingProvider()
    bundle = make_bundle(hook=True)
    answer = generate_answer("q?", bundle, provider)
    assert answer.insufficient_evidence_hook is True
    # Phase 5 never refuses itself: the provider is still called exactly once.
    assert len(provider.calls) == 1


def test_generate_answer_blank_query_raises():
    with pytest.raises(ValueError):
        generate_answer("   ", make_bundle(), RecordingProvider())


def test_generate_answer_empty_completion_raises():
    with pytest.raises(RuntimeError, match="empty completion"):
        generate_answer("q?", make_bundle(), RecordingProvider(text="   "))


def test_generate_answer_propagates_provider_errors():
    class ExplodingProvider:
        name = "boom"

        def generate(self, **kwargs):
            raise ConnectionError("network down")

    with pytest.raises(ConnectionError, match="network down"):
        generate_answer("q?", make_bundle(), ExplodingProvider())


# --- key resolution ---------------------------------------------------------------

def test_resolve_api_key_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("UNITTEST_API_KEY", "from-env")
    assert _resolve_api_key("explicit", "UNITTEST_API_KEY") == "explicit"
    assert _resolve_api_key(None, "UNITTEST_API_KEY") == "from-env"


def test_resolve_api_key_reads_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("UNITTEST_API_KEY2", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "# comment\nOTHER=1\nUNITTEST_API_KEY2=\"file-key\"  \n", encoding="utf-8"
    )
    assert _read_env_file("UNITTEST_API_KEY2") == "file-key"
    assert _resolve_api_key(None, "UNITTEST_API_KEY2") == "file-key"


def test_resolve_api_key_missing_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("UNITTEST_MISSING_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="UNITTEST_MISSING_KEY"):
        _resolve_api_key(None, "UNITTEST_MISSING_KEY")


def test_groq_provider_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqProvider()


def test_groq_provider_accepts_explicit_key():
    provider = GroqProvider(api_key="gsk_test")
    assert provider.name == "groq"


def test_groq_provider_invalid_params_raise():
    with pytest.raises(ValueError):
        GroqProvider(api_key="gsk_test", timeout_s=0)
    with pytest.raises(ValueError):
        GroqProvider(api_key="gsk_test", max_retries=-1)


def test_openai_provider_hint_when_package_absent():
    if importlib.util.find_spec("openai") is not None:
        pytest.skip("openai package installed; lazy-import path not exercisable")
    with pytest.raises(RuntimeError, match="pip install 'openai>=1.0.0'"):
        OpenAIProvider(api_key="sk_test").generate(
            system="s", user="u", model="m", temperature=0.0, timeout_s=60.0
        )


# --- helpers -----------------------------------------------------------------------

def test_is_transient_error_by_name():
    class RateLimitError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    assert _is_transient_error(RateLimitError("slow down")) is True
    assert _is_transient_error(TimeoutError("timed out")) is False  # stdlib name, not SDK
    assert _is_transient_error(AuthenticationError("bad key")) is False


def test_extract_text_rejects_empty():
    choice = SimpleNamespace(message=SimpleNamespace(content="  "))
    with pytest.raises(RuntimeError, match="empty completion"):
        _extract_text(choice)


def test_extract_text_reads_message_content():
    choice = SimpleNamespace(message=SimpleNamespace(content=" grounded. "))
    assert _extract_text(choice) == " grounded. "


def test_retry_on_transient_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    class FlakyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    attempts["n"] += 1
                    if attempts["n"] < 3:
                        err = type("APIConnectionError", (Exception,), {})("down")
                        raise err
                    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=7)
                    choice = SimpleNamespace(message=SimpleNamespace(content="recovered"))
                    return SimpleNamespace(choices=[choice], usage=usage, model="m")

    monkeypatch.setattr(GroqProvider, "_client", lambda self: FlakyClient())
    monkeypatch.setattr("time.sleep", lambda s: None)
    provider = GroqProvider(api_key="gsk_test", max_retries=DEFAULT_MAX_RETRIES)
    response = provider.generate(system="s", user="u", model="m",
                                 temperature=0.0, timeout_s=60.0)
    assert response.text == "recovered"
    assert attempts["n"] == 3


def test_no_retry_on_auth_error(monkeypatch):
    attempts = {"n": 0}

    class StrictClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    attempts["n"] += 1
                    raise type("AuthenticationError", (Exception,), {})("bad key")

    monkeypatch.setattr(GroqProvider, "_client", lambda self: StrictClient())
    provider = GroqProvider(api_key="gsk_test", max_retries=3)
    with pytest.raises(Exception, match="bad key"):
        provider.generate(system="s", user="u", model="m", temperature=0.0, timeout_s=60.0)
    assert attempts["n"] == 1
