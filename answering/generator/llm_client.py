"""Phase 5: LLM integration and orchestration.

Sends the Phase 4 ``PromptBundle`` (separate ``system`` + ``user``
messages) to a chat-completions LLM and returns the raw generated
text. Groq (default model ``openai/gpt-oss-120b``) is the primary
provider; OpenAI is supported through the same protocol with a lazy
import so Phase 4's dependency-free footprint is preserved (the
``openai`` package is not in ``requirements.txt``).

Scope:
  * Provider protocol + ``GroqProvider`` / ``GeminiProvider`` /
    ``OpenAIProvider`` behind ``build_provider()``.
  * ``generate_answer()`` orchestration: message passthrough,
    deterministic defaults, usage capture.
  * Timeout + small dependency-free retry on transient errors.

Out of scope: citation parsing (Phase 6), refusal wording and
abstention thresholds (Phase 7), FastAPI wiring (Phase 8). The
``insufficient_evidence_hook`` flag is carried through untouched for
Phase 7; this module never refuses or rewrites the prompt.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from answering.generator.prompts import DEFAULT_MODEL, PromptBundle
from answering.generator.telemetry import Telemetry

__all__ = [
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "GEMINI_DEFAULT_MODEL",
    "LLMResponse",
    "GeneratedAnswer",
    "LLMProvider",
    "GroqProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "build_provider",
    "generate_answer",
]

#: Deterministic default: grounded answers must not be creative.
DEFAULT_TEMPERATURE = 0.0

#: Per-attempt HTTP timeout (seconds).
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Retries after the first attempt, transient errors only.
DEFAULT_MAX_RETRIES = 2

#: Default Gemini model for the alternative provider (overridable with
#: ``GEMINI_MODEL``). The Groq default stays ``prompts.DEFAULT_MODEL``
#: (overridable with ``GROQ_MODEL``).
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"

#: Environment variable selecting the provider (``groq`` default).
LLM_PROVIDER_ENV_VAR = "LLM_PROVIDER"

#: Optional model overrides; unset preserves historical defaults.
GROQ_MODEL_ENV_VAR = "GROQ_MODEL"
GEMINI_MODEL_ENV_VAR = "GEMINI_MODEL"

#: Exception class names treated as transient for retry. Matched by
#: name so neither SDK needs to be imported to decide retryability.
_TRANSIENT_ERROR_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "Timeout",
        "ConnectError",
        "ReadTimeout",
    }
)


@dataclass
class LLMResponse:
    """Raw provider reply for one chat-completion call."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw_usage: dict[str, Any] | None = None


@dataclass
class GeneratedAnswer:
    """Orchestration result handed to Phase 6+ post-processing."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    insufficient_evidence_hook: bool = False


class LLMProvider(Protocol):
    """Minimal contract for a chat-completions backend."""

    name: str
    default_model: str

    def generate(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        timeout_s: float,
        telemetry: Telemetry | None = None,
    ) -> LLMResponse:
        """Complete one ``(system, user)`` turn."""
        ...


def _read_env_file(key: str, filename: str = ".env") -> str | None:
    """Read one key from a dotenv file in the working directory.

    Delegates to :func:`retrieval.store.read_env_file` (lazy import
    preserves this module's dependency-free footprint) so provider and
    storage configuration share identical parsing rules. Kept under
    this name as the monkeypatch seam used by tests.
    """
    from retrieval.store import read_env_file

    return read_env_file(key, filename)


def _resolve_api_key(explicit: str | None, env_var: str) -> str:
    """Explicit argument wins, then process env, then ``.env`` file."""
    if explicit:
        return explicit
    from_env = os.environ.get(env_var)
    if from_env:
        return from_env
    from_file = _read_env_file(env_var)
    if from_file:
        return from_file
    raise RuntimeError(
        f"{env_var} is not set: pass api_key explicitly, export it, "
        f"or define it in .env. Refusing to call the API without credentials."
    )


def _is_transient_error(exc: BaseException) -> bool:
    """True for retryable provider errors (timeout/connection/rate-limit/5xx)."""
    return type(exc).__name__ in _TRANSIENT_ERROR_NAMES


def _extract_text(choice: Any) -> str:
    """Pull the assistant message text from a chat-completion choice."""
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):  # segmented content parts
        parts = [p.text for p in content if getattr(p, "text", None)]
        joined = "".join(parts)
        if joined.strip():
            return joined
    raise RuntimeError("Provider returned an empty completion; refusing to continue with no text.")


def _extract_usage(completion: Any) -> dict[str, Any | None]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "raw": None}
    raw = usage.to_dict() if hasattr(usage, "to_dict") else dict(usage) if isinstance(usage, dict) else None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "raw": raw,
    }


class _ChatCompletionsProvider:
    """Shared chat-completions call loop (timeout + transient retry)."""

    name = "chat-completions"
    env_var = ""
    missing_hint = ""
    default_model = DEFAULT_MODEL

    def __init__(
        self,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._api_key = _resolve_api_key(api_key, self.env_var)
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries!r}")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._cached_client: Any | None = None

    def _client(self) -> Any:
        raise NotImplementedError

    def _cached(self, build: Callable[[], Any]) -> Any:
        """Reuse one SDK client per provider (same key/timeout/params).

        Previously a fresh client (fresh connection pool + TLS handshake)
        was built on every generation. Caching only reuses transport;
        requests, model, and retry policy are byte-identical.
        """
        if self._cached_client is None:
            self._cached_client = build()
        return self._cached_client

    def generate(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        timeout_s: float,
        telemetry: Telemetry | None = None,
    ) -> LLMResponse:
        client = self._client()
        attempts = 1 + self._max_retries
        for attempt in range(attempts):
            try:
                # Telemetry: count every actual provider request. This
                # sits immediately before create() so application-level
                # transient re-attempts are counted individually, while
                # hidden SDK-internal HTTP retries stay unobserved.
                if telemetry is not None:
                    telemetry.llm_api_calls += 1
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    timeout=min(timeout_s, self._timeout_s),
                )
                text = _extract_text(completion.choices[0])
                usage = _extract_usage(completion)
                return LLMResponse(
                    text=text,
                    model=getattr(completion, "model", model),
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    raw_usage=usage["raw"],
                )
            except Exception as exc:  # noqa: BLE001 - classified below, then re-raised
                if not _is_transient_error(exc):
                    raise
                if attempt == attempts - 1:
                    # Retries exhausted on a rate-limit/timeout/5xx:
                    # surface a uniform provider error (HTTP 502
                    # downstream) carrying the original failure, instead
                    # of leaking SDK-specific types that map to opaque
                    # HTTP 500s.
                    made = attempt + 1
                    suffix = f" after {made} attempts" if made > 1 else ""
                    raise RuntimeError(
                        f"{self.name} API error{suffix}: {exc}"
                    ) from exc
                time.sleep(2**attempt)  # 1s, 2s, ... backoff, dependency-free


class GroqProvider(_ChatCompletionsProvider):
    """Primary provider: Groq chat-completions (OpenAI-compatible)."""

    name = "groq"
    env_var = "GROQ_API_KEY"

    @property
    def default_model(self) -> str:
        """GROQ_MODEL override, else the historical default (unchanged)."""
        return os.environ.get(GROQ_MODEL_ENV_VAR) or DEFAULT_MODEL

    def _client(self) -> Any:
        def _build() -> Any:
            try:
                import groq
            except ImportError as exc:
                raise RuntimeError(
                    "The 'groq' package is required for GroqProvider "
                    "(pip install 'groq>=0.4.0')."
                ) from exc
            return groq.Groq(api_key=self._api_key, timeout=self._timeout_s)

        return self._cached(_build)


#: HTTP statuses treated as transient for the Gemini provider (429 /
#: 5xx only). Mirrors the application-level retry policy of the
#: chat-completions providers: same attempt count, same backoff.
_TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _gemini_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction from Gemini SDK errors."""
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


class GeminiProvider(_ChatCompletionsProvider):
    """Alternative provider: Gemini via the official ``google-genai`` SDK.

    Same orchestration contract as ``GroqProvider`` (system + user in,
    raw text out); only the transport differs (``generate_content`` with
    a system instruction instead of chat-completions). Prompts,
    temperature default, verification, refusal, and retry behavior are
    unchanged. One SDK client is reused per provider instance, matching
    the Groq reuse pattern.
    """

    name = "gemini"
    env_var = "GEMINI_API_KEY"

    @property
    def default_model(self) -> str:
        """GEMINI_MODEL override, else ``gemini-3.5-flash-lite``."""
        return os.environ.get(GEMINI_MODEL_ENV_VAR) or GEMINI_DEFAULT_MODEL

    def _client(self) -> Any:
        def _build() -> Any:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "The 'google-genai' package is required for GeminiProvider "
                    "(pip install 'google-genai>=1.0.0')."
                ) from exc
            return genai.Client(api_key=self._api_key)

        return self._cached(_build)

    def generate(
        self,
        *,
        system: str,
        user: str,
        model: str,
        temperature: float,
        timeout_s: float,
        telemetry: Telemetry | None = None,
    ) -> LLMResponse:
        from google.genai import types

        client = self._client()
        attempts = 1 + self._max_retries
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                # Telemetry: count every actual provider request, exactly
                # like the chat-completions providers. Hidden SDK-internal
                # HTTP retries stay unobserved (documented in telemetry.py).
                if telemetry is not None:
                    telemetry.llm_api_calls += 1
                response = client.models.generate_content(
                    model=model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        http_options=types.HttpOptions(
                            timeout=max(1, int(timeout_s * 1000))
                        ),
                    ),
                )
                text = response.text or ""
                if not text.strip():
                    raise RuntimeError(
                        "Provider returned an empty completion; "
                        "refusing to continue with no text."
                    )
                usage = getattr(response, "usage_metadata", None)
                return LLMResponse(
                    text=text,
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    completion_tokens=getattr(usage, "candidates_token_count", None),
                    raw_usage=None,
                )
            except Exception as exc:  # noqa: BLE001 - classified below, then re-raised
                last_error = exc
                transient = _is_transient_error(exc) or (
                    _gemini_status_code(exc) in _TRANSIENT_HTTP_STATUSES
                )
                if transient and attempt < attempts - 1:
                    time.sleep(2**attempt)  # 1s, 2s, ... same policy as Groq
                    continue
                if isinstance(exc, RuntimeError):
                    raise
                made = attempt + 1
                suffix = f" after {made} attempts" if made > 1 else ""
                raise RuntimeError(f"Gemini API error{suffix}: {exc}") from exc
        raise last_error  # pragma: no cover - loop always raises first


def build_provider(
    name: str | None = None,
    *,
    api_key: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> LLMProvider:
    """Construct the configured LLM provider (Groq default).

    Args:
        name: ``"groq"`` or ``"gemini"`` (case-insensitive); when
            omitted, ``LLM_PROVIDER`` decides (default ``"groq"``).
        api_key: explicit key, else the provider's ``*_API_KEY`` variable.
        timeout_s / max_retries: same transport knobs for both backends.

    Raises:
        ValueError: unknown provider name.
        RuntimeError: missing API key (from the provider constructor).
    """
    want = (name or os.environ.get(LLM_PROVIDER_ENV_VAR) or "groq").strip().lower()
    if want == "groq":
        return GroqProvider(api_key=api_key, timeout_s=timeout_s, max_retries=max_retries)
    if want == "gemini":
        return GeminiProvider(api_key=api_key, timeout_s=timeout_s, max_retries=max_retries)
    raise ValueError(f"unknown LLM provider {want!r}; expected 'groq' or 'gemini'")


class OpenAIProvider(_ChatCompletionsProvider):
    """Secondary provider: OpenAI chat-completions via lazy import.

    The ``openai`` package is intentionally not a hard dependency
    (absent from ``requirements.txt``); it is imported only when this
    provider is instantiated.
    """

    name = "openai"
    env_var = "OPENAI_API_KEY"

    def _client(self) -> Any:
        def _build() -> Any:
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError(
                    "The 'openai' package is required for OpenAIProvider "
                    "(pip install 'openai>=1.0.0')."
                ) from exc
            return openai.OpenAI(api_key=self._api_key, timeout=self._timeout_s)

        return self._cached(_build)


def generate_answer(
    query: str,
    bundle: PromptBundle,
    provider: LLMProvider,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    telemetry: Telemetry | None = None,
) -> GeneratedAnswer:
    """Generate one grounded answer for a prompt bundle.

    Pure passthrough orchestration: ``bundle.system`` / ``bundle.user``
    are sent unchanged with ``bundle.model``. Provider errors propagate
    to the caller (Phase 8 maps them to HTTP statuses). The
    ``insufficient_evidence_hook`` flag is carried through for Phase 7;
    no refusal or prompt rewriting happens here.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-blank string")
    # Telemetry: one logical generation attempt per execution, plus the
    # provider/model identity behind it (fixed per request, idempotent).
    resolved_model = bundle.model or DEFAULT_MODEL
    if telemetry is not None:
        telemetry.llm_generation_attempts += 1
        telemetry.llm_provider = provider.name
        telemetry.llm_model = resolved_model
    # Telemetry is forwarded only when present, so backends implementing
    # the original protocol (without the telemetry kwarg) keep working;
    # observability must never break generation.
    generate_kwargs: dict[str, object] = {
        "system": bundle.system,
        "user": bundle.user,
        "model": resolved_model,
        "temperature": temperature,
        "timeout_s": timeout_s,
    }
    if telemetry is not None:
        generate_kwargs["telemetry"] = telemetry
    response = provider.generate(**generate_kwargs)  # type: ignore[arg-type]
    if telemetry is not None:
        if response.prompt_tokens:
            telemetry.prompt_tokens_total += response.prompt_tokens
        if response.completion_tokens:
            telemetry.completion_tokens_total += response.completion_tokens
    if not response.text or not response.text.strip():
        raise RuntimeError("Provider returned an empty completion; refusing to continue with no text.")
    return GeneratedAnswer(
        text=response.text,
        model=response.model,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        insufficient_evidence_hook=bundle.insufficient_evidence_hook,
    )
