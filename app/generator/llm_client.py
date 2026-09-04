"""Phase 5: LLM integration and orchestration.

Sends the Phase 4 ``PromptBundle`` (separate ``system`` + ``user``
messages) to a chat-completions LLM and returns the raw generated
text. Groq (default model ``openai/gpt-oss-120b``) is the primary
provider; OpenAI is supported through the same protocol with a lazy
import so Phase 4's dependency-free footprint is preserved (the
``openai`` package is not in ``requirements.txt``).

Scope:
  * Provider protocol + ``GroqProvider`` / ``OpenAIProvider``.
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
from pathlib import Path
from typing import Any, Callable, Protocol

from app.generator.prompts import DEFAULT_MODEL, PromptBundle
from app.generator.telemetry import Telemetry

__all__ = [
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "LLMResponse",
    "GeneratedAnswer",
    "LLMProvider",
    "GroqProvider",
    "OpenAIProvider",
    "generate_answer",
]

#: Deterministic default: grounded answers must not be creative.
DEFAULT_TEMPERATURE = 0.0

#: Per-attempt HTTP timeout (seconds).
DEFAULT_TIMEOUT_SECONDS = 60.0

#: Retries after the first attempt, transient errors only.
DEFAULT_MAX_RETRIES = 2

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

    Minimal reader (no new dependency): ``KEY=value`` lines only,
    ``#`` comments and surrounding quotes stripped.
    """
    path = Path(filename)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'") or None
    return None


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
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                # Telemetry: count every actual provider request. This
                # sits immediately before create() so application-level
                # transient re-attempts are counted individually, while
                # hidden SDK-internal HTTP retries stay unobserved.
                if telemetry is not None:
                    telemetry.groq_api_calls += 1
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
                last_error = exc
                if not _is_transient_error(exc) or attempt == attempts - 1:
                    raise
                time.sleep(2**attempt)  # 1s, 2s, ... backoff, dependency-free
        raise last_error  # pragma: no cover - loop always raises first


class GroqProvider(_ChatCompletionsProvider):
    """Primary provider: Groq chat-completions (OpenAI-compatible)."""

    name = "groq"
    env_var = "GROQ_API_KEY"

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
    # Telemetry: one logical generation attempt per execution.
    if telemetry is not None:
        telemetry.llm_generation_attempts += 1
    # Telemetry is forwarded only when present, so backends implementing
    # the original protocol (without the telemetry kwarg) keep working;
    # observability must never break generation.
    generate_kwargs: dict[str, object] = {
        "system": bundle.system,
        "user": bundle.user,
        "model": bundle.model or DEFAULT_MODEL,
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
