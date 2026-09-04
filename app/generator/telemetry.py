"""Request-scoped generation telemetry (observability only).

Tracks per-query LLM usage so evaluations can account for actual
provider traffic:

* ``llm_generation_attempts``: logical ``generate_answer()`` executions.
* ``groq_api_calls``: actual ``client.chat.completions.create(...)``
  invocations (one per application-level attempt, including the
  application-level transient retry in ``llm_client``).
* ``correction_retry`` / ``widen_retry`` / ``retrieval_expansion``:
  which pipeline fallback paths actually executed.
* ``latency_ms``: total request latency measured with
  ``time.perf_counter()``.

Scope: counting only. This module never influences retrieval,
reranking, prompts, verification, thresholds, refusal, or retry
behavior. One instance per user query; no global mutable state —
callers create it and pass it explicitly down the call chain.

Groq SDK internals: the installed ``groq`` SDK defaults to
``max_retries=2``, so a single ``create()`` invocation may perform up
to 3 hidden HTTP transport attempts. Those are not observable from
application code and are deliberately NOT counted here:
``groq_api_calls`` counts observable ``create()`` invocations.
"Groq SDK internal retries are not separately observable."
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Telemetry"]


@dataclass
class Telemetry:
    """Per-query counters, reset by construction for every request."""

    llm_generation_attempts: int = 0
    # Observable LLM API invocations (any provider): incremented once per
    # actual provider request (``create()`` / ``generate_content()``),
    # including application-level transient re-attempts.
    llm_api_calls: int = 0
    # Provider/model identity behind this request (set by generate_answer).
    llm_provider: str = ""
    llm_model: str = ""
    correction_retry: bool = False
    widen_retry: bool = False
    retrieval_expansion: bool = False
    latency_ms: float = 0.0
    # Stage timings in milliseconds, measured with time.perf_counter().
    # Keys (when that stage executes): retrieval_ms,
    # retrieval_expansion_ms, context_ms, context_widen_ms, prompt_ms,
    # gen_initial_ms, gen_widen_ms, gen_correction_ms, verify_ms
    # (verify_ms accumulates parse+verify across passes).
    stages: dict[str, float] = field(default_factory=dict)
    # Prompt size proxy (characters of the user message incl. evidence;
    # no extra tokenizer — prompts.py already estimates tokens as chars/4).
    prompt_chars: int = 0
    # Cumulative provider-reported token usage across all attempts in
    # this request (None-safe: providers that omit usage contribute 0).
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0

    def add_stage(self, name: str, ms: float) -> None:
        """Record (or accumulate, for repeated stages) a stage duration."""
        self.stages[name] = self.stages.get(name, 0.0) + ms

    @property
    def groq_api_calls(self) -> int:
        """Backward-compatible alias of ``llm_api_calls`` (read-only)."""
        return self.llm_api_calls

    def to_dict(self) -> dict[str, object]:
        """Plain-JSON snapshot for API responses and evaluation records."""
        return {
            "llm_generation_attempts": self.llm_generation_attempts,
            "llm_api_calls": self.llm_api_calls,
            "groq_api_calls": self.llm_api_calls,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "correction_retry": self.correction_retry,
            "widen_retry": self.widen_retry,
            "retrieval_expansion": self.retrieval_expansion,
            "latency_ms": self.latency_ms,
            "stages": dict(self.stages),
            "prompt_chars": self.prompt_chars,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
        }
