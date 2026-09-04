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

from dataclasses import dataclass

__all__ = ["Telemetry"]


@dataclass
class Telemetry:
    """Per-query counters, reset by construction for every request."""

    llm_generation_attempts: int = 0
    groq_api_calls: int = 0
    correction_retry: bool = False
    widen_retry: bool = False
    retrieval_expansion: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Plain-JSON snapshot for API responses and evaluation records."""
        return {
            "llm_generation_attempts": self.llm_generation_attempts,
            "groq_api_calls": self.groq_api_calls,
            "correction_retry": self.correction_retry,
            "widen_retry": self.widen_retry,
            "retrieval_expansion": self.retrieval_expansion,
            "latency_ms": self.latency_ms,
        }
