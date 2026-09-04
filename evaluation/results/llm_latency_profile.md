# LLM Latency Profile — BIS Answering Pipeline

## 1. Objective
Determine what actually causes LLM wall-clock latency and test only
evidence-backed optimizations, preserving the protected 27/30 quality
baseline. No full benchmark, no quota-heavy runs.

## 2. Architecture relevant to latency
`answer()` → `run_query()` → frozen retrieval (BGE-M3 encode →
2081×1024 dot-product → bge-reranker-base top-10, all CUDA) →
pre-refusal → optional expansion → context/prompt assembly →
1–3 one-shot generations (initial, widen, correction) → citation
verification. Local stages measured in ms; one SDK client reused per
provider; app-level transient retries (2, 1s/2s backoff).

## 3. Instrumentation
Pre-existing request-scoped telemetry already covered 10/11 required
metrics (retrieval/context/prompt per-stage ms, per-pass generation ms,
attempts, API calls, provider/model, flags, in/out tokens, prompt
chars). The only gap was output tok/s, added as a pure derived metric
in `evaluation/compare_llm_latency.py` (`tok_per_s()`; aggregate over
multi-attempt requests by construction). No production-path change.

## 4. Methodology
`python evaluation/compare_llm_latency.py --provider <groq|gemini> --runs 1`
over S001 (simple), C002 (clause), N001 (numerical), M001
(multi-clause), N003 (widen recovery). Retrieval-only warmup discarded.
Identical code/prompts/thresholds; only the backend differs.
Total live spend: 6 Groq + 5 Gemini calls.

## 5. Baseline measurements (Groq, `openai/gpt-oss-120b`)

| Query | Total | LLM | InTok | OutTok | tok/s | Calls | Att |
|---|---:|---:|---:|---:|---:|---:|---:|
| S001 | 1.6s | 1.4s | 3166 | 182 | 129 | 1 | 1 |
| C002 | 4.5s | 4.3s | 2924 | 1939 | 450 | 1 | 1 |
| N001 | 37.0s | 36.4s | 3281 | 237 | 7 | 1 | 1 |
| M001 | 23.1s | 22.4s | 2605 | 1108 | 49 | 1 | 1 |
| N003 | 72.3s | 71.8s | 8586* | 452* | 6 | 2 | 2 |

*Cumulative over 2 attempts (widen abstention path).
Avg total 27.7s, median 23.1s, min 1.6s, max 72.3s, avg gen 27.3s.
Retrieval ≈0.3–0.5s; verify ≤5ms; context/prompt <1ms.

## 6. Optimization experiments
None adopted (see §11). One error-message accuracy fix: failure
messages now report actual attempts made instead of the attempt budget.

## 7. Before/after latency table
No production latency behavior was changed, so there is no
before/after delta to report. The Groq column above *is* the current
production baseline (cf. ~27.3s clean-run average — consistent).

## 8. Token/sec analysis
Throughput spreads ~64× on identical infrastructure (Groq 6–450 tok/s;
Gemini 1–124 tok/s this window). **No size correlation in either
direction**: the largest output (C002, 1939 tok) was fastest (450
tok/s); a 237-token output took 36.4s (7 tok/s). Equal-scale inputs
(S001 3166 tok → 1.4s vs N001 3281 tok → 36.4s) rule out prompt size
as a driver. Conclusion: per-request provider-side speed variance
(queue/throttle), not input length, output length, or client overhead.

## 9. Retry/widen/correction impact
Only N003 widened (honest-abstention path, +1 generation ≈40s
unavoidable without removing the recovery mechanism — rejected).
Zero corrections fired. Retries do not explain the 1.4–36.4s
single-attempt spread.

## 10. Quality checks
No RAG-affecting change was made, so no quality re-validation was
required. All 5 Groq rows answered (refused=False); N003 followed its
established abstention path. Evaluator untouched; 203/203 offline
tests pass.

## 11. Accepted optimizations
None. Every candidate was measured or statically ruled out (below).

## 12. Rejected optimizations (with evidence)
* **Prompt trimming** (379-word system prompt): input scale shows zero
  latency correlation; rule B's forbidden-style list fixed real
  M001/P001/R001 failures → cutting risks quality for no gain.
* **Context/top-k reduction**: inputs are not the bottleneck; would
  endanger table/parent-clause evidence (S002/N002 class).
* **max_tokens cap**: output length shows zero latency correlation
  (1939 tok fastest); risks truncating citations.
* **Temperature change**: determinism is a grounding control, not a
  speed knob.
* **Retry/widen removal**: sole multi-call source is the justified
  abstention-recovery path; removal trades quality for tail latency.
* **Streaming**: reduces time-to-first-token only, not generation
  time; answers must be complete before verification. Not implemented.
* **Provider switch to Gemini**: this window favored Gemini on
  average (22.1s vs 27.3s gen) but the previous window favored Groq —
  ranking flips run to run. No switch without quality proof.

## 13. Remaining bottleneck
Provider-side per-request generation speed variance (both vendors).
Nothing client-side remains: local path is ~0.5s, retries minimal and
justified, client reused, warmup active.

## 14. Recommendation
Hold production as-is. Revisit only if (a) Groq offers a
lower-variance tier/endpoint, or (b) a future quality-validated reason
to switch providers emerges. p95 is not statistically meaningful at
n=5 and is intentionally omitted.
