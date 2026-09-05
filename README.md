# BIS Standards Assistant — Answering & RAG Repository (`data-answering-bis`)

The **Answering / RAG / LLM pipeline** for the Bureau of Indian Standards (BIS) Assistant. Given a natural-language question about BIS documents, it retrieves grounded context chunks, assembles a structured prompt, generates an answer with a configured LLM, and verifies every technical claim against retrieved evidence with inline page/clause citations — or refuses when the evidence is insufficient.

The canonical programmatic interface is the `app.generator` library (analogous to Repo 2's `retrieve()`). **Direct function calls are the primary backend integration — FastAPI is an optional HTTP adapter and is never required by the answering pipeline:**

```python
from app.generator import answer, warmup

warmup()  # optional, once at backend startup; preloads retrieval
          # models so the first query skips ~40s of model load.
          # Makes zero LLM calls and needs no API key.

result = answer(
    "What is the minimum pH value of water for mixing concrete in IS 456?",
    top_k=3,
    mode="ask",  # or mode="product_match" (prompt framing only;
                 # retrieval, verification, refusal are identical)
)
```

`POST /api/v1/query` (FastAPI, `app/main.py`) only validates the request, calls this same `answer()`, and serializes the result.

---

## 1. Multi-Repository Architecture

| Repo | Responsibility |
|---|---|
| **Repo 1 — Ingestion** | Raw PDF extraction, OCR normalization, Markdown conversion. |
| **Repo 2 — Chunking & Retrieval** | Structure extraction, semantic chunking (3000 chars / 300 overlap), metadata, `BAAI/bge-m3` embeddings, vector indexing, retrieval benchmarks. |
| **Repo 3 — This repository (`data-answering-bis`)** | Vector-store connection, runtime retrieval, reranking, context assembly, prompt engineering, LLM orchestration (Groq default, Gemini alternative), citation verification, refusal/correction logic, telemetry, optional API serving. |

### What THIS repository owns
* Consuming the `retrieval/` package (`retrieve()`, `Retriever`, `ChunkStore`, `LocalNpyStore`, `PgVectorStore`).
* Query-time candidate filtering, dense retrieval, CrossEncoder reranking.
* Context assembly, neighbor expansion, table widening, retrieval-expansion fallback.
* System prompts, grounding rules, LLM orchestration via a provider abstraction.
* Citation parsing/repair/verification, abstention & refusal, one-shot correction retry.
* Explicit `warmup()` for direct-library backends (server needs no warmup endpoint).
* Request-scoped telemetry, error mapping, optional FastAPI adapter, evaluation tooling.

### Frozen (belongs to earlier repos — do not touch)
* Chunking: **3000 chars / 300 overlap, 2,081 chunks in 101 files** (`data/chunks/*.json`).
* Candidate embedder: **`BAAI/bge-m3` pinned revision `5617a9f`** (1024-dim); reranker `BAAI/bge-reranker-base`.
* Corpus vectors are pre-computed (`data/vectors/bge_m3_enriched_vectors.npy`); never re-embed at runtime.
* Rejected approaches stay rejected: BM25-primary, hybrid RRF, raw-text reranking.

---

## 2. Repository Layout (actual)

```text
data-answering-bis/
├── AGENTS.md                    # Master spec & agent instructions (authoritative)
├── README.md                    # This file
├── requirements.txt             # Runtime dependencies
├── .env.example                 # Configuration template (copy to .env; never commit .env)
├── retrieval/                   # Query-time retrieval package (from Repo 2)
│   ├── __init__.py              # Exports: retrieve, RetrievedEvidence, ChunkStore, ...
│   ├── retrieve.py              # retrieve(query, top_k=10): dense + rerank pipeline
│   ├── types.py                 # RetrievedEvidence, ChunkRecord dataclasses
│   ├── store.py                 # ChunkStore protocol + LocalNpyStore
│   ├── pg_store.py              # PgVectorStore (PostgreSQL/pgvector production path)
│   └── models.py                # Pinned model loaders (CUDA preferred, loud CPU fallback)
├── app/
│   ├── main.py                  # Optional FastAPI adapter: validates, calls answer(), serializes
│   └── generator/               # Standalone library: never imports FastAPI/Uvicorn
│       ├── __init__.py          # Canonical answer()/warmup() API + process singletons
│       ├── adapters.py          # to_ask_response()/to_match_response() frontend contracts
│       ├── pipeline.py          # run_query(): the one end-to-end orchestration
│       ├── llm_client.py        # LLMProvider protocol, Groq/Gemini/OpenAI providers
│       ├── prompts.py           # System prompt + grounding/citation rules
│       ├── context_builder.py   # Context blocks, neighbor expansion, dedup
│       ├── citations.py         # Citation parsing, repair, verification
│       ├── refusal.py           # Abstention logic (τ = 0.5), canonical refusal text
│       └── telemetry.py         # Request-scoped LLM/stage telemetry
├── data/
│   ├── chunks/                  # 101 chunk JSON files (+ human-readable .md mirrors)
│   ├── vectors/                 # Pre-computed BGE-M3 cache (2081, 1024)
│   └── provenance_index.json    # (standard_no, clause) / (source, page) lookup asset
├── indexing/
│   └── migrate_from_artifacts.py# PostgreSQL + pgvector migration helper (deploy-time;
│                                # defaults reference the old output_chunks/artifacts layout —
│                                # pass explicit paths for this data/ layout)
├── evaluation/
│   ├── test_queries.json        # 30-query regression suite (8 categories, stable IDs)
│   ├── progress.py              # Shared QueryProgress display (any query count, stdlib only)
│   ├── run_queries.py           # Runs the suite against a live API; records telemetry
│   ├── compare_llm_latency.py   # 5-query Groq-vs-Gemini latency experiment
│   ├── run_benchmark.py         # Generic query-file harness (answer() + backend adapters,
│   │                            # --mode/--delay-secs/--no-progress, timestamped responses file)
│   └── results/                 # latest_results.json (27/30 baseline), responses_<ts>.json runs
└── tests/                       # Offline unit tests (no keys/GPU/network required)
    ├── test_api.py, test_pipeline.py, test_citations.py, test_refusal.py, ...
    ├── test_provider.py         # Provider selection, answer() API, delegation
    ├── test_warmup.py           # Direct answer()/warmup() usage, zero-LLM-call warmup
    ├── test_modes.py            # ask vs product_match behavior
    ├── test_progress.py         # Generic progress/counter/countdown behavior
    ├── test_telemetry.py        # Telemetry counters/flags/stages
    ├── test_evaluation.py       # Suite loading + verdict classification
    └── retrieval_queries.md     # Chunking-evaluation query-type notes (Repo-2 era doc)
```

---

## 3. Setup & Configuration

### Prerequisites
* Python 3.10+, PyTorch (CUDA recommended; CPU works with a warning).
* `pip install -r requirements.txt` (includes `groq`, `google-genai`, `fastapi`, `sentence-transformers`).

### Use the library directly (primary integration)
```python
from app.generator import answer, warmup

warmup()  # once at backend startup; loads chunk index + BGE-M3 +
          # reranker + vector store (~40s once). No LLM call, no key.
          # Safe to skip: answer() lazily initializes on first call.

result = answer("What is the minimum pH of water per IS 456?", mode="ask")
result.answer       # grounded text with [IS ...] citations
result.citations    # verified CitationOut list
result.refused      # True when evidence is insufficient
result.telemetry    # per-query counters (no secrets)
```
`mode="product_match"` selects product-matching prompt instructions only; retrieval, verification, refusal, and retries are identical. `to_ask_response()` / `to_match_response()` (`adapters.py`) map `QueryResult` to the frontend contracts without new LLM calls.

### Configuration (`.env`, see `.env.example`)
```ini
LLM_PROVIDER=groq              # "groq" (default) or "gemini"
GROQ_API_KEY=...               # required for the default provider
GROQ_MODEL=                    # optional override (default: openai/gpt-oss-120b)
GEMINI_API_KEY=...             # required for LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash-lite   # optional override (same default)
BIS_WARMUP=0                   # server-only shortcut for warmup(); direct-library
                               # backends call warmup() explicitly instead
DATABASE_URL=...               # optional; only for the PgVectorStore production path
```
Key resolution order: explicit argument → process env → `.env` file. `BIS_DEVICE=cpu|cuda` optionally pins the retrieval device (CUDA preferred by default).
No `PORT`/`HOST` variables exist — pass host/port to uvicorn directly.

### Run the optional API
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
With `BIS_WARMUP=1`, server startup calls `warmup()` so the first query serves in ~0.5s instead of ~40s. The API adds no RAG behavior — it is the same `answer()` behind HTTP.

---

## 4. Answering Pipeline (actual call graph)

```text
warmup()                                    # app/generator/__init__.py (optional preload,
                                            # no LLM call; answer() works without it)
answer(query, top_k, mode)                  # app/generator/__init__.py (canonical API)
  └─ run_query()                            # app/generator/pipeline.py (one implementation)
       ├─ retrieve(query, top_k=10)         # frozen retrieval package
       │    ├─ query_side_candidate_mask    # IS-number filter (unmasked product queries skip it)
       │    ├─ BGE-M3 encode → dot-product  # 2081×1024, CUDA
       │    └─ bge-reranker-base re-score   # top-10 enriched pairs, CUDA
       ├─ pre-generation refusal (top rerank < τ=0.5 → refuse; sigmoid scale)
       ├─ near-miss expansion (unmasked queries scoring in [0.20, 0.5) only)
       ├─ build_context (top_k blocks) → build_prompt (system + user)
       ├─ generate_answer → provider (Groq/Gemini), temperature 0.0
       ├─ verify_answer (citation ↔ evidence linkage; table-fragment repair,
       │                 Foreword→N/A front-matter rule)
       ├─ widen once (zero-cite pass + reserve evidence) → re-verify
       ├─ correction retry once (mismatch, or non-abstention zero-cite) → re-verify
       └─ QueryResult(answer, citations, refused, refusal_reason,
                      retrieval_meta, telemetry)
```

FastAPI (`app/main.py`, optional) only validates the request, calls `answer()` with its
lifespan singletons, and serializes `QueryResult` to JSON (`/healthz`,
`/api/v1/device`, `/api/v1/query`). Error mapping: `ValueError` → 400,
provider `RuntimeError` → 502, anything else → 500. The pipeline never
imports FastAPI — stopping the server changes nothing about `answer()`.

---

## 5. LLM Providers

| Provider | Class | Default model | Key | Status |
|---|---|---|---|---|
| Groq | `GroqProvider` | `openai/gpt-oss-120b` | `GROQ_API_KEY` | **Default** |
| Gemini | `GeminiProvider` | `gemini-3.5-flash-lite` | `GEMINI_API_KEY` | Alternative (no full quality benchmark yet — see §7) |
| OpenAI | `OpenAIProvider` | `openai/gpt-oss-120b` | `OPENAI_API_KEY` | Lazy optional (package not required) |

Select with `LLM_PROVIDER=groq|gemini` (or `build_provider(name)` in code,
or `answer(..., provider=...)` per call). API keys resolve as explicit
argument → process env → `.env` file; SDKs are imported lazily so the
unselected provider needs neither package nor key.
The same pipeline runs either way — only the transport differs
(chat-completions vs `generate_content` with system instruction).
One SDK client is reused per provider instance; the app-level retry
policy (2 transient retries, 1s/2s backoff) is identical for both.

---

## 6. Telemetry (per query, in every API response)

```json
"telemetry": {
  "llm_provider": "groq", "llm_model": "openai/gpt-oss-120b",
  "llm_generation_attempts": 2, "llm_api_calls": 2,
  "correction_retry": false, "widen_retry": true, "retrieval_expansion": false,
  "latency_ms": 35935.3,
  "stages": {"retrieval_ms": 496.1, "gen_initial_ms": 18524.0,
             "gen_widen_ms": 40674.5, "verify_ms": 5.3, "...": 0},
  "prompt_chars": 8232, "prompt_tokens_total": 8586, "completion_tokens_total": 420
}
```
(`groq_api_calls` is retained as a read alias of `llm_api_calls`.)
Counts cover observable provider invocations; hidden SDK-internal HTTP
retries are documented as unobservable. No secrets are ever exposed.

Exhausted transient provider errors (rate-limit/timeout/5xx after all
retries) raise a uniform `"<provider> API error after N attempts: ..."`
→ **HTTP 502** with the cause in the detail, instead of an opaque 500.
Non-transient errors (auth, bad request) propagate unchanged.

---

## 7. Evaluation

### Protected baseline: 27/30 (clean run, `evaluation/results/latest_results.json`)
| Category | Score |
|---|---|
| Supported Exact | 4/4 |
| Supported Clause | 5/5 |
| Supported Numerical | 4/5 (N002 *or* N003 — same IS-1005-table abstention variance, alternates per run) |
| Multi-Clause | 3/3 |
| Product → Standard | 5/5 |
| Product Requirement | 3/3 |
| Out-of-Corpus refusals | 3/3 |
| General BIS | 2 REVIEW (by design) |

Run (server must be up): `python evaluation/run_queries.py --url http://127.0.0.1:8000/api/v1/query`.
Scoring is grounding-based (refusal flags, present + verified citations, retrieval metadata) — never naive substring matching. Quota-interrupted runs record `ERROR`, never PASS/FAIL; do not confuse them with quality results.
All runners share `evaluation/progress.py` (`QueryProgress`: `04/20 (20%)` counters, waiting/finish lines with latency, live `--delay-secs` countdown, stderr only, auto-off without a terminal, `--no-progress` override) — display only, never affecting execution, results, or timing defaults.

### Gemini-only 20-query validation (`evaluation/results/responses_20260905T144223Z.json`)
Sequential `answer()` run (18 `ask` + 2 `product_match`, 8s inter-query delays, no code changes), each entry saved as `{query, mode, response, evaluation}` where `response` is the exact `to_ask_response()` / `to_match_response()` output:
* 2/2 out-of-context queries correctly refused pre-generation (scores ~0.05/0.001, zero LLM spend).
* 0 pipeline errors; no rate-limit (429) errors.
* Every answered query fully verified: 100% `verified:true` citations, no negative pages, no missing years.
* Q19 (tyres product_match) was a correct conservative abstention with `matches: []` — IS 2415 covers tubes while the true tyres standard (IS 2414) has no chunks in the corpus.
* Q20 (fitness-app product_match) exposed a borderline applicability call (IS 17737 @0.556 on truncated scope evidence). This motivated the stricter product_match scope rule now in production (applicability requires stated scope coverage; similarity alone never qualifies) — covered offline and live re-verified on 9 queries with no regressions.

### Latency experiment (`evaluation/compare_llm_latency.py`, 5 queries)
`python evaluation/compare_llm_latency.py --provider both` — same pipeline, only the backend differs. Measured: no Gemini speedup (4 mutually answered queries; Gemini ~24% slower on average, Groq far more variable at 1.9–81.7s). N003/Gemini returned a provider-side HTTP 404 in that run (same key/model answers the other 4) — an infrastructure anomaly, not a quality signal. Note: Gemini 404s have also proven transient on retry for other queries, so a 404 alone never implies a deterministic model/query failure. **No full Gemini quality benchmark has been run; do not claim one.**

### Known limitations
* N002/N003 alternate PASS/FAIL across runs (reranker buries the IS-1005 Table-1 chunk; honest abstention follows).
* Groq on-demand latency varies ~70× run to run (5–400 tok/s observed); multi-call tails (up to ~212s seen) are provider-side.
* Groq daily token quota interrupts long runs; the 502 mapping above keeps these diagnosable.

---

## 8. Verified Performance Snapshot

| Stage | Measured |
|---|---|
| Model load | ~38.6s once (eliminated from requests via `BIS_WARMUP=1`) |
| Warm retrieval (encode + matmul + mask + rerank) | ~0.3–0.7s (matmul 0.6ms, mask 6.6ms) |
| Context / prompt / verification | ~0.1–30ms |
| Groq generation | ~90–99% of request time (avg ~22.6s initial, ~37.6s widen) |
| Clean-run average | ~27.3s (was ~35s before client reuse + warmup) |

No promises beyond these measurements. Both models stay on CUDA (`/api/v1/device` reports placement); the retriever is a process singleton — never loaded per request.

---

## 9. Tests

```bash
python -m pytest tests/ -q   # offline: no keys, GPU, network, or quota needed
```
**273 passed.** Covers pipeline orchestration, citations/repair/verification, refusal,
prompts, context assembly, providers + `answer()` delegation, `warmup()` direct-library
usage (including zero-LLM-call verification), `ask` vs `product_match` modes, runner
progress display, telemetry, evaluation verdicts, latency-experiment guards, and device/store paths. The suite runs with
no provider API keys set — any live LLM call would fail instead of passing silently.

---

## 10. Frozen Invariants

* Chunking, chunk JSONs, embedding model/revision, enriched representation, pre-computed vectors.
* `retrieve(query, top_k=10)` signature and the BGE-M3 → CrossEncoder architecture.
* Refusal threshold τ = 0.5; strict citation linkage; one-shot correction/widening; no fabricated evidence or hardcoded answers.
* Rejected retrievers (BM25-primary, hybrid RRF, raw-text rerank) stay rejected.
* Library-first: `answer()`/`warmup()` never import FastAPI/Uvicorn; `app/main.py` stays an optional adapter.
