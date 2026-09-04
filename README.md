# BIS Standards Assistant — Answering & RAG Repository (`data-answering-bis`)

The production **Answering, RAG (Retrieval-Augmented Generation), and LLM Pipeline** for the Bureau of Indian Standards (BIS) Assistant.

This repository (`data-answering-bis`) consumes grounded context candidates from the production retrieval package, assembles context-enriched system prompts, enforces 100% strict engineering grounding rules, and orchestrates LLM generation to produce accurate, verifiable technical answers with inline clause and page citations.

---

## 1. Project Overview & Multi-Repository Architecture

The BIS Standards Assistant system is partitioned into three decoupled repositories to maintain strict software boundaries and modular development:

```text
┌─────────────────────────┐    ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│  Repo 1: Ingestion      │    │  Repo 2: Chunking & Retrieval    │    │  Repo 3: data-answering-bis      │
│  - Raw PDF extraction   │ ──►│  - Structure & Semantic chunking │ ──►│  - Grounded LLM Context Builder │
│  - OCR normalization    │    │  - BGE-M3 1024-dim Vector Cache  │    │  - System Prompt Formatting      │
│  - Markdown conversion  │    │  - Cross-Encoder Reranker        │    │  - Groq / OpenAI API Generation  │
│                         │    │  - retrieval/ Package & Indexer  │    │  - Citation Parser & Verification│
│                         │    │  - 87-Query Benchmark Suite      │    │  - FastAPI REST Endpoint         │
└─────────────────────────┘    └──────────────────────────────────┘    └──────────────────────────────────┘
```

1. **Repo 1 — Ingestion Repository:** Raw PDF extraction, OCR normalization, and Markdown conversion.
2. **Repo 2 — Processing, Chunking & Retrieval Repository:** Markdown structure extraction, semantic chunking (3,000 chars / 300 overlap), metadata attachment, vector embedding (`BAAI/bge-m3`), vector storage indexing, and retrieval evaluation benchmarks.
3. **data-answering-bis — Answering / RAG Repository (THIS REPOSITORY):** Storage layer connection, context assembly, neighbor chunk expansion, grounding prompt engineering, LLM orchestration, inline citation post-processing, abstention/refusal evaluation, and API serving.

---

## 2. Repository Boundary & Separation of Concerns

To preserve system stability and maintain clean software boundaries:

### What THIS Repository (`data-answering-bis`) Owns
*   **Retrieval Package Consumption:** Importing the query-time retrieval package (`retrieval/` package containing `retrieve()`, `Retriever`, `ChunkStore`, `LocalNpyStore`, `PgVectorStore`).
*   **Dual Storage Seam Connection:** Connecting to candidate storage via `ChunkStore` (local `.npy` memory-mapped vector cache for development or PostgreSQL / pgvector / Supabase for production).
*   **Abstention & Refusal Evaluation:** Inspecting candidate `rerank_score` values, score margins, and `is_mask_restricted` flags to decide whether a query is unanswerable from the corpus before passing context to the LLM.
*   **Context Window Assembly & Expansion:** Deduplicating 300-char overlapping text, preserving clause metadata headers, and executing adjacent chunk stitching (`chunk_index ± 1`) for split tables or clauses.
*   **Prompt Engineering & Grounding Directives:** Injecting strict system prompts that constrain LLM output strictly to retrieved evidence with zero unstated extrapolation.
*   **LLM Orchestration:** Interfacing with LLM APIs (Groq `openai/gpt-oss-120b` / OpenAI API) for answer generation.
*   **Citation Parsing & Verification:** Appending and validating inline citations (`[IS <no>:<year>, Clause <cl>, Page <p>]`) against retrieved metadata.
*   **FastAPI REST Web Service:** Serving the end-to-end RAG pipeline via `POST /api/v1/query`.

### What Belongs to Previous Repositories (FROZEN & IMMUTABLE)
*   **Raw Markdown Ingestion:** Raw PDF parsing and Markdown text live in Repo 1/Repo 2 (`input_md/`).
*   **Chunking Pipeline (`scripts/chunker.py`):** Chunking is **FROZEN at 3,000 chars / 300 overlap** (2,081 chunks across 101 files).
*   **Corpus Vector Generation & Migration:** Pre-computed vector embeddings (`artifacts/bge_m3_enriched_vectors_gpu.npy`) and database loading (`indexing/migrate_from_artifacts.py`).

---

## 3. Production Retrieval Architecture

Retrieval is production-validated on the 87-query benchmark suite (Tier 2B). Repo 3 consumes candidate evidence via the unified entry point:

```python
from retrieval import retrieve, RetrievedEvidence

evidence_list: list[RetrievedEvidence] = retrieve("What is the minimum pH value of water in IS 456?", top_k=10)
```

### Retrieval Execution Flow

```text
User Query
   │
   ▼
1. Query-Side Candidate Filtering (query_side_candidate_mask)
   │ Restricts candidate set if IS standard number is explicitly named in query (is_mask_restricted=True)
   ▼
2. Dense Candidate Retrieval (BAAI/bge-m3@5617a9f, 1024-dim)
   │ Query encoded via BGE-M3 (pinned commit 5617a9f); dot-product or pgvector cosine similarity search
   ▼ Top-10 Candidates Selected
3. Cross-Encoder Reranker (BAAI/bge-reranker-base)
   │ Re-scores (query, enriched_text) pairs using SAME enriched context format
   ▼ Ranked Candidates (list[RetrievedEvidence])
4. Generation Layer (data-answering-bis) Context Assembly & Abstention
```

### Storage Seam (`ChunkStore` Protocol)
The system decouples retrieval logic from physical vector storage via the `ChunkStore` seam:
*   **`LocalNpyStore` (Development / Fallback):** Loads `artifacts/bge_m3_enriched_vectors_gpu.npy` (shape `(2081, 1024)` float32) for fast, zero-dependency local memory-mapped vector search (< 0.05s).
*   **`PgVectorStore` (Production):** Connects to PostgreSQL / Supabase with the `pgvector` extension enabled using `DATABASE_URL` (direct connection via `psycopg` v3 with `prepare_threshold=None` for transaction pooler compatibility). Cosine distance (`<=>`) matches local dot-product similarity scores identically (`score = 1 - dist`).

### Environment-Specific Storage Selection

| Environment | Backend | Purpose |
|---|---|---|
| **SIH Demo / Showcase** | `LocalNpyStore` | Fast, simple, reliable local retrieval |
| **Production / Deployment** | `PgVectorStore` | Supabase/PostgreSQL + pgvector for persistent and scalable storage |

The SIH demo uses `LocalNpyStore` to minimize latency and avoid dependency on database/network availability. The production deployment uses `PgVectorStore` with Supabase/PostgreSQL + pgvector.

---

## 4. End-to-End Retrieval $\to$ Answering Data Flow

The runtime execution flow within `data-answering-bis` proceeds through eight structured stages:

```text
1. User Query Received (POST /api/v1/query)
   │
   ▼
2. Candidate Retrieval (retrieval.retrieve(query, top_k=10))
   │ Output: list[RetrievedEvidence] from LocalNpyStore or PgVectorStore
   ▼
3. Generation-Layer Abstention Check
   │ Inspect top rerank_score logit and score margin against confidence threshold τ (e.g. τ < -2.0)
   │ If scores indicate unanswerable / missing evidence, return standard refusal response
   ▼
4. Neighbor Expansion (Optional)
   │ Fetch adjacent chunk_index ± 1 from output_chunks/ if candidate ends mid-sentence or mid-table
   ▼
5. Context Assembly & Deduplication
   │ Merge overlapping text and format Top-3 to Top-5 evidence items into structured Markdown context blocks
   ▼
6. System Prompt Injection
   │ Inject strict engineering grounding rules & standard identity into LLM prompt
   ▼
7. LLM Answer Generation
   │ Execute API call (Groq / OpenAI) to generate grounded technical text
   ▼
8. Post-Processing & Citation Rendering
   │ Verify facts and append formatted inline citations: [IS 456:2000, Clause 5.4, Page 15]
```

---

## 5. Retrieval $\to$ Answering Data Contract

The retrieval package supplies the generation layer with strictly typed data structures exported by `retrieval.types`:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class RetrievedEvidence:
    """One ranked candidate returned to the generation layer."""
    chunk_id: str                          # Stable chunk identifier (e.g. "456_2000_amd5_reff2021_0014")
    text: str                              # Chunk text with 300-char overlap
    source: str                            # Source document filename (e.g. "456_2000_amd5_reff2021.md")
    clause: Optional[str]                  # Clause label (e.g. "5.4") or None
    heading: Optional[str]                 # Section heading leaf or None
    standard_no: Optional[str]             # Standard ID (e.g. "IS 456") or None
    page_start: Optional[int]              # Physical PDF start page
    page_end: Optional[int]                # Physical PDF end page
    dense_score: Optional[float]           # BGE-M3 cosine similarity score
    rerank_score: Optional[float]          # Cross-Encoder output logit score
    is_mask_restricted: bool               # True if query named IS number and candidate mask fired
    extra: Dict[str, Any] = field(default_factory=dict)
```

---

## 6. Migrated Assets (Received from Repo 2)

This repository includes and depends on the following frozen data assets and production modules:

1.  **`output_chunks/*.json`** (2,081 files): Ground-truth text, clause numbers, headings, and page boundaries for all corpus chunks.
2.  **`artifacts/bge_m3_enriched_vectors_gpu.npy`** (8.5 MB, shape `(2081, 1024)`): Pre-computed BGE-M3 embeddings for local memory-mapped vector search.
3.  **`retrieval/` Package**: Canonical query-time retrieval package containing `retrieve.py`, `types.py`, `store.py`, `pg_store.py`, and `models.py`.
4.  **`indexing/migrate_from_artifacts.py`**: Idempotent database migration script loading `output_chunks/*.json` and vector cache into PostgreSQL + pgvector `chunks` table.
5.  **`artifacts/provenance_index.json`**: Fast lookup index mapping `(standard_no, clause)` and `(source, page)` to chunk IDs.
6.  **`retrieval_queries.md`**: Official 87-query benchmark test suite (80 scored queries + 6 probes + 1 missing).
7.  **`artifacts/chunk_hashes.json`**: SHA-256 hashes verifying zero drift between chunk JSONs and vector cache.

---

## 7. Answering & RAG Components (To Be Implemented in Repo 3)

The following core answering modules are defined in `AGENTS.md` and will be implemented in `data-answering-bis`:

*   **Context Builder & Neighbor Stitcher (`app/generator/context_builder.py`):** Assembles Top-3 to Top-5 evidence blocks, formats metadata headers, and stitches adjacent chunks (`chunk_index ± 1`) when tables or clauses cross chunk boundaries.
*   **Grounding Directives & System Prompts (`app/generator/prompts.py`):** Formats prompts enforcing 100% strict engineering grounding — no unstated extrapolation, exact numerical values/units, and conditional requirement handling.
*   **LLM Service Client (`app/generator/llm_client.py`):** Integrates Groq (`openai/gpt-oss-120b`) and OpenAI API clients.
*   **Refusal & Abstention Evaluator (`app/generator/refusal.py`):** Evaluates top `rerank_score` logit and score margins against threshold $\tau$ (e.g. $\text{logit} < -2.0$) to return standard refusals for out-of-corpus queries:
    > *"The provided Indian Standards documents do not contain sufficient technical information to answer this query."*
*   **Citation Parser & Post-Processor (`app/generator/citations.py`):** Extracts, verifies, and formats inline citations (`[IS <no>:<year>, Clause <cl>, Page <p>]`).
*   **FastAPI Endpoint (`app/main.py`):** Exposes `POST /api/v1/query` REST API serving end-to-end RAG responses.

---

## 8. Repository Layout

```text
data-answering-bis/
├── AGENTS.md                         # Master Development Plan & Instructions
├── README.md                         # Project Overview & Architecture Guide
├── requirements.txt                  # Python runtime dependencies
├── retrieval_queries.md              # Official 87-Query Benchmark Suite
├── output_chunks/                    # 2,081 ground-truth chunk JSON files
├── artifacts/
│   ├── bge_m3_enriched_vectors_gpu.npy # Pre-computed BGE-M3 vector cache (2081, 1024)
│   ├── provenance_index.json         # Standard/clause/page lookup index
│   └── chunk_hashes.json             # SHA-256 chunk integrity hashes
├── indexing/
│   └── migrate_from_artifacts.py     # PostgreSQL + pgvector migration script
└── retrieval/                        # Production retrieval package (from Repo 2)
    ├── __init__.py                   # Package exports (retrieve, RetrievedEvidence, ChunkStore)
    ├── types.py                      # Dataclasses (RetrievedEvidence, ChunkRecord)
    ├── store.py                      # ChunkStore protocol & LocalNpyStore implementation
    ├── pg_store.py                   # PgVectorStore implementation (PostgreSQL/pgvector)
    ├── models.py                     # Pinned model loaders (BGE-M3@5617a9f, BGE-reranker-base)
    └── retrieve.py                   # Main backend retrieval entry point
```

---

## 9. Environment Setup & Configuration

### Prerequisites
*   Python 3.10+
*   PyTorch (CUDA recommended for GPU reranking, CPU supported)
*   PostgreSQL 15+ with `pgvector` extension (optional for database-backed deployment)

### Dependencies (`requirements.txt`)
```ini
torch>=2.0.0
sentence-transformers>=2.2.2
transformers>=4.30.0
numpy>=1.24.0
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
groq>=0.4.0
openai>=1.0.0
psycopg[binary]>=3.1.0
```

### Environment Variables (`.env`)
Create a `.env` file in the repository root:

```ini
# Storage Backend (Optional: set for PostgreSQL/Supabase deployment, omit for LocalNpyStore fallback)
DATABASE_URL=postgresql://user:password@localhost:5432/bis_standards

# Generation LLM API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Service Settings
PORT=8000
HOST=0.0.0.0
```

---

## 10. Database Migration (Optional PostgreSQL / Supabase Path)

To populate a PostgreSQL database with `pgvector` from the local artifacts:

```bash
# Ensure DATABASE_URL is set in .env
python indexing/migrate_from_artifacts.py --drop-first
```

This creates the `chunks` table with a `vector(1024)` column, loads all 2,081 chunks and vectors, and verifies post-load row parity.

---

## 11. Frozen Invariants & Rejection Matrix

Future development in `data-answering-bis` MUST obey the following frozen invariants:

1.  **Chunking is Frozen:** Do NOT modify chunk sizes (3,000 chars), overlap (300 chars), or chunk JSON files in `output_chunks/`.
2.  **Embedding Model is Frozen:** Do NOT change candidate embedder (`BAAI/bge-m3` pinned to revision `5617a9f`) or vector dimensions (1024-dim).
3.  **Enriched Representation is Frozen:** Embeddings and reranking MUST use the `Standard + Clause + Heading + Text` representation.
4.  **Re-Use Pre-Computed Vectors:** Always load `artifacts/bge_m3_enriched_vectors_gpu.npy` or query PostgreSQL via `PgVectorStore`. Do NOT re-embed the corpus.

### Rejected Architectures (DO NOT REVIVE)

| Rejected Approach | Benchmark Outcome | Rejection Rationale |
|---|---|---|
| **BM25 Primary** | Clause @1: 25% | Fails on semantic paraphrases and clause headers. |
| **all-MiniLM-L6-v2** | Answerability: 68.4% | 512-token limit truncated 72% of chunks, losing tail context. |
| **Hybrid BM25 + Dense RRF** | Clause @1 dropped 100% $\to$ 50% | Naive rank fusion pulled up keyword-heavy wrong-clause chunks. |
| **Raw-Text Cross-Encoder** | Answerability fell 89.5% $\to$ 63.2% | Without metadata headers, reranker favors generic prose over clauses. |
| **Table Router** | Over-triggered on prose | Dense enriched retrieval natively handles tabular chunks without routing. |

---

## 12. Evaluation Framework & Benchmark Targets

System evaluation is conducted against the **87-query benchmark suite** (`retrieval_queries.md`):

*   **Retrieval Metrics:** Document Retrieval @1 (Target: 100%), Clause Retrieval @1 (Target: 100%), Clause MRR@3 (Target: 1.000).
*   **Strict Answerability:** Target $\ge 89.5\%$ phrase-hit verification against ground-truth answer keys.
*   **Citation Accuracy:** 100% verification that cited clauses and page numbers match retrieved chunk metadata.
*   **Zero Hallucination:** 0% tolerance for ungrounded numerical values, fabricated clauses, or unverified standard numbers.

### Reusable End-to-End Query Suite (`evaluation/`)

A 30-query regression suite covering expert users (IS/clause-specific) and
normal users who describe only their product (`product_to_standard`,
`product_requirement`):

*   `evaluation/test_queries.json` — 8 categories (`supported_exact`,
    `supported_clause`, `supported_numerical`, `supported_multi_clause`,
    `product_to_standard`, `product_requirement`, `out_of_corpus`,
    `general_bis`) with stable IDs and `expected` (`answer`/`refusal`/`review`).
*   Start the API first: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
*   Run: `python evaluation/run_queries.py` (see `--help` for
    `--url/--top-k/--output/--category/--limit`); results go to
    `evaluation/results/latest_results.json`.
*   Verdicts: `PASS` (grounded answer / correct refusal), `FAIL` (refused or
    unverified when an answer was expected, or answered when refusal was
    expected), `REVIEW` (general-BIS questions; `product_*` passes are still
    flagged for manual semantic review since no substring matching is used).
    `C002` (Clause 26.5) is a known `citation_mismatch` regression kept
    permanently as `expected: answer` — do not change RAG behavior to force it.

---

## 13. Implementation Roadmap & Current Status

Development in `data-answering-bis` proceeds in 8 structured phases:

- [x] **Phase 1: Repository Architecture & Migration Plan** — *(Completed: Asset inventory, data contract definition, `retrieval/` package alignment, and master `AGENTS.md` specification).*
- [x] **Phase 2: Data Asset & Retrieval Package Integration** — *(Completed: Imported `output_chunks/`, vector cache, `indexing/` script, and `retrieval/` package supporting `LocalNpyStore` and `PgVectorStore`).*
- [ ] **Phase 3: Context Assembly & Neighbor Expansion** — Implement prompt context builder, metadata formatting, and adjacent chunk stitcher.
- [ ] **Phase 4: System Prompt Engineering & Grounding Directives** — Implement system prompts enforcing 100% strict engineering grounding.
- [ ] **Phase 5: LLM Integration & Orchestration** — Wire Groq (`openai/gpt-oss-120b`) / OpenAI API connectors for answer generation.
- [ ] **Phase 6: Citation Parser & Post-Processor** — Implement automated post-processing to append verified inline citations (`[IS ..., Clause ..., Page ...]`).
- [ ] **Phase 7: Refusal & Abstention Logic** — Wire confidence-gated refusal when top reranker score is below threshold $\tau$ or margins indicate unanswerable query.
- [ ] **Phase 8: FastAPI Service & Benchmark Evaluation** — Build `POST /api/v1/query` REST API and evaluate end-to-end performance on the 87-query benchmark.

---

### Component Status Summary

| Component | Status | Location / Details |
|---|---|---|
| **Chunking Pipeline** | **FROZEN & COMPLETED** | 2,081 chunks across 101 files (`output_chunks/`) |
| **Vector Storage Assets** | **FROZEN & COMPLETED** | `artifacts/bge_m3_enriched_vectors_gpu.npy` & `indexing/migrate_from_artifacts.py` |
| **Production Retrieval Package** | **MIGRATED & COMPLETED** | `retrieval/` package (`retrieve()`, `RetrievedEvidence`, `LocalNpyStore`, `PgVectorStore`) |
| **Master Specifications** | **COMPLETED** | `AGENTS.md` & `README.md` |
| **LLM Context Builder & Prompts** | **CURRENT WORK (Phase 3–5)** | `app/generator/` (to be implemented) |
| **Citation Post-Processor & Refusal** | **FUTURE (Phase 6–7)** | `app/generator/` (to be implemented) |
| **FastAPI REST Service** | **FUTURE (Phase 8)** | `app/main.py` (to be implemented) |
