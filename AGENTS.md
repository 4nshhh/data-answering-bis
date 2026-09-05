# AGENTS.md — Master Development Plan & Instructions for the Answering / RAG Repository

This document is the **authoritative master specification, development blueprint, and instruction manual** for the **Answering / RAG / LLM Repository (`data-answering-bis`)** of the BIS Standards Assistant.

Any future Antigravity AI agent session working in this repository **MUST** read and strictly follow the directives, data contracts, grounding rules, and architectural boundaries defined in this document.

---

## 1. Project Purpose

The primary goal of this repository (`data-answering-bis: Answering & RAG`) is to take natural language user queries regarding Bureau of Indian Standards (BIS) documents, retrieve relevant grounded context chunks, assemble structured prompts, and execute an LLM-based generation pipeline to produce **highly accurate, technically precise, and verifiable grounded answers with inline page and clause citations**.

### Multi-Repository Architecture
The overall BIS Standards Assistant is partitioned into three decoupled repositories:

1. **Repo 1 — Ingestion Repository:** Raw PDF extraction, OCR normalization, and Markdown conversion.
2. **Repo 2 — Processing, Chunking & Retrieval Repository:** Markdown structure extraction, semantic chunking (3000 chars / 300 overlap), metadata attachment, vector embedding (`BAAI/bge-m3`), vector storage indexing, and retrieval evaluation benchmarks.
3. **data-answering-bis — Answering / RAG Repository (THIS REPOSITORY):** Vector store connection / cache loading, runtime candidate retrieval, reranking, context assembly, LLM prompt engineering, grounded answer generation, citation formatting, and API serving.

---

## 2. Repository Boundary & Separation of Concerns

To preserve system stability and maintain clean software boundaries, the responsibilities of this repository are strictly scoped:

### What THIS Repository (`data-answering-bis`) Owns
*   Importing the production query-time retrieval package (`retrieval/` package containing `retrieve()`, `Retriever`, `ChunkStore`, `LocalNpyStore`, `PgVectorStore`).
*   Connecting to the storage layer via the `ChunkStore` seam: either local vector cache (`artifacts/bge_m3_enriched_vectors_gpu.npy`) or PostgreSQL/pgvector (`PgVectorStore`).
*   Runtime query candidate filtering (`query_side_candidate_mask` / `is_mask_restricted`).
*   Dense candidate retrieval (`BAAI/bge-m3@5617a9f`) and Cross-Encoder reranking (`BAAI/bge-reranker-base`).
*   Context window assembly, chunk ordering, and optional neighboring-chunk expansion (`chunk_index ± 1`).
*   System prompt formatting, grounding constraints, and LLM orchestration (Groq / OpenAI API).
*   Inline citation generation (`[IS <no>:<year>, Clause <cl>, Page <p>]`) and evidence verification.
*   **Abstention & Refusal Handling:** Owning the "not in corpus" or insufficient-evidence decision based on `rerank_score` values, score margins, and `is_mask_restricted` (since `retrieve()` explicitly returns candidates unconditionally without internal abstention).
*   Optional FastAPI HTTP adapter (`POST /api/v1/query`) around the `answer()` library, and end-to-end RAG benchmark evaluation.

### What BELONGS TO PREVIOUS REPOSITORIES (DO NOT RE-IMPLEMENT OR MODIFY)
*   **Raw Markdown Ingestion & Extraction:** Raw document text lives in Repo 1/Repo 2 (`input_md/`). Do NOT re-parse raw PDFs or Markdown source files.
*   **Chunking Pipeline (`scripts/chunker.py`):** Chunking is **FROZEN at 3000 chars / 300 overlap** (2,081 chunks across 101 files). Do NOT re-chunk documents or modify chunk boundaries.
*   **Chunk Invariants (`scripts/validate.py`):** Validation rules and chunk ID structures are immutable.
*   **Corpus Vector Generation & Migration:** Embeddings for all 2,081 chunks are pre-computed in `artifacts/bge_m3_enriched_vectors_gpu.npy` and populated into PostgreSQL via `indexing/migrate_from_artifacts.py`. Do NOT re-embed the corpus at startup.

---

## 3. Completed & Frozen Retrieval Pipeline

Retrieval in this repository is **frozen and production-validated** on the 87-query benchmark suite (Tier 2B). Future agents MUST NOT alter this pipeline signature (`retrieve(query, top_k=10) -> list[RetrievedEvidence]`) without explicit benchmark justification.

```text
User Query
   │
   ▼
1. Query-Side Candidate Filtering (query_side_candidate_mask)
   │ Restricts candidates if IS standard number is explicitly mentioned (is_mask_restricted=True)
   ▼
2. BGE-M3 Dense Retrieval (1024-dim, 8192 token window, 0% truncation)
   │ Encodes query using BAAI/bge-m3@5617a9f; dot-product / pgvector cosine against store
   ▼ Top-10 Candidates Selected
3. Cross-Encoder Reranker (BAAI/bge-reranker-base)
   │ Re-scores (query, enriched_chunk_text) pairs using SAME Enriched representation
   ▼ Ranked Evidence (RetrievedEvidence)
4. Top-K Selection & Abstention Check for LLM Generation Context
```

### Frozen Pipeline Specifications

*   **Candidate Embedding Model:** `BAAI/bge-m3` pinned to revision `5617a9f61b028005a4858fdac845db406aefb181` (1024-dimensional, 8192-token context limit, float32, L2-normalized).
*   **Enriched Text Representation:**
    ```text
    Standard: <standard_no>
    Clause: <clause>
    Heading: <heading_path joined by ' > ' or heading>

    <raw chunk text>
    ```
    *CRITICAL: Prepending this structured metadata header at embedding and reranking time is MANDATORY. Dropping it degrades clause accuracy.*
*   **Vector Storage Seam (`ChunkStore`):**
    *   *Default / Development Path:* `LocalNpyStore` using `artifacts/bge_m3_enriched_vectors_gpu.npy` (or `bge_m3_enriched_vectors.npy`, shape `(2081, 1024)`).
    *   *PostgreSQL / Production Path:* `PgVectorStore` querying PostgreSQL table `chunks` with `vector(1024)` column using cosine distance (`<=>`), matching dot-product scores identically (`score = 1 - dist`).
*   **Candidate Window (Top-N):** **Top-10 dense candidates** are passed to the reranker.
*   **Reranker Model:** `BAAI/bge-reranker-base` (`sentence_transformers.CrossEncoder`).
*   **Reranker Input:** Passed as `(query_text, enriched_chunk_text)` pairs.
*   **Query Filtering Rule:** `query_side_candidate_mask` extracts explicit standard numbers (e.g. `"IS 456"`, `"IS 18000"`) from user queries and masks out chunks from non-matching standards.

### Environment-Specific Storage Selection

* **SIH Demo / Showcase:** Use `LocalNpyStore` with the local `.npy` vector cache for maximum speed, simplicity, and reliability during live demonstrations.
* **Production / Deployment:** Use `PgVectorStore` with PostgreSQL/Supabase + pgvector for persistent centralized storage and scalability.

---

## 4. Retrieval Experiment History & Rejection Matrix

Ten retrieval experiments were conducted in Repo 2. Future agents MUST NOT re-introduce rejected configurations:

| Approach / Model | Performance Summary | Status | Rejection Rationale / Directive |
|---|---|---|---|
| **BM25 Keyword Baseline** | Doc @1: 100%, Clause @1: 25%, Ans: 84.2% | **Excluded as Primary** | Fails on semantic paraphrases and clause headers. |
| **all-MiniLM-L6-v2 (Truncated)** | Doc @1: 94.7%, Ans: 68.4% | **Excluded** | 512-token limit truncated 72% of chunks, losing tail context. |
| **BGE-M3 Raw Dense** | Doc @1: 100%, Clause @1: 75%, Ans: 84.2% | **Baseline Only** | Lacks explicit clause header anchors. |
| **BGE-M3 Context-Enriched Dense** | Doc @1: 100%, Clause @1: **100%**, Ans: 89.5% | **Selected Candidate Retriever** | Metadata header gives exact semantic anchoring. |
| **Hybrid BM25 + Dense RRF ($k=60$)** | Clause @1 dropped **100% $\to$ 50%** | **REJECTED** | Naive rank fusion pulled up keyword-heavy wrong-clause chunks. |
| **Raw-Text Cross-Encoder** | Answerability fell **89.5% $\to$ 63.2%** | **REJECTED** | Without metadata headers, reranker favors generic prose over clauses. |
| **Enriched Cross-Encoder** | Clause @1: **100%**, Ans: **89.5%**, Solves Q8 | **SELECTED RERANKER** | Passing enriched text to Cross-Encoder achieves peak precision. |
| **MultiQuery Expansion** | Bit-identical output under reranker | **Adopted Fallback Only** | Use only as fallback if Cross-Encoder reranker is disabled. |
| **Confidence Clause Filter** | Tied reranker on Doc/Ans (73/80) | **Adopted Extra** | Intersects extracted clause labels with IS mask. |
| **Table Router** | Evaluated on structural queries | **REJECTED** | Dense enriched retrieval natively handles tabular chunks without routing. |

---

## 5. Migrated Assets (Received from Repo 2)

This repository receives and depends on the following static data assets and production retrieval modules generated by Repo 2:

1.  **`output_chunks/*.json`** (2,081 files)
    *   *Purpose:* Ground-truth text, clause numbers, headings, and page boundaries for all corpus chunks.
2.  **`artifacts/bge_m3_enriched_vectors_gpu.npy`** (or `bge_m3_enriched_vectors.npy`, 8.5 MB numpy array, shape `(2081, 1024)`)
    *   *Purpose:* Pre-computed BGE-M3 embeddings for local memory-mapped vector search (< 0.05s).
3.  **`retrieval/` Package (`retrieval/retrieve.py`, `retrieval/types.py`, `retrieval/store.py`, `retrieval/pg_store.py`, `retrieval/models.py`)**
    *   *Purpose:* Canonical production retrieval package exporting `retrieve()`, `RetrievedEvidence`, `ChunkRecord`, and `ChunkStore`.
4.  **`indexing/migrate_from_artifacts.py`**
    *   *Purpose:* Database migration script loading `output_chunks/*.json` and vector cache into PostgreSQL + pgvector `chunks` table.
5.  **`artifacts/provenance_index.json`**
    *   *Purpose:* Fast lookup index mapping `(standard_no, clause)` and `(source, page)` to chunk IDs.
6.  **`retrieval_queries.md`** (87-query benchmark file)
    *   *Purpose:* Official benchmark test suite (80 scored queries + 6 probes + 1 missing) for RAG system evaluation.
7.  **`artifacts/chunk_hashes.json`**
    *   *Purpose:* SHA-256 verification hashes ensuring zero drift between chunk JSONs and vector cache.

---

## 6. Retrieval $\to$ Answering Data Contract

The retrieval layer MUST supply the LLM answering layer with a strictly typed data payload. The canonical dataclasses are exported directly by `retrieval.types`:

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ChunkRecord:
    id: str
    text: str
    source: str
    clause: Optional[str]
    heading: Optional[str]
    standard_no: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    low_confidence: bool

@dataclass
class RetrievedEvidence:
    """One ranked candidate returned to the generation layer."""
    chunk_id: str                          # Stable chunk identifier (e.g. "12182_2025_0005")
    text: str                              # Chunk text with 300-char overlap
    source: str                            # Source document filename (e.g. "12182_2025.md")
    clause: Optional[str]                  # Clause label (e.g. "5.2.2.1") or None
    heading: Optional[str]                 # Section heading leaf or None
    standard_no: Optional[str]             # Standard ID (e.g. "IS 12182") or None
    page_start: Optional[int]              # Physical PDF start page
    page_end: Optional[int]                # Physical PDF end page
    dense_score: Optional[float]           # BGE-M3 cosine similarity score
    rerank_score: Optional[float]          # Cross-Encoder output score (sigmoid probability in [0, 1])
    is_mask_restricted: bool               # True if query named IS number and mask fired
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "clause": self.clause,
            "heading": self.heading,
            "standard_no": self.standard_no,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "dense_score": self.dense_score,
            "rerank_score": self.rerank_score,
            "is_mask_restricted": self.is_mask_restricted,
            "extra": dict(self.extra),
        }
```

---

## 7. Answering / RAG Architecture & Context Flow

The end-to-end execution flow within data-answering-bis follows eight structured stages:

```text
1. User Query Received
   │
   ▼
2. Candidate Retrieval (`retrieve(query, top_k=10)`)
   │ Returns list[RetrievedEvidence] using ChunkStore (LocalNpyStore or PgVectorStore)
   ▼
3. Abstention & Refusal Check (Generation Layer)
   │ Inspect top rerank_score and score margin against threshold τ
   │ If scores indicate unanswerable query, return standard refusal response
   ▼
4. Neighbor Expansion (Optional)
   │ Check if top chunks need chunk_index ± 1 for incomplete tables/clauses
   ▼
5. Context Assembly & Deduplication
   │ Format Top-K evidence into structured Markdown context blocks with headings
   ▼
6. System Prompt Injection
   │ Inject strict grounding directives & IS standard context into LLM prompt
   ▼
7. LLM Answer Generation (Groq / OpenAI API)
   │ Generate response constrained strictly to retrieved evidence
   ▼
8. Post-Processing & Citation Rendering
   │ Append verified inline citations: [IS 456:2000, Clause 5.4, Page 15]
```

---

## 8. Context Construction & Neighbor Expansion Rules

When formatting retrieved evidence into the prompt context for the LLM:

1.  **Top-K Budget:** Include the **Top-3 to Top-5** re-ranked evidence items by default. Keep all 10 candidates in reserve for complex queries.
2.  **Preserve Context Headers:** Format each context block clearly so the LLM understands standard identity and clause hierarchy:
    ```markdown
    ### Context Block [1]
    - **Standard:** IS 456:2000
    - **Clause:** 5.4
    - **Heading:** 5 MATERIALS > 5.4 Water
    - **Location:** Page 15-16 (Chunk ID: 456_2000_amd5_reff2021_0014)

    ```text
    <evidence text>
    ```
    ```
3.  **Neighbor Expansion Rule:** If a retrieved chunk has a table split or ends mid-sentence in a critical specification, inspect `source` and fetch `chunk_index - 1` or `chunk_index + 1` from `output_chunks/` to concatenate adjacent context.
4.  **Deduplication:** If two top chunks overlap significantly (due to 300-char overlap), merge them seamlessly before passing to the LLM.

---

## 9. LLM Answering & Grounding Directives

The LLM prompt MUST enforce strict grounding rules:

*   **Grounding Constraint:** The LLM MUST answer questions **ONLY** using facts explicitly stated in the provided BIS context blocks.
*   **No Extrapolation:** The LLM MUST NOT infer or extrapolate unstated technical limits, safety factors, or engineering tolerances.
*   **Technical Precision:** Numerical values, units (e.g., $\text{N/mm}^2$, $\text{pH} \ge 6$, $\text{mm}$), clause numbers, and vehicle categories (e.g., $M_1$, $N_2$) MUST be preserved with 100% exactness.
*   **Explicit vs. Interpretation:** If a standard specifies a requirement conditionally (e.g. *"subject to agreement between purchaser and manufacturer"*), the LLM must explicitly state the condition.

---

## 10. Inline Citations & Provenance Rules

Every technical assertion in the generated answer MUST be accompanied by an inline citation referencing the source standard, clause, and page number:

*   **Standard Citation Format:** `[IS <standard_no>:<year>, Clause <clause>, Page <page_start>]`
*   **Example Output:**
    > *"According to Indian Standards, water used for mixing concrete shall have a pH value of not less than 6 [IS 456:2000, Clause 5.4, Page 15]. For reinforced concrete, the maximum permissible chloride content is 500 mg/L [IS 456:2000, Clause 5.4, Page 16]."*
*   **Verifiability:** Citations must map directly back to the `RetrievedEvidence` metadata present in the retrieval response.

---

## 11. Hallucination & Grounding Violation Rules

To prevent unsafe engineering hallucinations:

1.  **Generation Layer Abstention Ownership:** `retrieve()` does not filter out unanswerable queries internally. The generation layer MUST inspect `rerank_score` values, score margins, and `is_mask_restricted`.
2.  **Insufficient Evidence Refusal:** If top `rerank_score` is below confidence threshold $\tau$ (default `0.5` on the sigmoid `[0, 1]` scale) or if candidates lack facts to answer the question, the LLM MUST respond with a standard refusal:
    > *"The provided Indian Standards documents do not contain sufficient technical information to answer this query."*
3.  **Negative Grounding Check:** The system must reject answers that cite non-existent clauses or invent standard numbers not present in the retrieved context.

---

## 12. API & Service Design

The primary interface for data-answering-bis is the **`app.generator` library** (`answer()` / `warmup()`, usable without any server); FastAPI below is an optional HTTP adapter around the same pipeline:

### Endpoint Specification: `POST /api/v1/query`

#### Request Payload
```json
{
  "query": "What is the minimum pH value of water for mixing concrete in IS 456?",
  "top_k": 3,
  "expand_neighbors": false,
  "confidence_threshold": 0.5
}
```

#### Response Payload
```json
{
  "query": "What is the minimum pH value of water for mixing concrete in IS 456?",
  "answer": "According to Indian Standards, water used for mixing concrete shall have a pH value of not less than 6 [IS 456:2000, Clause 5.4, Page 15].",
  "citations": [
    {
      "standard_no": "IS 456",
      "year": "2000",
      "clause": "5.4",
      "page": 15,
      "chunk_id": "456_2000_amd5_reff2021_0014"
    }
  ],
  "retrieval_meta": {
    "filtered_standard": "IS 456",
    "is_mask_restricted": true,
    "candidates_retrieved": 10,
    "top_reranker_score": 0.9957,
    "execution_time_ms": 312.4
  }
}
```

---

## 13. Dependencies & Runtime Environment

data-answering-bis maintains a lightweight runtime environment:

### Required Runtime Dependencies (`requirements.txt`)
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
psycopg[binary]>=3.1.0  # Optional/Required for PgVectorStore PostgreSQL support
```

### Excluded Dependencies (DO NOT INSTALL IN data-answering-bis)
*   `semantic-text-splitter` *(Chunking only — frozen in Repo 2)*
*   `faiss-cpu` / `faiss-gpu` *(Flat dot product on 2,081 vectors in NumPy takes < 2ms; pgvector / NumPy replace FAISS)*
*   `rank-bm25` *(BM25 rejected as primary retriever)*

---

## 14. Testing & Evaluation Framework

System evaluation in data-answering-bis will be conducted against the **87-query benchmark suite** (`retrieval_queries.md`):

1.  **Retrieval Metrics:** Document Retrieval @1 (Target: 100%), Clause Retrieval @1 (Target: 100%), Clause MRR@3 (Target: 1.000).
2.  **Answer Quality Metrics:**
    *   *Strict Answerability:* Phrase-hit verification against ground-truth answer keys.
    *   *Citation Accuracy:* Verification that 100% of cited clauses and page numbers match the underlying chunk metadata.
    *   *Hallucination Rate:* 0% tolerance for ungrounded numerical values or fabricated clauses.

---

## 15. Implementation Roadmap

Development in data-answering-bis proceeds in 8 structured phases:

- [x] **Phase 1: Repository Architecture & Migration Plan** — *(Completed: Asset inventory, data contract definition, `retrieval/` package alignment, and master `AGENTS.md` specification).*
- [ ] **Phase 2: Storage Connection & Retrieval Engine Integration** — Wire `retrieval` package with `LocalNpyStore` and `PgVectorStore` backend options.
- [ ] **Phase 3: Context Assembly & Neighbor Expansion** — Implement prompt context builder, chunk metadata formatting, and adjacent chunk stitcher.
- [ ] **Phase 4: System Prompt Engineering & Grounding Constraints** — Draft and validate system prompts enforcing 100% strict grounding.
- [ ] **Phase 5: LLM Integration & Orchestration** — Wire Groq (`openai/gpt-oss-120b`) / OpenAI API connectors for answer generation.
- [ ] **Phase 6: Citation Parser & Post-Processor** — Implement automated post-processing to append verified inline citations (`[IS ..., Clause ..., Page ...]`).
- [ ] **Phase 7: Refusal & Abstention Logic** — Wire confidence-gated refusal when top reranker score is below threshold $\tau$ or margins indicate unanswerable query.
- [x] **Phase 8: FastAPI Service & Benchmark Evaluation** — *(Completed: optional `POST /api/v1/query` adapter around `answer()`; direct-library usage is primary).*

---

## 16. Frozen Components & Invariants (DO NOT CHANGE)

Future Antigravity sessions working in data-answering-bis MUST obey the following frozen invariants:

1.  **Chunking is Frozen:** Do NOT modify chunk sizes (3000 chars), overlap (300 chars), or chunk JSON files in `output_chunks/`.
2.  **Embedding Model is Frozen:** Do NOT change candidate embedder (`BAAI/bge-m3` pinned to revision `5617a9f`) or vector dimensions (1024-dim).
3.  **Enriched Representation is Frozen:** Embeddings and reranking MUST use the `Standard + Clause + Heading + Text` representation.
4.  **Re-Use Pre-Computed Vectors:** Always load `artifacts/bge_m3_enriched_vectors_gpu.npy` or query PostgreSQL `chunks` table via `PgVectorStore`. Do NOT re-embed the corpus.
5.  **Do Not Re-Introduce Rejected Retrievers:** BM25-primary, Hybrid RRF, and Raw-text reranking are proven inferior and MUST NOT be revived.
6.  **`ChunkStore` Protocol Seam:** The `retrieve()` function interface must remain decoupled from specific vector storage backends.

---

## 17. Current Status Summary

| Component | Status | Location / Details |
|---|---|---|
| **Chunking Pipeline** | **FROZEN** | 2,081 chunks across 101 files (`output_chunks/`) |
| **Vector Storage Assets** | **FROZEN** | `artifacts/bge_m3_enriched_vectors_gpu.npy` & `indexing/migrate_from_artifacts.py` |
| **Production Retrieval Package** | **COMPLETED** | `retrieval/` package (`retrieve()`, `RetrievedEvidence`, `LocalNpyStore`, `PgVectorStore`) |
| **Migration Specification** | **COMPLETED** | Documented in `AGENTS.md` & `README.md` |
| **LLM Answering & Context Builder** | **COMPLETED** | `app/generator` library (`answer()` / `warmup()`, ask + product_match modes) |
| **FastAPI REST Endpoint** | **COMPLETED (optional adapter)** | `app/main.py` (`POST /api/v1/query`) around `answer()` |
