# Representative Retrieval Queries

Benchmark queries for evaluating whether the current chunking pipeline
supports the intended RAG use cases. Each query maps to a query type
that drives different chunking and indexing requirements.

Use these before adding more chunking heuristics. A retrieval smoke test
should surface whether the current 3000-char semantic chunks are
adequate, or whether a hybrid clause index is needed.

---

## Query types

| Type | Description | Chunking implication |
|---|---|---|
| **clause_lookup** | User cites or asks about a specific clause number | Needs accurate `clause` metadata or a side index |
| **definition** | User asks what a defined term means | Definition sections (3.x, 2.x) must stay intact |
| **tabular_limit** | User asks for a numeric limit from a table | Tables must not be split mid-row |
| **paraphrase** | User describes a requirement in plain language | Semantic chunks with overlap are appropriate |
| **cross_standard** | User references another IS standard by number | `standard_no` metadata must be correct |
| **annex** | User asks about annex content or formulae | Annex provenance (`ANNEX A`, `A-1`) must be preserved |
| **natural** (style) | Intent-phrased, no IS/clause numbers ("I make helmets…") | Tests full-corpus discrimination without the IS filter |
| **adversarial** (style) | Near-duplicate standards (IS 15858–15862, IS 18000 vs 18002, Part 1 vs Part 2) | `standard_no`/title discrimination under lexical overlap |
| **unanswerable** (style) | Out-of-corpus standard/topic, or stripped back-matter | Excluded from aggregates; any strict hit is a false positive |

### Phase 0 query distribution (Q1–Q77) + Tier 2B hardening (Q78–Q87)

| `query_style` | Count | Notes |
|---|---:|---|
| natural | 15 | Incl. phrasing twins of Q13/11/20/16/15/19/8/14 (same gold, no IS number, unfiltered) |
| clause_lookup | 14 | 24 clause-labeled queries total (incl. definition/provenance/adversarial entries with `expected_clause`) |
| adversarial | 14 | Phase 0 set (11) + clause-labeled Q85–Q87 (15859 cl.6, 15860 cl.7, 17428-2 cl.4.1) — fills the clause×adversarial cell |
| table_lookup | 10 | Incl. multi-hop probe Q40 (spans 5.4 + 8.2.5, expected single-chunk fail) |
| paraphrase | 10 | |
| definition | 12 | Incl. unfiltered twin Q32 of Q5; Tier 2B Q82–Q84 (15859/15860 terminology, 17428-2 privacy) |
| unanswerable | 6 | Q63–Q67 + Q74 (back-matter); reported separately, never aggregated |
| cross_standard | 3 | Original Q17 + Tier 2B Q78 (IS 14272 via IS 15986), Q79 (IS 383 via IS 10262) |
| provenance | 3 | Original Q18 + Tier 2B Q80 (10262 Annex A, p.18, low-conf), Q81 (10254 cl.4.2, p.4, low-conf) |

Known-hard labels (`failure_class`): Q1 `expected_fail` (doc has 4.1, not 4.2),
Q2 `demoted_heading` (5.4.2 demoted by design), Q3 `heading_pollution`
(7.3.1 text present, clause metadata polluted by "125" mis-heading —
Phase 0 audit verdict: hard chunking case, not a ground-truth bug),
Q14 `strict_artifact` (correct Annex E lacks literal token `M1`; strict
fail + soft pass = benchmark artifact), Q40 `multi_chunk`.

---

## Benchmark queries (87: 80 scored + 6 unanswerable probes + 1 corpus-missing)

Source of truth is `QUERIES` in `scripts/retrieval_eval.py` — every entry
carries a `query_style` slice key, a secondary `answer_phrases_soft` set,
and (for known-hard cases) a `failure_class` label. All strict + soft
phrase sets are programmatically verified to co-occur in a single chunk
of the expected document. This file describes the categories; see
`retrieval_eval.py` for the exact list.

### Clause lookup (4)

1. What does clause 4.2 of IS 12182 : 2025 say about reservoir sedimentation effects?
   - **Type:** clause_lookup
   - **Target doc:** `12182_2025.md`
   - **Expected clause:** `4.2`

2. What are the requirements in clause 5.4.2 of IS 456 : 2000 regarding pH value of water?
   - **Type:** clause_lookup
   - **Target doc:** `456_2000_amd5_reff2021.md`
   - **Note:** Heading may be demoted if extracted inline; chunk text should still contain the requirement.

3. What does section 7.3.1 of IS 4151 say about retention systems?
   - **Type:** clause_lookup
   - **Target doc:** `4151_2015_amd1_reff2020.md`

4. What test procedures are specified in Annex A of IS 15858?
   - **Type:** clause_lookup / annex
   - **Target doc:** `15858.md`
   - **Expected clause:** `ANNEX A` or `A-1`

### Definition lookup (4)

5. How is "sediment yield" defined in IS 12182?
   - **Type:** definition
   - **Target doc:** `12182_2025.md`

6. What is the definition of "placement and workability" in IS 10262?
   - **Type:** definition
   - **Target doc:** `10262.md`
   - **Expected clause:** `9.10` (definition-section numbering)

7. Define "trap efficiency" in the context of reservoir sedimentation.
   - **Type:** definition / paraphrase
   - **Target doc:** `12182_2025.md` (Annex A)

8. What does IS 18000 mean by "big data storage"?
   - **Type:** definition
   - **Target doc:** `18000_2020.md`
   - **Expected clause:** `2.134` or similar definition-section number

### Tabular / numeric limits (3)

9. What is the maximum chloride content permitted in concrete according to IS 456?
   - **Type:** tabular_limit
   - **Target doc:** `456_2000_amd5_reff2021.md`
   - **Risk:** Table may be split across chunks

10. What are the grading limits for fine aggregate in IS 383?
    - **Type:** tabular_limit
    - **Target doc:** (if present in corpus)

11. What pH range is required for water used in concrete mixing per IS 456?
    - **Type:** tabular_limit / paraphrase
    - **Target doc:** `456_2000_amd5_reff2021.md`

### Paraphrase / topic (5)

12. How should sedimentation effects on reservoir capacity be assessed?
    - **Type:** paraphrase
    - **Target doc:** `12182_2025.md`

13. What precautions apply when designing penstocks for high head conditions?
    - **Type:** paraphrase
    - **Target doc:** `11639_1_1986_reff2020.md`

14. What are the braking requirements for M1 category vehicles?
    - **Type:** paraphrase
    - **Target doc:** `15986_2015_amd1_reff2020.md`

15. How is post-harvest grain loss by rodents assessed?
    - **Type:** paraphrase
    - **Target doc:** `11261_2_1985_reff2020.md`

16. What sampling methods apply to fish and fishery products?
    - **Type:** paraphrase
    - **Target doc:** `11427_2001_reaff2023.md`

### Cross-standard and provenance (2)

17. Which Indian Standard governs rounding off of numerical values, as referenced in IS 15858?
    - **Type:** cross_standard
    - **Expected answer:** IS 2 : 1960 (referenced in foreword text)

18. On which page of IS 12182 does the trap efficiency example appear?
    - **Type:** provenance / citation
    - **Target doc:** `12182_2025.md`
    - **Expected:** `page_start`/`page_end` near Annex A, `low_confidence: true`

### Edge cases (2)

19. What does the National Flag standard say about khadi material requirements?
    - **Type:** paraphrase
    - **Target doc:** `1.md` (IS 1511 : 1961)
    - **Note:** Tests filename/title-page ID fallback

20. What is edible maize starch specified as under IS 1005?
    - **Type:** paraphrase
    - **Target doc:** `1005.md`
    - **Note:** Tests OCR-variant title page (`1S` → `IS`)

---

## Evaluation checklist

For each query, record:

- [ ] Did retrieval return a relevant chunk (manual or embedding top-k)?
- [ ] Is the answer complete without needing adjacent chunks?
- [ ] Is `standard_no` / `clause` / `page_start` correct for citation?
- [ ] If the answer is in a table, is the full table row present?
- [ ] For definition queries, is the definition term and body in the same chunk?

Pass threshold for moving to embedding/indexing: **≥ 15/20 queries** return a
single chunk sufficient to answer, or a clearly identified chunk pair where
overlap bridges the gap.
