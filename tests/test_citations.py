"""Offline unit tests for Phase 6 citation parsing / verification.

No models, API keys, database, or network required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval.types import RetrievedEvidence

from app.generator.citations import (
    parse_citations,
    verify_answer,
)
from app.generator.context_builder import build_context, load_chunk_index
from app.generator.llm_client import GeneratedAnswer


def make_evidence(
    chunk_id: str,
    text: str,
    source: str = "456_2000_amd5_reff2021.md",
    clause: str | None = "5.4",
    heading: str | None = "5.4 Water",
    standard_no: str | None = "IS 456",
    page_start: int | None = 15,
    page_end: int | None = 16,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        text=text,
        source=source,
        clause=clause,
        heading=heading,
        standard_no=standard_no,
        page_start=page_start,
        page_end=page_end,
        dense_score=0.9,
        rerank_score=1.5,
        is_mask_restricted=True,
    )


def make_chunk(chunk_id: str, chunk_index: int, text: str, **overrides) -> dict:
    meta = {
        "source": "456_2000_amd5_reff2021.md",
        "chunk_index": chunk_index,
        "clause": "5.4",
        "heading": "5.4 Water",
        "heading_path": ["5 MATERIALS", "5.4 Water"],
        "standard_no": "IS 456",
        "year": "2000",
        "page_start": 15,
        "page_end": 16,
        "tail_truncated": False,
        "table_repaired": False,
    }
    meta.update(overrides)
    return {"id": chunk_id, "text": text, "metadata": meta}


@pytest.fixture()
def context_and_index(tmp_path: Path):
    chunks = [
        make_chunk("456_2000_amd5_reff2021_0014", 14, "Water shall have pH not less than 6."),
        make_chunk("456_2000_amd5_reff2021_0015", 15, "Chloride limits are tabulated.", clause="8.2.5",
                   heading="8.2.5 Chloride", heading_path=["8 DURABILITY", "8.2.5 Chloride"],
                   page_start=17, page_end=18),
    ]
    (tmp_path / "456_3000_ov300.json").write_text(json.dumps(chunks), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    evidence = [
        make_evidence("456_2000_amd5_reff2021_0014", "Water shall have pH not less than 6."),
        make_evidence("456_2000_amd5_reff2021_0015", "Chloride limits are tabulated.",
                      clause="8.2.5", heading="8.2.5 Chloride", page_start=17, page_end=18),
    ]
    return build_context(evidence, chunk_index=index, top_k=2), index


def answer_with(text: str) -> GeneratedAnswer:
    return GeneratedAnswer(text=text, model="test-model")


# --- parsing ---------------------------------------------------------------------

def test_parse_single_citation_with_spans():
    text = "pH shall be not less than 6 [IS 456:2000, Clause 5.4, Page 15]."
    (c,) = parse_citations(text)
    assert (c.standard_no, c.year, c.clause, c.page) == ("456", "2000", "5.4", 15)
    assert text[c.span_start:c.span_end] == "[IS 456:2000, Clause 5.4, Page 15]"


def test_parse_multiple_citations_in_order():
    text = ("A [IS 456:2000, Clause 5.4, Page 15] and "
            "B [IS 456:2000, Clause 8.2.5, Page 18].")
    cites = parse_citations(text)
    assert [c.clause for c in cites] == ["5.4", "8.2.5"]
    assert [c.page for c in cites] == [15, 18]
    assert cites[0].span_start < cites[1].span_start


def test_parse_annex_clause_and_extra_spacing():
    (c,) = parse_citations("[IS   17428 : 2020,  Clause   ANNEX C , Page 30]")
    assert (c.standard_no, c.year, c.clause, c.page) == ("17428", "2020", "ANNEX C", 30)


def test_parse_ignores_non_canonical_brackets():
    assert parse_citations("See clause 5.4 of IS 456 (page 15).") == []
    assert parse_citations("No citations here.") == []


# --- verification ------------------------------------------------------------------

def test_verified_citation(context_and_index):
    context, index = context_and_index
    result = verify_answer(
        answer_with("pH shall be not less than 6 [IS 456:2000, Clause 5.4, Page 15]."),
        context, index,
    )
    assert result.has_citations is True
    assert result.all_verified is True
    (v,) = result.citations
    assert v.verdict == "verified"
    assert v.chunk_id == "456_2000_amd5_reff2021_0014"
    assert result.text == "pH shall be not less than 6 [IS 456:2000, Clause 5.4, Page 15]."


def test_page_end_of_range_verifies(context_and_index):
    context, index = context_and_index
    result = verify_answer(
        answer_with("pH rule [IS 456:2000, Clause 5.4, Page 16]."), context, index
    )
    assert result.all_verified is True


def test_wrong_clause_is_mismatch(context_and_index):
    context, index = context_and_index
    result = verify_answer(
        answer_with("pH rule [IS 456:2000, Clause 9.9, Page 15]."), context, index
    )
    assert result.all_verified is False
    (v,) = result.citations
    assert v.verdict == "mismatch"
    assert "9.9" in v.detail
    assert v.chunk_id == "456_2000_amd5_reff2021_0014"  # linked via standard match


def test_page_outside_range_is_mismatch(context_and_index):
    context, index = context_and_index
    result = verify_answer(
        answer_with("pH rule [IS 456:2000, Clause 5.4, Page 99]."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "mismatch"
    assert "99" in v.detail


def test_unknown_standard_is_mismatch(context_and_index):
    context, index = context_and_index
    result = verify_answer(
        answer_with("Rule [IS 9999:2020, Clause 1, Page 1]."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "mismatch"
    assert v.chunk_id is None


def test_wrong_year_is_mismatch(context_and_index):
    context, index = context_and_index
    result = verify_answer(
        answer_with("pH rule [IS 456:1999, Clause 5.4, Page 15]."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "mismatch"
    assert "1999" in v.detail


def test_missing_year_metadata_is_unverifiable_not_mismatch(tmp_path: Path):
    chunks = [make_chunk("x_0000", 0, "Some text.", year=None)]
    (tmp_path / "x_3000_ov300.json").write_text(json.dumps(chunks), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("x_0000", "Some text.", source="456_2000_amd5_reff2021.md")
    context = build_context([ev], chunk_index=index, top_k=1)
    result = verify_answer(
        answer_with("Text [IS 456:2000, Clause 5.4, Page 15]."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "unverifiable"
    assert result.all_verified is False


def test_missing_page_metadata_is_unverifiable(tmp_path: Path):
    chunks = [make_chunk("y_0000", 0, "Some text.", page_start=None, page_end=None)]
    (tmp_path / "y_3000_ov300.json").write_text(json.dumps(chunks), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("y_0000", "Some text.", source="456_2000_amd5_reff2021.md",
                       page_start=None, page_end=None)
    context = build_context([ev], chunk_index=index, top_k=1)
    result = verify_answer(
        answer_with("Text [IS 456:2000, Clause 5.4, Page 15]."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "unverifiable"


def test_mixed_verdicts_not_all_verified(context_and_index):
    context, index = context_and_index
    result = verify_answer(answer_with(
        "Good [IS 456:2000, Clause 5.4, Page 15] "
        "bad [IS 456:2000, Clause 9.9, Page 15]."
    ), context, index)
    assert [v.verdict for v in result.citations] == ["verified", "mismatch"]
    assert result.all_verified is False
    assert result.has_citations is True


def test_answer_without_citations(context_and_index):
    context, index = context_and_index
    result = verify_answer(answer_with("Plain answer with no citations."), context, index)
    assert result.has_citations is False
    assert result.all_verified is False  # uncited answers are not "verified"
    assert result.citations == []
    assert result.text == "Plain answer with no citations."


def test_text_carried_through_byte_identical(context_and_index):
    context, index = context_and_index
    original = "pH rule [IS 456:2000, Clause 5.4, Page 15] plus [IS 9999:2020, Clause 1, Page 1]."
    result = verify_answer(answer_with(original), context, index)
    assert result.text == original


# --- live-model Unicode variants (Tier 3 regression) -------------------------------
#
# The production model emits citations with CJK brackets, narrow
# no-break spaces (U+202F), and non-breaking-hyphen page ranges, e.g.
# the exact live answer for the canonical IS 456 pH query:
#   "...not less than\u202f6**\u3010IS 456:2000, Clause\u202f5.4,
#   Page\u202f15\u201116\u3011."

# Codepoint-built (ASCII source, no literal non-ASCII chars below).
NBSP = chr(0x00A0)
NARROW_NBSP = chr(0x202F)
CJK_OPEN = chr(0x3010)
CJK_CLOSE = chr(0x3011)
NB_HYPHEN = chr(0x2011)
FULLWIDTH_COLON = chr(0xFF1A)
FULLWIDTH_COMMA = chr(0xFF0C)

LIVE_CITATION = (
    f"{CJK_OPEN}IS 456:2000, Clause{NARROW_NBSP}5.4, "
    f"Page{NARROW_NBSP}15{NB_HYPHEN}16{CJK_CLOSE}"
)
LIVE_ANSWER = (
    "The standard requires that the water used for mixing concrete have a pH "
    f"**not less than{NARROW_NBSP}6**{LIVE_CITATION}."
)


def test_parse_live_cjk_brackets_nbsp_page_range():
    (c,) = parse_citations(LIVE_ANSWER)
    assert (c.standard_no, c.year, c.clause, c.page) == ("456", "2000", "5.4", 15)
    # Spans stay valid against the original (folding is 1:1).
    assert LIVE_ANSWER[c.span_start:c.span_end] == LIVE_CITATION


def test_live_answer_verifies_to_canonical_chunk(context_and_index):
    context, index = context_and_index
    result = verify_answer(answer_with(LIVE_ANSWER), context, index)
    assert result.has_citations is True
    assert result.all_verified is True
    (v,) = result.citations
    assert v.verdict == "verified"
    assert v.chunk_id == "456_2000_amd5_reff2021_0014"
    assert result.text == LIVE_ANSWER  # never mutated


def test_ascii_hyphen_page_range_yields_first_page():
    (c,) = parse_citations("Rule [IS 456:2000, Clause 5.4, Page 15-16].")
    assert c.page == 15


def test_pages_plural_and_fullwidth_punctuation():
    text = (
        f"[IS 456{FULLWIDTH_COLON}2000{FULLWIDTH_COMMA} "
        f"Clause 5.4{FULLWIDTH_COMMA} Pages 15]"
    )
    (c,) = parse_citations(text)
    assert (c.standard_no, c.year, c.clause, c.page) == ("456", "2000", "5.4", 15)


def test_mixed_brackets_and_regular_nbsp():
    text = f"[IS{NBSP}456:2000, Clause{NBSP}5.4, Page{NBSP}15{CJK_CLOSE}"
    (c,) = parse_citations(text)
    assert (c.standard_no, c.year, c.clause, c.page) == ("456", "2000", "5.4", 15)


def test_unicode_technical_symbols_do_not_break_parsing():
    text = "Strength taču 6 N/mm² [IS 456:2000, Clause 5.4, Page 15]."
    (c,) = parse_citations(text)
    assert (c.standard_no, c.year, c.clause, c.page) == ("456", "2000", "5.4", 15)


def test_unicode_technical_symbols_do_not_break_parsing():
    text = "Strength " + chr(0x2265) + " 6 N/mm" + chr(0xB2) + " [IS 456:2000, Clause 5.4, Page 15]."
    (c,) = parse_citations(text)
    assert (c.standard_no, c.year, c.clause, c.page) == ("456", "2000", "5.4", 15)


# --- sub-clause text anchoring (Tier 3 nominal-cover regression) -------------------
#
# Chunk 456_2000_amd5_reff2021_0074 carries metadata clause 26.4 while its
# text contains 26.4.1/26.4.2/26.4.2.1/26.4.2.2. Citations to those
# sub-clauses are grounded and must verify against that chunk.

SUBCLAUSE_TEXT = (
    "### 26.4 Nominal Cover to Reinforcement\n"
    "\n"
    "#### 26.4.1 Nominal Cover\n"
    "\n"
    "Nominal cover is the design depth of cover. It shall be not less "
    "than the diameter of the bar.\n"
    "\n"
    "#### 26.4.2 Nominal Cover to Meet Durability Requirement\n"
    "\n"
    "Minimum values shall be as given in Table 16.\n"
    "\n"
    "26.4.2.1 However for a column nominal cover shall not be less than 40 mm.\n"
    "\n"
    "26.4.2.2 For footings minimum cover shall be 50mm.\n"
)


def _subclause_context(tmp_path: Path):
    chunk = {
        "id": "456_0014",
        "text": SUBCLAUSE_TEXT,
        "metadata": {
            "source": "456.md",
            "chunk_index": 74,
            "clause": "26.4",
            "heading": "26.4 Nominal Cover to Reinforcement",
            "heading_path": ["26 REINFORCEMENT", "26.4 Nominal Cover to Reinforcement"],
            "standard_no": "IS 456",
            "year": "2000",
            "page_start": 47,
            "page_end": 47,
            "tail_truncated": False,
            "table_repaired": False,
        },
    }
    (tmp_path / "s_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("456_0014", SUBCLAUSE_TEXT, source="456.md",
                       clause="26.4", heading="26.4 Nominal Cover to Reinforcement",
                       page_start=47, page_end=47)
    return build_context([ev], chunk_index=index, top_k=1), index


def test_header_subclause_verifies_to_parent_chunk(tmp_path: Path):
    context, index = _subclause_context(tmp_path)
    result = verify_answer(
        answer_with("Cover is defined [IS 456:2000, Clause 26.4.1, Page 47]."),
        context, index,
    )
    assert result.all_verified is True
    (v,) = result.citations
    assert v.verdict == "verified"
    assert v.chunk_id == "456_0014"


def test_bare_child_label_verifies(tmp_path: Path):
    context, index = _subclause_context(tmp_path)
    result = verify_answer(
        answer_with("Columns need 40 mm [IS 456:2000, Clause 26.4.2.1, Page 47]."),
        context, index,
    )
    assert result.all_verified is True


def test_multi_subclause_answer_all_verified(tmp_path: Path):
    context, index = _subclause_context(tmp_path)
    result = verify_answer(answer_with(
        "Defined [IS 456:2000, Clause 26.4.1, Page 47], tabulated "
        "[IS 456:2000, Clause 26.4.2, Page 47], columns "
        "[IS 456:2000, Clause 26.4.2.1, Page 47], footings "
        "[IS 456:2000, Clause 26.4.2.2, Page 47]."
    ), context, index)
    assert [v.verdict for v in result.citations] == ["verified"] * 4
    assert result.all_verified is True
    assert {v.chunk_id for v in result.citations} == {"456_0014"}


def test_non_child_bare_number_does_not_verify(tmp_path: Path):
    # Guards the 0075 trap: line-initial "0.5" is a measurement, and must
    # never verify as Clause 0.5.
    chunk = {
        "id": "m_0001",
        "text": "### 26.5.1 Some Requirement\n\n0.5 mm tolerance applies here.\n",
        "metadata": {
            "source": "m.md",
            "chunk_index": 1,
            "clause": "26.5.1",
            "heading": "26.5.1 Some Requirement",
            "heading_path": [],
            "standard_no": "IS 456",
            "year": "2000",
            "page_start": 47,
            "page_end": 48,
            "tail_truncated": False,
            "table_repaired": False,
        },
    }
    (tmp_path / "m_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("m_0001", chunk["text"], source="m.md",
                       clause="26.5.1", heading="26.5.1 Some Requirement",
                       page_start=47, page_end=48)
    context = build_context([ev], chunk_index=index, top_k=1)
    result = verify_answer(
        answer_with("Tolerance [IS 456:2000, Clause 0.5, Page 47]."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "mismatch"


def test_genuine_wrong_subclause_still_mismatches(tmp_path: Path):
    context, index = _subclause_context(tmp_path)
    result = verify_answer(
        answer_with("Rule [IS 456:2000, Clause 26.4.9, Page 47]."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "mismatch"
    assert "26.4.9" in v.detail


# --- parent-cite acceptance (Tier 3 Clause 26.5 regression) ---------------------------
#
# Live: chunk 456_2000_amd5_reff2021_0078 (metadata 26.5.3, pages 49-50)
# shows "##### 26.5.2.2 Maximum diameter" plus its requirement sentence.
# The model cited [IS 456:2000, Clause 26.5.2, Page 49-50] — the parent of
# the shown label, same pages. Grounded provenance, must verify.

PARENT_TEXT = (
    "tail of previous matter here.\n"
    "\n"
    "##### 26.5.2.2 Maximum diameter\n"
    "\n"
    "The diameter of reinforcing bars shall not exceed one-eight of the "
    "total thickness of the slab.\n"
    "\n"
    "#### 26.5.3 Columns\n"
    "\n"
    ", 26.5.3.1 Longitudinal reinforcement\n"
)


def _parent_context(tmp_path: Path):
    chunk = {
        "id": "456_0078",
        "text": PARENT_TEXT,
        "metadata": {
            "source": "456.md",
            "chunk_index": 78,
            "clause": "26.5.3",
            "heading": "26.5.3 Columns",
            "heading_path": [],
            "standard_no": "IS 456",
            "year": "2000",
            "page_start": 49,
            "page_end": 50,
            "tail_truncated": False,
            "table_repaired": False,
        },
    }
    (tmp_path / "q_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("456_0078", PARENT_TEXT, source="456.md",
                       clause="26.5.3", heading="26.5.3 Columns",
                       page_start=49, page_end=50)
    return build_context([ev], chunk_index=index, top_k=1), index


def test_parent_clause_of_shown_label_verifies(tmp_path: Path):
    context, index = _parent_context(tmp_path)
    result = verify_answer(
        answer_with("Diameter rule [IS 456:2000, Clause 26.5.2, Page 49]."),
        context, index,
    )
    assert result.all_verified is True
    (v,) = result.citations
    assert v.verdict == "verified"
    assert v.chunk_id == "456_0078"


def test_parent_page_range_first_page_verifies(tmp_path: Path):
    context, index = _parent_context(tmp_path)
    result = verify_answer(
        answer_with("Diameter rule [IS 456:2000, Clause 26.5.2, Page 49-50]."),
        context, index,
    )
    assert result.all_verified is True


def test_sibling_clause_still_mismatches(tmp_path: Path):
    # 26.5.2 and 26.5.3 are siblings: sharing a prefix is not grounding.
    context, index = _parent_context(tmp_path)
    result = verify_answer(
        answer_with("Other rule [IS 456:2000, Clause 26.5.9, Page 49]."),
        context, index,
    )
    (v,) = result.citations
    assert v.verdict == "mismatch"


def test_two_level_ancestor_still_mismatches(tmp_path: Path):
    # "26" is two levels above shown "26.5.2.2" — too coarse to verify.
    context, index = _parent_context(tmp_path)
    result = verify_answer(
        answer_with("General rule [IS 456:2000, Clause 26, Page 49]."),
        context, index,
    )
    assert result.citations[0].verdict == "mismatch"


def test_meta_none_block_admits_headers_only(tmp_path: Path):
    chunk = {
        "id": "n_0000",
        "text": "### 7.1 Scope\n\n7.2 text without header status.\n",
        "metadata": {
            "source": "n.md",
            "chunk_index": 0,
            "clause": None,
            "heading": None,
            "heading_path": [],
            "standard_no": "IS 999",
            "year": "2020",
            "page_start": 5,
            "page_end": 5,
            "tail_truncated": False,
            "table_repaired": False,
        },
    }
    (tmp_path / "n_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("n_0000", chunk["text"], source="n.md", clause=None,
                       heading=None, standard_no="IS 999", page_start=5, page_end=5)
    context = build_context([ev], chunk_index=index, top_k=1)
    header_hit = verify_answer(
        answer_with("Scope [IS 999:2020, Clause 7.1, Page 5]."), context, index)
    assert header_hit.all_verified is True
    bare_miss = verify_answer(
        answer_with("Other [IS 999:2020, Clause 7.2, Page 5]."), context, index)
    assert bare_miss.citations[0].verdict == "mismatch"


# --- harmless qualifier normalization (live-model fidelity) ------------------------
#
# The production model refines clause cites with subdivision qualifiers
# ("26.5.3.1(a)", "26.5.1 (Table 16)", "8 (Table 1)"). The base clause
# must still be attested — qualifiers never excuse an unshown clause.

QUALIFIER_TEXT = (
    "#### 26.5.3 Columns\n"
    "\n"
    ", 26.5.3.1 Longitudinal reinforcement\n"
    "\n"
    "- a) Area shall be not less than 0.8 percent.\n"
)


def _qualifier_context(tmp_path: Path):
    chunk = {
        "id": "q_0078",
        "text": QUALIFIER_TEXT,
        "metadata": {
            "source": "q.md",
            "chunk_index": 78,
            "clause": "26.5.3",
            "heading": "26.5.3 Columns",
            "heading_path": [],
            "standard_no": "IS 456",
            "year": "2000",
            "page_start": 49,
            "page_end": 50,
            "tail_truncated": False,
            "table_repaired": False,
        },
    }
    (tmp_path / "qq_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("q_0078", QUALIFIER_TEXT, source="q.md",
                       clause="26.5.3", heading="26.5.3 Columns",
                       page_start=49, page_end=50)
    return build_context([ev], chunk_index=index, top_k=1), index


def test_parenthesized_subdivision_verifies_to_base_clause(tmp_path: Path):
    context, index = _qualifier_context(tmp_path)
    result = verify_answer(
        answer_with("Area rule [IS 456:2000, Clause 26.5.3.1(a), Page 49]."),
        context, index,
    )
    assert result.all_verified is True


def test_table_qualifier_verifies_to_base_clause(tmp_path: Path):
    context, index = _qualifier_context(tmp_path)
    result = verify_answer(
        answer_with("Columns [IS 456:2000, Clause 26.5.3 (Table 16), Page 49]."),
        context, index,
    )
    assert result.all_verified is True


def test_qualified_but_unshown_clause_still_mismatches(tmp_path: Path):
    context, index = _qualifier_context(tmp_path)
    result = verify_answer(
        answer_with("Rule [IS 456:2000, Clause 26.5.3.9(z), Page 49]."),
        context, index,
    )
    assert result.citations[0].verdict == "mismatch"


def test_inline_deep_label_verifies_without_qualifier(tmp_path: Path):
    # ", 26.5.3.1 Longitudinal" is mid-line (line-wrap casualty), yet a
    # genuine sub-clause title attested by the chunk text.
    context, index = _qualifier_context(tmp_path)
    result = verify_answer(
        answer_with("Longitudinal [IS 456:2000, Clause 26.5.3.1, Page 49]."),
        context, index,
    )
    assert result.all_verified is True


# --- single-level header clauses -------------------------------------------------
#
# Front-matter chunks (scope, sampling, tests) carry clause=None metadata
# while their text holds genuine single-level titles ("## 1 SCOPE").


def test_single_level_header_clause_verifies(tmp_path: Path):
    chunk = {
        "id": "s_0000",
        "text": "## 1 SCOPE\n\nThis standard covers the requirements.\n",
        "metadata": {
            "source": "s.md", "chunk_index": 0, "clause": None, "heading": None,
            "heading_path": [], "standard_no": "IS 2415", "year": "2025",
            "page_start": 1, "page_end": 3,
            "tail_truncated": False, "table_repaired": False,
        },
    }
    (tmp_path / "s1_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("s_0000", chunk["text"], source="s.md", clause=None,
                       heading=None, standard_no="IS 2415", page_start=1, page_end=3)
    context = build_context([ev], chunk_index=index, top_k=1)
    result = verify_answer(
        answer_with("Scope [IS 2415:2025, Clause 1, Page 1]."), context, index
    )
    assert result.all_verified is True


def test_na_clause_links_only_to_clauseless_block(tmp_path: Path):
    chunk = {
        "id": "s_0000",
        "text": "## 1 SCOPE\n\nThis standard covers the requirements.\n",
        "metadata": {
            "source": "s.md", "chunk_index": 0, "clause": None, "heading": None,
            "heading_path": [], "standard_no": "IS 2415", "year": "2025",
            "page_start": 1, "page_end": 3,
            "tail_truncated": False, "table_repaired": False,
        },
    }
    (tmp_path / "s2_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("s_0000", chunk["text"], source="s.md", clause=None,
                       heading=None, standard_no="IS 2415", page_start=1, page_end=3)
    context = build_context([ev], chunk_index=index, top_k=1)
    result = verify_answer(
        answer_with("Scope [IS 2415:2025, Clause N/A, Page 1]."), context, index
    )
    assert result.all_verified is True
    assert result.citations[0].chunk_id == "s_0000"


def test_na_clause_mismatches_labelled_block(context_and_index):
    # The 5.4 block genuinely knows its clause; "N/A" must not attach to it.
    context, index = context_and_index
    result = verify_answer(
        answer_with("Rule [IS 456:2000, Clause N/A, Page 15]."), context, index
    )
    assert result.citations[0].verdict == "mismatch"


# --- clause ranges -----------------------------------------------------------------
#
# "3.4-3.6" expands to member labels; every member must be attested.


def test_clause_range_verifies_when_all_members_attested(tmp_path: Path):
    chunk = {
        "id": "r_0001",
        "text": ("## 3 DEFINITIONS\n\n3.3 Tube Size Designation.\n\n"
                 "3.4 Flat Length.\n\n### 3.5 Flat Width.\n\n### 3.6 Thickness.\n"),
        "metadata": {
            "source": "r.md", "chunk_index": 1, "clause": "3", "heading": "3 DEFINITIONS",
            "heading_path": [], "standard_no": "IS 2415", "year": "2025",
            "page_start": 3, "page_end": 3,
            "tail_truncated": False, "table_repaired": False,
        },
    }
    (tmp_path / "r_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("r_0001", chunk["text"], source="r.md", clause="3",
                       heading="3 DEFINITIONS", standard_no="IS 2415",
                       page_start=3, page_end=3)
    context = build_context([ev], chunk_index=index, top_k=1)
    result = verify_answer(
        answer_with("Definitions [IS 2415:2025, Clause 3.4-3.6, Page 3]."),
        context, index,
    )
    assert result.all_verified is True


def test_clause_range_mismatch_lists_missing(tmp_path: Path):
    chunk = {
        "id": "r_0001",
        "text": ("## 3 DEFINITIONS\n\n3.3 Tube Size Designation.\n\n"
                 "3.4 Flat Length.\n\n### 3.5 Flat Width.\n\n### 3.6 Thickness.\n"),
        "metadata": {
            "source": "r.md", "chunk_index": 1, "clause": "3", "heading": "3 DEFINITIONS",
            "heading_path": [], "standard_no": "IS 2415", "year": "2025",
            "page_start": 3, "page_end": 3,
            "tail_truncated": False, "table_repaired": False,
        },
    }
    (tmp_path / "r2_3000_ov300.json").write_text(json.dumps([chunk]), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("r_0001", chunk["text"], source="r.md", clause="3",
                       heading="3 DEFINITIONS", standard_no="IS 2415",
                       page_start=3, page_end=3)
    context = build_context([ev], chunk_index=index, top_k=1)
    result = verify_answer(
        answer_with("Definitions [IS 2415:2025, Clause 3.4-3.9, Page 3]."),
        context, index,
    )
    (v,) = result.citations
    assert v.verdict == "mismatch"
    assert "3.7" in v.detail

def test_partial_pair_verifies_with_linked_standard(tmp_path: Path):
    context, index = _subclause_context(tmp_path)
    result = verify_answer(
        answer_with("Values as given in Clause 26.4.2 on Page 47."),
        context, index,
    )
    assert result.has_citations is True
    (v,) = result.citations
    assert v.verdict == "verified"
    assert v.chunk_id == "456_0014"
    assert v.citation.partial is True
    # Standard attribution completed from the linked evidence.
    assert v.citation.standard_no == "456"


def test_partial_pair_mismatch(tmp_path: Path):
    context, index = _subclause_context(tmp_path)
    result = verify_answer(
        answer_with("Rule per Clause 9.9 on Page 47."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "mismatch"
    assert result.all_verified is False


def test_partial_paren_clause_verifies(tmp_path: Path):
    context, index = _subclause_context(tmp_path)
    result = verify_answer(
        answer_with("Cover (see clause 26.4.1) shall be adequate."), context, index
    )
    (v,) = result.citations
    assert v.verdict == "verified"
    assert v.chunk_id == "456_0014"


def test_bare_prose_clause_mention_ignored(tmp_path: Path):
    context, index = _subclause_context(tmp_path)
    result = verify_answer(
        answer_with("The clause requires adequate cover for durability."),
        context, index,
    )
    assert result.has_citations is False


# --- adjacent-page scope (neighbor-expanded text safety) ---------------------------------

def test_adjacent_neighbor_page_accepted(tmp_path: Path):
    chunks = [
        {
            "id": "p_0000",
            "text": "### 5.4 Water\n\nIntro text here.\n",
            "metadata": {
                "source": "p.md", "chunk_index": 0, "clause": "5.4",
                "heading": "5.4 Water", "heading_path": ["5.4 Water"],
                "standard_no": "IS 456", "year": "2000",
                "page_start": 15, "page_end": 15,
                "tail_truncated": False, "table_repaired": False,
            },
        },
        {
            "id": "p_0001",
            "text": "### 5.4 Water continued\n\nMore text here.\n",
            "metadata": {
                "source": "p.md", "chunk_index": 1, "clause": "5.4",
                "heading": "5.4 Water", "heading_path": ["5.4 Water"],
                "standard_no": "IS 456", "year": "2000",
                "page_start": 16, "page_end": 16,
                "tail_truncated": False, "table_repaired": False,
            },
        },
    ]
    (tmp_path / "p_3000_ov300.json").write_text(json.dumps(chunks), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    ev = make_evidence("p_0000", chunks[0]["text"], source="p.md",
                       clause="5.4", heading="5.4 Water",
                       page_start=15, page_end=15)
    context = build_context([ev], chunk_index=index, top_k=1)
    result = verify_answer(
        answer_with("Rule [IS 456:2000, Clause 5.4, Page 16]."), context, index
    )
    assert result.all_verified is True
