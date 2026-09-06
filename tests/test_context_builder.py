"""Offline unit tests for Phase 3 context assembly / neighbor expansion.

No models, GPU, database, or network required. Fixtures are hand-built
``RetrievedEvidence`` objects plus tiny synthetic chunk JSONs written to
``tmp_path``. One read-only smoke test touches the real
``data/chunks`` corpus (skipped when absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retrieval.types import RetrievedEvidence

from answering.generator.context_builder import (
    BuiltContext,
    build_context,
    format_block,
    load_chunk_index,
)


def make_evidence(
    chunk_id: str = "doc_0001",
    text: str = "Some complete requirement statement.",
    source: str = "doc.md",
    clause: str | None = "5.4",
    heading: str | None = "5.4 Water",
    standard_no: str | None = "IS 456",
    page_start: int | None = 15,
    page_end: int | None = 16,
    dense_score: float | None = 0.9,
    rerank_score: float | None = 1.5,
    is_mask_restricted: bool = True,
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
        dense_score=dense_score,
        rerank_score=rerank_score,
        is_mask_restricted=is_mask_restricted,
    )


def write_chunks_file(
    directory: Path,
    filename: str,
    chunks: list[dict],
) -> Path:
    path = directory / filename
    path.write_text(json.dumps(chunks), encoding="utf-8")
    return path


def make_chunk(
    chunk_id: str,
    source: str,
    chunk_index: int,
    text: str,
    clause: str | None = None,
    heading: str | None = None,
    heading_path: list[str] | None = None,
    standard_no: str | None = "IS 456",
    year: str | None = "2000",
    page_start: int | None = 15,
    page_end: int | None = 16,
    tail_truncated: bool = False,
    table_repaired: bool = False,
) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "source": source,
            "chunk_index": chunk_index,
            "clause": clause,
            "heading": heading,
            "heading_path": heading_path or [],
            "standard_no": standard_no,
            "year": year,
            "page_start": page_start,
            "page_end": page_end,
            "tail_truncated": tail_truncated,
            "table_repaired": table_repaired,
        },
    }


@pytest.fixture()
def three_chunk_index(tmp_path: Path):
    """doc.md with indices 0..2; middle chunk flagged tail_truncated."""
    chunks = [
        make_chunk("doc_0000", "doc.md", 0, "First chunk ends cleanly.",
                   clause="5.3", heading="5.3 Aggregates",
                   heading_path=["5 MATERIALS", "5.3 Aggregates"]),
        make_chunk("doc_0001", "doc.md", 1, "Middle chunk ends mid-sentence, with trailing overlap " + "x" * 60,
                   clause="5.4", heading="5.4 Water",
                   heading_path=["5 MATERIALS", "5.4 Water"],
                   tail_truncated=True),
        make_chunk("doc_0002", "doc.md", 2, "Third chunk ends cleanly.",
                   clause="5.5", heading="5.5 Admixtures",
                   heading_path=["5 MATERIALS", "5.5 Admixtures"]),
    ]
    write_chunks_file(tmp_path, "doc_3000_ov300.json", chunks)
    return load_chunk_index(tmp_path)


# --- formatting -----------------------------------------------------------

def test_format_block_exact_shape():
    ev = make_evidence()
    out = format_block(
        1, ev, "Water shall have pH not less than 6.",
        year="2000", heading_path=["5 MATERIALS", "5.4 Water"],
    )
    assert out == (
        "### Context Block [1]\n"
        "- **Standard:** IS 456:2000\n"
        "- **Clause:** 5.4\n"
        "- **Heading:** 5 MATERIALS > 5.4 Water\n"
        "- **Location:** Page 15-16 (Chunk ID: doc_0001)\n"
        "\n"
        "```text\n"
        "Water shall have pH not less than 6.\n"
        "```"
    )


def test_format_block_missing_metadata_renders_na():
    ev = make_evidence(clause=None, heading=None, standard_no=None,
                       page_start=None, page_end=None)
    out = format_block(2, ev, "Some text.")
    assert "- **Standard:** N/A" in out
    assert "- **Clause:** N/A" in out
    assert "- **Heading:** N/A" in out
    assert "- **Location:** Page N/A (Chunk ID: doc_0001)" in out
    assert "### Context Block [2]" in out


def test_format_block_single_page():
    ev = make_evidence(page_start=15, page_end=15)
    out = format_block(1, ev, "t", year="2000")
    assert "- **Location:** Page 15 (Chunk ID: doc_0001)" in out


# --- top-k budget ----------------------------------------------------------

def test_topk_slicing_and_reserve():
    evidence = [make_evidence(chunk_id=f"doc_{i:04d}", source="doc.md") for i in range(10)]
    ctx = build_context(evidence, top_k=3)
    assert isinstance(ctx, BuiltContext)
    assert [b.rank for b in ctx.blocks] == [1, 2, 3]
    assert [b.evidence.chunk_id for b in ctx.blocks] == ["doc_0000", "doc_0001", "doc_0002"]
    assert [e.chunk_id for e in ctx.reserve] == [f"doc_{i:04d}" for i in range(3, 10)]
    assert ctx.prompt_text.count("### Context Block") == 3


def test_topk_clamps_when_larger_than_evidence():
    evidence = [make_evidence(chunk_id="a"), make_evidence(chunk_id="b")]
    ctx = build_context(evidence, top_k=10)
    assert len(ctx.blocks) == 2
    assert ctx.reserve == []


def test_topk_invalid_and_empty_raise():
    with pytest.raises(ValueError):
        build_context([make_evidence()], top_k=0)
    with pytest.raises(ValueError):
        build_context([], top_k=3)


def test_evidence_order_preserved_never_resorted():
    evidence = [
        make_evidence(chunk_id="low", rerank_score=-5.0),
        make_evidence(chunk_id="high", rerank_score=99.0),
    ]
    ctx = build_context(evidence, top_k=2)
    assert [b.evidence.chunk_id for b in ctx.blocks] == ["low", "high"]


# --- neighbor expansion ----------------------------------------------------

def test_neighbor_stitch_same_source(three_chunk_index):
    ev = make_evidence(chunk_id="doc_0001",
                       text="Middle chunk ends mid-sentence, with trailing overlap " + "x" * 60)
    ctx = build_context([ev], chunk_index=three_chunk_index, top_k=1, expand_neighbors=True)
    block = ctx.blocks[0]
    assert block.expanded_from == ["doc_0000", "doc_0002"]
    assert block.text.startswith("First chunk ends cleanly.\n")
    assert block.text.endswith("\nThird chunk ends cleanly.")
    assert "Middle chunk ends mid-sentence" in block.text


def test_neighbor_no_cross_source(tmp_path: Path):
    write_chunks_file(tmp_path, "a_3000_ov300.json", [
        make_chunk("a_0000", "a.md", 0, "A zero ends cleanly."),
        make_chunk("a_0001", "a.md", 1, "A one trails off without ending",
                   tail_truncated=True),
    ])
    write_chunks_file(tmp_path, "b_3000_ov300.json", [
        make_chunk("b_0000", "b.md", 0, "B zero belongs elsewhere."),
    ])
    index = load_chunk_index(tmp_path)
    ev = make_evidence(chunk_id="a_0001", source="a.md",
                       text="A one trails off without ending")
    ctx = build_context([ev], chunk_index=index, top_k=1, expand_neighbors=True)
    block = ctx.blocks[0]
    # Only a.md index 0 is stitched; b.md is never pulled in.
    assert block.expanded_from == ["a_0000"]
    assert "B zero" not in block.text


def test_neighbor_edge_chunk_stitches_available_side_only(three_chunk_index):
    ev = make_evidence(chunk_id="doc_0000", text="First chunk trails off,")
    ctx = build_context([ev], chunk_index=three_chunk_index, top_k=1, expand_neighbors=True)
    assert ctx.blocks[0].expanded_from == ["doc_0001"]


def test_neighbor_disabled_by_default_leaves_text_untouched(three_chunk_index):
    original = "Middle chunk ends mid-sentence, with trailing overlap " + "x" * 60
    ev = make_evidence(chunk_id="doc_0001", text=original)
    ctx = build_context([ev], chunk_index=three_chunk_index, top_k=1)
    assert ctx.blocks[0].text == original
    assert ctx.blocks[0].expanded_from == []


def test_complete_chunk_does_not_expand(three_chunk_index):
    ev = make_evidence(chunk_id="doc_0002", text="Third chunk ends cleanly.")
    ctx = build_context([ev], chunk_index=three_chunk_index, top_k=1, expand_neighbors=True)
    assert ctx.blocks[0].text == "Third chunk ends cleanly."
    assert ctx.blocks[0].expanded_from == []


def test_evidence_not_mutated_by_expansion(three_chunk_index):
    original = "Middle chunk ends mid-sentence, with trailing overlap " + "x" * 60
    ev = make_evidence(chunk_id="doc_0001", text=original)
    build_context([ev], chunk_index=three_chunk_index, top_k=1, expand_neighbors=True)
    assert ev.text == original


# --- deduplication ----------------------------------------------------------

def test_dedup_300char_overlap_same_source():
    seam = "S" * 300
    ev0 = make_evidence(chunk_id="d_0000", source="d.md", text="Head " + seam)
    ev1 = make_evidence(chunk_id="d_0001", source="d.md", text=seam + " tail.")
    ctx = build_context([ev0, ev1], top_k=2)
    assert ctx.blocks[0].text == "Head " + seam
    assert ctx.blocks[1].text == " tail."
    assert ctx.prompt_text.count(seam) == 1


def test_dedup_does_not_merge_across_sources():
    seam = "S" * 300
    ev0 = make_evidence(chunk_id="d_0000", source="d.md", text="Head " + seam)
    ev1 = make_evidence(chunk_id="e_0000", source="e.md", text=seam + " tail.")
    ctx = build_context([ev0, ev1], top_k=2)
    assert ctx.blocks[1].text == seam + " tail."


# --- metadata passthrough ----------------------------------------------------

def test_metadata_passthrough_and_enrichment(three_chunk_index):
    ev = make_evidence(chunk_id="doc_0001")
    ctx = build_context([ev], chunk_index=three_chunk_index, top_k=1)
    block = ctx.blocks[0]
    assert block.evidence.chunk_id == "doc_0001"
    assert block.evidence.standard_no == "IS 456"
    assert block.evidence.clause == "5.4"
    assert (block.evidence.page_start, block.evidence.page_end) == (15, 16)
    # year + heading_path enriched from the chunk index for display
    assert "- **Standard:** IS 456:2000" in ctx.prompt_text
    assert "- **Heading:** 5 MATERIALS > 5.4 Water" in ctx.prompt_text


# --- real-corpus smoke test (read-only) --------------------------------------

def test_real_chunk_fixture_smoke():
    corpus_file = Path("data/chunks/456_2000_amd5_reff2021_3000_ov300.json")
    if not corpus_file.exists():
        pytest.skip("corpus chunk file absent")
    chunks = json.loads(corpus_file.read_text(encoding="utf-8"))
    first = chunks[0]
    meta = first["metadata"]
    ev = make_evidence(
        chunk_id=first["id"],
        text=first["text"],
        source=meta["source"],
        clause=meta["clause"],
        heading=meta["heading"],
        standard_no=meta["standard_no"],
        page_start=meta["page_start"],
        page_end=meta["page_end"],
    )
    ctx = build_context([ev], top_k=1)
    assert first["id"] in ctx.prompt_text
    assert "IS 456" in ctx.prompt_text


def test_load_chunk_index_default_survives_foreign_cwd(tmp_path, monkeypatch):
    """Bare load_chunk_index() must resolve the repo-anchored default
    from any working directory (JSON reads only — no models)."""
    monkeypatch.chdir(tmp_path)
    assert len(load_chunk_index().by_id) == 2081
