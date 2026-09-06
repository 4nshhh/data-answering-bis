"""Offline unit tests for Phase 4 prompts / grounding directives.

No models, API keys, database, or network required. Context fixtures
are built with the Phase 3 builder over synthetic chunk JSONs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from retrieval.types import RetrievedEvidence

from answering.generator.context_builder import build_context, load_chunk_index
from answering.generator.prompts import (
    SYSTEM_PROMPT,
    PromptBundle,
    _estimate_tokens,
    build_prompt,
)


def make_evidence(chunk_id: str, text: str, source: str = "doc.md") -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        text=text,
        source=source,
        clause="5.4",
        heading="5.4 Water",
        standard_no="IS 456",
        page_start=15,
        page_end=16,
        dense_score=0.9,
        rerank_score=1.5,
        is_mask_restricted=True,
    )


def make_chunk(chunk_id: str, source: str, chunk_index: int, text: str) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "source": source,
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
        },
    }


@pytest.fixture()
def context_and_index(tmp_path: Path):
    chunks = [
        make_chunk("doc_0000", "doc.md", 0, "Water shall have pH not less than 6."),
        make_chunk("doc_0001", "doc.md", 1, "Chloride content shall not exceed 500 mg/L."),
        make_chunk("doc_0002", "doc.md", 2, "Admixtures shall comply with IS 9103."),
    ]
    (tmp_path / "doc_3000_ov300.json").write_text(json.dumps(chunks), encoding="utf-8")
    index = load_chunk_index(tmp_path)
    evidence = [
        make_evidence("doc_0000", "Water shall have pH not less than 6."),
        make_evidence("doc_0001", "Chloride content shall not exceed 500 mg/L."),
        make_evidence("doc_0002", "Admixtures shall comply with IS 9103."),
    ]
    return build_context(evidence, chunk_index=index, top_k=3), index


# --- estimator ---------------------------------------------------------------

def test_estimate_tokens_chars4_ceiling():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("abcde") == 2
    assert _estimate_tokens("x" * 400) == 100


def test_estimate_tokens_deterministic_and_monotonic():
    assert _estimate_tokens("hello world") == _estimate_tokens("hello world")
    assert _estimate_tokens("x" * 1000) > _estimate_tokens("x" * 10)


# --- grounding rules ----------------------------------------------------------

def test_system_prompt_contains_grounding_rules():
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered and "explicitly stated" in lowered  # facts-only rule
    assert "do not infer" in lowered or "extrapolat" in lowered  # no-extrapolation rule
    assert "100%" in SYSTEM_PROMPT or "exactness" in lowered  # exact-numbers rule
    assert "condition" in lowered  # conditional-requirements rule


def test_system_prompt_has_citation_format_but_no_refusal_sentence():
    assert "Clause" in SYSTEM_PROMPT and "Page" in SYSTEM_PROMPT
    assert "do not contain sufficient technical information" not in SYSTEM_PROMPT


# --- message assembly ----------------------------------------------------------

def test_user_message_contains_query_and_blocks(context_and_index):
    context, index = context_and_index
    bundle = build_prompt("What is the minimum pH of water?", context, chunk_index=index)
    assert isinstance(bundle, PromptBundle)
    assert "What is the minimum pH of water?" in bundle.user
    assert bundle.user.count("### Context Block") == 3
    assert "pH not less than 6" in bundle.user
    assert "IS 456:2000" in bundle.user  # enriched header matches Phase 3 rendering


def test_roles_separated(context_and_index):
    context, _ = context_and_index
    bundle = build_prompt("Some query?", context)
    assert "Some query?" not in bundle.system
    assert "### Context Block" not in bundle.system
    assert "Grounding rules" in bundle.system or "grounding" in bundle.system.lower()
    assert "Grounding rules" not in bundle.user


def test_used_ids_track_blocks_in_order(context_and_index):
    context, _ = context_and_index
    bundle = build_prompt("q?", context)
    assert bundle.used_block_ids == ["doc_0000", "doc_0001", "doc_0002"]
    assert bundle.reserve_block_ids == []
    assert bundle.insufficient_evidence_hook is False
    assert bundle.model == "openai/gpt-oss-120b"


def test_reserve_ids_propagated(context_and_index):
    context, _ = context_and_index
    assert len(context.reserve) == 0
    evidence = [make_evidence(f"extra_{i}", f"Extra text {i}.", source="other.md") for i in range(2)]
    from answering.generator.context_builder import build_context as build_ctx

    base = [make_evidence("doc_0000", "Water shall have pH not less than 6.")]
    ctx = build_ctx(base + evidence, top_k=1)
    bundle = build_prompt("q?", ctx)
    assert bundle.used_block_ids == ["doc_0000"]
    assert bundle.reserve_block_ids == ["extra_0", "extra_1"]


def test_empty_query_raises(context_and_index):
    context, _ = context_and_index
    with pytest.raises(ValueError):
        build_prompt("   ", context)


def test_invalid_budget_params_raise(context_and_index):
    context, _ = context_and_index
    with pytest.raises(ValueError):
        build_prompt("q?", context, max_context_tokens=0)
    with pytest.raises(ValueError):
        build_prompt("q?", context, reserve_margin_tokens=-1)


# --- budgeting ------------------------------------------------------------------

def test_budget_drops_trailing_blocks_first(context_and_index):
    from answering.generator.prompts import _render_blocks

    context, _ = context_and_index
    rendered = _render_blocks(context)
    overhead = _estimate_tokens(SYSTEM_PROMPT) + _estimate_tokens("q?") + 8192
    # Room for the first block plus a 1-token separator, but not the second block.
    tiny = overhead + _estimate_tokens(rendered[0]) + 1
    assert tiny < overhead + sum(_estimate_tokens(r) + 1 for r in rendered)
    bundle = build_prompt(
        "q?",
        context,
        max_context_tokens=tiny,
        reserve_margin_tokens=8192,
    )
    assert bundle.used_block_ids == ["doc_0000"]  # rerank order preserved
    assert bundle.reserve_block_ids == ["doc_0001", "doc_0002"]
    assert _estimate_tokens(bundle.system) + _estimate_tokens(bundle.user) <= tiny
    assert bundle.insufficient_evidence_hook is False


def test_zero_budget_sets_hook_without_refusal_text(context_and_index):
    context, _ = context_and_index
    bundle = build_prompt("q?", context, max_context_tokens=10, reserve_margin_tokens=8192)
    assert bundle.used_block_ids == []
    assert bundle.insufficient_evidence_hook is True
    assert "Evidence:\n" in bundle.user
    assert "do not contain sufficient technical information" not in bundle.user
    assert "do not contain sufficient technical information" not in bundle.system
    # Dropped ids remain visible for Phase 7 / logging.
    assert bundle.reserve_block_ids == ["doc_0000", "doc_0001", "doc_0002"]


def test_context_not_mutated(context_and_index):
    context, _ = context_and_index
    before = [b.text for b in context.blocks]
    build_prompt("q?", context, max_context_tokens=10, reserve_margin_tokens=8192)
    assert [b.text for b in context.blocks] == before
    assert len(context.blocks) == 3
