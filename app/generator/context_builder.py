"""Phase 3: context assembly and neighbor expansion.

Consumes rerank-ordered ``RetrievedEvidence`` from the frozen retrieval
package (``retrieval.retrieve``) and produces deterministic Markdown
context blocks for the generation layer (Phase 4+).

Scope (AGENTS.md section 8):
  1. Top-K budget: select Top-N (default 3) evidence items, keep the
     remainder in reserve.
  2. Context-block formatting with the exact header template.
  3. Neighbor expansion: stitch ``chunk_index +/- 1`` chunks from the
     same source document when a chunk ends mid-sentence or mid-table.
  4. Deduplication of the 300-char chunking overlap between selected
     blocks.

Out of scope: abstention/refusal (Phase 7), system prompts (Phase 4),
LLM calls (Phase 5), citation parsing (Phase 6), token/char budgeting
(deferred to Phase 4 by design decision).

The ``RetrievedEvidence`` contract is treated as immutable input; this
module copies text and never mutates evidence objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from retrieval.types import RetrievedEvidence

__all__ = [
    "DEFAULT_TOP_K",
    "OVERLAP_CHARS",
    "ChunkIndex",
    "ContextBlock",
    "BuiltContext",
    "load_chunk_index",
    "format_block",
    "build_context",
]

#: Default number of evidence items selected for the prompt context.
DEFAULT_TOP_K = 3

#: Frozen chunking overlap (chars). Used for seam deduplication.
OVERLAP_CHARS = 300

#: Trailing characters treated as a complete ending. Anything else
#: (trailing letter/digit/comma/semicolon/colon/hyphen/pipe/...) is
#: treated as a potential mid-sentence or mid-table split.
_TERMINAL_CHARS = frozenset('.!?"\'\u201d\u2019)]')

#: Radius for neighbor lookup (``chunk_index +/- 1`` per AGENTS.md 8.3).
_NEIGHBOR_RADIUS = 1


@dataclass
class ChunkIndex:
    """Read-only lookup over ``data/chunks/*.json``.

    Attributes:
        by_position: ``(source, chunk_index) -> raw chunk dict``.
        by_id: ``chunk_id -> raw chunk dict``.
        chunks_dir: directory the index was loaded from.
    """

    by_position: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    chunks_dir: Optional[Path] = None


@dataclass
class ContextBlock:
    """One selected evidence item with its assembled context text."""

    rank: int  # 1-based display rank (rerank order preserved)
    evidence: RetrievedEvidence  # original object, unmutated
    text: str  # post-expansion + post-dedup text for this block
    expanded_from: list[str] = field(default_factory=list)  # stitched neighbor ids


@dataclass
class BuiltContext:
    """Result of :func:`build_context`."""

    blocks: list[ContextBlock]
    reserve: list[RetrievedEvidence]  # unselected remainder, order preserved
    prompt_text: str  # formatted blocks joined, ready for Phase 4


def load_chunk_index(chunks_dir: Path | str = Path("data/chunks")) -> ChunkIndex:
    """Load a read-only positional index over chunk JSON files.

    Each ``*.json`` file holds a list of ``{"id", "text", "metadata"}``
    dicts. Only reads; never modifies chunk files or boundaries.
    """
    root = Path(chunks_dir)
    index = ChunkIndex(chunks_dir=root)
    for path in sorted(root.glob("*.json")):
        chunks = json.loads(path.read_text(encoding="utf-8"))
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            index.by_id[chunk["id"]] = chunk
            chunk_pos = meta.get("chunk_index")
            source = meta.get("source")
            if source is not None and chunk_pos is not None:
                index.by_position[(source, int(chunk_pos))] = chunk
    return index


def _needs_expansion(chunk_dict: Optional[dict[str, Any]], text: str) -> bool:
    """Decide whether a chunk needs ``chunk_index +/- 1`` stitching."""
    if chunk_dict is not None:
        meta = chunk_dict.get("metadata", {})
        if bool(meta.get("tail_truncated")) or bool(meta.get("table_repaired")):
            return True
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.count("|") % 2 == 1:
        return True  # unbalanced table pipes: probable split table
    return stripped[-1] not in _TERMINAL_CHARS


def _fetch_neighbors(
    evidence: RetrievedEvidence,
    chunk_dict: dict[str, Any],
    index: ChunkIndex,
    radius: int = _NEIGHBOR_RADIUS,
) -> list[tuple[str, dict[str, Any]]]:
    """Fetch ``(label, chunk_dict)`` neighbors within the same source.

    Returns at most ``("prev", ...)`` and ``("next", ...)`` entries, in
    document order. Never crosses into another ``source`` document and
    silently skips missing edge positions.
    """
    meta = chunk_dict.get("metadata", {})
    source = meta.get("source", evidence.source)
    try:
        pos = int(meta["chunk_index"])
    except (KeyError, TypeError, ValueError):
        return []
    neighbors: list[tuple[str, dict[str, Any]]] = []
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        candidate = index.by_position.get((source, pos + offset))
        if candidate is None or candidate["id"] == evidence.chunk_id:
            continue
        neighbors.append(("prev" if offset < 0 else "next", candidate))
    neighbors.sort(key=lambda item: 0 if item[0] == "prev" else 1)
    return neighbors


def _expand_text(
    evidence: RetrievedEvidence,
    chunk_dict: Optional[dict[str, Any]],
    index: ChunkIndex,
) -> tuple[str, list[str]]:
    """Stitch ``chunk_index +/- 1`` context around one evidence text."""
    if chunk_dict is None:
        return evidence.text, []
    if not _needs_expansion(chunk_dict, evidence.text):
        return evidence.text, []
    neighbors = _fetch_neighbors(evidence, chunk_dict, index)
    if not neighbors:
        return evidence.text, []
    text = evidence.text
    expanded_from: list[str] = []
    for label, neighbor in neighbors:
        neighbor_text = neighbor.get("text", "")
        if label == "prev":
            text = neighbor_text + "\n" + _strip_leading_overlap(text, neighbor_text)
        else:
            text = _strip_trailing_overlap(text, neighbor_text) + "\n" + neighbor_text
        expanded_from.append(neighbor["id"])
    return text, expanded_from


def _strip_leading_overlap(text: str, prev_text: str, max_overlap: int = OVERLAP_CHARS) -> str:
    """Strip a duplicated seam where ``text`` starts with ``prev_text``'s tail."""
    limit = min(max_overlap, len(text), len(prev_text))
    for size in range(limit, 0, -1):
        if text[:size] == prev_text[-size:]:
            return text[size:]
    return text


def _strip_trailing_overlap(text: str, next_text: str, max_overlap: int = OVERLAP_CHARS) -> str:
    """Strip a duplicated seam where ``text`` ends with ``next_text``'s head."""
    limit = min(max_overlap, len(text), len(next_text))
    for size in range(limit, 0, -1):
        if text[-size:] == next_text[:size]:
            return text[:-size]
    return text


def _deduplicate_texts(texts: list[str], sources: list[str]) -> list[str]:
    """Remove 300-char overlap seams between consecutive same-source blocks."""
    if not texts:
        return []
    result = [texts[0]]
    for i in range(1, len(texts)):
        if sources[i] == sources[i - 1]:
            result.append(_strip_leading_overlap(texts[i], result[i - 1]))
        else:
            result.append(texts[i])
    return result


def _display_year(evidence: RetrievedEvidence, chunk_dict: Optional[dict[str, Any]]) -> Optional[str]:
    if chunk_dict is not None:
        year = chunk_dict.get("metadata", {}).get("year")
        if year:
            return str(year)
    extra = evidence.extra or {}
    for key in ("year",):
        if extra.get(key):
            return str(extra[key])
    return None


def _display_heading(evidence: RetrievedEvidence, chunk_dict: Optional[dict[str, Any]]) -> str:
    if chunk_dict is not None:
        heading_path = chunk_dict.get("metadata", {}).get("heading_path") or []
        if heading_path:
            return " > ".join(str(h) for h in heading_path)
    if evidence.heading:
        return evidence.heading
    return "N/A"


def _display_pages(evidence: RetrievedEvidence) -> str:
    start, end = evidence.page_start, evidence.page_end
    if start is not None and end is not None:
        return f"Page {start}-{end}" if start != end else f"Page {start}"
    if start is not None:
        return f"Page {start}"
    if end is not None:
        return f"Page {end}"
    return "Page N/A"


def format_block(
    rank: int,
    evidence: RetrievedEvidence,
    text: str,
    year: Optional[str] = None,
    heading_path: Optional[list[str]] = None,
) -> str:
    """Format one evidence item as a Markdown context block (AGENTS.md 8.2)."""
    standard = evidence.standard_no or "N/A"
    if year and evidence.standard_no:
        standard = f"{evidence.standard_no}:{year}"
    if heading_path:
        heading = " > ".join(str(h) for h in heading_path)
    elif evidence.heading:
        heading = evidence.heading
    else:
        heading = "N/A"
    clause = evidence.clause or "N/A"
    location = f"{_display_pages(evidence)} (Chunk ID: {evidence.chunk_id})"
    return (
        f"### Context Block [{rank}]\n"
        f"- **Standard:** {standard}\n"
        f"- **Clause:** {clause}\n"
        f"- **Heading:** {heading}\n"
        f"- **Location:** {location}\n"
        f"\n"
        f"```text\n"
        f"{text}\n"
        f"```"
    )


def build_context(
    evidence: list[RetrievedEvidence],
    chunk_index: Optional[ChunkIndex] = None,
    top_k: int = DEFAULT_TOP_K,
    expand_neighbors: bool = False,
) -> BuiltContext:
    """Assemble Top-K evidence into deterministic Markdown context.

    Args:
        evidence: rerank-ordered candidates from ``retrieve()``; order
            is preserved, never re-sorted.
        chunk_index: optional read-only index from :func:`load_chunk_index`,
            required for neighbor expansion and year/heading enrichment.
            When ``None``, blocks use evidence fields only.
        top_k: number of items to select (default 3). Values above
            ``len(evidence)`` select everything; values below 1 raise.
        expand_neighbors: stitch ``chunk_index +/- 1`` context when a
            selected chunk ends mid-sentence or mid-table (default False,
            matching the API request schema).

    Returns:
        ``BuiltContext`` with selected blocks, reserve, and joined prompt text.
    """
    if not evidence:
        raise ValueError("evidence must contain at least one item")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")

    selected = list(evidence[:top_k])
    reserve = list(evidence[top_k:])

    texts: list[str] = []
    expanded: list[list[str]] = []
    enrich: list[tuple[Optional[str], Optional[list[str]]]] = []
    for item in selected:
        chunk_dict = chunk_index.by_id.get(item.chunk_id) if chunk_index is not None else None
        year = _display_year(item, chunk_dict)
        heading_path: Optional[list[str]] = None
        if chunk_dict is not None:
            raw_path = chunk_dict.get("metadata", {}).get("heading_path") or []
            heading_path = [str(h) for h in raw_path] or None
        if expand_neighbors and chunk_index is not None:
            text, from_ids = _expand_text(item, chunk_dict, chunk_index)
        else:
            text, from_ids = item.text, []
        texts.append(text)
        expanded.append(from_ids)
        enrich.append((year, heading_path))

    deduped = _deduplicate_texts(texts, [item.source for item in selected])

    blocks = [
        ContextBlock(rank=i + 1, evidence=item, text=deduped[i], expanded_from=expanded[i])
        for i, item in enumerate(selected)
    ]
    formatted = [
        format_block(block.rank, block.evidence, block.text, year=enrich[i][0], heading_path=enrich[i][1])
        for i, block in enumerate(blocks)
    ]
    return BuiltContext(blocks=blocks, reserve=reserve, prompt_text="\n\n".join(formatted))
