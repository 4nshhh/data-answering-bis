"""Chunk-store abstraction: candidate retrieval over persisted chunks.

`ChunkStore` is the seam that lets the storage backend change (local
`.npy` today, PGVector tomorrow) without touching `retrieve()` or the
backend contract. `LocalNpyStore` is the current file-based
implementation; logic mirrors the validated benchmark path exactly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import ChunkRecord

IS_IN_QUERY_RE = re.compile(r"\bIS\s+(\d+)\b", re.IGNORECASE)
EMBEDDING_DIM = 1024


class ChunkStore(Protocol):
    """Minimal contract for candidate retrieval + record fetch."""

    def search(self, query_vector: np.ndarray, k: int,
               mask: set[int] | None) -> list[tuple[int, float]]:
        """Top-k (row_index, score) over masked candidates, rank-ordered."""
        ...

    def fetch(self, indices: list[int]) -> list[ChunkRecord]:
        """Records for row indices, in the requested order."""
        ...

    def enriched_text(self, index: int) -> str:
        """Reranker-ready representation for one row (enriched, not raw)."""
        ...

    def __len__(self) -> int:
        ...


def load_chunks(chunks_dir: Path) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for path in sorted(chunks_dir.glob("*.json")):
        for chunk in json.loads(path.read_text(encoding="utf-8")):
            meta = chunk["metadata"]
            records.append(
                ChunkRecord(
                    id=chunk["id"],
                    text=chunk["text"],
                    source=meta.get("source", ""),
                    clause=meta.get("clause"),
                    heading=meta.get("heading"),
                    standard_no=meta.get("standard_no"),
                    page_start=meta.get("page_start"),
                    page_end=meta.get("page_end"),
                    low_confidence=bool(meta.get("low_confidence")),
                )
            )
    return records


def load_chunk_dicts(chunks_dir: Path) -> list[dict]:
    chunks = []
    for path in sorted(chunks_dir.glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


def build_enriched_text(chunk_dict: dict) -> str:
    meta = chunk_dict["metadata"]
    parts = []
    if meta.get("standard_no"):
        parts.append(f"Standard: {meta['standard_no']}")
    if meta.get("clause"):
        parts.append(f"Clause: {meta['clause']}")
    heading_path = meta.get("heading_path")
    if heading_path:
        parts.append("Heading: " + " > ".join(heading_path))
    elif meta.get("heading"):
        parts.append(f"Heading: {meta['heading']}")

    prefix = "\n".join(parts)
    if prefix:
        return f"{prefix}\n\n{chunk_dict['text']}"
    return chunk_dict["text"]


def _standard_number_match(number: str, record: ChunkRecord) -> bool:
    """Exact IS-number match against chunk metadata or source filename.

    The number must match the whole standard number, not be a fragment
    of a longer one (e.g. '456' must not match 'IS 4560').
    """
    if record.standard_no:
        core = re.sub(r"\D", "", record.standard_no)
        if core == number:
            return True
    fn = record.source
    stem = fn.split(".", 1)[0]
    head = re.match(r"^(\d+)", stem)
    if head and head.group(1) == number:
        return True
    return False


def query_side_candidate_mask(query_text: str,
                              records: list[ChunkRecord]) -> set[int] | None:
    """Restrict candidates when the user names a specific IS number.

    Query-text-derived only (never gold metadata); exact-number matching
    with full-corpus (`None`) fallback. Returns the set of allowed row
    indices, and whether the mask fired is reported per evidence item.
    """
    is_match = IS_IN_QUERY_RE.search(query_text)
    if not is_match:
        return None
    number = is_match.group(1)
    indices = {
        i for i, r in enumerate(records) if _standard_number_match(number, r)
    }
    return indices or None


def rank_indices(scores: list[float], top_k: int) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]


class LocalNpyStore:
    """File-backed store: `data/chunks/*.json` + precomputed `.npy` cache.

    Row alignment invariant (validated): sorted-glob chunk order ==
    `chunks_combined.jsonl` order == `.npy` rows.
    """

    def __init__(self, chunks_dir: Path | str = "data/chunks",
                  vector_cache: Path | str = "data/vectors/bge_m3_enriched_vectors.npy",
                  expected_dim: int = EMBEDDING_DIM) -> None:
        self._chunks_dir = Path(chunks_dir)
        self._records = load_chunks(self._chunks_dir)
        self._dicts = load_chunk_dicts(self._chunks_dir)
        assert [d["id"] for d in self._dicts] == [r.id for r in self._records], \
            "chunk dict/record order mismatch"
        cache_path = Path(vector_cache)
        vectors = np.load(cache_path)
        if not (isinstance(vectors, np.ndarray)
                and vectors.shape == (len(self._records), expected_dim)
                and np.isfinite(vectors).all()):
            raise ValueError(
                f"Vector cache '{cache_path}' incompatible: shape "
                f"{getattr(vectors, 'shape', None)}, expected "
                f"({len(self._records)}, {expected_dim}).")
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._enriched = [build_enriched_text(d) for d in self._dicts]

    def search(self, query_vector: np.ndarray, k: int,
               mask: set[int] | None) -> list[tuple[int, float]]:
        sims = (self._vectors @ query_vector).tolist()
        if mask is not None:
            sims = [s if i in mask else float("-inf")
                    for i, s in enumerate(sims)]
        return [(i, float(sims[i])) for i in rank_indices(sims, k)]

    def fetch(self, indices: list[int]) -> list[ChunkRecord]:
        return [self._records[i] for i in indices]

    def enriched_text(self, index: int) -> str:
        return self._enriched[index]

    def mask_for(self, query_text: str) -> set[int] | None:
        return query_side_candidate_mask(query_text, self._records)

    def __len__(self) -> int:
        return len(self._records)
