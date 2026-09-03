"""Shared retrieval data types — canonical home (query-time package).

`ChunkRecord` was moved here from `scripts/retrieval_eval.py`, which keeps
a compatibility re-export. Backend code must import from `retrieval`,
never from `scripts/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkRecord:
    id: str
    text: str
    source: str
    clause: str | None
    heading: str | None
    standard_no: str | None
    page_start: int | None
    page_end: int | None
    low_confidence: bool


@dataclass
class RetrievedEvidence:
    """One ranked candidate returned to the backend (generation input)."""

    chunk_id: str
    text: str
    source: str
    clause: str | None
    heading: str | None
    standard_no: str | None
    page_start: int | None
    page_end: int | None
    dense_score: float | None
    rerank_score: float | None
    is_mask_restricted: bool
    extra: dict = field(default_factory=dict)

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
