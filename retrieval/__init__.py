"""Backend-facing retrieval package (query-time only).

Public interface — the backend depends on these names, never on
individual research scripts::

    from retrieval import retrieve, ChunkStore, LocalNpyStore, RetrievedEvidence

    results = retrieve("What does clause 9.10 of IS 10262 require?")
"""

from .retrieve import Retriever, retrieve
from .store import ChunkStore, LocalNpyStore
from .types import ChunkRecord, RetrievedEvidence

__all__ = [
    "retrieve",
    "Retriever",
    "ChunkStore",
    "LocalNpyStore",
    "RetrievedEvidence",
    "ChunkRecord",
]
