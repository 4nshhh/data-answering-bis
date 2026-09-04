"""Backend-facing retrieval package (query-time only).

Public interface — the backend depends on these names, never on
individual research scripts::

    from retrieval import retrieve, ChunkStore, LocalNpyStore, RetrievedEvidence

    results = retrieve("What does clause 9.10 of IS 10262 require?")
"""

from .models import device_info, resolve_device
from .retrieve import Retriever, loaded_retriever, retrieve
from .store import ChunkStore, LocalNpyStore
from .types import ChunkRecord, RetrievedEvidence

__all__ = [
    "retrieve",
    "Retriever",
    "loaded_retriever",
    "resolve_device",
    "device_info",
    "ChunkStore",
    "LocalNpyStore",
    "RetrievedEvidence",
    "ChunkRecord",
]
