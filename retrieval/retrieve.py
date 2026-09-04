"""Backend-facing retrieval entry point: the frozen winning pipeline.

query -> BGE-M3 enriched dense retrieval -> top-10 candidates
      -> cross-encoder reranker (same enriched representation)
      -> ranked evidence with full provenance.

Depends only on the `ChunkStore` protocol: swapping `LocalNpyStore` for a
future PGVector implementation changes nothing backend-facing. No
benchmark concepts here (no QUERIES, gold labels, or reports).
"""

from __future__ import annotations

import logging

from .models import device_info, load_query_encoder, load_reranker
from .store import ChunkStore, LocalNpyStore
from .types import RetrievedEvidence

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 10


class Retriever:
    """Reusable retrieval pipeline; construct once, call per query."""
    def __init__(self, store: ChunkStore | None = None, top_n: int = DEFAULT_TOP_N,
                 device: str | None = None) -> None:
        self.store = store or LocalNpyStore()
        self.top_n = top_n
        self._model = load_query_encoder(device=device)
        self._reranker = load_reranker(device=device)
        info = device_info()
        logger.info(
            "Retriever ready: encoder=%s reranker=%s (cuda_available=%s, gpu=%s)",
            getattr(self._model, "device", "?"),
            getattr(self._reranker, "device", "?"),
            info["cuda_available"],
            info["gpu_name"],
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedEvidence]:
        """Ranked evidence for one user query (raw query text only)."""
        import numpy as np

        from .store import LocalNpyStore

        q_vec = np.asarray(
            self._model.encode([query], normalize_embeddings=True),
            dtype=np.float32)[0]
        mask = self.store.mask_for(query) \
            if hasattr(self.store, "mask_for") else None
        if isinstance(self.store, LocalNpyStore):
            sims = (self.store._vectors @ q_vec).tolist()
        else:  # protocol-based store: score via search()
            sims = [float("-inf")] * len(self.store)
            for i, s in self.store.search(q_vec, len(self.store), mask):
                sims[i] = s
        if mask is not None:
            sims = [s if i in mask else float("-inf")
                    for i, s in enumerate(sims)]
        dense_order = [i for i, _ in
                       sorted(enumerate(sims), key=lambda kv: kv[1], reverse=True)]
        cand_idx = [i for i in dense_order if np.isfinite(sims[i])][:self.top_n]
        cand_texts = [self.store.enriched_text(i) for i in cand_idx]
        cross = self._reranker.predict([(query, t) for t in cand_texts])
        if isinstance(cross, np.ndarray):
            cross = cross.tolist()
        order = sorted(range(len(cand_idx)),
                       key=lambda r: cross[r], reverse=True)[:top_k]
        recs = self.store.fetch([cand_idx[r] for r in order])
        out = []
        for r, rec in zip(order, recs):
            row = cand_idx[r]
            out.append(RetrievedEvidence(
                chunk_id=rec.id, text=rec.text, source=rec.source,
                clause=rec.clause, heading=rec.heading,
                standard_no=rec.standard_no, page_start=rec.page_start,
                page_end=rec.page_end, dense_score=float(sims[row]),
                rerank_score=float(cross[r]),
                is_mask_restricted=mask is not None,
            ))
        return out


_default_retriever: Retriever | None = None


def loaded_retriever() -> Retriever | None:
    """Return the shared retriever if already constructed, else None.

    Never triggers model loading; for read-only device diagnostics.
    """
    return _default_retriever


def retrieve(query: str, top_k: int = 10) -> list[RetrievedEvidence]:
    """Backend entry point: ranked evidence for one user query.

    Uses a lazily-constructed shared `Retriever` (models load once on
    first call). For explicit lifecycle control, construct `Retriever`
    directly instead.
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever.retrieve(query, top_k=top_k)
