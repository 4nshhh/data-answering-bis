"""Path-configuration tests for LocalNpyStore (SIH demo backend).

Verifies the store resolves the migrated assets at their current
locations (``data/chunks/*.json`` + ``data/vectors/*.npy``) with no
file duplication or renaming, that explicit paths still override the
defaults, and that the frozen shape/dtype guards remain active.

The default-init test loads the real 8.5 MB validated cache (fast,
memory-mapped read by numpy); no embeddings are regenerated.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from retrieval.store import EMBEDDING_DIM, LocalNpyStore


def test_default_paths_match_migrated_layout():
    from retrieval.store import DEFAULT_CHUNKS_DIR, DEFAULT_VECTOR_CACHE

    signature = inspect.signature(LocalNpyStore.__init__)
    assert signature.parameters["chunks_dir"].default == DEFAULT_CHUNKS_DIR
    assert signature.parameters["vector_cache"].default == DEFAULT_VECTOR_CACHE
    assert signature.parameters["expected_dim"].default == EMBEDDING_DIM == 1024
    # Anchored to the repo root (not CWD) with the migrated layout intact.
    assert Path(DEFAULT_CHUNKS_DIR).is_absolute()
    assert Path(DEFAULT_VECTOR_CACHE).is_absolute()
    assert Path(DEFAULT_CHUNKS_DIR).name == "chunks"
    assert Path(DEFAULT_VECTOR_CACHE).name == "bge_m3_enriched_vectors.npy"


def test_default_init_works_from_foreign_cwd(tmp_path: Path, monkeypatch):
    """Direct-library backends start from any directory: defaults must
    still resolve (proves CWD-independence without loading models)."""
    monkeypatch.chdir(tmp_path)
    store = LocalNpyStore()
    assert len(store) == 2081


def test_default_init_loads_validated_cache():
    store = LocalNpyStore()
    assert len(store) == 2081
    assert store._vectors.shape == (2081, 1024)
    assert store._vectors.dtype == np.float32
    assert bool(np.isfinite(store._vectors).all())
    # Spot checks: enriched header, record fetch, IS-mask helper.
    assert store.enriched_text(0).startswith("Standard: ")
    (record,) = store.fetch([0])
    assert record.id and record.text
    assert isinstance(store.mask_for("What does IS 456 require?"), (set, type(None)))


def _write_chunks(directory: Path, name: str, n: int) -> Path:
    chunks = [
        {
            "id": f"t_{i:04d}",
            "text": f"Chunk text {i}.",
            "metadata": {
                "source": "t.md",
                "chunk_index": i,
                "clause": "1",
                "heading": "H",
                "heading_path": ["H"],
                "standard_no": "IS 1",
                "year": "2000",
                "page_start": 1,
                "page_end": 1,
                "low_confidence": False,
            },
        }
        for i in range(n)
    ]
    path = directory / name
    path.write_text(json.dumps(chunks), encoding="utf-8")
    return path


def test_explicit_paths_override_defaults(tmp_path: Path):
    _write_chunks(tmp_path, "t_3000_ov300.json", 2)
    cache = tmp_path / "custom.npy"
    rng = np.random.default_rng(0)
    np.save(cache, (rng.random((2, 1024)) - 0.5).astype(np.float32))
    store = LocalNpyStore(chunks_dir=tmp_path, vector_cache=cache)
    assert len(store) == 2
    assert [r.id for r in store.fetch([0, 1])] == ["t_0000", "t_0001"]


def test_missing_cache_raises(tmp_path: Path):
    _write_chunks(tmp_path, "t_3000_ov300.json", 1)
    with pytest.raises((FileNotFoundError, OSError)):
        LocalNpyStore(chunks_dir=tmp_path, vector_cache=tmp_path / "absent.npy")


def test_shape_mismatch_guard_still_active(tmp_path: Path):
    # 1 chunk dir against the real 2081-row cache must fail loudly,
    # proving the frozen row-alignment guard was not weakened.
    _write_chunks(tmp_path, "t_3000_ov300.json", 1)
    with pytest.raises(ValueError, match="incompatible"):
        LocalNpyStore(chunks_dir=tmp_path)


def test_enriched_texts_matches_loop():
    store = LocalNpyStore()
    rows = [0, 7, 2080]
    assert store.enriched_texts(rows) == [store.enriched_text(i) for i in rows]
    assert store.enriched_texts([]) == []


def test_retriever_prefers_batch_texts():
    """Retriever must use one enriched_texts() call (not N single calls)
    so remote-backed stores avoid per-candidate round trips. No models,
    keys, or network: the Retriever is assembled with fakes."""
    from retrieval.retrieve import Retriever
    from retrieval.types import ChunkRecord

    class BatchStore:
        def __init__(self):
            self.batch_calls: list = []
            self.single_calls: list = []

        def __len__(self):
            return 2

        def search(self, query_vector, k, mask):
            return [(0, 0.9), (1, 0.8)][:k]

        def fetch(self, indices):
            return [ChunkRecord(id=f"c{i}", text=f"t{i}", source="s.md",
                                clause="1", heading="H", standard_no="IS 1",
                                page_start=1, page_end=1,
                                low_confidence=False) for i in indices]

        def enriched_text(self, index):
            self.single_calls.append(index)
            raise AssertionError("batch path must be preferred")

        def enriched_texts(self, indices):
            self.batch_calls.append(list(indices))
            return [f"enriched-{i}" for i in indices]

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True):
            return np.zeros((len(texts), 4), dtype=np.float32)

    class FakeReranker:
        def predict(self, pairs):
            return np.array([0.9, 0.1][:len(pairs)])

    store = BatchStore()
    retriever = Retriever.__new__(Retriever)
    retriever.store = store
    retriever.top_n = 10
    retriever._model = FakeEncoder()
    retriever._reranker = FakeReranker()

    out = retriever.retrieve("q?", top_k=2)
    assert store.batch_calls == [[0, 1]]
    assert store.single_calls == []
    assert [e.chunk_id for e in out] == ["c0", "c1"]
    assert [e.rerank_score for e in out] == [0.9, 0.1]


def test_retriever_falls_back_without_batch_hook():
    """Minimal ChunkStore implementations without enriched_texts keep
    working through the single-row method (protocol backward compat)."""
    from retrieval.retrieve import Retriever
    from retrieval.types import ChunkRecord

    class MinimalStore:
        def __len__(self):
            return 1

        def search(self, query_vector, k, mask):
            return [(0, 0.5)]

        def fetch(self, indices):
            return [ChunkRecord(id="c0", text="t", source="s.md", clause="1",
                                heading="H", standard_no="IS 1", page_start=1,
                                page_end=1, low_confidence=False)]

        def enriched_text(self, index):
            return "enriched-0"

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True):
            return np.zeros((len(texts), 4), dtype=np.float32)

    class FakeReranker:
        def predict(self, pairs):
            return np.array([0.7])

    retriever = Retriever.__new__(Retriever)
    retriever.store = MinimalStore()
    retriever.top_n = 10
    retriever._model = FakeEncoder()
    retriever._reranker = FakeReranker()

    (evidence,) = retriever.retrieve("q?", top_k=1)
    assert evidence.chunk_id == "c0"
    assert evidence.rerank_score == 0.7
