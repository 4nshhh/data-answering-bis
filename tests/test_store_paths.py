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
    signature = inspect.signature(LocalNpyStore.__init__)
    assert signature.parameters["chunks_dir"].default == "data/chunks"
    assert signature.parameters["vector_cache"].default == \
        "data/vectors/bge_m3_enriched_vectors.npy"
    assert signature.parameters["expected_dim"].default == EMBEDDING_DIM == 1024


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
