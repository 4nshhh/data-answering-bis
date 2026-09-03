"""One-time migration: canonical chunk JSON + .npy cache -> PG chunks table.

Reads `output_chunks/*.json` (sorted-glob order) and
`artifacts/bge_m3_enriched_vectors_gpu.npy` row-aligned, verifies
count (2081) and shape/dtype/finiteness, then upserts all rows.
Idempotent re-runs via chunk_id upsert. Verifies post-load count +
spot vector equality before exiting.

Usage (from repo root, repo venv only; needs DATABASE_URL in env/.env):
    .\\venv\\Scripts\\python.exe indexing/migrate_from_artifacts.py
    .\\venv\\Scripts\\python.exe indexing/migrate_from_artifacts.py --drop-first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.pg_store import _dsn  # noqa: E402


def ddl(table: str) -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id TEXT PRIMARY KEY,
    row_pos INT NOT NULL,
    embedding vector(1024) NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_index INT,
    clause TEXT,
    heading TEXT,
    heading_path TEXT[],
    standard_no TEXT,
    year TEXT,
    part TEXT,
    page_start INT,
    page_end INT,
    char_start INT,
    char_end INT,
    low_confidence BOOL NOT NULL DEFAULT FALSE,
    tail_truncated BOOL NOT NULL DEFAULT FALSE,
    table_repaired BOOL NOT NULL DEFAULT FALSE
);
"""


COLS = ("chunk_id, row_pos, embedding, text, source, chunk_index, clause, "
        "heading, heading_path, standard_no, year, part, "
        "page_start, page_end, char_start, char_end, "
        "low_confidence, tail_truncated, table_repaired")
UPDATABLE = ("row_pos, embedding, text, source, chunk_index, clause, heading, "
             "heading_path, standard_no, year, part, page_start, page_end, "
             "char_start, char_end, low_confidence, tail_truncated, "
             "table_repaired")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, default=Path("output_chunks"))
    parser.add_argument("--vectors", type=Path,
                        default=Path("artifacts/bge_m3_enriched_vectors_gpu.npy"))
    parser.add_argument("--table", type=str, default="chunks")
    parser.add_argument("--drop-first", action="store_true",
                        help="DROP TABLE before loading (fresh start).")
    parser.add_argument("--hnsw", action="store_true",
                        help="Build HNSW cosine index after load (default: exact scan).")
    args = parser.parse_args()

    chunks = []
    for p in sorted(args.chunks_dir.glob("*.json")):
        chunks.extend(json.loads(p.read_text(encoding="utf-8")))
    vectors = np.load(args.vectors)
    assert len(chunks) == 2081, len(chunks)
    assert vectors.shape == (len(chunks), 1024) and vectors.dtype == np.float32
    assert bool(np.isfinite(vectors).all())
    print(f"chunks={len(chunks)} vectors={vectors.shape} (validated)")

    import psycopg

    # prepare_threshold=None: Supavisor/pgbouncer transaction pooling does
    # not support server-side prepared statements (executemany would fail
    # with DuplicatePreparedStatement).
    with psycopg.connect(_dsn(), connect_timeout=30, autocommit=True,
                         prepare_threshold=None) as conn:
        if args.drop_first:
            conn.execute(f"DROP TABLE IF EXISTS {args.table}")
            print("dropped existing table.")
        conn.execute(ddl(args.table))
        rows = []
        for pos, (ch, vec) in enumerate(zip(chunks, vectors)):
            m = ch["metadata"]
            rows.append((
                ch["id"], pos,
                "[" + ",".join(f"{x:.6f}" for x in vec) + "]",
                ch["text"], m.get("source", ""), m.get("chunk_index"),
                m.get("clause"), m.get("heading"),
                m.get("heading_path"), m.get("standard_no"), m.get("year"),
                m.get("part"), m.get("page_start"), m.get("page_end"),
                m.get("char_start"), m.get("char_end"),
                bool(m.get("low_confidence")), bool(m.get("tail_truncated")),
                bool(m.get("table_repaired"))))
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}"
                               for c in UPDATABLE.split(", "))
        placeholders = ("%s, %s, %s::vector, " + "%s, " * 15 + "%s").rstrip(", ")
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {args.table} ({COLS}) VALUES ({placeholders}) "
                f"ON CONFLICT (chunk_id) DO UPDATE SET {set_clause}",
                rows)
        n = conn.execute(f"SELECT count(*) FROM {args.table}").fetchone()[0]
        assert n == len(chunks), (n, len(chunks))
        # spot vector equality: first + last row round-trip
        for cid in (chunks[0]["id"], chunks[-1]["id"]):
            db = conn.execute(
                f"SELECT embedding::text FROM {args.table} "
                f"WHERE chunk_id = %s", (cid,)).fetchone()[0]
            local = vectors[[c["id"] for c in chunks].index(cid)]
            back = np.array([float(x) for x in db.strip("[]").split(",")],
                            dtype=np.float32)
            assert np.allclose(local, back, atol=1e-5), cid
        if args.hnsw:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {args.table}_emb_hnsw ON "
                f"{args.table} USING hnsw (embedding vector_cosine_ops)")
            print("HNSW index built.")
    print(f"migration OK: {n} rows verified (count + spot vectors).")


if __name__ == "__main__":
    main()
