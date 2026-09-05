"""PgVectorStore reference implementation of the ChunkStore protocol.

Reads connection from `DATABASE_URL` (direct connection). Sync psycopg v3.
Table layout (see `indexing/migrate_from_artifacts.py`):

    chunks(chunk_id TEXT PK, embedding vector(1024),
           text TEXT, source TEXT, chunk_index INT, clause TEXT,
           heading TEXT, heading_path TEXT[], standard_no TEXT,
           year TEXT, part TEXT, page_start INT, page_end INT,
           char_start INT, char_end INT, low_confidence BOOL,
           tail_truncated BOOL, table_repaired BOOL)

Similarity is cosine distance (`<=>`); stored vectors are unit-norm, so
score = 1 - dist matches the dot-product scoring of `LocalNpyStore`.
Row indices are positional over `ORDER BY row_pos` (NOT `ORDER BY
chunk_id`: Postgres collations sort ``_`` before digits, unlike Python
codepoint order); the migration assigns ``row_pos`` in sorted-glob chunk
order, so local row i == remote row i.
"""

from __future__ import annotations

import os

import numpy as np

from .store import ChunkStore
from .types import ChunkRecord

EMBEDDING_DIM = 1024


def _dsn() -> str:
    from .store import read_env_file

    dsn = os.environ.get("DATABASE_URL", "") or read_env_file("DATABASE_URL") or ""
    if not dsn:
        raise SystemExit("DATABASE_URL not set (env or .env); refusing to guess.")
    return dsn


def _vec_literal(query_vector: np.ndarray) -> str:
    v = np.asarray(query_vector, dtype=np.float32).tolist()
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


class PgVectorStore:
    """ChunkStore over Supabase/Postgres + pgvector. Sync, short-lived conns."""

    def __init__(self, dsn: str | None = None, table: str = "chunks") -> None:
        from psycopg.rows import dict_row

        import psycopg

        self._psycopg = psycopg
        self._dict_row = dict_row
        self._dsn = dsn or _dsn()
        self._table = table
        with self._connect() as conn:
            n = conn.execute(f"SELECT count(*) AS n FROM {self._table}").fetchone()["n"]
            # NOTE: ORDER BY row_pos, NOT chunk_id — Postgres collations
            # sort '_' before digits ('1_0000' < '1005_0000'), while Python
            # codepoint order puts '1005_0000' first. Positional identity
            # (local row i == remote row i) holds only via row_pos.
            self._ids = [r["chunk_id"] for r in conn.execute(
                f"SELECT chunk_id FROM {self._table} ORDER BY row_pos")]
            from .store import query_side_candidate_mask
            from types import SimpleNamespace

            meta = conn.execute(
                f"SELECT source, standard_no FROM {self._table} "
                f"ORDER BY row_pos").fetchall()
            self._shims = [SimpleNamespace(source=r["source"],
                                           standard_no=r["standard_no"])
                           for r in meta]
            self._mask_fn = query_side_candidate_mask
        if n == 0:
            raise ValueError(f"Table '{self._table}' is empty; run migration first.")
        if n != len(self._ids):
            raise ValueError("Row-count/ID-list mismatch; aborting.")
        self._n = n

    def _connect(self):
        # prepare_threshold=None: Supavisor/pgbouncer transaction pooling
        # does not support server-side prepared statements.
        return self._psycopg.connect(self._dsn, connect_timeout=20,
                                     row_factory=self._dict_row,
                                     prepare_threshold=None)

    @staticmethod
    def _row_to_record(row: dict) -> ChunkRecord:
        return ChunkRecord(
            id=row["chunk_id"], text=row["text"], source=row["source"],
            clause=row["clause"], heading=row["heading"],
            standard_no=row["standard_no"], page_start=row["page_start"],
            page_end=row["page_end"],
            low_confidence=bool(row["low_confidence"]))

    @staticmethod
    def _row_to_enriched(row: dict) -> str:
        parts = []
        if row["standard_no"]:
            parts.append(f"Standard: {row['standard_no']}")
        if row["clause"]:
            parts.append(f"Clause: {row['clause']}")
        hp = row["heading_path"] or []
        if hp:
            parts.append("Heading: " + " > ".join(hp))
        elif row["heading"]:
            parts.append(f"Heading: {row['heading']}")
        prefix = "\n".join(parts)
        return f"{prefix}\n\n{row['text']}" if prefix else row["text"]

    def search(self, query_vector: np.ndarray, k: int,
               mask: set[int] | None) -> list[tuple[int, float]]:
        lit = _vec_literal(query_vector)
        pos = {cid: i for i, cid in enumerate(self._ids)}
        with self._connect() as conn:
            if mask is None:
                rows = conn.execute(
                    f"SELECT chunk_id, embedding <=> %s::vector AS dist "
                    f"FROM {self._table} ORDER BY dist LIMIT %s",
                    (lit, k)).fetchall()
            else:
                allowed = [self._ids[i] for i in sorted(mask)
                           if 0 <= i < len(self._ids)]
                if not allowed:
                    return []
                rows = conn.execute(
                    f"SELECT chunk_id, embedding <=> %s::vector AS dist "
                    f"FROM {self._table} WHERE chunk_id = ANY(%s) "
                    f"ORDER BY dist LIMIT %s",
                    (lit, allowed, k)).fetchall()
        return [(pos[r["chunk_id"]], float(1.0 - r["dist"])) for r in rows]

    def fetch(self, indices: list[int]) -> list[ChunkRecord]:
        want = [self._ids[i] for i in indices]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {self._table} WHERE chunk_id = ANY(%s)",
                (want,)).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        return [self._row_to_record(by_id[c]) for c in want]

    def enriched_text(self, index: int) -> str:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {self._table} WHERE chunk_id = %s",
                (self._ids[index],)).fetchone()
        if row is None:
            raise KeyError(f"No row at index {index}")
        return self._row_to_enriched(row)

    def enriched_texts(self, indices: list[int]) -> list[str]:
        """Batch reranker texts in one round trip (F5: avoids one
        connection per candidate on every query). Order follows
        ``indices``; output is identical to looping :meth:`enriched_text`."""
        if not indices:
            return []
        want = [self._ids[i] for i in indices]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {self._table} WHERE chunk_id = ANY(%s)",
                (want,)).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        try:
            ordered = [by_id[c] for c in want]
        except KeyError as exc:
            raise KeyError(f"No row for chunk_id {exc}") from exc
        return [self._row_to_enriched(r) for r in ordered]

    def mask_for(self, query_text: str) -> set[int] | None:
        """Same IS-number candidate mask as the local path, over row metadata."""
        return self._mask_fn(query_text, self._shims)

    def __len__(self) -> int:
        return self._n
