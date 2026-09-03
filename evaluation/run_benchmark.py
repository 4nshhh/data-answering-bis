"""Phase 8: end-to-end benchmark harness.

Runs ``run_query`` over a plain-text query file (one query per line,
``#`` comments and blank lines ignored) and writes per-query JSON
records plus a summary to stdout.

What this measures (honest scope): end-to-end latency, refusal rate
and reasons, citation presence/verification rates, and mask usage.
What it does NOT measure: gold-label retrieval metrics (Doc@1,
Clause@1, MRR) — the 87-query gold answer keys live in Repo 2's
``scripts/retrieval_eval.py`` and are not vendored in this
repository. When those keys are available, extend ``score_row()``
with phrase-hit verification against them.

Usage (from the repository root; needs GROQ_API_KEY for answerable
queries, models load lazily on first query)::

    python evaluation/run_benchmark.py --queries tests/sample_queries.txt --limit 5
    python evaluation/run_benchmark.py --queries queries.txt --out results.jsonl --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.generator.context_builder import load_chunk_index  # noqa: E402
from app.generator.llm_client import GroqProvider  # noqa: E402
from app.generator.pipeline import run_query  # noqa: E402


def load_queries(path: Path) -> list[str]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            queries.append(line)
    return queries


def score_row(result) -> dict:
    """Pipeline-stat scoring for one result (no gold labels required)."""
    citations = result.citations
    return {
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "n_citations": len(citations),
        "n_verified": sum(1 for c in citations if c.verified),
        "all_verified": bool(citations) and all(c.verified for c in citations),
        "is_mask_restricted": result.retrieval_meta.is_mask_restricted if result.retrieval_meta else False,
        "execution_time_ms": result.retrieval_meta.execution_time_ms if result.retrieval_meta else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end RAG pipeline benchmark")
    parser.add_argument("--queries", required=True, help="Text file, one query per line")
    parser.add_argument("--out", default=None, help="JSONL output path (default: stdout records only)")
    parser.add_argument("--limit", type=int, default=None, help="Max queries to run")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--expand-neighbors", action="store_true")
    parser.add_argument("--threshold", type=float, default=-2.0)
    parser.add_argument("--chunks-dir", default="data/chunks")
    args = parser.parse_args(argv)

    queries = load_queries(Path(args.queries))
    if args.limit is not None:
        queries = queries[: args.limit]
    if not queries:
        print("no queries found", file=sys.stderr)
        return 2

    chunk_index = load_chunk_index(Path(args.chunks_dir))
    provider = GroqProvider()

    out_fh = open(args.out, "w", encoding="utf-8") if args.out else None
    rows = []
    try:
        for i, query in enumerate(queries):
            try:
                result = run_query(
                    query,
                    top_k=args.top_k,
                    expand_neighbors=args.expand_neighbors,
                    threshold=args.threshold,
                    chunk_index=chunk_index,
                    provider=provider,
                )
                record = {"i": i, "query": query, "answer": result.answer,
                          "citations": [asdict(c) for c in result.citations],
                          **score_row(result), "error": None}
            except Exception as exc:  # noqa: BLE001 - recorded per row, run continues
                record = {"i": i, "query": query, "answer": None, "citations": [],
                          "refused": None, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(record)
            if out_fh:
                out_fh.write(json.dumps(record) + "\n")
    finally:
        if out_fh:
            out_fh.close()

    n = len(rows)
    answered = [r for r in rows if r["error"] is None and not r["refused"]]
    summary = {
        "n_queries": n,
        "n_errors": sum(1 for r in rows if r["error"] is not None),
        "refusal_rate": sum(1 for r in rows if r["refused"]) / n,
        "mean_latency_ms": (sum(r["execution_time_ms"] for r in answered) / len(answered)) if answered else None,
        "citation_rate": (sum(1 for r in answered if r["n_citations"] > 0) / len(answered)) if answered else None,
        "all_verified_rate": (sum(1 for r in answered if r["all_verified"]) / len(answered)) if answered else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
