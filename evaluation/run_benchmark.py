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
import datetime as _dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from answering.generator.adapters import to_ask_response, to_match_response  # noqa: E402
from answering.generator.context_builder import load_chunk_index  # noqa: E402
from answering.generator.llm_client import GroqProvider  # noqa: E402
from answering.generator.pipeline import run_query  # noqa: E402
from answering.generator.prompts import validate_mode  # noqa: E402
from answering.generator.refusal import DEFAULT_THRESHOLD  # noqa: E402
from evaluation.progress import QueryProgress  # noqa: E402


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


def adapt_result(query: str, mode: str, result, chunk_index=None) -> dict:
    """Convert one :class:`QueryResult` with the exact backend adapter.

    ``mode="ask"`` uses :func:`to_ask_response`, ``mode="product_match"``
    uses :func:`to_match_response`. The returned dict is the adapter's
    own output, unmodified — the backend response contract, verbatim.
    """
    validate_mode(mode)
    if mode == "product_match":
        return to_match_response(result, query)
    return to_ask_response(result, chunk_index=chunk_index)


def default_responses_path() -> Path:
    """Timestamped responses file; every run gets its own path."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "evaluation" / "results" / f"responses_{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end RAG pipeline benchmark")
    parser.add_argument("--queries", required=True, help="Text file, one query per line")
    parser.add_argument("--out", default=None, help="JSONL output path (default: stdout records only)")
    parser.add_argument("--limit", type=int, default=None, help="Max queries to run")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", default="ask", choices=("ask", "product_match"),
                        help="Answering mode for every query (selects the backend adapter)")
    parser.add_argument("--expand-neighbors", action="store_true")
    # Calibrated sigmoid-scale default (refusal.py): the old -2.0
    # logit-scale default could never fire, silently disabling refusal.
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--delay-secs", type=float, default=0.0,
                        help="Opt-in pause between queries (rate-limit courtesy); 0 disables")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the terminal progress display")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
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
    prog = QueryProgress(len(queries), enabled=False) if args.no_progress \
        else QueryProgress(len(queries))
    rows = []
    saved = []
    try:
        for pos, query in enumerate(queries, 1):
            i = pos - 1
            prog.waiting(pos, "Waiting for LLM response...")
            try:
                result = run_query(
                    query,
                    top_k=args.top_k,
                    expand_neighbors=args.expand_neighbors,
                    threshold=args.threshold,
                    chunk_index=chunk_index,
                    provider=provider,
                    mode=args.mode,
                )
                record = {"i": i, "query": query, "answer": result.answer,
                          "citations": [asdict(c) for c in result.citations],
                          **score_row(result), "error": None}
                saved.append({
                    "query": query,
                    "mode": args.mode,
                    "response": adapt_result(query, args.mode, result, chunk_index),
                    "evaluation": {
                        "refused": result.refused,
                        "refusal_reason": result.refusal_reason,
                        "n_citations": len(result.citations),
                        "n_verified": sum(1 for c in result.citations if c.verified),
                        "top_reranker_score": result.retrieval_meta.top_reranker_score
                        if result.retrieval_meta else None,
                        "latency_ms": record["execution_time_ms"],
                        "error": None,
                    },
                })
            except Exception as exc:  # noqa: BLE001 - recorded per row, run continues
                record = {"i": i, "query": query, "answer": None, "citations": [],
                          "refused": None, "error": f"{type(exc).__name__}: {exc}"}
                saved.append({
                    "query": query,
                    "mode": args.mode,
                    "response": None,
                    "evaluation": {"error": record["error"]},
                })
            rows.append(record)
            if out_fh:
                out_fh.write(json.dumps(record) + "\n")
            if record["error"] is not None:
                status, ok, latency = "ERROR", False, None
            elif record["refused"]:
                status, ok = "REFUSED", False
                latency = (record["execution_time_ms"] or 0.0) / 1000.0 \
                    if isinstance(record.get("execution_time_ms"), (int, float)) else None
            else:
                status, ok = "OK", True
                latency = (record["execution_time_ms"] or 0.0) / 1000.0 \
                    if isinstance(record.get("execution_time_ms"), (int, float)) else None
            short = query if len(query) <= 47 else query[:47] + "..."
            prog.finish(pos, status, latency, note=f"q{i} {short}", ok=ok)
            if args.delay_secs > 0 and pos < len(queries):
                prog.delay(args.delay_secs)
    finally:
        prog.close()
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
    responses_path = default_responses_path()
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with open(responses_path, "w", encoding="utf-8") as fh:
        json.dump(saved, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Responses written to {responses_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
