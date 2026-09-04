"""Groq vs Gemini 3.5 Flash-Lite latency comparison (5 queries).

Latency experiment ONLY: identical pipeline (retrieval, reranking,
context, prompts, verification, refusal, retries) with only the LLM
provider/model swapped. Does not touch the 30-query benchmark
(``run_queries.py``) or its scoring.

Usage (from the repository root)::

    python evaluation/compare_llm_latency.py --provider both
    python evaluation/compare_llm_latency.py --provider groq
    python evaluation/compare_llm_latency.py --provider gemini --runs 1

Requires ``GROQ_API_KEY`` for Groq rows and ``GEMINI_API_KEY`` for
Gemini rows (see ``.env.example``). Missing keys fail fast with a
clear message; transport/quota failures are reported per row, never
retried by this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.generator import answer  # noqa: E402
from app.generator.context_builder import load_chunk_index  # noqa: E402
from app.generator.llm_client import build_provider  # noqa: E402
from app.generator.telemetry import Telemetry  # noqa: E402

# Five stable benchmark queries covering distinct pipeline shapes.
# IDs double as labels; texts load from evaluation/test_queries.json so
# no new benchmark semantics are introduced.
QUERY_IDS = ("S001", "C002", "N001", "M001", "N003")
QUERY_ROLES = {
    "S001": "simple supported factual / normal path",
    "C002": "clause-specific query",
    "N001": "numerical / exact-value query",
    "M001": "larger multi-clause query",
    "N003": "recovery-path query (typically triggers widening)",
}


def load_benchmark_queries() -> dict[str, str]:
    """Map the five comparison IDs to their benchmark texts."""
    path = REPO_ROOT / "evaluation" / "test_queries.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {}
    for items in data.values():
        for entry in items:
            by_id[entry["id"]] = entry["query"]
    missing = [qid for qid in QUERY_IDS if qid not in by_id]
    if missing:
        raise ValueError(f"benchmark queries missing from test set: {missing}")
    return {qid: by_id[qid] for qid in QUERY_IDS}


def run_once(query_id: str, query: str, provider, chunk_index, top_k: int) -> dict:
    """One measured answer() call; failures become error rows."""
    telemetry = Telemetry()
    started = time.perf_counter()
    try:
        result = answer(
            query,
            top_k=top_k,
            chunk_index=chunk_index,
            provider=provider,
            telemetry=telemetry,
        )
    except Exception as exc:  # noqa: BLE001 - reported per row, run continues
        return {
            "id": query_id,
            "provider": getattr(provider, "name", "?"),
            "model": getattr(provider, "default_model", "?"),
            "error": f"{type(exc).__name__}: {exc}",
            "total_ms": (time.perf_counter() - started) * 1000.0,
        }
    stages = dict(telemetry.stages)
    gen_ms = sum(v for k, v in stages.items() if k.startswith("gen_"))
    meta = result.retrieval_meta
    return {
        "id": query_id,
        "provider": telemetry.llm_provider or getattr(provider, "name", "?"),
        "model": telemetry.llm_model or getattr(provider, "default_model", "?"),
        "total_ms": telemetry.latency_ms,
        "retrieval_ms": stages.get("retrieval_ms", 0.0),
        "generation_ms": gen_ms,
        "verify_ms": stages.get("verify_ms", 0.0),
        "attempts": telemetry.llm_generation_attempts,
        "api_calls": telemetry.llm_api_calls,
        "correction": telemetry.correction_retry,
        "widen": telemetry.widen_retry,
        "expansion": telemetry.retrieval_expansion,
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "top_reranker_score": meta.top_reranker_score if meta else None,
        "answer_chars": len(result.answer or ""),
        "error": None,
    }


def print_table(rows: list[dict]) -> None:
    print("Query   Provider       Model                  Total(s)   Gen(s)   Attempts   Refused")
    print("-" * 95)
    for row in rows:
        if row["error"] is not None:
            print(f"{row['id']:<8}{row['provider']:<15}{row['model']:<23}ERROR: {row['error'][:60]}")
            continue
        print(
            f"{row['id']:<8}{row['provider']:<15}{row['model']:<23}"
            f"{row['total_ms'] / 1000.0:<11.1f}{row['generation_ms'] / 1000.0:<9.1f}"
            f"{row['attempts']:<11}{row['refused']}"
        )


def print_aggregates(rows: list[dict], provider: str, label: str) -> dict:
    ok = [r for r in rows if r["provider"] == provider and r["error"] is None and not r["refused"]]
    print(f"\n{label}")
    if not ok:
        print("  no successful answered queries")
        return {}
    totals = [r["total_ms"] for r in ok]
    gens = [r["generation_ms"] for r in ok]
    stats = {
        "n": len(ok),
        "avg_total": sum(totals) / len(totals),
        "median_total": statistics.median(totals),
        "min_total": min(totals),
        "max_total": max(totals),
        "avg_gen": sum(gens) / len(gens),
        "total_calls": sum(r["api_calls"] for r in ok),
    }
    print(f"  Answered queries: {stats['n']}")
    print(f"  Average total latency: {stats['avg_total'] / 1000.0:.1f}s")
    print(f"  Median total latency: {stats['median_total'] / 1000.0:.1f}s")
    print(f"  Min: {stats['min_total'] / 1000.0:.1f}s")
    print(f"  Max: {stats['max_total'] / 1000.0:.1f}s")
    print(f"  Average generation latency: {stats['avg_gen'] / 1000.0:.1f}s")
    print(f"  Total logical LLM calls: {stats['total_calls']}")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Groq vs Gemini latency comparison (5 queries)")
    parser.add_argument("--provider", default="both", choices=("both", "groq", "gemini"))
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per query/provider")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunks-dir", default="data/chunks")
    args = parser.parse_args(argv)

    if args.runs < 1:
        print("error: --runs must be >= 1", file=sys.stderr)
        return 2

    names = ("groq", "gemini") if args.provider == "both" else (args.provider,)
    providers = {}
    for name in names:
        try:
            providers[name] = build_provider(name)
        except RuntimeError as exc:
            print(f"error: cannot build {name} provider: {exc}", file=sys.stderr)
            print("hint: set GROQ_API_KEY / GEMINI_API_KEY (see .env.example)", file=sys.stderr)
            return 2

    queries = load_benchmark_queries()
    print("Selected queries:")
    for qid, text in queries.items():
        print(f"  {qid} ({QUERY_ROLES[qid]}): {text}")

    chunk_index = load_chunk_index(Path(args.chunks_dir))

    # Warmup (not measured): one retrieval-only call warms BGE-M3 +
    # CrossEncoder without spending LLM quota.
    from retrieval import retrieve as retrieve_fn

    retrieve_fn("Bureau of Indian Standards specification warmup")
    print("Retrieval warmup done (discarded).")

    rows: list[dict] = []
    for run in range(args.runs):
        for name in names:
            for qid, text in queries.items():
                rows.append(run_once(qid, text, providers[name], chunk_index, args.top_k))

    print()
    print_table(rows)
    groq_stats = print_aggregates(rows, "groq", "Groq") if "groq" in providers else {}
    gemini_stats = print_aggregates(rows, "gemini", "Gemini 3.5 Flash-Lite") if "gemini" in providers else {}

    problems = [r for r in rows if r["error"] is not None or r["refused"]]
    if problems:
        print("\nRefused/failed rows (excluded from speedup):")
        for row in problems:
            print(f"  {row['id']} {row['provider']}: "
                  f"{row['error'] or ('refused: ' + str(row['refusal_reason']))}")

    if groq_stats and gemini_stats:
        mutual = set(
            r["id"] for r in rows
            if r["provider"] == "groq" and r["error"] is None and not r["refused"]
        ) & set(
            r["id"] for r in rows
            if r["provider"] == "gemini" and r["error"] is None and not r["refused"]
        )
        if mutual:
            g = [r["total_ms"] for r in rows if r["provider"] == "groq" and r["id"] in mutual]
            m = [r["total_ms"] for r in rows if r["provider"] == "gemini" and r["id"] in mutual]
            avg_g, avg_m = sum(g) / len(g), sum(m) / len(m)
            print(f"\nGemini speedup vs Groq (over {len(mutual)} mutually answered queries):")
            print(f"  Average latency reduction: {(avg_g - avg_m) / 1000.0:.1f}s")
            print(f"  Percentage latency reduction: {100.0 * (avg_g - avg_m) / avg_g:.1f}%")
        else:
            print("\nNo mutually answered queries; speedup not computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
