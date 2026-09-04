"""Reusable end-to-end evaluation suite for the BIS Answering/RAG API.

Calls the existing ``POST /api/v1/query`` endpoint once per query in
``evaluation/test_queries.json`` and writes observable-behavior verdicts to
``evaluation/results/latest_results.json``.

Evaluation principle (no naive substring matching): supported questions are
judged only on observable API behavior (HTTP success, refusal flags,
non-empty answers, present + verified citations, retrieval metadata, no
provider/server errors). Refusal questions are judged on refusal signals.
Review questions are always surfaced for manual inspection. Product-to-standard
questions never check for literal standard numbers in the answer text; passing
grounding checks are still flagged for manual semantic review.

Usage (API server must already be running; this runner needs no Groq key)::

    python evaluation/run_queries.py
    python evaluation/run_queries.py --url http://127.0.0.1:8000/api/v1/query --top-k 3
    python evaluation/run_queries.py --category supported_clause --limit 5
    python evaluation/run_queries.py --help
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "evaluation" / "test_queries.json"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "results" / "latest_results.json"
DEFAULT_URL = "http://127.0.0.1:8000/api/v1/query"

VALID_CATEGORIES = (
    "supported_exact",
    "supported_clause",
    "supported_numerical",
    "supported_multi_clause",
    "product_to_standard",
    "product_requirement",
    "out_of_corpus",
    "general_bis",
)
VALID_EXPECTED = ("answer", "refusal", "review")

# Categories where a passing grounding verdict still needs a human to confirm
# the semantic mapping (e.g. product description -> right standard).
SEMANTIC_REVIEW_CATEGORIES = ("product_to_standard", "product_requirement")

__all__ = [
    "VALID_CATEGORIES",
    "VALID_EXPECTED",
    "load_dataset",
    "validate_dataset",
    "flatten_dataset",
    "classify_result",
    "post_query",
    "run_suite",
]


def load_dataset(path: Path | str = DEFAULT_DATASET) -> dict:
    """Load and JSON-parse the query dataset."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"dataset root must be a JSON object, got {type(data).__name__}")
    return data


def validate_dataset(data: dict) -> list[dict]:
    """Validate structure and return flattened records.

    Raises:
        ValueError: unknown category, missing/invalid fields, duplicate IDs.
    """
    if not isinstance(data, dict):
        raise ValueError("dataset must be a JSON object of category -> list")
    seen_ids: set[str] = set()
    flat: list[dict] = []
    for category, items in data.items():
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"unknown category {category!r}; valid: {list(VALID_CATEGORIES)}"
            )
        if not isinstance(items, list) or not items:
            raise ValueError(f"category {category!r} must be a non-empty list")
        for pos, entry in enumerate(items):
            if not isinstance(entry, dict):
                raise ValueError(f"{category}[{pos}] must be an object")
            for field in ("id", "query", "expected"):
                if field not in entry:
                    raise ValueError(f"{category}[{pos}] missing required field {field!r}")
            qid, query, expected = entry["id"], entry["query"], entry["expected"]
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError(f"{category}[{pos}].id must be a non-blank string")
            if qid in seen_ids:
                raise ValueError(f"duplicate query id {qid!r}")
            seen_ids.add(qid)
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"query {qid!r} must be a non-blank string")
            if expected not in VALID_EXPECTED:
                raise ValueError(
                    f"query {qid!r} has invalid expected {expected!r}; "
                    f"valid: {list(VALID_EXPECTED)}"
                )
            flat.append(
                {
                    "id": qid,
                    "category": category,
                    "query": query.strip(),
                    "expected": expected,
                }
            )
    if not flat:
        raise ValueError("dataset contains no queries")
    return flat


def flatten_dataset(data: dict) -> list[dict]:
    """Validate and flatten the dataset, preserving category + ID order."""
    return validate_dataset(data)


def _citation_stats(citations: object) -> tuple[int, int]:
    """Return (total, verified) citation counts for list- or dict-shaped payloads."""
    if isinstance(citations, dict):
        items = list(citations.values()) if citations else []
        # API returns a list; a dict shape is tolerated for forward-compat.
        total = len(items)
        verified = sum(1 for c in items if isinstance(c, dict) and c.get("verified") is True)
        return total, verified
    if isinstance(citations, list):
        total = len(citations)
        verified = sum(1 for c in citations if isinstance(c, dict) and c.get("verified") is True)
        return total, verified
    return 0, 0


def classify_result(
    *,
    expected: str,
    category: str,
    http_status: int | None,
    success: bool,
    answer: str,
    refused: bool | None,
    refusal_reason: object,
    citations: object,
    retrieval_meta: object,
    error: str | None,
) -> tuple[str, str]:
    """Classify one query outcome without inspecting answer wording.

    Returns:
        (evaluation_status, evaluation_notes) where status is one of
        ``PASS`` / ``FAIL`` / ``REVIEW`` / ``ERROR``.
    """
    n_cit, n_verified = _citation_stats(citations)
    if error is not None or not success:
        return "ERROR", f"transport/API failure prevented evaluation: {error or 'unsuccessful call'}"
    if expected == "review":
        return "REVIEW", "general question recorded for manual review; no automatic verdict applied"
    if expected == "refusal":
        problems = []
        if refused is not True:
            problems.append("expected refusal but refused != true")
        if isinstance(citations, list) and len(citations) != 0:
            problems.append("expected empty citations on refusal")
        if isinstance(citations, dict) and len(citations) != 0:
            problems.append("expected empty citations on refusal")
        if not refusal_reason:
            problems.append("expected a refusal_reason")
        if problems:
            return "FAIL", "; ".join(problems)
        return "PASS", "refusal with empty citations and refusal_reason present"
    # expected == "answer": observable grounding properties only.
    problems = []
    if refused is not False and refused is not True:
        problems.append("refused flag missing")
    elif refused:
        problems.append(f"expected an answer but got refusal ({refusal_reason})")
    if not isinstance(answer, str) or not answer.strip():
        problems.append("answer is empty")
    if n_cit == 0:
        problems.append("no citations present")
    elif n_verified != n_cit:
        problems.append(f"only {n_verified}/{n_cit} citations verified")
    if not isinstance(retrieval_meta, dict) or not retrieval_meta:
        problems.append("retrieval_meta missing")
    if problems:
        return "FAIL", "; ".join(problems)
    if category in SEMANTIC_REVIEW_CATEGORIES:
        return (
            "PASS",
            f"grounded answer with {n_verified}/{n_cit} verified citations; "
            "semantic standard-mapping still requires manual review",
        )
    return "PASS", f"grounded answer with {n_verified}/{n_cit} verified citations"


def post_query(
    url: str,
    query: str,
    top_k: int,
    timeout_s: float,
) -> tuple[int | None, dict | None, str | None]:
    """POST one query to the API. Returns (http_status, body, error)."""
    payload = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.getcode()
            try:
                body = json.loads(response.read().decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                return status, None, f"malformed API response: {exc}"
            return status, body, None
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - best-effort error body
            detail = ""
        return exc.code, None, f"HTTP {exc.code}: {detail[:500]}"
    except urllib.error.URLError as exc:
        return None, None, f"connection failure: {exc.reason}"
    except TimeoutError as exc:
        return None, None, f"timeout after {timeout_s}s: {exc}"
    except Exception as exc:  # noqa: BLE001 - suite must continue per query
        return None, None, f"{type(exc).__name__}: {exc}"


def run_suite(
    records: list[dict],
    *,
    url: str,
    top_k: int,
    timeout_s: float,
) -> list[dict]:
    """Execute all queries against the live API; never raises per-query errors."""
    results: list[dict] = []
    for record in records:
        started = time.perf_counter()
        http_status, body, error = post_query(url, record["query"], top_k, timeout_s)
        latency_ms = (time.perf_counter() - started) * 1000.0
        timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
        if body is None:
            body = {}
        answer = body.get("answer", "") if isinstance(body, dict) else ""
        refused = body.get("refused") if isinstance(body, dict) else None
        refusal_reason = body.get("refusal_reason") if isinstance(body, dict) else None
        citations = body.get("citations", []) if isinstance(body, dict) else []
        retrieval_meta = body.get("retrieval_meta", {}) if isinstance(body, dict) else {}
        telemetry = body.get("telemetry", {}) if isinstance(body, dict) else {}
        if not isinstance(telemetry, dict):
            telemetry = {}
        success = error is None and http_status == 200 and isinstance(body, dict)
        status, notes = classify_result(
            expected=record["expected"],
            category=record["category"],
            http_status=http_status,
            success=success,
            answer=answer if isinstance(answer, str) else "",
            refused=refused,
            refusal_reason=refusal_reason,
            citations=citations,
            retrieval_meta=retrieval_meta,
            error=error,
        )
        results.append(
            {
                "id": record["id"],
                "category": record["category"],
                "query": record["query"],
                "expected": record["expected"],
                "timestamp": timestamp,
                "http_status": http_status,
                "success": success,
                "answer": answer if isinstance(answer, str) else "",
                "refused": refused,
                "refusal_reason": refusal_reason,
                "citations": citations,
                "retrieval_meta": retrieval_meta,
                "telemetry": telemetry,
                "latency_ms": round(latency_ms, 1),
                "evaluation_status": status,
                "evaluation_notes": notes,
                "error": error,
                "api_response": body,
            }
        )
    return results


CATEGORY_LABELS = {
    "supported_exact": "Supported Exact",
    "supported_clause": "Supported Clause",
    "supported_numerical": "Supported Numerical",
    "supported_multi_clause": "Multi-Clause",
    "product_to_standard": "Product -> Standard",
    "product_requirement": "Product Requirement",
    "out_of_corpus": "Out-of-Corpus",
    "general_bis": "General BIS",
}


def print_report(results: list[dict]) -> None:
    """Print the console summary plus per-FAIL/REVIEW/ERROR diagnostics."""
    print("=" * 50)
    print("BIS ANSWERING EVALUATION")
    print("=" * 50)
    order = [c for c in VALID_CATEGORIES if any(r["category"] == c for r in results)]
    for category in order:
        rows = [r for r in results if r["category"] == category]
        passed = sum(1 for r in rows if r["evaluation_status"] == "PASS")
        print(f"{CATEGORY_LABELS[category]:<22} {passed}/{len(rows)}   PASS")
    print("-" * 50)
    total = len(results)
    n_pass = sum(1 for r in results if r["evaluation_status"] == "PASS")
    n_fail = sum(1 for r in results if r["evaluation_status"] == "FAIL")
    n_review = sum(1 for r in results if r["evaluation_status"] == "REVIEW")
    n_error = sum(1 for r in results if r["evaluation_status"] == "ERROR")
    print(f"Total: {total}")
    print(f"Passed: {n_pass}")
    print(f"Failed: {n_fail}")
    print(f"Review: {n_review}")
    if n_error:
        print(f"Errors: {n_error}")
    latencies = [r["latency_ms"] for r in results if isinstance(r["latency_ms"], (int, float))]
    if latencies:
        print(
            f"Latency ms: avg {sum(latencies) / len(latencies):.1f} | "
            f"min {min(latencies):.1f} | max {max(latencies):.1f}"
        )
    # Telemetry aggregate: sums of per-query measured values only.
    teles = [r.get("telemetry") for r in results]
    teles = [t for t in teles if isinstance(t, dict)]
    if teles:
        def _api_calls(t: dict) -> int:
            # Prefer the provider-neutral counter; fall back to the legacy
            # alias for servers predating it.
            for key in ("llm_api_calls", "groq_api_calls"):
                value = t.get(key)
                if isinstance(value, (int, float)):
                    return int(value or 0)
            return 0

        n_groq = sum(_api_calls(t) for t in teles)
        n_gen = sum(int(t.get("llm_generation_attempts", 0) or 0) for t in teles)
        n_corr = sum(1 for t in teles if t.get("correction_retry") is True)
        n_wide = sum(1 for t in teles if t.get("widen_retry") is True)
        n_exp = sum(1 for t in teles if t.get("retrieval_expansion") is True)
        print(f"LLM API calls: {n_groq}")
        print(f"Logical generation attempts: {n_gen}")
        print(f"Correction retries: {n_corr} | Widen generations: {n_wide} | Retrieval expansions: {n_exp}")
        stage_totals: dict[str, float] = {}
        stage_counts: dict[str, int] = {}
        for t in teles:
            stages = t.get("stages")
            if isinstance(stages, dict):
                for name, ms in stages.items():
                    if isinstance(ms, (int, float)):
                        stage_totals[name] = stage_totals.get(name, 0.0) + float(ms)
                        stage_counts[name] = stage_counts.get(name, 0) + 1
        if stage_totals:
            breakdown = " | ".join(
                f"{name}: avg {stage_totals[name] / stage_counts[name]:.1f}ms (n={stage_counts[name]})"
                for name in sorted(stage_totals)
            )
            print(f"Stage breakdown: {breakdown}")
        prompt_sizes = [t.get("prompt_chars") for t in teles if isinstance(t.get("prompt_chars"), (int, float))]
        if prompt_sizes:
            print(f"Prompt chars: avg {sum(prompt_sizes) / len(prompt_sizes):.0f} | max {max(prompt_sizes):.0f}")
    print("=" * 50)
    for r in results:
        if r["evaluation_status"] == "PASS":
            continue
        meta = r["retrieval_meta"] if isinstance(r["retrieval_meta"], dict) else {}
        citations = r["citations"]
        if isinstance(citations, list):
            n_cit = len(citations)
            n_ver = sum(1 for c in citations if isinstance(c, dict) and c.get("verified") is True)
        elif isinstance(citations, dict):
            n_cit = len(citations)
            n_ver = sum(1 for c in citations.values() if isinstance(c, dict) and c.get("verified") is True)
        else:
            n_cit, n_ver = 0, 0
        print(f"[{r['evaluation_status']}] {r['id']} ({r['category']})")
        print(f"  query: {r['query']}")
        print(f"  refused: {r['refused']} | reason: {r['refusal_reason']}")
        print(
            f"  top_reranker_score: {meta.get('top_reranker_score')} | "
            f"filtered_standard: {meta.get('filtered_standard')} | "
            f"is_mask_restricted: {meta.get('is_mask_restricted')} | "
            f"candidates_retrieved: {meta.get('candidates_retrieved')}"
        )
        print(f"  citations: {n_cit} | verified: {n_ver} | latency_ms: {r['latency_ms']}")
        print(f"  notes: {r['evaluation_notes']}")
        if r["error"]:
            print(f"  error: {r['error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BIS Answering/RAG end-to-end evaluation")
    parser.add_argument("--url", default=DEFAULT_URL, help="Full API endpoint URL")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to test_queries.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Results JSON output path")
    parser.add_argument("--top-k", type=int, default=3, help="Context blocks per query (1-5)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-query HTTP timeout in seconds")
    parser.add_argument("--category", default=None, choices=list(VALID_CATEGORIES),
                        help="Run a single category only")
    parser.add_argument("--limit", type=int, default=None, help="Max queries to run")
    args = parser.parse_args(argv)

    if not 1 <= args.top_k <= 5:
        print(f"error: --top-k must be 1-5, got {args.top_k}", file=sys.stderr)
        return 2

    try:
        data = load_dataset(args.dataset)
        records = flatten_dataset(data)
    except (OSError, ValueError) as exc:
        print(f"error: invalid dataset: {exc}", file=sys.stderr)
        return 2

    if args.category is not None:
        records = [r for r in records if r["category"] == args.category]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print("error: no queries selected", file=sys.stderr)
        return 2

    print(f"Running {len(records)} queries against {args.url} ...")
    results = run_suite(records, url=args.url, top_k=args.top_k, timeout_s=args.timeout)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "endpoint": args.url,
                "top_k": args.top_k,
                "n_queries": len(results),
                "results": results,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Results written to {out_path}")
    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
