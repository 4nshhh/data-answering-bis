"""Offline tests for the reusable evaluation suite.

No server, Groq/OpenAI keys, GPU, or network required: dataset handling
and verdict classification are exercised with synthetic API payloads, and
HTTP failures are simulated by monkeypatching ``post_query``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.run_queries import (
    VALID_CATEGORIES,
    classify_result,
    flatten_dataset,
    load_dataset,
    post_query,
    run_suite,
    validate_dataset,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "test_queries.json"


def _grounded_body() -> dict:
    return {
        "query": "q",
        "answer": "Water pH not less than 6 [IS 456:2000, Clause 5.4, Page 15].",
        "citations": [
            {"standard_no": "IS 456", "year": "2000", "clause": "5.4",
             "page": 15, "chunk_id": "c1", "verified": True}
        ],
        "retrieval_meta": {"filtered_standard": "IS 456", "is_mask_restricted": True,
                           "candidates_retrieved": 10, "top_reranker_score": 0.99,
                           "execution_time_ms": 10.0},
        "refused": False,
        "refusal_reason": None,
    }


def test_dataset_loads():
    data = load_dataset(DATASET_PATH)
    assert isinstance(data, dict)
    assert set(data) == set(VALID_CATEGORIES)


def test_all_ids_unique_and_fields_present():
    records = flatten_dataset(load_dataset(DATASET_PATH))
    assert len(records) == 30
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)
    for record in records:
        assert record["id"] and record["category"] and record["query"] and record["expected"]
    # Permanent regression IDs remain stable.
    by_id = {r["id"]: r for r in records}
    for permanent in ("S001", "C001", "C002", "M001", "N001", "O001", "G001"):
        assert permanent in by_id, f"missing permanent case {permanent}"
    assert by_id["C002"]["expected"] == "answer"


def test_category_counts():
    records = flatten_dataset(load_dataset(DATASET_PATH))
    counts: dict[str, int] = {}
    for record in records:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    assert counts == {
        "supported_exact": 4,
        "supported_clause": 5,
        "supported_numerical": 5,
        "supported_multi_clause": 3,
        "product_to_standard": 5,
        "product_requirement": 3,
        "out_of_corpus": 3,
        "general_bis": 2,
    }


def test_flattening_preserves_category_and_id():
    records = flatten_dataset({"supported_exact": [{"id": "S001", "query": "q?", "expected": "answer"}]})
    assert records == [{"id": "S001", "category": "supported_exact",
                        "query": "q?", "expected": "answer"}]


def test_malformed_records_raise_clear_errors():
    with pytest.raises(ValueError, match="missing required field"):
        validate_dataset({"supported_exact": [{"id": "S001", "query": "q?"}]})
    with pytest.raises(ValueError, match="duplicate query id"):
        validate_dataset({"supported_exact": [
            {"id": "S001", "query": "a", "expected": "answer"},
            {"id": "S001", "query": "b", "expected": "answer"},
        ]})
    with pytest.raises(ValueError, match="unknown category"):
        validate_dataset({"nope": [{"id": "X1", "query": "q?", "expected": "answer"}]})
    with pytest.raises(ValueError, match="non-blank string"):
        validate_dataset({"supported_exact": [{"id": "S001", "query": "   ", "expected": "answer"}]})


def test_classify_answer_pass_and_fail_modes():
    body = _grounded_body()
    status, _ = classify_result(expected="answer", category="supported_exact",
                                http_status=200, success=True, answer=body["answer"],
                                refused=False, refusal_reason=None,
                                citations=body["citations"],
                                retrieval_meta=body["retrieval_meta"], error=None)
    assert status == "PASS"
    # Refused supported question -> FAIL (e.g. C002 citation_mismatch regression).
    status, notes = classify_result(expected="answer", category="supported_clause",
                                    http_status=200, success=True,
                                    answer="The provided Indian Standards documents do not contain "
                                            "sufficient technical information to answer this query.",
                                    refused=True, refusal_reason="citation_mismatch",
                                    citations=[], retrieval_meta=body["retrieval_meta"], error=None)
    assert status == "FAIL"
    assert "citation_mismatch" in notes
    # Unverified citations -> FAIL without reading answer text.
    bad_cit = [dict(body["citations"][0], verified=False)]
    status, _ = classify_result(expected="answer", category="supported_exact",
                                http_status=200, success=True, answer=body["answer"],
                                refused=False, refusal_reason=None,
                                citations=bad_cit,
                                retrieval_meta=body["retrieval_meta"], error=None)
    assert status == "FAIL"


def test_classify_refusal_and_review():
    status, _ = classify_result(expected="refusal", category="out_of_corpus",
                                http_status=200, success=True, answer="refusal text",
                                refused=True, refusal_reason="below_threshold",
                                citations=[], retrieval_meta={}, error=None)
    assert status == "PASS"
    status, _ = classify_result(expected="refusal", category="out_of_corpus",
                                http_status=200, success=True, answer="Paris.",
                                refused=False, refusal_reason=None,
                                citations=[], retrieval_meta={}, error=None)
    assert status == "FAIL"
    status, _ = classify_result(expected="review", category="general_bis",
                                http_status=200, success=True, answer="anything",
                                refused=False, refusal_reason=None,
                                citations=[], retrieval_meta={}, error=None)
    assert status == "REVIEW"


def test_http_failure_does_not_terminate_suite(monkeypatch):
    calls = {"n": 0}

    def flaky(url: str, query: str, top_k: int, timeout_s: float):
        calls["n"] += 1
        if calls["n"] == 1:
            return None, None, "connection failure: refused"
        return 200, _grounded_body(), None

    monkeypatch.setattr("evaluation.run_queries.post_query", flaky)
    records = [
        {"id": "S001", "category": "supported_exact", "query": "a", "expected": "answer"},
        {"id": "S002", "category": "supported_exact", "query": "b", "expected": "answer"},
    ]
    results = run_suite(records, url="http://x", top_k=3, timeout_s=5.0)
    assert [r["evaluation_status"] for r in results] == ["ERROR", "PASS"]
    assert results[0]["error"] is not None
    assert results[1]["error"] is None


def test_result_serialization_round_trip(tmp_path):
    records = [{"id": "O001", "category": "out_of_corpus", "query": "q?", "expected": "refusal"}]
    refused_body = {"query": "q?", "answer": "refusal text", "citations": [],
                    "retrieval_meta": {"candidates_retrieved": 10}, "refused": True,
                    "refusal_reason": "below_threshold"}
    import evaluation.run_queries as runner

    original = runner.post_query
    runner.post_query = lambda url, query, top_k, timeout_s: (200, refused_body, None)
    try:
        results = runner.run_suite(records, url="http://x", top_k=3, timeout_s=5.0)
    finally:
        runner.post_query = original
    out = tmp_path / "results.json"
    out.write_text(json.dumps({"results": results}), encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8"))["results"][0]
    for key in ("id", "category", "query", "timestamp", "http_status", "success",
                "answer", "refused", "refusal_reason", "citations", "retrieval_meta",
                "latency_ms", "evaluation_status", "evaluation_notes", "error", "api_response"):
        assert key in reloaded
    assert reloaded["evaluation_status"] == "PASS"


def test_post_query_signature_without_network():
    # post_query must accept these kwargs; no call is made here.
    import inspect

    params = inspect.signature(post_query).parameters
    assert set(params) == {"url", "query", "top_k", "timeout_s"}


def test_benchmark_threshold_defaults_to_calibrated_tau():
    """run_benchmark must refuse on the same sigmoid scale as the API:
    the old -2.0 logit-scale default could never fire."""
    import evaluation.run_benchmark as harness
    from app.generator.refusal import DEFAULT_THRESHOLD

    assert harness.build_parser().get_default("threshold") == DEFAULT_THRESHOLD == 0.5
