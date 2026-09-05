"""Offline tests for the shared runner progress helper.

No models, keys, GPU, network, or waiting: clocks are stubbed, streams
are in-memory, and run_suite's HTTP layer is monkeypatched.
"""

from __future__ import annotations

import io

import pytest

import evaluation.run_queries as runner
from evaluation.progress import QueryProgress


def _buf():
    return io.StringIO()


def test_counter_scales_to_any_total():
    assert QueryProgress(1, stream=_buf(), enabled=True).counter(1) == "1/1 (100%)"
    assert QueryProgress(5, stream=_buf(), enabled=True).counter(4) == "4/5 (80%)"
    assert QueryProgress(20, stream=_buf(), enabled=True).counter(4) == "04/20 (20%)"
    assert QueryProgress(50, stream=_buf(), enabled=True).counter(4) == "04/50 (8%)"
    assert QueryProgress(100, stream=_buf(), enabled=True).counter(4) == "004/100 (4%)"
    assert QueryProgress(137, stream=_buf(), enabled=True).counter(137) == "137/137 (100%)"


def test_counter_clamps_and_empty_total():
    prog = QueryProgress(3, stream=_buf(), enabled=True)
    assert prog.counter(0) == "0/3 (0%)"
    assert prog.counter(99) == "3/3 (100%)"
    assert QueryProgress(0, stream=_buf(), enabled=True).counter(1) == "0/0"


def test_waiting_and_finish_lines():
    buf = _buf()
    prog = QueryProgress(20, stream=buf, enabled=True)
    prog.waiting(4, "Table-based query — waiting for response...")
    prog.finish(4, "PASS", 6.7, note="Table-based query")
    out = buf.getvalue()
    assert "04/20 (20%) | Running" in out
    assert "6.7s | PASS" in out
    assert "Table-based query" in out


def test_finish_marks_and_latency_optional():
    buf = _buf()
    prog = QueryProgress(2, stream=buf, enabled=True)
    prog.finish(1, "FAIL", 1.25, note="q")
    prog.finish(2, "REVIEW", note="q2", ok=False)
    prog.finish(2, "OK")
    out = buf.getvalue()
    assert "1.2s | FAIL" in out  # latency rounded to 0.1s
    assert "| OK" in out  # no latency segment when omitted


def test_ascii_fallback_stream_never_raises():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    prog = QueryProgress(2, stream=stream, enabled=True)
    prog.waiting(1, "note")
    prog.finish(1, "PASS", 0.5, note="note")
    prog.delay(2)
    stream.flush()
    out = raw.getvalue().decode("cp1252")
    assert "1/2" in out and "PASS" in out


def test_disabled_produces_no_output_but_still_sleeps():
    buf = _buf()
    slept: list = []
    prog = QueryProgress(5, stream=buf, enabled=False, sleep_fn=slept.append)
    prog.waiting(1, "x")
    prog.finish(1, "PASS", 1.0, note="x")
    prog.delay(2.5)
    prog.close()
    assert buf.getvalue() == ""
    assert slept == [2.5]  # timing behavior identical with display off


def test_delay_counts_down_whole_seconds():
    buf = _buf()
    slept: list = []
    prog = QueryProgress(3, stream=buf, enabled=True, sleep_fn=slept.append)
    prog.delay(2.5)
    assert slept == [1.0, 1.0, 0.5]
    out = buf.getvalue()
    assert "Waiting 2s before next query" in out
    assert "Waiting 1s before next query" in out
    prog.delay(0)
    prog.delay(-3)
    assert slept == [1.0, 1.0, 0.5]  # non-positive delays sleep nothing


def test_auto_disables_off_terminal():
    prog = QueryProgress(4, stream=_buf())  # StringIO: not a tty
    assert prog.enabled is False


def _records(n: int = 3) -> list:
    return [
        {"id": f"Q{i:03d}", "category": "supported_exact",
         "query": f"query {i}", "expected": "answer"}
        for i in range(1, n + 1)
    ]


def _grounded_body():
    return {
        "answer": "Water pH not less than 6 [IS 456:2000, Clause 5.4, Page 15].",
        "citations": [
            {"standard_no": "IS 456", "year": "2000", "clause": "5.4",
             "page": 15, "chunk_id": "c1", "verified": True}
        ],
        "refused": False,
        "refusal_reason": None,
        "retrieval_meta": {"filtered_standard": "IS 456", "is_mask_restricted": True,
                           "candidates_retrieved": 10, "top_reranker_score": 0.99,
                           "execution_time_ms": 10.0},
        "telemetry": {},
    }


def test_run_suite_results_identical_with_and_without_progress(monkeypatch):
    monkeypatch.setattr(runner, "post_query", lambda *a, **k: (200, _grounded_body(), None))
    plain = runner.run_suite(_records(), url="http://x", top_k=3, timeout_s=5.0,
                             progress=QueryProgress(3, stream=_buf(), enabled=False))
    buf = _buf()
    shown = runner.run_suite(_records(), url="http://x", top_k=3, timeout_s=5.0,
                             progress=QueryProgress(3, stream=buf, enabled=True))
    for row in list(plain) + list(shown):
        row.pop("timestamp", None)  # wall-clock only; everything else must match
    assert shown == plain  # display never alters results
    out = buf.getvalue()
    assert "1/3" in out and "3/3 (100%)" in out and "PASS" in out


def test_run_suite_delay_uses_countdown_without_real_wait(monkeypatch):
    monkeypatch.setattr(runner, "post_query", lambda *a, **k: (200, _grounded_body(), None))
    slept: list = []
    buf = _buf()
    results = runner.run_suite(_records(2), url="http://x", top_k=3, timeout_s=5.0,
                               delay_s=2.0,
                               progress=QueryProgress(2, stream=buf, enabled=True,
                                                      sleep_fn=slept.append))
    assert len(results) == 2  # delay only between queries: one 2s pause
    assert slept == [1.0, 1.0]
    assert "Waiting 2s before next query" in buf.getvalue()


def test_runner_parsers_expose_display_flags():
    import evaluation.run_benchmark as harness

    for parser in (runner.build_parser(), harness.build_parser()):
        assert parser.get_default("delay_secs") == 0.0
        assert parser.get_default("no_progress") is False
