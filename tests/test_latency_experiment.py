"""Offline tests for the latency experiment script.

No models, keys, GPU, network, or quota required: only the pure
measurement/aggregation helpers are exercised with synthetic rows.
Live runs (run_once/main) are never invoked here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.compare_llm_latency import (
    QUERY_IDS,
    print_aggregates,
    tok_per_s,
)


def test_tok_per_s_basic():
    assert tok_per_s(164, 1450.0) == 164 / 1.45
    assert tok_per_s(0, 1450.0) is None
    assert tok_per_s(164, 0.0) is None
    assert tok_per_s(0, 0.0) is None


def _row(qid="S001", provider="groq", total=2000.0, gen=1600.0,
         out=160, refused=False, error=None):
    return {
        "id": qid, "provider": provider, "model": "m",
        "total_ms": total, "generation_ms": gen,
        "api_calls": 1, "tok_per_s": tok_per_s(out, gen),
        "refused": refused, "refusal_reason": None, "error": error,
    }


def test_aggregates_cover_gen_distribution(capsys):
    rows = [_row(total=1000.0, gen=800.0, out=80),
            _row(total=3000.0, gen=2500.0, out=500)]
    stats = print_aggregates(rows, "groq", "Groq")
    assert stats["n"] == 2
    assert stats["avg_total"] == 2000.0
    assert stats["median_total"] == 2000.0
    assert stats["min_gen"] == 800.0
    assert stats["max_gen"] == 2500.0
    assert stats["median_gen"] == 1650.0
    assert stats["avg_tok_per_s"] == (100.0 + 200.0) / 2
    out = capsys.readouterr().out
    assert "Median generation latency" in out
    assert "Average output tok/s" in out


def test_aggregates_exclude_refused_and_errors():
    rows = [_row(total=1000.0, gen=800.0, out=80),
            _row(qid="X", refused=True),
            _row(qid="Y", error="boom")]
    stats = print_aggregates(rows, "groq", "Groq")
    assert stats["n"] == 1


def test_query_ids_cover_pipeline_shapes():
    assert QUERY_IDS == ("S001", "C002", "N001", "M001", "N003")
