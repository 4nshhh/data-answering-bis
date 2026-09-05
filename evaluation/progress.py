"""Generic terminal progress for query/test runners (stdlib only).

Shared by ``run_queries``, ``run_benchmark``, and
``compare_llm_latency`` so every validation runner shows the same
progress UX for ANY query count (1, 5, 20, 100+, ...): totals come
from the collection being executed, counters are zero-padded to the
total's width, and nothing is hardcoded to a specific count.

Deliberately stdlib-only: ``tqdm`` is not a declared dependency, so
depending on it would break fresh installs. Display goes to stderr so
piped stdout (JSON summaries, result files) is never disturbed. When
the stream is not a terminal, display disables itself automatically;
timing behavior (including opt-in inter-query delays) is identical
either way.

This module never executes queries, never touches results, files,
telemetry, or error handling — display only.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, TextIO

__all__ = ["QueryProgress"]

_OK = "\u2713"  # ✓
_BAD = "\u2717"  # ✗
_WAIT = "\u23f3"  # ⏳
_CLOCK = "\u23f1"  # ⏱

#: ASCII fallbacks for consoles whose encoding lacks the symbols above
#: (e.g. Windows cp1252 powershell defaults). Applied only when the
#: stream cannot encode the original text.
_FALLBACK = {
    _WAIT: "...",
    _OK: "ok",
    _BAD: "FAIL",
    _CLOCK: "wait",
    "\u2014": "-",
    "\u2502": "|",
}

#: Statuses that render with the success mark; anything else renders
#: with the failure mark unless the caller passes ``ok`` explicitly.
_OK_STATUSES = frozenset({"PASS", "OK"})


def _fit(text: str, stream: TextIO) -> str:
    """Replace unsupported symbols when the stream cannot encode them."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        for original, replacement in _FALLBACK.items():
            text = text.replace(original, replacement)
        return text


class QueryProgress:
    """Per-run progress display for a known-size query collection.

    Args:
        total: number of items to execute (any non-negative int).
        label: noun used in the header line (display only).
        stream: where display lines go (default stderr).
        enabled: force on/off; when omitted, on only for terminals.
        sleep_fn: clock used by :meth:`delay` (injectable for tests).
    """

    def __init__(
        self,
        total: int,
        *,
        label: str = "queries",
        stream: TextIO | None = None,
        enabled: bool | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        if enabled is None:
            try:
                enabled = bool(self.stream.isatty())
            except Exception:  # noqa: BLE001 - exotic streams stay quiet
                enabled = False
        self.enabled = bool(enabled) and self.total > 0
        self._sleep = sleep_fn
        self._width = len(str(max(self.total, 1)))
        if self.enabled:
            self._emit(f"Running {self.total} {self.label} ...")

    def _emit(self, text: str, end: str = "\n") -> None:
        if not self.enabled:
            return
        self.stream.write(_fit(text, self.stream) + end)
        try:
            self.stream.flush()
        except Exception:  # noqa: BLE001 - display must never break a run
            pass

    def counter(self, done: int) -> str:
        """``04/20 (20%)``-style counter; width follows the total."""
        if self.total <= 0:
            return "0/0"
        clamped = min(max(int(done), 0), self.total)
        pct = 100.0 * clamped / self.total
        return f"{clamped:0{self._width}d}/{self.total} ({pct:.0f}%)"

    def waiting(self, index: int, message: str = "Waiting for LLM response...") -> None:
        """Show that item ``index`` (1-based) is still running."""
        self._emit(f"{_WAIT} {self.counter(index)} | Running — {message}")

    def finish(
        self,
        index: int,
        status: str,
        latency_s: float | None = None,
        note: str = "",
        ok: bool | None = None,
    ) -> None:
        """Record one completed item: ``✓ 04/20 (20%) | <note> | 6.7s | PASS``."""
        if ok is None:
            ok = str(status) in _OK_STATUSES
        mark = _OK if ok else _BAD
        parts = [f"{mark} {self.counter(index)}"]
        if note:
            parts.append(str(note))
        if latency_s is not None:
            parts.append(f"{float(latency_s):.1f}s")
        parts.append(str(status))
        self._emit(" | ".join(parts))

    def delay(self, seconds: float, message: str = "Waiting {s}s before next query...") -> None:
        """Sleep ``seconds`` with a live whole-second countdown display.

        The full duration is always slept — display on or off — so
        timing behavior never depends on the terminal.
        """
        seconds = float(seconds or 0.0)
        if seconds <= 0:
            return
        if not self.enabled:
            self._sleep(seconds)
            return
        whole = int(seconds)
        for remaining in range(whole, 0, -1):
            self._emit(f"{_CLOCK} " + message.format(s=remaining), end="\r")
            self._sleep(1.0)
        leftover = seconds - whole
        if leftover > 0:
            self._sleep(leftover)
        self._emit(" " * 72, end="\r")

    def close(self) -> None:
        """Finish display output (clears any partial countdown line)."""
        if self.enabled:
            self._emit(f"Done: {self.counter(self.total)}")
