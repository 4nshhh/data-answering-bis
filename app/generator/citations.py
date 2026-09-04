"""Phase 6: citation parser and post-processor.

Extracts inline citations in the canonical format
``[IS <standard_no>:<year>, Clause <clause>, Page <page>]`` (AGENTS.md
section 10) from generated answer text and verifies each one against
the retrieved evidence that grounded the answer.

Scope:
  * ``parse_citations()``: deterministic regex extraction.
  * ``verify_answer()``: per-citation verdicts linked to chunk IDs.

Out of scope: refusal or answer repair on failed verification
(Phase 7 owns abstention; Phase 8 owns HTTP mapping). This module
never mutates answer text — it verifies and reports, so later phases
decide on complete information.

Verdict model (three states, deliberately):
  * ``verified``: every checkable field matches the linked evidence.
  * ``mismatch``: at least one checkable field contradicts evidence
    (wrong clause, page outside the chunk range, unknown standard).
  * ``unverifiable``: the evidence record lacks the metadata needed
    for a field (e.g. no year or no page numbers stored), so the field
    is skipped rather than failed.

Clause matching is text-anchored, not metadata-only: chunks carry one
metadata clause label but their text routinely contains sub-clauses
(e.g. chunk ``26.4`` containing ``26.4.1``/``26.4.2.1``). A cited label
verifies when it equals the metadata clause, a markdown header label
(single- or multi-level, e.g. ``## 7 SAMPLING``), or a bare line-initial
label extending the metadata clause or any header label — and,
additionally, when it is the one-level parent of such a label (cited
``26.5.2`` against shown ``26.5.2.2``). Inline (mid-line) occurrences of
deep dotted labels (3+ parts, e.g. OCR line-wrap casualties like
``, 26.5.3.1 Longitudinal``) extending the metadata clause also attest,
since 3+-part dotted numbers are clause references in practice, never
measurements. Sibling labels and coarser ancestors never match.
Harmless formatting qualifiers are normalized away before comparison:
parenthesized subdivisions (``26.5.3.1(a)`` -> ``26.5.3.1``,
``26.5.1 (Table 16)`` -> ``26.5.1``) and clause ranges (``3.4-3.6`` ->
``3.4``, ``3.5``, ``3.6``, every member must be attested). A cited
``N/A`` clause verifies only against a block whose own clause metadata
is unknown, with standard/year/page still checked. Page scope and year
rules apply unchanged. Formal partial references (``Clause X`` +
``Page Y`` pairs, parenthesized clause mentions) are extracted and
verified the same way; bare prose mentions are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.generator.context_builder import BuiltContext, ChunkIndex
from app.generator.llm_client import GeneratedAnswer

__all__ = [
    "CITATION_RE",
    "Citation",
    "VerifiedCitation",
    "VerifiedAnswer",
    "parse_citations",
    "parse_references",
    "verify_answer",
]

#: Canonical inline citation (AGENTS.md section 10), e.g.
#: ``[IS 456:2000, Clause 5.4, Page 15]``. Live models emit Unicode
#: variants (CJK brackets U+3010/U+3011, narrow no-break spaces,
#: non-breaking hyphens in page ranges like ``Page 15-16``); these are
#: normalized 1:1 before matching so the canonical shape still parses.
#: Clause labels may contain dots, dashes, spaces, and Annex names.
CITATION_RE = re.compile(
    r"\[\s*IS\s+(?P<standard_no>[0-9][A-Za-z0-9.\-]*)"
    r"\s*:\s*(?P<year>\d{4})\s*,\s*"
    r"Clause\s+(?P<clause>[^,\]]+?)\s*,\s*"
    r"Pages?\s+(?P<page>\d+)(?:\s*[-\u2010-\u2015\u2212]\s*\d+)?\s*\]"
)

#: 1:1 Unicode confusables folded before parsing (length-preserving,
#: so match spans stay valid against the original answer text).
_PARSE_FOLD = str.maketrans(
    {
        "\u00a0": " ",  # no-break space
        "\u202f": " ",  # narrow no-break space (live model emits this)
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u3000": " ",  # ideographic space
        "\u3010": "[",  # left black lenticular bracket (live model emits this)
        "\u3011": "]",  # right black lenticular bracket
        "\uff1a": ":",  # fullwidth colon
        "\uff0c": ",",  # fullwidth comma
    }
)


@dataclass
class Citation:
    """One raw citation span extracted from answer text."""

    standard_no: str  # digits/code as written, e.g. "456" ("" for partial refs)
    year: str  # e.g. "2000" ("" for partial refs, which state no year)
    clause: str  # e.g. "5.4" or "ANNEX B"
    page: int  # first page for ranges; -1 when no page stated
    span_start: int  # char offsets into the answer text
    span_end: int
    partial: bool = False  # True for formal Clause/Page refs without the full triple


@dataclass
class VerifiedCitation:
    """A citation plus its verification outcome against the evidence."""

    citation: Citation
    verdict: str  # "verified" | "mismatch" | "unverifiable"
    chunk_id: str | None  # linked evidence chunk, if identified
    detail: str  # human-readable reason, for Phase 7/8 logging
    checked_fields: list[str] = field(default_factory=list)


@dataclass
class VerifiedAnswer:
    """Post-processing result: original text plus citation verdicts."""

    text: str  # byte-identical to the input answer text
    citations: list[VerifiedCitation]
    all_verified: bool  # True when every citation is "verified"
    has_citations: bool  # False when the answer cites nothing


def _normalize_standard(value: str | None) -> str | None:
    """Normalize ``IS 456`` / ``IS456`` / ``456`` to digits ``456``."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", value)
    return digits or None


def parse_citations(text: str) -> list[Citation]:
    """Extract canonical citations in order of appearance.

    The text is folded through ``_PARSE_FOLD`` (strictly 1:1
    replacements) before matching, so spans remain valid against the
    original string while Unicode punctuation variants still parse.
    A page range (``Page 15-16``) yields the first page, matching the
    ``Page <page_start>`` response convention.
    """
    folded = text.translate(_PARSE_FOLD)
    found = []
    for match in CITATION_RE.finditer(folded):
        found.append(
            Citation(
                standard_no=match.group("standard_no").strip(),
                year=match.group("year"),
                clause=match.group("clause").strip(),
                page=int(match.group("page")),
                span_start=match.start(),
                span_end=match.end(),
            )
        )
    return found


#: Formal Clause+Page reference pairs without the full ``[IS ...]`` triple
#: (e.g. "as given in Clause 26.4.2 on Page 47"). The pair requirement is
#: deliberate: bare prose mentions of "clause" are ignored to avoid false
#: refusals, while explicit clause/page pairings are verifiable claims.
_PARTIAL_PAIR_RE = re.compile(
    r"Clause\s+(?P<clause>\d+(?:\.\d+)*|ANNEX\s+[A-Z0-9]+)\b"
    r".{0,80}?Pages?\s+(?P<page>\d+)",
    re.IGNORECASE,
)

#: Parenthesized single-clause references (e.g. "(see clause 26.4.2)").
#: Verified clause-only against the evidence (no page stated, none checked).
_PARTIAL_PAREN_RE = re.compile(
    r"\(\s*[^()]{0,24}?clause\s+"
    r"(?P<clause>\d+(?:\.\d+)*|ANNEX\s+[A-Z0-9]+)\b[^()]*\)",
    re.IGNORECASE,
)


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(start < span[1] and span[0] < end for start, end in spans)


def parse_references(text: str) -> list[Citation]:
    """Extract canonical citations plus formal partial references.

    Canonical triples come first in document order merged with partials
    (``partial=True``); partial spans overlapping a canonical span are
    skipped so no claim is counted twice. Spans are valid against the
    original text (folding is 1:1).
    """
    folded = text.translate(_PARSE_FOLD)
    canonical = parse_citations(text)
    covered = [(c.span_start, c.span_end) for c in canonical]
    partials: list[Citation] = []
    for pattern, has_page in (
        (_PARTIAL_PAIR_RE, True),
        (_PARTIAL_PAREN_RE, False),
    ):
        for match in pattern.finditer(folded):
            span = (match.start(), match.end())
            if _overlaps(span, covered):
                continue
            covered.append(span)
            partials.append(
                Citation(
                    standard_no="",
                    year="",
                    clause=match.group("clause").strip(),
                    page=int(match.group("page")) if has_page else -1,
                    span_start=span[0],
                    span_end=span[1],
                    partial=True,
                )
            )
    return sorted(canonical + partials, key=lambda c: c.span_start)


def _norm_clause(value: str | None) -> str | None:
    """Normalize a clause label for comparison.

    Case/terminal-dot insensitive; parenthesized subdivision qualifiers
    (``26.5.3.1(a)``, ``26.5.1 (Table 16)``, ``8 (Table 1)``) are stripped
    to the base clause they refine, and Unicode dashes are folded to
    ASCII so ranges (``3.4-3.6``) split deterministically downstream.
    The base clause must still be attested — qualifiers never excuse an
    unshown clause.
    """
    if value is None:
        return None
    cleaned = value.strip().rstrip(".")
    cleaned = re.sub(r"\s*\([^()]*\)", "", cleaned).strip()
    cleaned = re.sub(r"[\u2010-\u2015\u2212]", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).rstrip(".")
    return cleaned.upper() or None


#: Markdown clause headings (``#### 26.4.1 Nominal Cover``, ``## 7 SAMPLING``).
#: Single-level headings are genuine clause titles too (Scope, Sampling,
#: Tests); bare prose numbers are still never trusted from headers alone.
_HEADER_CLAUSE_RE = re.compile(r"^#{1,6}\s*(\d+(?:\.\d+)*)\b", re.MULTILINE)

#: Markdown annex headings (``ANNEX B``), optionally bare.
_HEADER_ANNEX_RE = re.compile(r"^#{0,6}\s*(ANNEX\s+[A-Z][A-Z0-9]*)\b", re.MULTILINE)

#: Bare line-initial dotted numbers (``26.4.2.1 However for, ...``).
#: Trusted as strict dotted-children of the chunk's metadata clause or of
#: any markdown header label in the chunk (a line-initial ``0.5`` is a
#: measurement, not Clause 0.5; ``7.2`` under header ``7.1`` is a sibling,
#: not a child).
_BARE_CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\b", re.MULTILINE)

#: Inline deep clause mentions (``..., 26.5.3.1 Longitudinal ...``).
#: Three-or-more-part dotted numbers occurring anywhere in the text that
#: strictly extend the metadata clause (typically line-wrap casualties of
#: genuine sub-clause titles). Two-part numbers are excluded: those are
#: routinely measurements (``0.5``) rather than clause references.
_INLINE_CLAUSE_RE = re.compile(r"\b(\d+(?:\.\d+){2,})\b")


def _block_label_sets(block: Any) -> tuple[str | None, set[str]]:
    """Clause labels a chunk attests: (metadata_label, scoped_text_labels).

    Scoped text labels = markdown header labels (any level) plus bare
    line-initial labels that extend the metadata clause or any header
    label (``26.4``/header ``8`` admit ``26.4.1``/``8.1`` but never a
    stray ``0.5`` or a sibling ``7.2``), plus inline deep labels
    extending the metadata clause. Blocks with no metadata clause admit
    header labels only, plus the literal ``N/A`` marker for their own
    unknown clause (so a ``Clause N/A`` citation links to the genuinely
    clause-less block it came from, with standard/year/page still
    checked — never to a labelled block).
    """
    meta = _norm_clause(block.evidence.clause)
    text = block.text
    headers = {_norm_clause(m.group(1)) for m in _HEADER_CLAUSE_RE.finditer(text)}
    headers |= {_norm_clause(m.group(1)) for m in _HEADER_ANNEX_RE.finditer(text)}
    headers.discard(None)
    scoped: set[str] = set(headers)
    if meta is not None:
        scoped.add(meta)
        parents = {meta} | headers
        for m in _BARE_CLAUSE_RE.finditer(text):
            bare = _norm_clause(m.group(1))
            if bare is not None and any(bare.startswith(p + ".") for p in parents):
                scoped.add(bare)
        for m in _INLINE_CLAUSE_RE.finditer(text):
            inline = _norm_clause(m.group(1))
            if inline is not None and inline.startswith(meta + "."):
                scoped.add(inline)
    else:
        scoped.add("N/A")
    return meta, scoped


def _expand_clause_range(cited: str) -> list[str]:
    """Expand a cited clause range into its member labels.

    ``3.4-3.6`` -> ``[3.4, 3.5, 3.6]``; ``5.1-5.2`` -> ``[5.1, 5.2]``.
    Anything that is not a same-prefix integer range returns the input
    unchanged as a single-element list. Every member must be attested
    for the citation to verify — a range never excuses an unshown clause.
    """
    if "-" not in cited:
        return [cited]
    start, _, end = cited.partition("-")
    start, end = start.strip(), end.strip()
    sparts, eparts = start.split("."), end.split(".")
    if (
        len(sparts) == len(eparts)
        and sparts[:-1] == eparts[:-1]
        and sparts[-1].isdigit()
        and eparts[-1].isdigit()
    ):
        lo, hi = int(sparts[-1]), int(eparts[-1])
        if 0 < hi - lo <= 50:
            prefix = ".".join(sparts[:-1])
            return [f"{prefix}.{i}" if prefix else str(i) for i in range(lo, hi + 1)]
    return [cited]


def _clause_supported(cited: str, attested: set[str]) -> bool:
    """True when a cited label is grounded in a chunk's attested labels.

    Accepts exact matches and one-level parent cites (cited ``26.5.2``
    against attested ``26.5.2.2``): the parent names the section whose
    shown text the claim comes from. Sibling labels (``26.5.2`` vs
    ``26.5.3``) and anything coarser than one level (``26`` vs
    ``26.5.3``) never match — those cite content the chunk does not show.
    """
    if cited in attested:
        return True
    cparts = cited.split(".")
    for label in attested:
        lparts = label.split(".")
        if len(lparts) == len(cparts) + 1 and lparts[: len(cparts)] == cparts:
            return True
    return False


def _page_in_scope(
    page: int,
    block: Any,
    chunk_index: ChunkIndex | None,
) -> bool | None:
    """Check a cited page against the block's scope.

    Returns True inside the chunk's page range, None when the record
    carries no page metadata, else consults adjacent same-source chunks
    (covers neighbor-expanded text) before reporting False.
    """
    start, end = block.evidence.page_start, block.evidence.page_end
    if start is None and end is None:
        return None
    low = start if start is not None else end
    high = end if end is not None else start
    assert low is not None and high is not None
    if low <= page <= high:
        return True
    if chunk_index is not None:
        chunk_dict = chunk_index.by_id.get(block.evidence.chunk_id)
        if chunk_dict is not None:
            meta = chunk_dict.get("metadata", {})
            try:
                pos = int(meta["chunk_index"])
            except (KeyError, TypeError, ValueError):
                return False
            source = meta.get("source", block.evidence.source)
            for neighbour in (
                chunk_index.by_position.get((source, pos - 1)),
                chunk_index.by_position.get((source, pos + 1)),
            ):
                if not neighbour:
                    continue
                nmeta = neighbour.get("metadata", {})
                ns, ne = nmeta.get("page_start"), nmeta.get("page_end")
                if ns is None and ne is None:
                    continue
                nlow = ns if ns is not None else ne
                nhigh = ne if ne is not None else ns
                if nlow is not None and nhigh is not None and nlow <= page <= nhigh:
                    return True
    return False


def _block_year(block: Any, chunk_index: ChunkIndex | None) -> str | None:

    if chunk_index is not None:
        chunk_dict = chunk_index.by_id.get(block.evidence.chunk_id)
        if chunk_dict is not None:
            year = chunk_dict.get("metadata", {}).get("year")
            if year:
                return str(year)
    extra = block.evidence.extra or {}
    if extra.get("year"):
        return str(extra["year"])
    return None


def _verify_one(
    citation: Citation,
    context: BuiltContext,
    chunk_index: ChunkIndex | None,
) -> VerifiedCitation:
    """Verify one citation against the selected context blocks.

    Clause rule (strict): the cited label must be attested by the linked
    chunk — either its metadata clause or a scoped text label (markdown
    header, or a bare line-initial label extending the metadata clause).
    Page rule: the cited page must fall in the chunk's page scope.
    Neither rule consults anything outside the retrieved selection.
    """
    checked: list[str] = []
    unverifiable: list[str] = []

    if citation.partial:
        candidates = list(context.blocks)
    else:
        want_std = _normalize_standard("IS " + citation.standard_no)
        candidates = [
            block for block in context.blocks
            if _normalize_standard(block.evidence.standard_no) == want_std
        ]
        checked.append("standard_no")
        if not candidates:
            return VerifiedCitation(
                citation=citation,
                verdict="mismatch",
                chunk_id=None,
                detail=f"standard IS {citation.standard_no} not present in retrieved context",
                checked_fields=checked,
            )

    want_clause = _norm_clause(citation.clause)
    want_labels = _expand_clause_range(want_clause) if want_clause is not None else []
    # Coverage per block: how many cited labels (single or range members)
    # each block attests. Single-label behavior is unchanged (first hit
    # wins); ranges link the block covering the most members.
    coverage: list[tuple[int, Any]] = []
    for block in candidates:
        _, scoped = _block_label_sets(block)
        hit = sum(
            1 for label in want_labels if _clause_supported(label, scoped)
        ) if want_labels else 0
        coverage.append((hit, block))
    best = max((hit for hit, _ in coverage), default=0)
    scoped_hits = [block for hit, block in coverage if hit == best and best > 0]
    # For ranges, every member must be attested *somewhere* in the
    # selection; the linked block is the one with the widest coverage.
    attested_elsewhere = set()
    for hit, block in coverage:
        _, scoped = _block_label_sets(block)
        for label in want_labels:
            if _clause_supported(label, scoped):
                attested_elsewhere.add(label)
    missing = [label for label in want_labels if label not in attested_elsewhere]
    checked.append("clause")
    if not scoped_hits or missing:
        if citation.partial:
            return VerifiedCitation(
                citation=citation,
                verdict="mismatch",
                chunk_id=None,
                detail=f"partial reference to clause {citation.clause!r} "
                f"not supported by retrieved context",
                checked_fields=checked,
            )
        if len(want_labels) > 1:
            return VerifiedCitation(
                citation=citation,
                verdict="mismatch",
                chunk_id=(scoped_hits[0].evidence.chunk_id if scoped_hits
                          else candidates[0].evidence.chunk_id),
                detail=f"clause range {citation.clause!r} not fully in retrieved "
                f"context (missing: {', '.join(missing)})",
                checked_fields=checked,
            )
        got = sorted({(b.evidence.clause or "N/A") for b in candidates})
        return VerifiedCitation(
            citation=citation,
            verdict="mismatch",
            chunk_id=candidates[0].evidence.chunk_id,
            detail=f"clause {citation.clause!r} not in retrieved context (standard has: {', '.join(got)})",
            checked_fields=checked,
        )
    block = scoped_hits[0]
    if citation.partial:
        digits = _normalize_standard(block.evidence.standard_no)
        if digits:
            citation.standard_no = digits

    if citation.page < 0:
        # Partial reference states no page: clause support is the whole check.
        checked.append("page:unstated")
    else:
        checked.append("page")
        scope = _page_in_scope(citation.page, block, chunk_index)
        if scope is None:
            unverifiable.append("page")
        elif not scope:
            return VerifiedCitation(
                citation=citation,
                verdict="mismatch",
                chunk_id=block.evidence.chunk_id,
                detail=f"page {citation.page} outside evidence page scope "
                f"{block.evidence.page_start}-{block.evidence.page_end}",
                checked_fields=checked,
            )

    if citation.partial:
        if unverifiable:
            return VerifiedCitation(
                citation=citation,
                verdict="unverifiable",
                chunk_id=block.evidence.chunk_id,
                detail=f"matched clause, but could not check: {', '.join(unverifiable)}",
                checked_fields=checked,
            )
        return VerifiedCitation(
            citation=citation,
            verdict="verified",
            chunk_id=block.evidence.chunk_id,
            detail="partial reference: clause and page supported by retrieved evidence",
            checked_fields=checked,
        )

    checked.append("year")
    have_year = _block_year(block, chunk_index)
    if have_year is None:
        unverifiable.append("year")
    elif have_year != citation.year:
        return VerifiedCitation(
            citation=citation,
            verdict="mismatch",
            chunk_id=block.evidence.chunk_id,
            detail=f"year {citation.year} does not match standard year {have_year}",
            checked_fields=checked,
        )

    if unverifiable:
        return VerifiedCitation(
            citation=citation,
            verdict="unverifiable",
            chunk_id=block.evidence.chunk_id,
            detail=f"matched clause, but could not check: {', '.join(unverifiable)}",
            checked_fields=checked,
        )
    return VerifiedCitation(
        citation=citation,
        verdict="verified",
        chunk_id=block.evidence.chunk_id,
        detail="standard, clause, page, and year all match retrieved evidence",
        checked_fields=checked,
    )


def verify_answer(
    answer: GeneratedAnswer,
    context: BuiltContext,
    chunk_index: ChunkIndex | None = None,
) -> VerifiedAnswer:
    """Parse and verify all citations in a generated answer.

    Args:
        answer: Phase 5 output; text is carried through byte-identical.
        context: Phase 3 context the answer was grounded in.
        chunk_index: optional read-only index for year enrichment.

    Returns:
        ``VerifiedAnswer`` with per-citation verdicts (canonical triples
        and formal partial references, in document order). ``all_verified``
        is True only when at least one citation exists and every one
        is ``verified`` (an uncited answer is not "verified").
    """
    raw = parse_references(answer.text)
    verified = [_verify_one(c, context, chunk_index) for c in raw]
    return VerifiedAnswer(
        text=answer.text,
        citations=verified,
        all_verified=bool(verified) and all(v.verdict == "verified" for v in verified),
        has_citations=bool(verified),
    )
