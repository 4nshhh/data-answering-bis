"""Phase 4: system prompt engineering and grounding directives.

Wraps the deterministic context assembled in Phase 3
(``context_builder.BuiltContext``) in strict grounding rules and
produces separate ``system`` + ``user`` chat messages for the LLM
client (Phase 5).

Scope (AGENTS.md section 9):
  1. System message with the BIS assistant's grounding/behavior rules.
  2. User message with the user's query and the assembled evidence.
  3. Explicit model-based token/context budgeting with greedy,
     order-preserving block selection.

Out of scope: LLM API calls (Phase 5), citation parsing/verification
(Phase 6), refusal wording and abstention thresholds (Phase 7), FastAPI
wiring (Phase 8). This module never reads ``rerank_score`` values and
contains no refusal sentence; when no evidence fits the budget it sets
``insufficient_evidence_hook`` and leaves the decision to Phase 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.generator.context_builder import BuiltContext, ChunkIndex, format_block

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RESERVE_MARGIN_TOKENS",
    "SYSTEM_PROMPT",
    "PromptBundle",
    "build_prompt",
]

#: Default generation model (Groq). Budgeting defaults below are sized
#: for this model; pass explicit values when targeting another model.
DEFAULT_MODEL = "openai/gpt-oss-120b"

#: Default total context window (tokens) assumed for budget fitting.
DEFAULT_MAX_CONTEXT_TOKENS = 131072

#: Tokens held back from the window for the model's own generation,
#: so the prompt (system + query + evidence) never consumes the full
#: window. Phase 5 may tune this against the live model.
DEFAULT_RESERVE_MARGIN_TOKENS = 8192

#: Frozen grounding directives (AGENTS.md section 9). The citation lines
#: state the required output *format* only; extraction and verification
#: of citations belong to Phase 6. The format rules are strict because
#: only the canonical triple shape can be verified against the evidence:
#: any other style (bare clause numbers, Foreword cites, non-bracket
#: markers) leaves the claim unverifiable and the answer refused.
SYSTEM_PROMPT = """\
You are the BIS Standards Assistant, a precise engineering aide for Bureau of Indian Standards documents.

Grounding rules — follow all of them without exception:

1. Answer ONLY using facts explicitly stated in the provided BIS context blocks below. If the blocks do not contain the answer, say so plainly instead of guessing.
2. Do NOT infer or extrapolate unstated technical limits, safety factors, tolerances, or requirements. Never invent values the text does not give.
3. Preserve technical details with 100% exactness: numerical values, units (e.g. N/mm^2, pH >= 6, mm), clause numbers, and categories (e.g. M1, N2) must be reproduced exactly as written in the context.
4. If a requirement is conditional (e.g. "subject to agreement between purchaser and manufacturer"), state the condition explicitly alongside the requirement.
5. If the blocks do not contain the answer, say so plainly in one sentence WITHOUT any citation (an uncited abstention is honest; a fabricated citation is a failure).

Citation rules — follow all of them without exception:

A. Cite every technical assertion using ONLY this exact format: [IS <standard_no>:<year>, Clause <clause>, Page <page>], copying the Standard, Clause, and Location headers of the context block it comes from. Example: [IS 456:2000, Clause 5.4, Page 15].
B. Never use any other citation style: no bare clause numbers, no "Foreword" or section-name cites, no parenthetical remarks inside the brackets (write [IS 456:2000, Clause 26.5.3.1, Page 49], never [IS 456:2000, Clause 26.5.3.1(a), Page 49]), and no non-bracket markers.
C. If a block's Clause header is "N/A", cite it as Clause N/A with that block's Standard and Page.
D. If the question names a parent clause (e.g. 26.5) but the blocks show numbered sub-clauses, cite the shown sub-clause numbers exactly as written.\
"""


@dataclass
class PromptBundle:
    """Assembled chat messages plus selection bookkeeping for later phases."""

    system: str  # grounding/behavior rules; never contains query or evidence
    user: str  # user query followed by the fitted evidence section
    model: str  # model the budget was computed against
    used_block_ids: list[str] = field(default_factory=list)  # chunk_ids included, in order
    reserve_block_ids: list[str] = field(default_factory=list)  # chunk_ids left out
    insufficient_evidence_hook: bool = False  # True when zero blocks fit; Phase 7 owns behavior


def _estimate_tokens(text: str) -> int:
    """Estimate token count as ``ceil(chars / 4)``.

    This is a deliberately conservative, dependency-free approximation
    for context fitting only — it is NOT exact API token accounting and
    must not be used for billing or precise window management. Phase 5
    may replace this function with the exact tokenizer for the selected
    Groq/OpenAI model; the fitting logic in :func:`build_prompt` depends
    only on this function's signature, not its implementation.
    """
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def _render_blocks(
    context: BuiltContext,
    chunk_index: ChunkIndex | None = None,
) -> list[str]:
    """Render each block with the Phase 3 header template.

    When ``chunk_index`` is provided, year and ``heading_path`` display
    enrichment matches ``BuiltContext.prompt_text`` exactly; otherwise
    headers fall back to the raw evidence fields.
    """
    rendered = []
    for block in context.blocks:
        year: str | None = None
        heading_path: list[str] | None = None
        if chunk_index is not None:
            chunk_dict = chunk_index.by_id.get(block.evidence.chunk_id)
            if chunk_dict is not None:
                raw_year = chunk_dict.get("metadata", {}).get("year")
                year = str(raw_year) if raw_year else None
                raw_path = chunk_dict.get("metadata", {}).get("heading_path") or []
                heading_path = [str(h) for h in raw_path] or None
        rendered.append(format_block(block.rank, block.evidence, block.text, year=year, heading_path=heading_path))
    return rendered


def _fit_blocks(
    rendered: list[str],
    query: str,
    max_context_tokens: int,
    reserve_margin_tokens: int,
) -> tuple[list[int], bool]:
    """Select block indices that fit the budget, preserving rerank order.

    Greedy from the front: keep blocks while they fit, drop trailing
    blocks first. Blocks are never re-ordered, split, or truncated.
    Returns ``(kept_indices, hook)`` where ``hook`` is True when no
    block fits and Phase 7 must decide how to respond.
    """
    overhead = _estimate_tokens(SYSTEM_PROMPT) + _estimate_tokens(query) + reserve_margin_tokens
    budget = max_context_tokens - overhead
    if budget <= 0:
        return [], True
    block_tokens = [_estimate_tokens(text) for text in rendered]
    kept: list[int] = []
    used = 0
    for i, cost in enumerate(block_tokens):
        separator = _estimate_tokens("\n\n") if kept else 0
        if used + separator + cost <= budget:
            used += separator + cost
            kept.append(i)
        else:
            break
    return kept, len(kept) == 0


def build_prompt(
    query: str,
    context: BuiltContext,
    *,
    model: str = DEFAULT_MODEL,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    reserve_margin_tokens: int = DEFAULT_RESERVE_MARGIN_TOKENS,
    chunk_index: ChunkIndex | None = None,
) -> PromptBundle:
    """Assemble ``system`` + ``user`` messages for the LLM client.

    Args:
        query: the user's natural-language question (non-blank).
        context: assembled evidence from ``build_context``; consumed
            read-only, never mutated or re-ordered.
        model: model identifier the budget is computed against.
        max_context_tokens: total context window for ``model``.
        reserve_margin_tokens: tokens held back for generation.
        chunk_index: optional read-only index from ``load_chunk_index``
            for year/``heading_path`` header enrichment matching
            ``BuiltContext.prompt_text``. When ``None``, headers use
            the raw evidence fields.

    Returns:
        ``PromptBundle`` with the two chat roles and id bookkeeping.
        When nothing fits, the evidence section is empty and
        ``insufficient_evidence_hook`` is True (Phase 7 decides).
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-blank string")
    if max_context_tokens < 1:
        raise ValueError(f"max_context_tokens must be positive, got {max_context_tokens!r}")
    if reserve_margin_tokens < 0:
        raise ValueError(f"reserve_margin_tokens must be non-negative, got {reserve_margin_tokens!r}")

    rendered = _render_blocks(context, chunk_index)
    kept, hook = _fit_blocks(rendered, query.strip(), max_context_tokens, reserve_margin_tokens)

    kept_blocks = [context.blocks[i] for i in kept]
    evidence_section = "\n\n".join(rendered[i] for i in kept)
    user = f"Question:\n{query.strip()}\n\nEvidence:\n{evidence_section}"

    used_ids = [block.evidence.chunk_id for block in kept_blocks]
    dropped_ids = [block.evidence.chunk_id for i, block in enumerate(context.blocks) if i not in kept]
    reserve_ids = dropped_ids + [item.chunk_id for item in context.reserve]

    return PromptBundle(
        system=SYSTEM_PROMPT,
        user=user,
        model=model,
        used_block_ids=used_ids,
        reserve_block_ids=reserve_ids,
        insufficient_evidence_hook=hook,
    )
