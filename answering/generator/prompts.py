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
from typing import Literal

from answering.generator.context_builder import BuiltContext, ChunkIndex, format_block

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RESERVE_MARGIN_TOKENS",
    "SYSTEM_PROMPT",
    "VALID_MODES",
    "MODE_INSTRUCTIONS",
    "Mode",
    "validate_mode",
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
You are the BIS Standards Assistant, a precise engineering aide for
Bureau of Indian Standards documents.

The task-specific instructions supplied separately determine the type
and format of response. The following grounding and citation rules
apply to every task without exception.

Grounding rules — follow all of them without exception:

1. Answer ONLY using facts explicitly stated in the provided BIS
   context blocks below. The context blocks are the complete
   authoritative evidence available for this response. Do not use
   your pretrained/general knowledge as evidence.

2. Do NOT infer or extrapolate unstated technical limits, safety
   factors, tolerances, requirements, applicability, or relationships.
   Never invent values the text does not give.

3. You may synthesize or combine facts explicitly stated across
   multiple context blocks, but the synthesis must not introduce any
   new technical fact, assumption, value, or relationship that is not
   supported by the provided context.

4. Preserve technical details with 100% exactness: numerical values,
   units (e.g. N/mm^2, pH >= 6, mm), clause numbers, and categories
   (e.g. M1, N2) must be reproduced exactly as written in the context.

5. If a requirement is conditional (e.g. "subject to agreement between
   purchaser and manufacturer"), state the condition explicitly
   alongside the requirement.

6. If context blocks contain conflicting values, requirements, or
   statements, do not resolve the conflict using prior knowledge or
   intuition. State that the retrieved context contains conflicting
   information and cite the relevant evidence.

7. If the blocks do not contain the answer, say so plainly in one
   sentence WITHOUT any citation. An uncited abstention is honest;
   a fabricated citation is a failure.

Citation rules — follow all of them without exception:

A. Cite every technical assertion using ONLY this exact format:
   [IS <standard_no>:<year>, Clause <clause>, Page <page>]

   Copy the Standard, Clause, and Location/Page headers of the
   context block it comes from.

   Example:
   [IS 456:2000, Clause 5.4, Page 15]

B. Never use any other citation style: no bare clause numbers,
   no "Foreword" or section-name cites, no parenthetical remarks
   inside the brackets, no non-bracket markers of any kind, and no
   shorthand citation fragments.

C. If a block's Clause header is "N/A", cite it as Clause N/A with
   that block's Standard and Page.

D. If the question names a parent clause (e.g. 26.5) but the blocks
   show numbered sub-clauses, cite the shown sub-clause numbers
   exactly as written.

E. If the fact comes from a table, cite the Clause header of the
   block containing that table exactly as written. Never use the
   table number itself as the clause.\
"""

#: Task modes selecting mode-specific instructions. The universal
#: grounding/citation rules above apply unchanged in every mode.
Mode = Literal["ask", "product_match"]

VALID_MODES: tuple[str, ...] = ("ask", "product_match")


def validate_mode(mode: str) -> str:
    """Return the mode, or raise for anything but an explicit task type."""
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {list(VALID_MODES)}, got {mode!r}"
        )
    return mode


#: Mode-specific instructions appended after the universal rules.
#: Task framing only — grounding, citation format, verification, and
#: refusal behavior are identical across modes.
MODE_INSTRUCTIONS: dict[str, str] = {
    "ask": """\
Task instructions (ask mode):
Answer the user's BIS question directly using ONLY the evidence blocks.
When asked which standard applies, name the standard with its year
and state the scope basis found in the blocks.

Synthesize information across multiple evidence blocks when needed,
but do not introduce any technical fact, value, requirement, or
relationship that is not supported by the blocks.
""",
    "product_match": """\
Task instructions (product_match mode):

Identify and rank the BIS standards applicable to the described
product using ONLY the evidence blocks.

For each applicable standard, state:
- its standard number and year,
- the scope basis found in the blocks, and
- the key applicable requirements explicitly supported by the blocks.

Order the most applicable standard first.

Do not infer applicability from general knowledge or from product
similarity alone. The evidence blocks must provide the basis for
identifying a standard as applicable.

A standard number merely mentioned inside another standard's text
(reference lists, bibliographies, and annex tables name standards
that are not necessarily applicable to the product) must never be
presented as applicable on that basis alone. Present a standard as
applicable only when the evidence blocks show its own scope or
requirements under that standard's `Standard:` header.

A standard is applicable to the user's product only when the evidence
blocks state, in that standard's own scope (its Scope clause,
applicability statement, or subject definition), that the standard
covers the user's product or product category. Technical similarity
alone — the standard discusses related technology, test methods, or
requirements that resemble the product — never establishes
applicability.
If the evidence shows relevant requirements but does not establish
that the user's specific product falls within the standard's stated
scope, do not list the standard as applicable. Say plainly that the
evidence does not establish applicability, without citing.

Cite every claim canonically. If the evidence blocks do not provide
enough basis to identify an applicable standard, abstain plainly
without citing.
""",
}


@dataclass
class PromptBundle:
    """Assembled chat messages plus selection bookkeeping for later phases."""

    system: str  # grounding/behavior rules; never contains query or evidence
    user: str  # user query followed by the fitted evidence section
    model: str  # model the budget was computed against
    used_block_ids: list[str] = field(default_factory=list)  # chunk_ids included, in order
    reserve_block_ids: list[str] = field(default_factory=list)  # chunk_ids left out
    insufficient_evidence_hook: bool = False  # True when zero blocks fit; Phase 7 owns behavior
    mode: str = "ask"  # task mode selecting MODE_INSTRUCTIONS


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
    mode: str = "ask",
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
        mode: task mode (``"ask"`` or ``"product_match"``) selecting
            ``MODE_INSTRUCTIONS``; the universal rules are identical.

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
    validate_mode(mode)

    rendered = _render_blocks(context, chunk_index)
    kept, hook = _fit_blocks(rendered, query.strip(), max_context_tokens, reserve_margin_tokens)

    kept_blocks = [context.blocks[i] for i in kept]
    evidence_section = "\n\n".join(rendered[i] for i in kept)
    user = f"Question:\n{query.strip()}\n\nEvidence:\n{evidence_section}"
    system = SYSTEM_PROMPT + "\n\n" + MODE_INSTRUCTIONS[mode]

    used_ids = [block.evidence.chunk_id for block in kept_blocks]
    dropped_ids = [block.evidence.chunk_id for i, block in enumerate(context.blocks) if i not in kept]
    reserve_ids = dropped_ids + [item.chunk_id for item in context.reserve]

    return PromptBundle(
        system=system,
        user=user,
        model=model,
        used_block_ids=used_ids,
        reserve_block_ids=reserve_ids,
        insufficient_evidence_hook=hook,
        mode=mode,
    )
