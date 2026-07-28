"""
Generalize agent — a separate pipeline stage that proposes a generalization
for each already-detected quasi-identifier, decoupled from the detection LLM
that found the spans.

Runs BEFORE the Label Studio export, so the proposed generalization is already
attached to every quasi span when a human annotator reviews the document — the
reviewer verifies span + label + risk + generalization in one pass, rather than
generalization being a second human-in-the-loop step.

Only quasi-identifiers are generalizable: direct identifiers always redact to a
placeholder and medications are kept verbatim (see entities.py). So the agent
targets exactly the LLM-sourced, generalizable spans — the same set the
detection LLM used to generalize inline. When this stage is enabled it is the
authoritative source of generalizations: it overwrites whatever the detection
call may have proposed inline (that field can be left on as a harmless no-op, or
removed from the detection prompt, without affecting this stage).
"""

import logging

from entities import Entity, ALWAYS_DIRECT_LABELS, NEVER_REDACT_LABELS
from llm_backend import LLMBackend

logger = logging.getLogger(__name__)


def _is_generalizable(e: Entity) -> bool:
    """A quasi-identifier the LLM found — not a direct identifier (placeholder)
    and not a medication (kept verbatim)."""
    return (
        e.source == "llm"
        and isinstance(e.label, str)
        and e.label not in ALWAYS_DIRECT_LABELS
        and e.label not in NEVER_REDACT_LABELS
    )


class GeneralizeAgent:
    def __init__(self, backend: LLMBackend, name: str | None = None, enable_thinking: bool = False):
        self.backend = backend
        self.name = name or backend.backend_name
        self.enable_thinking = enable_thinking

    def apply(self, text: str, entities: list[Entity]) -> int:
        """
        Fill `generalized` on every generalizable entity in `entities`, in
        place, from one batched call to the backend. Returns how many spans
        got a non-empty generalization. Mutates the entities directly, so the
        caller's list (and the downstream audit / Label Studio export) sees the
        proposals.
        """
        targets = [e for e in entities if _is_generalizable(e)]
        if not targets:
            return 0

        spans = [(e.text, e.label) for e in targets]
        proposals = self.backend.generalize(text, spans, enable_thinking=self.enable_thinking)

        filled = 0
        for entity, proposal in zip(targets, proposals):
            entity.generalized = proposal  # authoritative: overwrite any inline suggestion
            if proposal:
                filled += 1
        return filled
