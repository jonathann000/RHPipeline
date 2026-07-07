"""
Coreference propagation.
Extends already-detected entities to their other mentions in the document —
repeated exact phrases (a diagnosis named twice) and, for person names,
standalone name parts (a full name followed later by a first-name-only
reference). Runs once on the merged entity list after all detection stages,
so it applies equally regardless of which agents (rules/BERT/LLM) found the
original entity — i.e. it works the same in "full", "no_bert", and
"llm_only" modes.
"""

import re
from entities import Entity

_MIN_NAME_PART_LEN = 3
_TITLES = {"dr", "dr.", "professor", "prof", "prof."}


def propagate_entities(text: str, entities: list[Entity]) -> list[Entity]:
    """Return newly discovered entities; does not mutate `entities`."""
    covered = {(e.start, e.end) for e in entities}
    new_entities: list[Entity] = []

    for entity in entities:
        needles = [entity.text]
        if entity.label == "private_person":
            needles.extend(_name_parts(entity.text))

        for needle in needles:
            for span in _find_occurrences(text, needle, covered):
                new_entities.append(_clone_at(entity, text, span))
                covered.add(span)

    return new_entities


def _name_parts(name: str) -> list[str]:
    """Standalone name parts worth propagating (skips titles like 'Dr.')."""
    parts = [p.strip(".,") for p in name.split()]
    return [
        p for p in parts
        if len(p) >= _MIN_NAME_PART_LEN
        and p.lower() not in _TITLES
        and p[0].isupper()
    ]


def _find_occurrences(text: str, needle: str, covered: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Whole-word occurrences of `needle` not already covered by an existing span."""
    if not needle:
        return []
    spans = []
    for match in re.finditer(r"\b" + re.escape(needle) + r"\b", text):
        span = (match.start(), match.end())
        if not _overlaps(span, covered):
            spans.append(span)
    return spans


def _overlaps(span: tuple[int, int], covered: set[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < c_end and end > c_start for c_start, c_end in covered)


def _clone_at(entity: Entity, text: str, span: tuple[int, int]) -> Entity:
    start, end = span
    return Entity(
        text=text[start:end],
        label=entity.label,
        start=start,
        end=end,
        source="propagated",
        confidence=entity.confidence,
        generalized=entity.generalized,
        risk=entity.risk,
    )
