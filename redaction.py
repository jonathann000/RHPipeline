"""
Redaction module.
Applies entity redactions to text, preferring generalization over blanking.
Processes spans in reverse order to preserve character offsets.
"""

from entities import Entity, ALWAYS_DIRECT_LABELS, NEVER_REDACT_LABELS

# Fallback placeholders when no generalization is available
PLACEHOLDERS = {
    "private_person":  "[PERSON]",
    "private_email":   "[E-POST]",
    "private_phone":   "[TELEFON]",
    "private_address": "[ADRESS]",
    "account_number":  "[ID-NUMMER]",
    "private_date":    "[DATUM]",
    "secret":          "[HEMLIG-UPPGIFT]",
    "demographics":    "[DEMOGRAFISK-UPPGIFT]",
    "medical":         "[MEDICINSK-UPPGIFT]",
    "temporal":        "[TIDPUNKT]",
    "social":          "[SOCIAL-UPPGIFT]",
    # Added for the MBERTHIPAA model's expanded label set (see
    # bert_agent.py's _BERT_LABEL_MAP) — genuinely new direct-identifier
    # categories beyond what the old Roberta checkpoint distinguished.
    "private_vehicle":    "[FORDON]",
    "private_device":     "[ENHET]",
    "private_url":        "[URL]",
    "private_ip":         "[IP-ADRESS]",
    "private_biometric":  "[BIOMETRISK-UPPGIFT]",
    "private_photo":      "[FOTOGRAFI]",
    "private_other":      "[ÖVRIG-UPPGIFT]",
}


def resolve_replacement(entity: Entity, no_generalize: bool = False) -> str:
    """
    The actual replacement text for an entity — single source of truth so
    the audit log always matches what was written to the redacted document.

    no_generalize: when True, never trust `generalized` text for any label
        — every quasi-identifier falls back to its category placeholder,
        same as direct identifiers already do unconditionally. Exists
        because a generalization can be wrong in ways `_is_non_generalization`
        can't catch — not just "didn't abstract enough" but factually
        incorrect (e.g. "Hypotyreos" — hypothyroidism, a thyroid condition —
        generalized to "underfunktion av tjocktarmen", underfunction of the
        *large intestine*). A placeholder can never be factually wrong; it
        just costs the document some of the informativeness a correct
        generalization would have kept.
    """
    if entity.label in NEVER_REDACT_LABELS:
        return entity.text  # tracked in the audit trail, left verbatim in the document
    if (
        not no_generalize
        and entity.generalized
        and entity.label in PLACEHOLDERS  # only trust generalize for a label we recognize —
                                           # an unrecognized/hallucinated label (e.g. a model
                                           # inventing "quasi-identifierare" instead of a real
                                           # category) gets the least trust, not a free pass
        and entity.label not in ALWAYS_DIRECT_LABELS
        and not _is_non_generalization(entity.text, entity.generalized)
    ):
        return entity.generalized
    return PLACEHOLDERS.get(entity.label, "[REDAKTERAD]")


def _is_non_generalization(original: str, generalized: str) -> bool:
    """
    True if a "generalization" didn't actually generalize anything — the
    original identifying text still appears verbatim inside it (e.g.
    "Sahlgrenska" -> "Sahlgrenska sjukhus" just appends a word; the specific
    hospital is exactly as identifiable as before). Quasi-identifier labels
    trust `generalized` verbatim, so a model that merely extends the
    original instead of abstracting it would otherwise leak straight
    through with no other check catching it.
    """
    return original.strip().lower() in generalized.strip().lower()


def redact_document(text: str, entities: list[Entity], no_generalize: bool = False) -> str:
    """
    Replace entity spans in text with generalized form or placeholder.
    Processes in reverse to preserve offsets. See resolve_replacement for
    what no_generalize does.

    Adjacent (touching) spans that resolve to the exact same replacement
    text are merged into one occurrence of it first. This mainly matters
    after build_redaction_plan splits an entity around a higher-priority
    one embedded inside it (see entities.py) — the surviving fragments on
    either side of, say, a personnummer are separate spans by construction,
    but if they happen to carry the same label (so the same placeholder),
    showing that placeholder twice in a row conveys nothing a reader
    couldn't already tell from showing it once. This never hides distinct
    information: two spans producing identical text are, by definition,
    indistinguishable to a reader either way.
    """
    sorted_entities = sorted(entities, key=lambda e: e.start)
    merged: list[tuple[int, int, str]] = []
    for entity in sorted_entities:
        replacement = resolve_replacement(entity, no_generalize)
        if merged and merged[-1][1] == entity.start and merged[-1][2] == replacement:
            merged[-1] = (merged[-1][0], entity.end, replacement)
        else:
            merged.append((entity.start, entity.end, replacement))

    result = text
    for start, end, replacement in sorted(merged, key=lambda m: m[0], reverse=True):
        result = result[:start] + replacement + result[end:]

    return result
