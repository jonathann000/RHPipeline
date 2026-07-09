"""
Redaction module.
Applies entity redactions to text, preferring generalization over blanking.
Processes spans in reverse order to preserve character offsets.
"""

from entities import Entity, ALWAYS_DIRECT_LABELS

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
}


def resolve_replacement(entity: Entity) -> str:
    """
    The actual replacement text for an entity — single source of truth so
    the audit log always matches what was written to the redacted document.
    """
    if (
        entity.generalized
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


def redact_document(text: str, entities: list[Entity]) -> str:
    """
    Replace entity spans in text with generalized form or placeholder.
    Processes in reverse to preserve offsets.
    """
    # Sort by start position descending
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

    result = text
    for entity in sorted_entities:
        replacement = resolve_replacement(entity)
        result = result[:entity.start] + replacement + result[entity.end:]

    return result
