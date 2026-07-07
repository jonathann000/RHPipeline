"""
Redaction module.
Applies entity redactions to text, preferring generalization over blanking.
Processes spans in reverse order to preserve character offsets.
"""

from entities import Entity

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

# Labels that are always direct identifiers, regardless of which prompt/mode
# produced them — neither QUASI_ID_SYSTEM nor FULL_DETECTION_SYSTEM ever asks
# for a generalization on these, so any `generalized` value here is a model
# compliance slip, not an intentional design choice. Always placeholder them.
# ("private_address" is deliberately excluded — it's used as a legitimate
# quasi-identifier with a real generalization in "full" mode's LLM stage.)
ALWAYS_DIRECT_LABELS = {
    "private_person",
    "private_email",
    "private_phone",
    "account_number",
    "private_date",
    "secret",
}


def resolve_replacement(entity: Entity) -> str:
    """
    The actual replacement text for an entity — single source of truth so
    the audit log always matches what was written to the redacted document.
    """
    if entity.generalized and entity.label not in ALWAYS_DIRECT_LABELS:
        return entity.generalized
    return PLACEHOLDERS.get(entity.label, "[REDAKTERAD]")


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
