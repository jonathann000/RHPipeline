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


def redact_document(text: str, entities: list[Entity]) -> str:
    """
    Replace entity spans in text with generalized form or placeholder.
    Processes in reverse to preserve offsets.
    """
    # Sort by start position descending
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)

    result = text
    for entity in sorted_entities:
        replacement = (
            entity.generalized
            if entity.generalized
            else PLACEHOLDERS.get(entity.label, "[REDAKTERAD]")
        )
        result = result[:entity.start] + replacement + result[entity.end:]

    return result
