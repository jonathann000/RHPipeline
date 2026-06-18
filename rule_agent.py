"""
Rule-based agent for structured, unambiguous PII.
These patterns are fast, free, and highly reliable — no LLM needed.
"""

import re
from entities import Entity


PATTERNS: dict[str, list[str]] = {
    "private_email": [
        r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}",
    ],
    "private_phone": [
        r"(\+46|0046|0)[\s\-]?[0-9]{1,4}[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}",
        r"\b07[0-9][\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b",
    ],
    "account_number": [
        # Swedish personnummer: YYYYMMDD-XXXX or YYMMDD-XXXX
        r"\b(19|20)?\d{6}[\s\-]\d{4}\b",
        # Swedish passport / national ID (basic)
        r"\b[A-Z]{2}\d{7}\b",
    ],
    "private_date": [
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",          # ISO: 2024-03-15
        r"\b\d{1,2}\s+\w+\s+\d{4}\b",            # 15 mars 2024
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",    # 15/03/24
    ],
    "private_address": [
        r"\b\d{3}\s?\d{2}\b",                     # Swedish zip codes
    ],
}


class RuleAgent:
    def __init__(self):
        self._compiled = {
            label: [re.compile(p, re.IGNORECASE) for p in patterns]
            for label, patterns in PATTERNS.items()
        }

    def detect(self, text: str) -> list[Entity]:
        entities = []
        for label, regexes in self._compiled.items():
            for regex in regexes:
                for match in regex.finditer(text):
                    entities.append(Entity(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        source="rule",
                        confidence=1.0,
                    ))
        return self._remove_overlaps(entities)

    def _remove_overlaps(self, entities: list[Entity]) -> list[Entity]:
        """Deduplicate and remove spans fully contained within a longer span."""
        # Sort by start asc, then by span length desc (longer spans first)
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        kept: list[Entity] = []
        for e in entities:
            if kept and e.start >= kept[-1].start and e.end <= kept[-1].end:
                continue  # fully contained in previous — drop
            kept.append(e)
        return kept
