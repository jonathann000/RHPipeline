"""
Judge panel — audits a redacted document for residual PII and flags it for
another detection pass if anything's still exposed.

Deliberately simple aggregation for now: with only two judges, "did ANY
judge flag something" (a union of all judges' quotes, deduplicated by exact
text) is the only sensible rule — majority voting needs enough judges for a
majority to mean anything. Once the panel grows to 3+, JudgePanel.min_votes
can be raised, but that also needs a way to tell that two differently-worded
quotes from two judges are pointing at the same real leak before they can be
counted as agreeing — that fuzzy-matching step doesn't exist yet and is the
natural next addition when the panel grows.
"""

import re
import logging
from dataclasses import dataclass, field

from llm_backend import LLMBackend
from redaction import PLACEHOLDERS

logger = logging.getLogger(__name__)

_PLACEHOLDER_TOKENS = set(PLACEHOLDERS.values()) | {"[REDAKTERAD]"}


@dataclass
class JudgeFlag:
    quote: str
    reason: str
    judge_name: str


@dataclass
class Judge:
    """Wraps an LLMBackend with a name, so panel output can be attributed."""
    name: str
    backend: LLMBackend
    enable_thinking: bool = False

    def review(self, redacted_text: str) -> list[JudgeFlag]:
        raw_flags = self.backend.judge(redacted_text, enable_thinking=self.enable_thinking)
        return [
            JudgeFlag(
                quote=f.get("quote", ""),
                reason=f.get("reason", ""),
                judge_name=self.name,
            )
            for f in raw_flags
            if f.get("quote")
        ]


def _is_placeholder_misread(quote: str) -> bool:
    """
    True if a flag is almost certainly a false positive from the judge
    misreading its own already-redacted placeholder as still-exposed —
    i.e. the quote contains a known placeholder token, and whatever's left
    after removing all placeholder tokens has no digit sequence and no
    other capitalized, non-sentence-initial word that could itself be a
    real leftover name/place. (Trade-off: this can also suppress a genuine
    but vaguely-worded quasi-identifier concern that happens to reference
    an already-redacted placeholder — e.g. "member of [ADRESS] hunting
    club" — in exchange for reliably killing the much more common
    self-contradicting false positive.)
    """
    if not any(tok in quote for tok in _PLACEHOLDER_TOKENS):
        return False  # nothing to misread — let it through

    remainder = quote
    for tok in _PLACEHOLDER_TOKENS:
        remainder = remainder.replace(tok, "")

    if re.search(r"\d", remainder):
        return False  # leftover digits (phone/personnummer fragment) — real risk

    words = remainder.split()
    for i, word in enumerate(words):
        cleaned = word.strip(",.():;\"'")
        if cleaned and cleaned[0].isupper() and i != 0:
            return False  # possible real leftover proper noun — keep the flag

    return True


@dataclass
class JudgePanel:
    judges: list[Judge]
    min_votes: int = 1  # OR logic — see module docstring

    def review(self, redacted_text: str) -> list[JudgeFlag]:
        """Union of all judges' flags, deduplicated and placeholder-misreads filtered."""
        all_flags: list[JudgeFlag] = []
        for judge in self.judges:
            try:
                flags = judge.review(redacted_text)
                logger.info(f"Judge '{judge.name}': {len(flags)} flag(s)")
                all_flags.extend(flags)
            except Exception:
                logger.warning(f"Judge '{judge.name}' failed to produce a verdict — skipping", exc_info=True)

        seen_quotes: set[str] = set()
        deduped: list[JudgeFlag] = []
        for flag in all_flags:
            if flag.quote not in seen_quotes:
                seen_quotes.add(flag.quote)
                deduped.append(flag)

        survivors = [f for f in deduped if not _is_placeholder_misread(f.quote)]
        discarded = len(deduped) - len(survivors)
        if discarded:
            logger.info(f"Discarded {discarded} likely false positive(s) (placeholder misreads)")

        if survivors:
            logger.info(f"{len(survivors)} surviving issue(s):")
            for flag in survivors:
                logger.info(f"  [{flag.judge_name}] \"{flag.quote}\" — {flag.reason}")

        return survivors
