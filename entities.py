from dataclasses import dataclass
from typing import Optional


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    source: str                        # "rule", "bert", "llm"
    confidence: float = 1.0
    generalized: Optional[str] = None  # e.g. "45 år" -> "40-50 år"
    risk: str = "low"                  # "low", "medium", "high"


# Labels that are always direct identifiers, regardless of which prompt/mode
# produced them — neither QUASI_ID_SYSTEM nor FULL_DETECTION_SYSTEM ever asks
# for a generalization on these, so any `generalized` value here is a model
# compliance slip, not an intentional design choice. redaction.py always
# placeholders these, ignoring whatever `generalized` text was suggested.
# ("private_address" is deliberately excluded — it's used as a legitimate
# quasi-identifier with a real generalization in "full" mode's LLM stage.)
# Lives here (not redaction.py) so remove_overlapping_entities below can use
# it without a circular import.
ALWAYS_DIRECT_LABELS = {
    "private_person",
    "private_email",
    "private_phone",
    "account_number",
    "private_date",
    "secret",
}


def remove_overlapping_entities(entities: list[Entity]) -> list[Entity]:
    """
    Resolve entities whose spans overlap by keeping one representative per
    overlapping cluster. A single detection call can legitimately return
    overlapping/nested entities for the same mention (e.g. a name plus
    name+title) — redaction processes spans in reverse order assuming they
    never overlap, so leaving both in causes stale offsets and corrupted
    output on the second, now-misaligned replacement.

    Priority: an ALWAYS_DIRECT_LABELS entity beats a non-direct one sharing
    the span, even if shorter — a direct label is guaranteed to redact to a
    placeholder no matter what `generalized` text came with it, while a
    quasi label trusts that text verbatim. We've seen a quasi-labeled
    generalize suggestion fail to actually remove the identifying content it
    was overlapping (e.g. a "social"-labeled span whose own suggested
    rewrite still contained the person's name) — picking the longer-but-
    riskier span silently reintroduced the exact leak redaction exists to
    prevent. Longest span is only the tiebreaker among same-priority
    candidates.
    """
    ordered = sorted(entities, key=lambda e: e.start)
    clusters: list[list[Entity]] = []
    cluster_max_end = -1
    for e in ordered:
        if clusters and e.start < cluster_max_end:
            clusters[-1].append(e)
            cluster_max_end = max(cluster_max_end, e.end)
        else:
            clusters.append([e])
            cluster_max_end = e.end

    def priority(e: Entity) -> tuple[int, int]:
        is_direct = 0 if e.label in ALWAYS_DIRECT_LABELS else 1
        return (is_direct, -(e.end - e.start))

    return [min(cluster, key=priority) for cluster in clusters]
