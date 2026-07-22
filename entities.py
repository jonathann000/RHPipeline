from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    source: str                        # "rule", "bert", "llm", "gazetteer", "propagated"
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
    # From the MBERTHIPAA model's expanded label set — see bert_agent.py's
    # _BERT_LABEL_MAP. Never has a legitimate generalization; BERT doesn't
    # suggest one anyway, but this keeps remove_overlapping_entities'
    # priority tiebreak (direct beats non-direct) correct for these too.
    "private_vehicle",
    "private_device",
    "private_url",
    "private_ip",
    "private_biometric",
    "private_photo",
    "private_other",
}

# Labels that are tracked (found, logged in the audit trail) but never alter
# the visible text — redaction.py returns the original text unchanged for
# these instead of a placeholder or generalization. "medication" exists
# because medication names are useful for downstream clinical/research
# analysis and aren't meaningfully identifying on their own (a common
# antibiotic or asthma inhaler doesn't narrow down who someone is the way a
# name or rare diagnosis does) — the LLM finding and logging a medication
# mention is useful, but generalizing it away in the actual output isn't.
NEVER_REDACT_LABELS = {
    "medication",
}

# Deterministic, exact-match detectors — a rule regex, BERT's token
# classification, or a gazetteer lookup — as opposed to the LLM's own
# per-instance judgment call. Used to decide when a competing entity's
# `generalized` text can be trusted (see remove_overlapping_entities and
# build_redaction_plan): a rule/BERT/gazetteer hit is already treated as
# unambiguously sensitive (see their own risk="high" — rule_agent.py,
# bert_agent.py, gazetteer_agent.py), so there's no legitimate case where a
# lower-confidence LLM entity should be trusted to have safely paraphrased
# that exact text away.
GROUND_TRUTH_SOURCES = {"rule", "bert", "gazetteer"}

# Labels for which a GROUND_TRUTH_SOURCES entity earns elevated priority
# over an overlapping LLM entity in build_redaction_plan: ALWAYS_DIRECT_LABELS
# (never legitimately generalized regardless of source) plus private_address
# specifically — a gazetteer/BERT/rule place-name match is a confirmed,
# unambiguous location (see the Kungälv/Sahlgrenska case), even though
# private_address is deliberately excluded from ALWAYS_DIRECT_LABELS so the
# LLM can still use it as a legitimate, generalizable label on its OWN
# entities. Deliberately NOT every rule/BERT/gazetteer label — e.g. BERT's
# own numeric-age fix-up (see bert_agent.py's _fix_numeric_person) labels a
# bare age fragment like "34-årig" as "demographics", and that shouldn't
# force a wider, perfectly good LLM generalization like "34-årig
# ensamstående man" -> "30-40 år, ensamstående förälder" to be distrusted
# and split apart — a bare age number isn't a confirmed, non-negotiable
# identifier the way a personnummer or a named place is.
GROUND_TRUTH_PROTECTED_LABELS = ALWAYS_DIRECT_LABELS | {"private_address"}


def remove_overlapping_entities(entities: list[Entity]) -> list[Entity]:
    """
    Resolve entities whose spans overlap by greedily accepting them in
    priority order, skipping any candidate that overlaps an
    already-accepted entity — regardless of which stage (rules/BERT/LLM/
    gazetteer) produced which entity, or which one happened to be detected
    first. A single detection call can legitimately return overlapping/
    nested entities for the same mention (e.g. a name plus name+title), and
    different stages can also independently flag overlapping spans (e.g.
    BERT tagging a bare institution name inside a longer phrase the LLM
    flags as a quasi-identifier) — redaction processes spans in reverse
    order assuming they never overlap, so leaving more than one accepted
    entity overlapping causes stale offsets and corrupted output on the
    second, now-misaligned replacement.

    Priority: an ALWAYS_DIRECT_LABELS entity beats a non-direct one sharing
    the span, even if shorter — a direct label is guaranteed to redact to a
    placeholder no matter what `generalized` text came with it, while a
    quasi label trusts that text verbatim. Longest span is the tiebreaker
    among same-priority candidates (e.g. two quasi labels, or two direct
    ones) — a longer span is usually the more complete redaction.

    This is deliberately NOT "cluster every transitively-overlapping entity
    together and keep one representative for the whole cluster" — that
    approach has a real failure mode: entity A and entity C can be
    completely non-overlapping with each other, yet both get bridged into
    one cluster because entity B happens to overlap both of them. E.g. BERT
    correctly tags "Ingrid Andersson" (private_person, direct) and,
    separately, the rule agent correctly tags her personnummer a few words
    later (account_number, direct) — the two don't overlap each other at
    all. But the LLM, trying to describe the family context, proposes one
    wide "Ingrid Andersson (personnummer ...)" quasi entity spanning both.
    Clustering-then-one-winner would force a choice between the two
    perfectly valid, non-conflicting direct identifiers (picking whichever
    is longer) and silently drop the other's entire span with nothing left
    to redact it — worse than a placeholder, it's raw exposure. Processing
    candidates in priority order and only rejecting a candidate that
    overlaps something already accepted lets both survive: neither directly
    conflicts with the other, so both get taken before the lower-priority
    wide entity is even considered, and that one is correctly rejected
    against both.

    The longest-span tiebreak within a priority tier still lets one entity
    legitimately subsume another of the same tier (e.g. BERT's bare
    "Karolinska" vs. the LLM's longer "en framstående barnneurolog vid
    Karolinska Universitetssjukhuset" quasi phrase, both non-direct since
    private_address isn't in ALWAYS_DIRECT_LABELS — the wider one is
    processed first and wins). That tiebreak isn't automatically safe,
    though: picking the longer span assumes its own `generalized` text
    actually abstracts away whatever the shorter, rejected entity would
    have caught — not guaranteed (e.g. if "generalized" had lazily kept
    "Karolinska" in verbatim instead of abstracting it away). So whenever a
    candidate is rejected for overlapping an already-accepted entity, check
    whether the accepted entity's `generalized` text still contains the
    rejected entity's original text — if so, that generalization can no
    longer be trusted; clear it so redaction falls back to a safe
    placeholder instead of silently reintroducing the leak.

    That check tolerates a trailing Swedish genitive "-s" on either side
    (e.g. a gazetteer match "Kungälvs" losing a same-tier tiebreak to the
    LLM's wider "Kungälvs jaktklubb", whose own generalize text paraphrases
    to "'jaktklubb' i Kungälv" — reintroducing the same place name in its
    bare form). A plain verbatim check misses that: "kungälvs" is not a
    substring of "...i kungälv", even though it's obviously the same leaked
    place name just missing the possessive suffix. This is a general
    Swedish grammar rule (any proper noun can take "-s"), not a fix for one
    document's specific wording.

    But textual leak-detection is inherently reactive — it only catches
    paraphrase failures shaped like ones already seen (verbatim reappearance,
    now genitive inflection; the next one will be some other rewording).
    Whenever the discarded entity came from a GROUND_TRUTH_SOURCE (rule,
    BERT, gazetteer — a deterministic, exact-match detector, not the LLM's
    own per-instance judgment call), skip the textual check entirely and
    unconditionally distrust the winner's `generalized` text. A rule/BERT/
    gazetteer hit is treated as unambiguously sensitive already (see their
    own risk="high" — rule_agent.py, bert_agent.py, gazetteer_agent.py), so
    there is no legitimate case where the LLM's wider entity should be
    trusted to have safely paraphrased that exact text away; the wider
    entity can still win (it may genuinely cover more identifying context,
    e.g. an occupation phrase wrapped around a bare institution name a
    gazetteer also caught), but only with a placeholder fallback, never with
    an unverified rewrite of ground-truth-confirmed text.

    Used for intra-stage self-dedup (an agent cleaning up its own raw
    output — see bert_agent.py, gazetteer_agent.py, rule_agent.py,
    llm_backend.py's chunk merging), where fully discarding a redundant
    duplicate from the same detector is fine. For the cross-stage merge
    across rules/BERT/LLM/gazetteer, see build_redaction_plan instead —
    that one never discards a distinct finding, only decides what the
    final redacted text looks like.
    """
    def priority(e: Entity) -> tuple[int, int]:
        is_direct = 0 if e.label in ALWAYS_DIRECT_LABELS else 1
        return (is_direct, -(e.end - e.start))

    def leak_variants(text: str) -> list[str]:
        variants = [text]
        if len(text) > 3 and text.endswith(("s", "S")):
            variants.append(text[:-1])
        return variants

    accepted: list[Entity] = []
    for e in sorted(entities, key=priority):
        overlap_idxs = [
            i for i, a in enumerate(accepted)
            if e.start < a.end and e.end > a.start
        ]
        if not overlap_idxs:
            accepted.append(e)
            continue
        if e.text.strip():
            for i in overlap_idxs:
                a = accepted[i]
                if not a.generalized:
                    continue
                distrust = (
                    e.source in GROUND_TRUTH_SOURCES
                    or any(v.lower() in a.generalized.strip().lower() for v in leak_variants(e.text.strip()))
                )
                if distrust:
                    accepted[i] = replace(a, generalized=None)
    return accepted


def build_redaction_plan(text: str, entities: list[Entity]) -> list[Entity]:
    """
    Compute the non-overlapping spans to actually redact in the final
    document text. Deliberately separate from deduplicating the *findings*
    list: PipelineResult.entities keeps every detected entity, overlapping
    or not — the system's job is finding as many quasi-identifiers as
    possible, not settling on a single "cleanest" reading, so a genuine
    finding (e.g. a gazetteer's bare institution match) should never
    silently vanish from the audit/Label Studio view just because it
    overlaps a different one (the LLM's wider occupation-phrase entity).
    But the redacted text can only make one decision per character, so this
    function resolves that conflict without touching the entities list
    callers already hold.

    For every atomic sub-interval of the document (cut at every entity
    boundary, so the set of entities covering it is constant throughout),
    the highest-priority covering entity owns it: an ALWAYS_DIRECT_LABELS
    entity beats a non-direct one, then a GROUND_TRUTH_SOURCES entity whose
    own label is in GROUND_TRUTH_PROTECTED_LABELS (a rule/BERT/gazetteer
    hit for a confirmed, non-negotiable identifier — not just any
    rule/BERT/gazetteer label) beats an LLM one, then longest span wins.
    Adjacent atomic intervals owned by the same entity merge back into one
    contiguous redacted span.

    An entity that ends up owning its own full, exact span (nothing
    competed for any part of it) redacts exactly as it always has —
    `generalized` trusted if eligible, same as a document with no conflicts
    at all. An entity that only owns *part* of its original span (something
    higher-priority claimed the rest — e.g. a gazetteer hit for the
    institution name inside a wider LLM occupation-phrase entity) gets
    `generalized` cleared on every fragment it still owns: once a span is
    split, there's no way to know which part of one `generalized` string
    corresponds to which surviving fragment, so a sliced paraphrase can
    never be trusted. But the fragment is still redacted, via that entity's
    own category placeholder — never silently left exposed the way a
    fully-discarded entity would be. This can produce more, smaller
    placeholders butted up against each other (e.g. an institution's
    placeholder immediately followed by the occupation-phrase's own
    placeholder, with no literal connecting words preserved between them)
    rather than one clean sentence — an acceptable cost given the priority
    is never losing a finding, not prose quality.
    """
    if not entities:
        return []

    def priority(e: Entity) -> tuple[int, int, int]:
        is_direct = 0 if e.label in ALWAYS_DIRECT_LABELS else 1
        is_protected_ground_truth = 0 if (
            e.source in GROUND_TRUTH_SOURCES and e.label in GROUND_TRUTH_PROTECTED_LABELS
        ) else 1
        return (is_direct, is_protected_ground_truth, -(e.end - e.start))

    cuts = sorted({e.start for e in entities} | {e.end for e in entities})
    owned_intervals: dict[int, list[tuple[int, int]]] = {}
    owner_by_id: dict[int, Entity] = {}

    for lo, hi in zip(cuts, cuts[1:]):
        covering = [e for e in entities if e.start <= lo and e.end >= hi]
        if not covering:
            continue
        owner = min(covering, key=priority)
        owned_intervals.setdefault(id(owner), []).append((lo, hi))
        owner_by_id[id(owner)] = owner

    plan: list[Entity] = []
    for oid, intervals in owned_intervals.items():
        owner = owner_by_id[oid]
        intervals.sort()
        total_owned = sum(hi - lo for lo, hi in intervals)

        if total_owned == owner.end - owner.start:
            # Nothing competed for any part of this entity's span.
            plan.append(owner)
            continue

        merged: list[tuple[int, int]] = []
        for lo, hi in intervals:
            if merged and merged[-1][1] == lo:
                merged[-1] = (merged[-1][0], hi)
            else:
                merged.append((lo, hi))
        for lo, hi in merged:
            plan.append(replace(owner, start=lo, end=hi, text=text[lo:hi], generalized=None))

    return sorted(plan, key=lambda e: e.start)
