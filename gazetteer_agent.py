"""
Gazetteer agent — fast exact-match lookup against known Swedish place and
institution names (administrative entities, roads, hospitals, schools)
sourced from Wikidata (see wikidata_script.py).

Uses Aho-Corasick to scan text once regardless of gazetteer size — the
standard approach for matching many thousands of fixed strings against text
in a single pass, rather than searching the text once per gazetteer entry
(which would be far too slow at this scale).
"""

import ahocorasick
import pandas as pd

from entities import Entity, remove_overlapping_entities

# Place categories map to private_address — they're all specific, named
# real-world places, and none of our existing labels distinguish "hospital"
# from "street" from "school". Name categories map to private_person.
# Revisit and split these out further if a future guideline needs
# finer-grained categories.
CATEGORY_LABELS = {
    "Administrative_Entity": "private_address",
    "Street": "private_address",
    "Hospital": "private_address",
    "School": "private_address",
    "Given_Name": "private_person",
    "Family_Name": "private_person",
}

# Skip very short names — a 2-3 character gazetteer entry is far more likely
# to collide with an ordinary word than to be a meaningful match.
_MIN_NAME_LEN = 4

# Swedish institutions are almost always referred to informally by just their
# specific part — "Sahlgrenska" rather than the official "Sahlgrenska
# Universitetssjukhuset" — so exact-match alone misses the form people
# actually write. Registering the prefix before one of these generic
# category words as an additional pattern closes that gap. Trade-off: a
# short form can occasionally be a generic-sounding word on its own (a
# hospital literally named "X Sjukhus" where X is a common word) — the
# capitalization check in _is_plausible_match cuts a lot of that risk, but
# not all of it. Worth monitoring for false positives as the gazetteer sees
# more real documents.
_SHORT_FORM_SUFFIXES = {
    # Swedish
    "universitetssjukhuset", "universitetssjukhus", "sjukhuset", "sjukhus",
    "lasarettet", "lasarett",
    "kommunen", "kommun", "länet", "län",
    "skolan", "skola", "gymnasiet", "gymnasium", "högskolan", "högskola",
    "universitetet", "universitet",
    # English — some entries only have an English Wikidata label at all
    # (e.g. "Södertälje Hospital" with no Swedish label present), so the
    # Swedish suffix list alone would silently miss them.
    "hospital", "school", "municipality", "county", "university", "college",
}


# A short form that reduces to just one of these is almost certainly a
# generic modifier, not a distinctive place identifier — many towns have a
# "Västra skola"/"Norra skola" ("Western/Northern School") as a plain
# compass-direction name. This targets the actual problem (the word is
# generic) directly, rather than using length as an indirect proxy for it —
# length alone would also reject genuinely short, legitimate place names
# like "Gnesta" (as in "Gnesta Municipality" / "Gnesta kommun").
_GENERIC_SINGLE_WORDS = {
    "västra", "östra", "norra", "södra",  # compass directions
    "övre", "nedre",                       # upper/lower
    "stora", "lilla", "nya", "gamla",     # size/age
}


def _short_form(name: str) -> str | None:
    """
    Prefix before a trailing generic category word, e.g. "Sahlgrenska
    Universitetssjukhuset" -> "Sahlgrenska". None if there's no such suffix,
    or if the resulting single-word prefix is itself a generic modifier
    (see _GENERIC_SINGLE_WORDS) rather than a distinctive name.
    """
    words = name.split()
    if len(words) < 2 or words[-1].lower().strip(".,") not in _SHORT_FORM_SUFFIXES:
        return None
    short = " ".join(words[:-1]).strip()
    if not short or len(short) < _MIN_NAME_LEN:
        return None
    if len(words) == 2 and short.lower() in _GENERIC_SINGLE_WORDS:
        return None  # e.g. "Västra Skolan" -> "Västra" alone isn't a place
    return short


class GazetteerAgent:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.automaton = ahocorasick.Automaton()

        seen: set[str] = set()

        def register(name: str, label: str) -> None:
            key = name.lower()
            if key in seen:
                return  # first category to claim a name wins (all map to the same label today anyway)
            seen.add(key)
            self.automaton.add_word(key, (name, label))

        for _, row in df.iterrows():
            label = CATEGORY_LABELS.get(row.get("category", ""), "private_address")
            for name in (row.get("label_sv"), row.get("label_en")):
                if not isinstance(name, str):
                    continue
                name = name.strip()
                if len(name) >= _MIN_NAME_LEN:
                    register(name, label)

                short = _short_form(name)
                if short:
                    register(short, label)

        self.automaton.make_automaton()

    def detect(self, text: str) -> list[Entity]:
        lowered = text.lower()
        entities = []
        for end_idx, (matched_name, label) in self.automaton.iter(lowered):
            start = end_idx - len(matched_name) + 1
            end = end_idx + 1
            if not self._is_plausible_match(text, start, end):
                continue
            entities.append(Entity(
                text=text[start:end],
                label=label,
                start=start,
                end=end,
                source="gazetteer",
                confidence=0.9,
            ))
        return remove_overlapping_entities(entities)

    def _is_plausible_match(self, text: str, start: int, end: int) -> bool:
        """
        Aho-Corasick matches raw substrings — confirm this is a real word
        match (not part of a larger word) and that it's actually capitalized
        like the proper noun it's supposed to be, not a coincidental hit on
        an ordinary lowercase word that happens to share a gazetteer entry's
        spelling.
        """
        before_ok = start == 0 or not text[start - 1].isalnum()
        after_ok = end == len(text) or not text[end].isalnum()
        return before_ok and after_ok and text[start].isupper()
