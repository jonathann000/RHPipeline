"""
Swappable LLM backend for PII detection.

Supported backends (set via config["llm_configs"], a list — one entry runs
a single model, more than one runs an ensemble, see pipeline.py):
  - "llama"      : Llama 3.1 8B Instruct (multilingual baseline)
  - "mistral"    : Mistral 7B Instruct
  - "qwen"       : Qwen3 8B (native thinking mode support) or Qwen3 32B
  - "gemma"      : Gemma 2 9B Instruct or Gemma 2 27B Instruct

All backends use the same interface — swap by changing config only.

detect_direct=False (default): LLM only looks for quasi-identifiers
                               (used in "full" and "no_bert" modes)
detect_direct=True:            LLM handles all PII including direct identifiers
                               (used in "llm_only" mode, or "full"/"no_bert" with
                               llm_backstop enabled — see below)

scan_text=None (default):     the model sees `text` itself (today's
                               llm_only/no_bert behavior).
scan_text=<partial redaction>: the model sees a partially-redacted view
                               instead of the raw document (used by
                               llm_backstop and the judge retry loop) —
                               anything already redacted is invisible to the
                               model, so it can only report what's still
                               exposed, without needing a "skip already
                               found" instruction. `text` is still used to
                               resolve final offsets. Only meaningful when
                               detect_direct=True.

load_llm() caches loaded backends by (backend_name, model_path) so the
judge panel and the main detection LLM can share a checkpoint instead of
loading it twice.
"""

import json
import re
import logging
from abc import ABC, abstractmethod
from entities import Entity, remove_overlapping_entities
from redaction import PLACEHOLDERS
from chunking import chunk_by_sentences

logger = logging.getLogger(__name__)

# Built from the actual PLACEHOLDERS dict rather than hand-copied — a judge
# told about placeholders that don't match what redaction.py really produces
# would either flag legitimate placeholders as leaks or fail to recognize a
# real one as already-clean, and nothing would catch that drift.
_JUDGE_PLACEHOLDER_LIST = ", ".join(sorted(set(PLACEHOLDERS.values())) + ["[REDAKTERAD]"])


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Few-shot examples in Swedish clinical context
FEW_SHOT = """
Exempel 1:
Text: "Patienten Erik Svensson, 67-årig man bosatt i Lilla Edet, söker för svår bröstsmärta. Arbetar som snickare."
Quasi-identifierare:
[
  {"text": "67-årig man", "label": "demographics", "risk": "medium", "generalize": "65-70 år, man"},
  {"text": "Lilla Edet", "label": "private_address", "risk": "high", "generalize": "mindre ort i Västra Götaland"},
  {"text": "snickare", "label": "demographics", "risk": "low", "generalize": "hantverkare"}
]

Exempel 2:
Text: "Anna-Lena, 34 år, ensamstående med 3 barn, diagnostiserad med Huntingtons sjukdom 2019-03-12."
Quasi-identifierare:
[
  {"text": "34 år", "label": "demographics", "risk": "medium", "generalize": "30-40 år"},
  {"text": "ensamstående med 3 barn", "label": "demographics", "risk": "medium", "generalize": "ensamstående förälder"},
  {"text": "Huntingtons sjukdom", "label": "medical", "risk": "high", "generalize": "sällsynt neurologisk sjukdom"}
]

Exempel 3:
Text: "Bor tillsammans med sin make, Dr. Nilsson, en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset."
Quasi-identifierare:
[
  {"text": "en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset", "label": "social", "risk": "high", "generalize": "make med specialiserat läkaryrke vid större sjukhus"}
]
Notera: "Dr. Nilsson" är INTE med i listan — namnet i sig är en direkt
identifierare (hanteras av andra steg), inte en quasi-identifierare. Flagga
den BESKRIVANDE FRASEN (yrket + arbetsplatsen) separat, inte namnet den står
bredvid.

Exempel 4:
Text: "Patienten är en av mycket få personer med samisk bakgrund som bor kvar i kommunen."
Quasi-identifierare:
[
  {"text": "en av mycket få personer med samisk bakgrund som bor kvar i kommunen", "label": "demographics", "risk": "high", "generalize": "person med etnisk minoritetsbakgrund, ovanlig i kommunen"}
]

Exempel 5:
Text: "Patienten är aktiv i det lilla samhällets enda judiska församling."
Quasi-identifierare:
[
  {"text": "aktiv i det lilla samhällets enda judiska församling", "label": "social", "risk": "high", "generalize": "aktiv i en religiös minoritetsförsamling på orten"}
]

Exempel 6:
Text: "Patienten behandlas med Levofloxacin för en misstänkt sällsynt lungsjukdom."
Quasi-identifierare:
[
  {"text": "Levofloxacin", "label": "medication", "risk": "low", "generalize": null},
  {"text": "en misstänkt sällsynt lungsjukdom", "label": "medical", "risk": "high", "generalize": "en ovanlig lungsjukdom"}
]
Notera: läkemedelsnamnet flaggas (label: medication) men generaliseras INTE
— generalize är null och texten behålls oförändrad i dokumentet. Ett vanligt
läkemedelsnamn avslöjar inte i sig vem patienten är, till skillnad från en
sällsynt diagnos (label: medical), som fortfarande ska generaliseras.
"""

QUASI_ID_SYSTEM = """Du är ett system för att identifiera quasi-identifierare i svenska journalanteckningar.
Quasi-identifierare är uppgifter som ensamma kanske inte identifierar en person, men som i kombination med andra uppgifter kan göra det — särskilt i en liten svensk kommun.

Central fråga för VARJE uppgift du överväger, oavsett kategori: hur många
andra personer i en svensk kommun eller på detta sjukhus skulle troligen ha
exakt samma egenskap?
- Väldigt få (ett ovanligt eller framstående yrke, en sällsynt diagnos, en
  ovanlig kombination av fakta) — DÅ är det en quasi-identifierare, även om
  den inte liknar något exempel nedan.
- De flesta patienter med liknande vårdbehov (normala vitalparametrar som
  blodtryck/puls/andningsfrekvens inom normalvärden, vanliga sjukdomar som
  högt blodtryck eller depression, vanliga mediciner, normala eller
  negativa undersökningsfynd) — DÅ är det INTE en quasi-identifierare, hur
  specifikt eller tekniskt det än låter. Ett exakt tal (t.ex. "142/76" eller
  "leukocyter 19") är i sig INTE identifierande bara för att det är ett tal
  — fråga dig om just DETTA värde är ovanligt för denna typ av patient.

Detta gäller INTE bara patienten själv — ovanlig eller specifik information om
anhöriga (make/maka, förälder, barn) räknas också, eftersom den indirekt kan
identifiera patienten. Ett ovanligt eller framstående yrke hos en anhörig
("en framstående barnneurolog", "en välkänd hjärtkirurg", "kommunens enda
tandläkare") är en lika stark quasi-identifierare som samma uppgift om
patienten hade varit — oavsett vilket yrke eller vilken arbetsplats det
gäller i just detta fall.

Kategorier att leta efter:
- demographics: ålder, kön, etnicitet, yrke, familjesituation
- medical: sällsynta diagnoser, ovanliga ingrepp
- temporal: exakta datum, vårdtider, specifika tidpunkter
- private_address: klinik, avdelning, stadsdel, ort
- social: arbetsgivare, boendesituation, religiös tillhörighet, yrke eller titel hos anhöriga
- medication: läkemedelsnamn — flagga dessa så de syns i granskningsloggen,
  men generalize ska alltid vara null. Läkemedelsnamn är i sig sällan
  identifierande (vanliga mediciner som antibiotika eller astmainhalatorer
  säger inte vem patienten är) och är ofta viktiga att bevara oförändrade
  för analys- eller forskningssyften — de ska INTE ersättas med en
  läkemedelsklass eller generaliserad beskrivning.

Returnera ENBART giltig JSON — inga förklaringar, inga markdown-block.
Varje entitet ska ha: text, label, risk (low/medium/high), generalize (föreslagen generalisering, eller null för medication).
"""

# Used in llm_only / no_bert modes — LLM detects everything
FULL_DETECTION_SYSTEM = """Du är ett system för att identifiera ALL personlig och känslig information i svenska journalanteckningar, inklusive direkta identifierare och quasi-identifierare.

Direkta identifierare (hög risk — alltid maskera):
- private_person:  namn, titel
- private_email:   e-postadresser
- private_phone:   telefonnummer
- account_number:  personnummer, passnummer, körkort, kontonummer
- private_address: gatuadress, postnummer, stad
- private_date:    födelsedatum, specifika vårddatum
- secret:          lösenord, PIN-koder

Quasi-identifierare (kontextberoende risk) — gäller även ovanlig eller
specifik information om anhöriga (make/maka, förälder, barn), inte bara
patienten själv, eftersom det indirekt kan identifiera patienten.

Central fråga för VARJE uppgift du överväger, oavsett kategori: hur många
andra personer i en svensk kommun eller på detta sjukhus skulle troligen ha
exakt samma egenskap? Väldigt få (ett ovanligt eller framstående yrke — hos
patienten ELLER en anhörig — en sällsynt diagnos, en ovanlig kombination av
fakta) betyder att det är en quasi-identifierare, även om den inte liknar
något exempel nedan. De flesta patienter med liknande vårdbehov (normala
vitalparametrar, vanliga sjukdomar som högt blodtryck eller depression,
vanliga mediciner, normala eller negativa undersökningsfynd) betyder att
det INTE är en quasi-identifierare, hur specifikt eller tekniskt det än
låter — ett exakt tal är inte i sig identifierande bara för att det är ett
tal.

- demographics:    ålder, kön, etnicitet, yrke, familjesituation
- medical:         sällsynta diagnoser, ovanliga ingrepp
- temporal:        exakta tidpunkter, vårdlängd
- social:          arbetsgivare, boendesituation, religiös tillhörighet, yrke eller titel hos anhöriga
- medication:      läkemedelsnamn — flagga för granskningsloggen men generalize
                   ska alltid vara null; vanliga läkemedelsnamn är sällan
                   identifierande och är ofta viktiga att bevara oförändrade
                   för analys- eller forskningssyften

Returnera ENBART giltig JSON — inga förklaringar, inga markdown-block.
Varje entitet ska ha: text, label, risk (low/medium/high), generalize (föreslagen generalisering eller null för direkta identifierare/medication).
"""

FEW_SHOT_FULL = """
Exempel 1:
Text: "Patienten Erik Svensson, personnummer 850312-1234, tel 070-123 45 67, 67-årig man bosatt i Lilla Edet. Arbetar som snickare."
Entiteter:
[
  {"text": "Erik Svensson", "label": "private_person", "risk": "high", "generalize": null},
  {"text": "850312-1234", "label": "account_number", "risk": "high", "generalize": null},
  {"text": "070-123 45 67", "label": "private_phone", "risk": "high", "generalize": null},
  {"text": "67-årig man", "label": "demographics", "risk": "medium", "generalize": "65-70 år, man"},
  {"text": "Lilla Edet", "label": "private_address", "risk": "high", "generalize": "mindre ort i Västra Götaland"},
  {"text": "snickare", "label": "demographics", "risk": "low", "generalize": "hantverkare"}
]

Exempel 2:
Text: "Bor tillsammans med sin make, Dr. Nilsson, en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset."
Entiteter:
[
  {"text": "Dr. Nilsson", "label": "private_person", "risk": "high", "generalize": null},
  {"text": "en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset", "label": "social", "risk": "high", "generalize": "make med specialiserat läkaryrke vid större sjukhus"}
]

Exempel 3:
Text: "Patienten är en av mycket få personer med samisk bakgrund som bor kvar i kommunen."
Entiteter:
[
  {"text": "en av mycket få personer med samisk bakgrund som bor kvar i kommunen", "label": "demographics", "risk": "high", "generalize": "person med etnisk minoritetsbakgrund, ovanlig i kommunen"}
]

Exempel 4:
Text: "Patienten är aktiv i det lilla samhällets enda judiska församling."
Entiteter:
[
  {"text": "aktiv i det lilla samhällets enda judiska församling", "label": "social", "risk": "high", "generalize": "aktiv i en religiös minoritetsförsamling på orten"}
]

Exempel 5:
Text: "Patienten behandlas med Levofloxacin för en misstänkt sällsynt lungsjukdom."
Entiteter:
[
  {"text": "Levofloxacin", "label": "medication", "risk": "low", "generalize": null},
  {"text": "en misstänkt sällsynt lungsjukdom", "label": "medical", "risk": "high", "generalize": "en ovanlig lungsjukdom"}
]
Notera: läkemedelsnamnet flaggas (label: medication) men generaliseras INTE
— generalize är null och texten behålls oförändrad i dokumentet. Ett vanligt
läkemedelsnamn avslöjar inte i sig vem patienten är, till skillnad från en
sällsynt diagnos (label: medical), som fortfarande ska generaliseras.
"""

# Used for the judge retry pass — classifying specific already-flagged
# excerpts, NOT scanning for new ones. Deliberately its own prompt rather
# than reusing FULL_DETECTION_SYSTEM: that system prompt's whole framing
# ("identify ALL...") and few-shot example (which demonstrates scanning an
# entire document) pull the model back toward a full sweep even when the
# user message says "just classify these" — the system prompt matters more
# than the instruction layered on top of it.
TARGETED_CLASSIFY_SYSTEM = """Du är ett system som klassificerar EXAKT angivna textutdrag ur en svensk journalanteckning.
Du letar INTE efter ny information — en granskare har redan identifierat exakt vilka utdrag som är problematiska.

Kategorier:
Direkta identifierare (generalize ska alltid vara null):
- private_person, private_email, private_phone, account_number, private_address, private_date, secret
Quasi-identifierare (generalize: en riktig generalisering som INTE innehåller den ursprungliga texten):
- demographics, medical, temporal, social
Läkemedelsnamn (generalize ska alltid vara null — behålls oförändrat, flaggas bara för spårning):
- medication

Ditt enda jobb: klassificera VARJE angivet utdrag nedan. Lägg INTE till några
andra fynd, även om du ser annan information i den bifogade texten — texten
finns enbart som sammanhang för att avgöra rätt kategori och generalisering.

Returnera ENBART giltig JSON — exakt en post per angivet utdrag, i samma ordning:
[{"text": "utdraget exakt som angivet", "label": "...", "risk": "low/medium/high", "generalize": "... eller null"}]
"""

TARGETED_CLASSIFY_FEW_SHOT = """
Exempel:
Angivna utdrag:
1. "Erik Svensson"
2. "Lilla Edet"
Text (sammanhang): "Patienten Erik Svensson bosatt i Lilla Edet."
Entiteter:
[
  {"text": "Erik Svensson", "label": "private_person", "risk": "high", "generalize": null},
  {"text": "Lilla Edet", "label": "private_address", "risk": "high", "generalize": "mindre ort i Västra Götaland"}
]

Exempel 2 (quasi-identifierare, inte bara direkta):
Angivna utdrag:
1. "en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset"
Text (sammanhang): "Bor tillsammans med sin make, Dr. Nilsson, en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset."
Entiteter:
[
  {"text": "en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset", "label": "social", "risk": "high", "generalize": "make med specialiserat läkaryrke vid större sjukhus"}
]
"""

# Used by the judge panel — a different task from detection: audit a
# document that's already been redacted, rather than enumerate entities.
JUDGE_SYSTEM = f"""Du är en integritetsgranskare som kontrollerar avidentifierade svenska journalanteckningar.
Dokumentet du får har redan genomgått avidentifiering — vissa uppgifter är ersatta med platshållare:
{_JUDGE_PLACEHOLDER_LIST}.

Dessa platshållare räknas som redan rena — flagga dem ALDRIG.

Din uppgift: hitta text som INTE redan är maskerad men som ändå avslöjar vem patienten eller
andra namngivna personer i texten är. Detta gäller två typer av kvarvarande information:
1. Direkta identifierare: namn, adresser, telefonnummer, e-post, personnummer, eller annan
   direkt identifierande uppgift som blivit kvar av misstag.
2. Quasi-identifierare som är för specifika för att ha generaliserats bort: t.ex. ett ovanligt
   eller framstående yrke hos patienten eller en anhörig, en sällsynt diagnos, eller en specifik
   ort/institution — sådant som i kombination med annat kan avslöja vem patienten är, särskilt
   i en liten svensk kommun.

Läkemedelsnamn som förekommer oförändrat i texten (t.ex. "Levofloxacin", "Aspirin") är INTE ett
fynd — de flaggas medvetet inte bort, eftersom de sällan är identifierande i sig och är viktiga
att bevara för analys. Flagga dem ALDRIG.

Returnera ENBART giltig JSON — en lista, inga förklaringar utanför JSON.
Om dokumentet är rent: returnera en tom lista [].
Om något avslöjande kvarstår, en post per fynd:
[{{"quote": "exakt textutdrag som fortfarande avslöjar identitet", "reason": "kort motivering"}}]
"""

JUDGE_FEW_SHOT = """
Exempel 1 (rent dokument):
Text: "Patienten [PERSON], [DEMOGRAFISK-UPPGIFT], sökte för huvudvärk. Bosatt i [ADRESS]."
Bedömning:
[]

Exempel 2 (kvarvarande läckage, direkt identifierare):
Text: "Ansvarig läkare: Dr. Helena Björk, överläkare neurologi. Patienten [PERSON] har [DATUM]."
Bedömning:
[{"quote": "Dr. Helena Björk", "reason": "läkarens fullständiga namn är inte maskerat"}]

Exempel 3 (kvarvarande läckage, quasi-identifierare):
Text: "Patienten [PERSON] bor med sin make, [PERSON], en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset."
Bedömning:
[{"quote": "en framstående barnneurolog vid Sahlgrenska Universitetssjukhuset", "reason": "ovanligt specifikt yrke och arbetsplats för en anhörig kan indirekt identifiera patienten"}]
"""

# Two distinct reasoning-block delimiter conventions seen in practice:
# Qwen3 wraps it in <think>...</think>; Gemma 4 opens a channel with
# <|channel>thought and closes it with <channel|> (confirmed directly
# against google/gemma-4-12B-it's own chat template — it inserts this
# unconditionally, even when enable_thinking=False is passed). Each
# backend uses one or the other, never both, but there's no cheap way to
# know which ahead of a given raw response, so both patterns are always
# checked.
_THINKING_BLOCK_PATTERNS = [
    r"<think>(.*?)</think>",
    r"<\|channel>thought(.*?)<channel\|>",
]


def _extract_thinking(raw: str) -> str | None:
    """
    Pull out a reasoning block's content, if present — call this before
    _parse_llm_json, which strips the same block to get at the JSON
    payload. Used to persist the model's reasoning for explainability
    rather than just discarding it once parsing is done.
    """
    for pattern in _THINKING_BLOCK_PATTERNS:
        match = re.search(pattern, raw, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _parse_llm_json(raw: str) -> list[dict]:
    """
    Robustly extract entities from LLM output.
    Long documents can push generation past max_new_tokens, cutting the JSON
    array off mid-entity — salvage whichever top-level objects are individually
    complete rather than discarding the whole batch.
    """
    # Strip a reasoning block (see _THINKING_BLOCK_PATTERNS — Qwen3's
    # <think>...</think>, or Gemma 4's <|channel>thought...<channel|>) and
    # markdown fences before looking for the entity array, so stray
    # brackets in the model's own reasoning text (e.g. describing the JSON
    # shape it's about to produce) don't get mistaken for the JSON payload
    # itself, or corrupt the brace-matching salvage fallback below.
    for pattern in _THINKING_BLOCK_PATTERNS:
        raw = re.sub(pattern, "", raw, flags=re.DOTALL)
    raw = re.sub(r"```json|```", "", raw).strip()

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass  # malformed (likely truncated) — fall through to salvage below

    objects = [
        json.loads(m.group())
        for m in re.finditer(r"\{[^{}]*\}", raw)
        if _is_valid_json(m.group())
    ]
    if objects:
        logger.warning(f"LLM JSON output was truncated/malformed — salvaged {len(objects)} complete entities")
    else:
        logger.warning(f"Failed to parse LLM JSON output — raw length: {len(raw)} chars")
    return objects


def _is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


# A single character (e.g. the LLM reporting "f" for a "Sex: F" field) is
# never a safe redaction target on its own — a plain substring search for it
# is near-guaranteed to land on an unrelated occurrence elsewhere in the
# document (e.g. the "f" in "of"), silently corrupting unrelated text.
_MIN_ENTITY_LEN = 2

# Target chunk size (characters) for quasi-identifier detection sweeps —
# not a technical necessity like BERT's hard context limit (this model can
# read the whole document in one pass), but an empirically-motivated
# choice: the same prompt that reliably caught a quasi-identifier in a
# short, isolated test sentence missed it 4/4 times embedded in a ~12KB
# document — real evidence that a narrower per-call scope measurably helps
# recall even though nothing forces it. A first attempt at 800 chars was
# too narrow in practice — cutting a vitals/lab-values section down to a
# couple of bare sentences removed the surrounding context that signals
# "this is routine clinical data," and the model started flagging nearly
# every bare number as some quasi-identifier category instead. Sized here
# to data/notes.txt's full length (~1800 chars) — the original short test
# document this pipeline was tuned against from the start, and a size
# large enough to keep a full clinical section's context together.
_QUASI_CHUNK_CHARS = 1800


def _word_boundaries_ok(text: str, start: int, end: int) -> bool:
    """
    True if [start, end) isn't glued to more word characters on either side —
    same check as gazetteer_agent.py's _is_plausible_match. An exact
    substring match can still be wrong: the LLM occasionally quotes only the
    head noun of a Swedish compound (e.g. "neurolog" instead of the intended
    "barnneurolog"), and a plain text.find() then happily matches that
    fragment inside a completely unrelated word elsewhere in the document
    (e.g. "neurologiska" in a symptom list) — silently redacting the wrong
    sentence while leaving the real target untouched. Rejecting a
    mid-word match forces the search to keep looking for an actual
    whole-word/phrase occurrence instead of accepting the first coincidental
    substring hit.
    """
    before_ok = start == 0 or not text[start - 1].isalnum()
    after_ok = end == len(text) or not text[end].isalnum()
    return before_ok and after_ok


def _resolve_offsets(
    text: str, entity_text: str, claimed: set[tuple[int, int]]
) -> tuple[int, int] | None:
    """
    Find char offsets of entity_text in text, skipping spans already claimed
    by an earlier entity in this batch. The LLM can legitimately report the
    same phrase for two distinct occurrences (e.g. a diagnosis mentioned
    twice) — each should resolve to its own occurrence, not collapse onto
    the first match found.
    """
    if not entity_text or len(entity_text) < _MIN_ENTITY_LEN:
        return None
    search_from = 0
    while True:
        idx = text.find(entity_text, search_from)
        if idx == -1:
            return _fuzzy_find(text, entity_text, claimed)
        span = (idx, idx + len(entity_text))
        if span not in claimed and _word_boundaries_ok(text, *span):
            return span
        search_from = idx + 1


# Minimum similarity (difflib.SequenceMatcher ratio) for a fuzzy match to be
# trusted — high enough that it only catches near-verbatim reproductions
# (a transcription typo, not a genuinely different phrase).
_FUZZY_MATCH_THRESHOLD = 0.85


def _fuzzy_find(
    text: str, entity_text: str, claimed: set[tuple[int, int]]
) -> tuple[int, int] | None:
    """
    Fallback when no exact substring match exists — the LLM occasionally
    reproduces a span with a minor, consistent transcription error (e.g. an
    inserted/transposed letter in a long compound word) rather than
    verbatim. This isn't sampling noise to retry past — it's reproducible
    across calls, so an exact match will never appear.

    Finds the longest exactly-matching block shared between the document
    and entity_text (almost always most of the phrase, since a typo is
    usually a single localized edit) and uses its position as an anchor to
    estimate the full span, rather than assuming entity_text's length
    exactly matches the real span — an inserted/deleted character means it
    won't. Snaps both boundaries outward to the nearest word boundary so a
    length mismatch from the edit can't leave the span starting or ending
    mid-word into unrelated, legitimate text.
    """
    import difflib

    n = len(entity_text)
    if n == 0 or n > len(text):
        return None

    matcher = difflib.SequenceMatcher(None, text, entity_text, autojunk=False)
    match = matcher.find_longest_match(0, len(text), 0, n)
    if match.size < n // 2:
        return None  # too weak an anchor to trust

    start = max(0, match.a - match.b)
    end = min(len(text), start + n)

    while 0 < start < len(text) and text[start - 1].isalnum() and text[start].isalnum():
        start += 1
    while 0 < end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end += 1

    # entity_text's length is only an estimate of the real span — an
    # inserted/deleted character in the typo means it can overshoot by a
    # character or two onto trailing punctuation/whitespace that belongs to
    # the surrounding sentence, not the phrase itself (e.g. swallowing the
    # period that ends the sentence). Trim those back off the edges.
    while end > start and not text[end - 1].isalnum():
        end -= 1
    while start < end and not text[start].isalnum():
        start += 1

    span = (start, end)
    if span in claimed or start >= end:
        return None

    ratio = difflib.SequenceMatcher(None, text[start:end], entity_text).ratio()
    if ratio >= _FUZZY_MATCH_THRESHOLD:
        return span
    return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    # Every concrete backend must set these (HuggingFaceLLMBackend and
    # MockLLMBackend both do, in __init__) — declared here so callers typed
    # against the abstract base (e.g. pipeline.py's self.llms: list[LLMBackend])
    # can reference them.
    backend_name: str
    # Populated only when enable_thinking is on and the model actually
    # produced a <think> block; see detect()/judge().
    reasoning_log: list[dict]

    @abstractmethod
    def detect(
        self,
        text: str,
        existing: list[Entity],
        detect_direct: bool = False,
        enable_thinking: bool = False,
        scan_text: str | None = None,
        target_quotes: list[str] | None = None,
    ) -> list[Entity]:
        """
        text:                    original document — always used to resolve entity offsets
        detect_direct=False:     quasi-identifiers only (used alongside BERT/rules)
        detect_direct=True:      all PII including direct identifiers (llm_only mode,
                                  or backstopping a prior pass — see scan_text)
        enable_thinking=True:    ask the model to reason in a <think> block before
                                  answering (only meaningful on backends that support
                                  it, e.g. Qwen3 — ignored by everything else)
        scan_text:               what's shown to the model (defaults to `text`). Pass
                                  a partially-redacted version to backstop a prior
                                  pass — already-redacted spans are invisible to the
                                  model, so it can only report what's still exposed
        target_quotes:           specific excerpts to classify (e.g. a judge's flags)
                                  instead of a blind full sweep — takes priority over
                                  detect_direct's prompt selection when given
        """
        pass

    @abstractmethod
    def judge(self, redacted_text: str, enable_thinking: bool = False) -> list[dict]:
        """
        Audit an already-redacted document for residual PII — a different
        task from detect(): pass/fail on the final text, not enumeration.
        Returns a list of {"quote": ..., "reason": ...} dicts; empty means
        the judge considers the document clean.
        """
        pass


# ---------------------------------------------------------------------------
# HuggingFace backend (Llama / Mistral / Qwen / Gemma — all use same interface)
# ---------------------------------------------------------------------------

class HuggingFaceLLMBackend(LLMBackend):
    """
    Generic HuggingFace causal LM backend.
    Works with Llama, Mistral, Qwen, Gemma, and most instruct-tuned models.
    """

    def __init__(self, backend_name: str, model_path: str, approx_params_b: float = 9.0):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline as hf_pipeline
        from device import resolve_device_map, resolve_quantization_config

        self.backend_name = backend_name
        logger.info(f"Loading LLM: {model_path}")

        # clean_up_tokenization_spaces=True (the old default) is destructive
        # for BPE tokenizers and warns on every decode — set explicitly
        # rather than accept the warning on every single generation call.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        # Checks the GPU actually detected at runtime against this model's
        # approximate size, and quantizes to 8-bit only if needed to fit —
        # e.g. a ~32B model needs this on a 40GB card but not an 80GB one.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=resolve_device_map(),
            quantization_config=resolve_quantization_config(approx_params_b),
        )
        self.model.eval()

        self.pipe = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        # Prints generated tokens live to the terminal as they're produced —
        # otherwise a 30-90s generation call looks like a hang.
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Populated only when enable_thinking is on and the model actually
        # produced a <think> block — see _detect_once/judge. Explainability
        # trail for quasi-identifier decisions, not used by detection logic
        # itself. Cleared at the start of each PIIPipeline.run() (a cached
        # backend can be reused across multiple documents in one process),
        # so it always reflects only the most recent run.
        self.reasoning_log: list[dict] = []

    def _build_prompt(
        self,
        text: str,
        existing: list[Entity],
        detect_direct: bool,
        enable_thinking: bool = False,
        target_quotes: list[str] | None = None,
    ) -> str:
        if target_quotes:
            # Judge retry: don't re-run a blind full sweep and hope sampling
            # surfaces the same miss again — give the model the judge's
            # exact flagged excerpts and ask it to classify each one. Uses
            # its own narrow prompt (not FULL_DETECTION_SYSTEM) — see
            # TARGETED_CLASSIFY_SYSTEM's docstring for why that matters.
            system = TARGETED_CLASSIFY_SYSTEM
            quotes_list = "\n".join(f'{i+1}. "{q}"' for i, q in enumerate(target_quotes))
            user_msg = (
                f"{TARGETED_CLASSIFY_FEW_SHOT}\n"
                f"Angivna utdrag:\n{quotes_list}\n\n"
                f"Text (sammanhang — klassificera INTE något annat än utdragen ovan):\n\"{text}\"\n\n"
                f"Entiteter:"
            )
        elif detect_direct:
            # llm_only mode — detect everything from scratch.
            # Also used for backstop mode: the caller passes already-redacted
            # text here (see PIIPipeline), so anything rules/BERT already
            # caught is literally gone from what the model sees — no "skip
            # already found" instruction needed, since there's nothing left
            # to duplicate. Models weren't reliably following that instruction
            # anyway; removing the need for it is more robust than asking nicer.
            system = FULL_DETECTION_SYSTEM
            user_msg = (
                f"{FEW_SHOT_FULL}\n"
                f"Identifiera ALLA direkta identifierare och quasi-identifierare.\n\n"
                f"Text: \"{text}\"\n"
                f"Entiteter:"
            )
        else:
            # Hybrid mode — only quasi-identifiers, rules/BERT already ran
            already_found = ", ".join(set(e.label for e in existing)) or "inga ännu"
            system = QUASI_ID_SYSTEM
            user_msg = (
                f"{FEW_SHOT}\n"
                f"Redan identifierade entiteter: {already_found}\n"
                f"Identifiera ENBART quasi-identifierare som inte redan täcks.\n\n"
                f"Text: \"{text}\"\n"
                f"Quasi-identifierare:"
            )

        return self._apply_chat_template(system, user_msg, enable_thinking)

    def _apply_chat_template(
        self, system: str, user_msg: str, enable_thinking: bool = False
    ) -> str:
        """
        Use the tokenizer's own chat template rather than a hand-maintained
        per-model format string — every instruct-tuned checkpoint ships one,
        so a newly added backend works with zero changes here. Some
        templates (e.g. Gemma) reject a separate system role — fall back to
        folding it into the user turn, same as that model expects natively.

        enable_thinking is a Qwen3-specific chat-template kwarg; templates
        that don't reference it (everything except Qwen3) simply ignore it.
        """
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:
            messages = [{"role": "user", "content": f"{system}\n\n{user_msg}"}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

    def detect(
        self,
        text: str,
        existing: list[Entity],
        detect_direct: bool = False,
        enable_thinking: bool = False,
        scan_text: str | None = None,
        target_quotes: list[str] | None = None,
    ) -> list[Entity]:
        """
        text: original document — always used to resolve entity offsets.
        scan_text: what's actually shown to the model (defaults to `text`).
                   Pass a partially-redacted version to backstop a prior
                   pass — anything already redacted is invisible to the
                   model, so it can't re-derive duplicates of what's already
                   been found; it can only report what's still exposed.
        target_quotes: specific excerpts to classify (e.g. from a judge's
                   flags), instead of a blind full sweep. Takes priority
                   over detect_direct's prompt selection when given.

        Quasi-only blind sweeps (not target_quotes, not detect_direct) are
        split into sentence-group chunks once the scanned text exceeds
        _QUASI_CHUNK_CHARS — see that constant's comment for why. Targeted
        classification and full detect_direct sweeps are never chunked:
        the former is already scoped to specific excerpts, and there's no
        evidence yet the latter needs it — direct identifiers like names
        and dates are far more mechanically obvious regardless of how much
        surrounding text competes for the model's attention.
        """
        scanned_text = scan_text if scan_text is not None else text

        if not target_quotes and not detect_direct and len(scanned_text) > _QUASI_CHUNK_CHARS:
            claimed_spans: set[tuple[int, int]] = set()
            entities = []
            for start, end in chunk_by_sentences(scanned_text, _QUASI_CHUNK_CHARS):
                entities.extend(self._detect_once(
                    text, scanned_text[start:end], existing, detect_direct,
                    enable_thinking, target_quotes=None, claimed_spans=claimed_spans,
                ))
            # Chunks are non-overlapping by construction, so this doesn't
            # dedupe chunk-vs-chunk — it catches an entity that also got
            # separately found by rules/BERT/a prior pass (all now resolved
            # together downstream too, but this keeps this stage's own
            # output clean for logging/testing in isolation).
            return remove_overlapping_entities(entities)

        return self._detect_once(
            text, scanned_text, existing, detect_direct, enable_thinking,
            target_quotes, claimed_spans=set(),
        )

    def _detect_once(
        self,
        text: str,
        scanned_text: str,
        existing: list[Entity],
        detect_direct: bool,
        enable_thinking: bool,
        target_quotes: list[str] | None,
        claimed_spans: set[tuple[int, int]],
    ) -> list[Entity]:
        """
        A single model call over `scanned_text` — the whole document when
        not chunking, or one chunk's substring when chunking. Offsets are
        always resolved against the full `text`, never `scanned_text`
        directly: `_resolve_offsets`/`_fuzzy_find` do a content-based
        search (substring/fuzzy match), not a position-based one, so they
        don't need to know which slice of the document the model was
        actually looking at — they just need the full text to search
        within. `claimed_spans` is threaded in (rather than always starting
        fresh) so that across multiple chunk calls, two genuinely distinct
        occurrences of the same repeated phrase each still resolve to their
        own position instead of every chunk independently collapsing onto
        the first occurrence in the whole document.
        """
        prompt = self._build_prompt(
            scanned_text,
            existing, detect_direct, enable_thinking, target_quotes,
        )

        input_tokens = None  # computed lazily below, only when actually needed

        if target_quotes:
            # A handful of concrete items to classify — modest budget, but
            # scaled up a bit for judge rounds that flag many things at once.
            max_new_tokens = max(512, 200 * len(target_quotes))
        else:
            # A full-document (or chunk) sweep's JSON output grows with the
            # scanned text itself — a fixed ceiling silently truncates
            # longer input mid-array (each entity costs real generation
            # tokens, and more input just has more entities' worth of JSON
            # to emit than the short documents this was originally tuned
            # against). Scale off the actual tokenized input length rather
            # than a flat number — correctness matters more than shaving a
            # few generation tokens off the common case, and the model
            # still stops early via EOS once it's actually done, so a
            # generous ceiling doesn't cost anything when the output is
            # short. detect_direct enumerates ~2x the entries of
            # quasi-only, so it gets double the multiplier.
            input_tokens = len(self.tokenizer.encode(scanned_text))
            max_new_tokens = max(2048, input_tokens * 2) if detect_direct else max(512, input_tokens)

        # A reasoning block consumes real generation budget before the
        # answer even starts, on top of whatever the answer itself needs —
        # this used to be gated on our own enable_thinking flag, which
        # silently starved longer documents whenever a model reasons
        # regardless of what we asked for: Gemma 4's chat template opens a
        # "thought" channel unconditionally (confirmed directly — it's
        # still there with enable_thinking=False passed), unlike Qwen3
        # where the kwarg genuinely toggles it. A model that reasoned
        # anyway with no headroom budgeted for it ran out of tokens mid
        # JSON array on a real document (see llm_backend:_parse_llm_json's
        # salvage warning). Applying this unconditionally, not just when
        # enable_thinking=True, costs nothing for a model that doesn't
        # reason by default — it still stops early via EOS once actually
        # done, so a generous ceiling it never needs is free; it just also
        # correctly covers a model that reasons whether we ask it to or
        # not. Same headroom sizing as before: scaled by input length for
        # the same reason the base budget is — a longer document needs
        # more room to reason about, not a flat allowance sized for
        # whatever document this was last tuned against.
        if input_tokens is None:
            input_tokens = len(self.tokenizer.encode(scanned_text))
        max_new_tokens += max(4096, input_tokens * 2)

        generated = self._generate(prompt, max_new_tokens)
        if enable_thinking:
            reasoning = _extract_thinking(generated)
            if reasoning:
                self.reasoning_log.append({
                    "stage": "detect",
                    "backend": self.backend_name,
                    "scanned_text": scanned_text,
                    "reasoning": reasoning,
                })
        raw_entities = _parse_llm_json(generated)

        entities = []
        for ent in raw_entities:
            offsets = _resolve_offsets(text, ent.get("text", ""), claimed_spans)
            if offsets is None:
                logger.debug(f"Skipping hallucinated span: {ent.get('text')}")
                continue
            claimed_spans.add(offsets)

            entities.append(Entity(
                text=ent["text"],
                label=ent.get("label", "unknown"),
                start=offsets[0],
                end=offsets[1],
                source="llm",
                confidence=0.8,
                generalized=ent.get("generalize"),
                risk=ent.get("risk", "medium"),
            ))

        return entities

    def judge(self, redacted_text: str, enable_thinking: bool = False) -> list[dict]:
        user_msg = f"{JUDGE_FEW_SHOT}\nText: \"{redacted_text}\"\nBedömning:"
        prompt = self._apply_chat_template(JUDGE_SYSTEM, user_msg, enable_thinking)

        # Same rationale as _detect_once: applied unconditionally, not just
        # when enable_thinking=True, since some models (Gemma 4) reason
        # regardless of what we ask — see the long comment there.
        max_new_tokens = 512
        input_tokens = len(self.tokenizer.encode(redacted_text))
        max_new_tokens += max(4096, input_tokens * 2)
        generated = self._generate(prompt, max_new_tokens)
        if enable_thinking:
            reasoning = _extract_thinking(generated)
            if reasoning:
                self.reasoning_log.append({
                    "stage": "judge",
                    "backend": self.backend_name,
                    "scanned_text": redacted_text,
                    "reasoning": reasoning,
                })
        return _parse_llm_json(generated)

    def _generate(self, prompt: str, max_new_tokens: int, stream: bool = False) -> str:
        """
        Run generation via an explicit GenerationConfig — passing loose
        max_new_tokens/temperature/... kwargs alongside the model's own
        generation_config.json triggers a deprecation warning on every
        single call otherwise.

        stream=False by default: TextStreamer's only purpose is printing
        tokens live so a 30-90s generation call doesn't look like a hang
        during interactive local development (see self.streamer's comment)
        — real, if modest, overhead (incremental decode + stdout flush on
        every token) for zero benefit on a scripted/server run where
        nobody's watching the terminal live, and piped/redirected output
        (a log file, a non-interactive SSH session) can make that overhead
        considerably worse than on a local interactive tty. Pass
        stream=True explicitly for local interactive debugging if wanted.
        """
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.1,     # low temp for consistent structured output
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            # Hard block on repeating an exact 24-token run — guards against
            # the model falling into a degenerate loop re-emitting the same
            # JSON entity forever instead of finishing (measured: the fixed
            # JSON scaffold shared between two DIFFERENT entities with the
            # same label/risk is ~20 tokens, a real duplicated entity line
            # is ~43 — 24 sits between the two, so two distinct entities
            # that happen to share a label/risk/generalize combo can still
            # both be emitted, but the same entity repeating verbatim gets
            # cut off almost immediately instead of consuming the whole
            # generation budget).
            no_repeat_ngram_size=24,
        )
        output = self.pipe(prompt, generation_config=gen_config, streamer=self.streamer if stream else None)
        return output[0]["generated_text"][len(prompt):]


# ---------------------------------------------------------------------------
# Mock backend (for local testing without GPU/models)
# ---------------------------------------------------------------------------

class MockLLMBackend(LLMBackend):
    """Returns hardcoded quasi-identifiers for pipeline integration testing."""

    def __init__(self):
        self.backend_name = "mock"
        # Always empty — no real model, so nothing to reason about. Present
        # so code that reads self.llm.reasoning_log works the same
        # regardless of which backend is loaded.
        self.reasoning_log: list[dict] = []

    def detect(
        self,
        text: str,
        existing: list[Entity],
        detect_direct: bool = False,
        enable_thinking: bool = False,
        scan_text: str | None = None,
        target_quotes: list[str] | None = None,
    ) -> list[Entity]:
        entities = []
        # Simple keyword scan to simulate LLM detection
        keywords = {
            "snickare": ("demographics", "medium", "hantverkare"),
            "ensamstående": ("demographics", "medium", "familjesituation"),
            "Huntingtons": ("medical", "high", "sällsynt neurologisk sjukdom"),
        }
        claimed_spans: set[tuple[int, int]] = set()
        for kw, (label, risk, generalize) in keywords.items():
            offsets = _resolve_offsets(text, kw, claimed_spans)
            if offsets:
                entities.append(Entity(
                    text=kw,
                    label=label,
                    start=offsets[0],
                    end=offsets[1],
                    source="llm",
                    confidence=0.8,
                    generalized=generalize,
                    risk=risk,
                ))
        return entities

    def judge(self, redacted_text: str, enable_thinking: bool = False) -> list[dict]:
        """Always reports clean — no model to actually audit with."""
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_loaded_backends: dict[tuple[str, str], LLMBackend] = {}


def load_llm(backend_name: str, model_path: str, approx_params_b: float = 9.0) -> LLMBackend:
    """
    Load the appropriate LLM backend.

    backend_name options:
      "llama"    -> meta-llama/Meta-Llama-3.1-8B-Instruct
      "mistral"  -> mistralai/Mistral-7B-Instruct-v0.3
      "qwen"     -> Qwen/Qwen3-8B (or Qwen/Qwen3-32B — same backend, bigger checkpoint)
      "gemma"    -> google/gemma-2-9b-it (or google/gemma-2-27b-it — same backend, bigger checkpoint)
      "mock"     -> MockLLMBackend (no model, for testing)

    model_path can be a HuggingFace model ID or a local directory path.

    approx_params_b: rough parameter count in billions, used to decide
    whether this checkpoint needs 8-bit quantization on the GPU actually
    detected at runtime — see device.resolve_quantization_config().

    Cached by (backend_name, model_path) — calling this twice for the same
    checkpoint (e.g. the main detection LLM also being used as a judge)
    returns the already-loaded instance instead of loading it again.
    """
    if backend_name == "mock":
        return MockLLMBackend()

    supported = {"llama", "mistral", "qwen", "gemma"}
    if backend_name not in supported:
        raise ValueError(f"Unknown backend '{backend_name}'. Choose from: {supported}")

    key = (backend_name, model_path)
    if key not in _loaded_backends:
        _loaded_backends[key] = HuggingFaceLLMBackend(backend_name, model_path, approx_params_b)
    return _loaded_backends[key]
