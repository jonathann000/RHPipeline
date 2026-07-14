"""
Swappable LLM backend for PII detection.

Supported backends (set via config["llm_backend"]):
  - "llama"      : Llama 3.1 8B Instruct (multilingual baseline)
  - "mistral"    : Mistral 7B Instruct
  - "qwen"       : Qwen3 8B (native thinking mode support)
  - "gemma"      : Gemma 2 9B Instruct

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
from entities import Entity

logger = logging.getLogger(__name__)


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
"""

QUASI_ID_SYSTEM = """Du är ett system för att identifiera quasi-identifierare i svenska journalanteckningar.
Quasi-identifierare är uppgifter som ensamma kanske inte identifierar en person, men som i kombination med andra uppgifter kan göra det — särskilt i en liten svensk kommun.

Detta gäller INTE bara patienten själv — ovanlig eller specifik information om
anhöriga (make/maka, förälder, barn) räknas också, eftersom den indirekt kan
identifiera patienten. Ett ovanligt eller framstående yrke hos en anhörig
("en framstående barnneurolog", "kommunens enda tandläkare") är en lika stark
quasi-identifierare som samma uppgift om patienten hade varit.

Kategorier att leta efter:
- demographics: ålder, kön, etnicitet, yrke, familjesituation
- medical: sällsynta diagnoser, ovanliga ingrepp, specifika läkemedel
- temporal: exakta datum, vårdtider, specifika tidpunkter
- private_address: klinik, avdelning, stadsdel, ort
- social: arbetsgivare, boendesituation, religiös tillhörighet, yrke eller titel hos anhöriga

Returnera ENBART giltig JSON — inga förklaringar, inga markdown-block.
Varje entitet ska ha: text, label, risk (low/medium/high), generalize (föreslagen generalisering).
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
patienten själv, eftersom det indirekt kan identifiera patienten:
- demographics:    ålder, kön, etnicitet, yrke, familjesituation
- medical:         sällsynta diagnoser, ovanliga ingrepp, specifika läkemedel
- temporal:        exakta tidpunkter, vårdlängd
- social:          arbetsgivare, boendesituation, religiös tillhörighet, yrke eller titel hos anhöriga

Returnera ENBART giltig JSON — inga förklaringar, inga markdown-block.
Varje entitet ska ha: text, label, risk (low/medium/high), generalize (föreslagen generalisering eller null för direkta identifierare).
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
"""

# Used by the judge panel — a different task from detection: audit a
# document that's already been redacted, rather than enumerate entities.
JUDGE_SYSTEM = """Du är en integritetsgranskare som kontrollerar avidentifierade svenska journalanteckningar.
Dokumentet du får har redan genomgått avidentifiering — vissa uppgifter är ersatta med platshållare:
[PERSON], [ADRESS], [TELEFON], [E-POST], [ID-NUMMER], [DATUM], [HEMLIG-UPPGIFT],
[DEMOGRAFISK-UPPGIFT], [MEDICINSK-UPPGIFT], [TIDPUNKT], [SOCIAL-UPPGIFT], [REDAKTERAD].

Dessa platshållare räknas som redan rena — flagga dem ALDRIG.

Din enda uppgift: hitta text som INTE redan är maskerad men som ändå avslöjar vem patienten
eller andra namngivna personer i texten är — t.ex. namn, adresser, telefonnummer, e-post,
personnummer, eller annan direkt identifierande uppgift som blivit kvar av misstag.

Returnera ENBART giltig JSON — en lista, inga förklaringar utanför JSON.
Om dokumentet är rent: returnera en tom lista [].
Om något avslöjande kvarstår, en post per fynd:
[{"quote": "exakt textutdrag som fortfarande avslöjar identitet", "reason": "kort motivering"}]
"""

JUDGE_FEW_SHOT = """
Exempel 1 (rent dokument):
Text: "Patienten [PERSON], [DEMOGRAFISK-UPPGIFT], sökte för huvudvärk. Bosatt i [ADRESS]."
Bedömning:
[]

Exempel 2 (kvarvarande läckage):
Text: "Ansvarig läkare: Dr. Helena Björk, överläkare neurologi. Patienten [PERSON] har [DATUM]."
Bedömning:
[{"quote": "Dr. Helena Björk", "reason": "läkarens fullständiga namn är inte maskerat"}]
"""

def _parse_llm_json(raw: str) -> list[dict]:
    """
    Robustly extract entities from LLM output.
    Long documents can push generation past max_new_tokens, cutting the JSON
    array off mid-entity — salvage whichever top-level objects are individually
    complete rather than discarding the whole batch.
    """
    # Strip a reasoning block (e.g. Qwen3's <think>...</think>) and markdown
    # fences before looking for the entity array, so stray brackets in the
    # model's reasoning text don't get mistaken for the JSON payload.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
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
        if span not in claimed:
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
        """
        scanned_text = scan_text if scan_text is not None else text
        prompt = self._build_prompt(
            scanned_text,
            existing, detect_direct, enable_thinking, target_quotes,
        )

        if target_quotes:
            # A handful of concrete items to classify — modest budget, but
            # scaled up a bit for judge rounds that flag many things at once.
            max_new_tokens = max(512, 200 * len(target_quotes))
        elif enable_thinking:
            # A <think> block consumes generation budget before the actual
            # answer — reasoning models can use 5-20x more tokens per
            # response than non-reasoning ones, so this needs real headroom.
            max_new_tokens = 4096
        else:
            # A full-document sweep's JSON output grows with the document
            # itself — a fixed ceiling silently truncates longer documents
            # mid-array (each entity costs real generation tokens, and a
            # long document just has more entities' worth of JSON to emit
            # than the short documents this was originally tuned against).
            # Scale off the actual tokenized input length rather than a
            # flat number — correctness matters more than shaving a few
            # generation tokens off the common case, and the model still
            # stops early via EOS once it's actually done, so a generous
            # ceiling doesn't cost anything when the output is short.
            # detect_direct enumerates ~2x the entries of quasi-only, so it
            # gets double the multiplier.
            input_tokens = len(self.tokenizer.encode(scanned_text))
            max_new_tokens = max(2048, input_tokens * 2) if detect_direct else max(512, input_tokens)

        generated = self._generate(prompt, max_new_tokens)
        raw_entities = _parse_llm_json(generated)

        entities = []
        claimed_spans: set[tuple[int, int]] = set()
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

        max_new_tokens = 4096 if enable_thinking else 512
        # No live stream here: JudgePanel always prints a clean post-filter
        # summary of what actually survives, so streaming the raw unfiltered
        # verdict first would just show everything twice.
        generated = self._generate(prompt, max_new_tokens, stream=False)
        return _parse_llm_json(generated)

    def _generate(self, prompt: str, max_new_tokens: int, stream: bool = True) -> str:
        """
        Run generation via an explicit GenerationConfig — passing loose
        max_new_tokens/temperature/... kwargs alongside the model's own
        generation_config.json triggers a deprecation warning on every
        single call otherwise.
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
      "gemma"    -> google/gemma-2-9b-it
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
