"""
Swappable LLM backend for PII detection.

Supported backends (set via config["llm_backend"]):
  - "gpt-sw3"    : AI Sweden GPT-SW3 (recommended for Swedish clinical text)
  - "eurollm"    : EuroLLM-9B (strong Swedish benchmarks)
  - "llama"      : Llama 3.1 8B Instruct (multilingual baseline)

All backends use the same interface — swap by changing config only.

detect_direct=False (default): LLM only looks for quasi-identifiers
                               (used in "full" and "no_bert" modes)
detect_direct=True:            LLM handles all PII including direct identifiers
                               (used in "llm_only" mode, or "full"/"no_bert" with
                               llm_backstop enabled — see below)

backstop_existing=False (default): full detection re-enumerates everything
                               from scratch, ignoring `existing` (today's
                               llm_only/no_bert behavior).
backstop_existing=True:       full detection is told the specific entity
                               texts already found (by rules/BERT) and asked
                               to skip those, catching quasi-identifiers plus
                               any direct identifiers those stages missed —
                               without wasting output budget re-deriving
                               spans already covered. Only meaningful when
                               detect_direct=True.
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
"""

QUASI_ID_SYSTEM = """Du är ett system för att identifiera quasi-identifierare i svenska journalanteckningar.
Quasi-identifierare är uppgifter som ensamma kanske inte identifierar en person, men som i kombination med andra uppgifter kan göra det — särskilt i en liten svensk kommun.

Kategorier att leta efter:
- demographics: ålder, kön, etnicitet, yrke, familjesituation
- medical: sällsynta diagnoser, ovanliga ingrepp, specifika läkemedel
- temporal: exakta datum, vårdtider, specifika tidpunkter
- private_address: klinik, avdelning, stadsdel, ort
- social: arbetsgivare, boendesituation, religiös tillhörighet

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

Quasi-identifierare (kontextberoende risk):
- demographics:    ålder, kön, etnicitet, yrke, familjesituation
- medical:         sällsynta diagnoser, ovanliga ingrepp, specifika läkemedel
- temporal:        exakta tidpunkter, vårdlängd
- social:          arbetsgivare, boendesituation, religiös tillhörighet

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
    if not entity_text:
        return None
    search_from = 0
    while True:
        idx = text.find(entity_text, search_from)
        if idx == -1:
            return None
        span = (idx, idx + len(entity_text))
        if span not in claimed:
            return span
        search_from = idx + 1


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
        backstop_existing: bool = False,
        enable_thinking: bool = False,
    ) -> list[Entity]:
        """
        detect_direct=False:     quasi-identifiers only (used alongside BERT/rules)
        detect_direct=True:      all PII including direct identifiers (llm_only mode)
        backstop_existing=True:  (only with detect_direct=True) skip spans already
                                  found by rules/BERT instead of re-deriving them,
                                  while still catching quasi-identifiers plus any
                                  direct identifiers those stages missed
        enable_thinking=True:    ask the model to reason in a <think> block before
                                  answering (only meaningful on backends that support
                                  it, e.g. Qwen3 — ignored by everything else)
        """
        pass


# ---------------------------------------------------------------------------
# HuggingFace backend (GPT-SW3 / EuroLLM / Llama — all use same interface)
# ---------------------------------------------------------------------------

class HuggingFaceLLMBackend(LLMBackend):
    """
    Generic HuggingFace causal LM backend.
    Works with GPT-SW3, EuroLLM, Llama, and most instruct-tuned models.
    """

    def __init__(self, backend_name: str, model_path: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
        from device import resolve_device_map

        self.backend_name = backend_name
        logger.info(f"Loading LLM: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=resolve_device_map(),
        )
        self.model.eval()

        self.pipe = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def _build_prompt(
        self,
        text: str,
        existing: list[Entity],
        detect_direct: bool,
        backstop_existing: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        if detect_direct and backstop_existing:
            # Full detection, but told exactly what's already been found —
            # skip re-deriving those spans, catch quasi-identifiers plus any
            # direct identifiers rules/BERT missed.
            already_found = ", ".join(dict.fromkeys(e.text for e in existing)) or "inga ännu"
            system = FULL_DETECTION_SYSTEM
            user_msg = (
                f"{FEW_SHOT_FULL}\n"
                f"Redan identifierade (hoppa över dessa): {already_found}\n"
                f"Identifiera ALLA ÖVRIGA direkta identifierare och quasi-identifierare "
                f"som inte redan är med i listan ovan.\n\n"
                f"Text: \"{text}\"\n"
                f"Entiteter:"
            )
        elif detect_direct:
            # llm_only mode — detect everything from scratch
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
        backstop_existing: bool = False,
        enable_thinking: bool = False,
    ) -> list[Entity]:
        prompt = self._build_prompt(text, existing, detect_direct, backstop_existing, enable_thinking)

        if enable_thinking:
            # A <think> block consumes generation budget before the actual
            # answer — reasoning models can use 5-20x more tokens per
            # response than non-reasoning ones, so this needs real headroom.
            max_new_tokens = 4096
        elif detect_direct:
            # Full detection enumerates direct + quasi identifiers (~2x the
            # entries of quasi-only), so it needs more headroom too.
            max_new_tokens = 2048
        else:
            max_new_tokens = 512

        output = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,     # low temp for consistent structured output
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = output[0]["generated_text"][len(prompt):]
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
        backstop_existing: bool = False,
        enable_thinking: bool = False,
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_llm(backend_name: str, model_path: str) -> LLMBackend:
    """
    Load the appropriate LLM backend.

    backend_name options:
      "gpt-sw3"  -> AI-Sweden-Models/gpt-sw3-20b-instruct  (recommended)
      "eurollm"  -> utter-project/EuroLLM-9B-Instruct
      "llama"    -> meta-llama/Meta-Llama-3.1-8B-Instruct
      "mock"     -> MockLLMBackend (no model, for testing)

    model_path can be a HuggingFace model ID or a local directory path.
    """
    if backend_name == "mock":
        return MockLLMBackend()

    supported = {"gpt-sw3", "eurollm", "llama", "mistral", "qwen", "gemma"}
    if backend_name not in supported:
        raise ValueError(f"Unknown backend '{backend_name}'. Choose from: {supported}")

    return HuggingFaceLLMBackend(backend_name, model_path)
