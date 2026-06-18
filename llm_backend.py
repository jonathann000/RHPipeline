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
                               (used in "llm_only" mode)
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
    """Robustly extract a JSON list from LLM output."""
    # Strip markdown fences if present
    raw = re.sub(r"```json|```", "", raw).strip()
    # Find first [ ... ] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON output")
        return []


def _resolve_offsets(text: str, entity_text: str) -> tuple[int, int] | None:
    """Find char offsets of entity_text in text. Returns None if not found."""
    idx = text.find(entity_text)
    if idx == -1:
        return None
    return idx, idx + len(entity_text)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    @abstractmethod
    def detect(
        self,
        text: str,
        existing: list[Entity],
        detect_direct: bool = False
    ) -> list[Entity]:
        """
        detect_direct=False: quasi-identifiers only (used alongside BERT/rules)
        detect_direct=True:  all PII including direct identifiers (llm_only mode)
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

    # Model-specific prompt templates
    CHAT_TEMPLATES = {
        "gpt-sw3": {
            # GPT-SW3 uses a simple User/Assistant format
            "format": "User: {system}\n\n{user}\nAssistant:",
        },
        "eurollm": {
            # EuroLLM follows standard chat ML format
            "format": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        },
        "llama": {
            # Llama 3.1 instruct format
            "format": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        },
    }

    def __init__(self, backend_name: str, model_path: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

        self.backend_name = backend_name
        logger.info(f"Loading LLM: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        self.pipe = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def _build_prompt(self, text: str, existing: list[Entity], detect_direct: bool) -> str:
        template = self.CHAT_TEMPLATES.get(
            self.backend_name,
            self.CHAT_TEMPLATES["llama"]
        )

        if detect_direct:
            # llm_only mode — detect everything from scratch
            user_msg = (
                f"{FEW_SHOT_FULL}\n"
                f"Identifiera ALLA direkta identifierare och quasi-identifierare.\n\n"
                f"Text: \"{text}\"\n"
                f"Entiteter:"
            )
            return template["format"].format(system=FULL_DETECTION_SYSTEM, user=user_msg)
        else:
            # Hybrid mode — only quasi-identifiers, rules/BERT already ran
            already_found = ", ".join(set(e.label for e in existing)) or "inga ännu"
            user_msg = (
                f"{FEW_SHOT}\n"
                f"Redan identifierade entiteter: {already_found}\n"
                f"Identifiera ENBART quasi-identifierare som inte redan täcks.\n\n"
                f"Text: \"{text}\"\n"
                f"Quasi-identifierare:"
            )
            return template["format"].format(system=QUASI_ID_SYSTEM, user=user_msg)

    def detect(
        self,
        text: str,
        existing: list[Entity],
        detect_direct: bool = False
    ) -> list[Entity]:
        prompt = self._build_prompt(text, existing, detect_direct)

        output = self.pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.1,     # low temp for consistent structured output
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = output[0]["generated_text"][len(prompt):]
        raw_entities = _parse_llm_json(generated)

        entities = []
        for ent in raw_entities:
            offsets = _resolve_offsets(text, ent.get("text", ""))
            if offsets is None:
                logger.debug(f"Skipping hallucinated span: {ent.get('text')}")
                continue

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
        detect_direct: bool = False
    ) -> list[Entity]:
        entities = []
        # Simple keyword scan to simulate LLM detection
        keywords = {
            "snickare": ("demographics", "medium", "hantverkare"),
            "ensamstående": ("demographics", "medium", "familjesituation"),
            "Huntingtons": ("medical", "high", "sällsynt neurologisk sjukdom"),
        }
        for kw, (label, risk, generalize) in keywords.items():
            offsets = _resolve_offsets(text, kw)
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

    supported = {"gpt-sw3", "eurollm", "llama"}
    if backend_name not in supported:
        raise ValueError(f"Unknown backend '{backend_name}'. Choose from: {supported}")

    return HuggingFaceLLMBackend(backend_name, model_path)
