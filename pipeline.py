"""
PHI De-identification Pipeline
-------------------------------
Stage 1: Rule-based agent      — fast regex for structured PII
Stage 2: BERT NER agent        — token classification for direct identifiers (optional)
Stage 3: LLM agent             — direct identifiers + quasi-identifiers
Stage 4: Risk + redaction      — generalization and audit logging

Modes (set via config["mode"]):
  "full"     — Rules → BERT → LLM  (default, highest coverage)
  "no_bert"  — Rules → LLM only    (benchmark without BERT)
  "llm_only" — LLM only            (pure LLM baseline, no rules either)

LLM backend is swappable via config — see llm_backend.py
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from entities import Entity
from bert_agent import BERTAgent
from llm_backend import load_llm, LLMBackend
from rule_agent import RuleAgent
from redaction import redact_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    original_text: str
    redacted_text: str
    entities: list[Entity]
    audit_log: list[dict] = field(default_factory=list)


class PIIPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "full")

        self.rule_agent = RuleAgent() if self.mode != "llm_only" else None
        self.bert_agent = BERTAgent(config["bert_model_path"]) if self.mode == "full" else None
        self.llm: LLMBackend = load_llm(config["llm_backend"], config["llm_model_path"])

        logger.info(f"Pipeline mode: {self.mode}")

    def run(self, text: str) -> PipelineResult:
        all_entities: list[Entity] = []

        # --- Stage 1: Rule-based (skipped in llm_only mode) ---
        if self.rule_agent:
            rule_entities = self.rule_agent.detect(text)
            all_entities.extend(rule_entities)
            logger.info(f"Rule agent found {len(rule_entities)} entities")

        # --- Stage 2: BERT NER (full mode only) ---
        if self.bert_agent:
            bert_entities = self.bert_agent.detect(text)
            bert_entities = self._deduplicate(bert_entities, all_entities)
            all_entities.extend(bert_entities)
            logger.info(f"BERT agent found {len(bert_entities)} new entities")

        # --- Stage 3: LLM ---
        # In full/no_bert mode: LLM handles quasi-identifiers only
        # In llm_only mode:     LLM handles everything
        llm_entities = self.llm.detect(
            text,
            existing=all_entities,
            detect_direct=(self.mode == "llm_only")
        )
        llm_entities = self._deduplicate(llm_entities, all_entities)
        all_entities.extend(llm_entities)
        logger.info(f"LLM agent found {len(llm_entities)} entities")

        # --- Stage 4: Redaction + audit ---
        redacted = redact_document(text, all_entities)
        audit = self._build_audit(text, all_entities)

        return PipelineResult(
            original_text=text,
            redacted_text=redacted,
            entities=all_entities,
            audit_log=audit
        )

    def _deduplicate(self, new: list[Entity], existing: list[Entity]) -> list[Entity]:
        """Remove entities that overlap with already-found spans."""
        existing_spans = {(e.start, e.end) for e in existing}
        return [
            e for e in new
            if not any(
                e.start < ex_end and e.end > ex_start
                for ex_start, ex_end in existing_spans
            )
        ]

    def _build_audit(self, text: str, entities: list[Entity]) -> list[dict]:
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "original": e.text,
                "label": e.label,
                "start": e.start,
                "end": e.end,
                "source": e.source,
                "risk": e.risk,
                "generalized_to": e.generalized or "[REDACTED]",
            }
            for e in sorted(entities, key=lambda x: x.start)
        ]
