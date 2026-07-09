"""
PHI De-identification Pipeline
-------------------------------
Stage 1: Rule-based agent      — fast regex for structured PII
Stage 2: BERT NER agent        — token classification for direct identifiers (optional)
Stage 3: LLM agent             — direct identifiers + quasi-identifiers
Stage 4: Gazetteer agent       — fast exact-match lookup against known Swedish
                                  places/institutions from Wikidata (optional)
Stage 5: Coreference           — propagate entities to their other mentions
Stage 6: Risk + redaction      — generalization and audit logging
Stage 7: Judge panel           — audit the redacted output, retry if flagged (optional)

config["gazetteer_path"] (default None, meaning the stage is off): path to a
  CSV of known place/institution names (see wikidata_script.py +
  gazetteer_agent.py). Skipped in "llm_only" mode, like rules. Deliberately
  runs LAST among detection stages (a "gap filler"), not first — it has no
  context awareness, just exact string matching, so a bad blind match (e.g.
  a place name that's also a common surname) must never be allowed to claim
  a span before a context-aware stage (BERT/LLM) gets a chance to find the
  correct interpretation there first.

Modes (set via config["mode"]):
  "full"     — Rules → BERT → LLM  (default, highest coverage)
  "no_bert"  — Rules → LLM only    (benchmark without BERT)
  "llm_only" — LLM only            (pure LLM baseline, no rules either)

config["llm_backstop"] (default False), independent of mode:
  False — LLM only does the job described for its mode above (unchanged
          default behavior).
  True  — LLM instead scans the text already redacted by rules/BERT so far
          (not the original) and does full detection on that — anything
          already caught is invisible to it, so it can only report what's
          still exposed: quasi-identifiers plus any direct identifiers
          those stages missed. Kept as an opt-in flag (not the default) so
          both strategies can be compared against each other — see
          llm_backend.py.

config["llm_thinking"] (default False): ask the LLM to reason in a <think>
  block before answering. Only meaningful on backends that support it
  (e.g. Qwen3) — ignored by everything else. Off by default since reasoning
  costs significantly more output tokens per call.

config["judges"] (default [], meaning the judge panel is off): a list of
  {"name": ..., "llm_backend": ..., "llm_model_path": ...} dicts, each
  loaded as a Judge on the panel (see judge.py). After the pipeline
  produces its normal redacted output, every judge reviews it; if any judge
  flags residual PII, the LLM runs another detection pass scanning the
  current redacted text (same mechanism as llm_backstop) and the whole
  document is re-redacted, up to config["judge_max_rounds"] (default 2)
  times. If still flagged after the cap, PipelineResult.needs_human_review
  is set instead of looping forever. A backend already loaded elsewhere
  (e.g. as the main detection LLM) is reused rather than loaded twice —
  see load_llm()'s cache in llm_backend.py.

LLM backend is swappable via config — see llm_backend.py
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

from entities import Entity, remove_overlapping_entities
from bert_agent import BERTAgent
from gazetteer_agent import GazetteerAgent
from llm_backend import load_llm, LLMBackend
from rule_agent import RuleAgent
from coreference import propagate_entities
from judge import Judge, JudgePanel
from redaction import redact_document, resolve_replacement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    original_text: str
    redacted_text: str
    entities: list[Entity]
    audit_log: list[dict] = field(default_factory=list)
    needs_human_review: bool = False
    judge_flags: list = field(default_factory=list)


class PIIPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "full")
        self.llm_backstop = config.get("llm_backstop", False)
        self.llm_thinking = config.get("llm_thinking", False)
        self.judge_max_rounds = config.get("judge_max_rounds", 2)

        self.rule_agent = RuleAgent() if self.mode != "llm_only" else None
        gazetteer_path = config.get("gazetteer_path")
        self.gazetteer_agent = (
            GazetteerAgent(gazetteer_path) if gazetteer_path and self.mode != "llm_only" else None
        )
        self.bert_agent = BERTAgent(config["bert_model_path"]) if self.mode == "full" else None
        self.llm: LLMBackend = load_llm(
            config["llm_backend"], config["llm_model_path"], config.get("approx_params_b", 9)
        )

        judge_configs = config.get("judges", [])
        self.judge_panel = JudgePanel(judges=[
            Judge(
                name=jc.get("name", jc["llm_backend"]),
                backend=load_llm(jc["llm_backend"], jc["llm_model_path"], jc.get("approx_params_b", 9)),
                enable_thinking=jc.get("enable_thinking", self.llm_thinking),
            )
            for jc in judge_configs
        ]) if judge_configs else None

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
        # In full mode:               LLM handles quasi-identifiers only (BERT covers direct)
        #                              — unless llm_backstop is on, then it scans the
        #                              text already redacted so far instead, catching
        #                              anything rules/BERT missed.
        # In no_bert/llm_only mode:   LLM handles direct + quasi-identifiers (no BERT to cover direct)
        scan_text = redact_document(text, all_entities) if self.llm_backstop else None
        llm_entities = self.llm.detect(
            text,
            existing=all_entities,
            detect_direct=(self.mode in ("llm_only", "no_bert")) or self.llm_backstop,
            enable_thinking=self.llm_thinking,
            scan_text=scan_text,
        )
        llm_entities = self._deduplicate(llm_entities, all_entities)
        all_entities.extend(llm_entities)
        logger.info(f"LLM agent found {len(llm_entities)} entities")

        # --- Stage 4: Gazetteer (skipped in llm_only mode, off unless configured) ---
        # Runs last among detection stages, purely as a gap-filler: it has
        # no context awareness (just exact string matching), so it must
        # never claim a span before a context-aware stage gets to it —
        # e.g. "Lindberg" is both a real Wikidata locality and an ordinary
        # Swedish surname; only BERT/LLM's context can tell "Maria Lindberg"
        # is a person, not a place.
        if self.gazetteer_agent:
            gazetteer_entities = self.gazetteer_agent.detect(text)
            gazetteer_entities = self._deduplicate(gazetteer_entities, all_entities)
            all_entities.extend(gazetteer_entities)
            logger.info(f"Gazetteer agent found {len(gazetteer_entities)} new entities")

        # --- Stage 5 + 6: Coreference propagation, then redaction ---
        redacted = self._propagate_and_redact(text, all_entities)

        # --- Stage 7: Judge panel (optional) ---
        # Each round: judges review the current redacted text; if any flags
        # it, run a targeted retry pass on exactly those flagged excerpts
        # (not a blind full sweep — a concrete target is far more reliable
        # than hoping sampling surfaces the same miss again) and re-redact.
        # Capped so a judge that never agrees the document is clean can't
        # loop forever — if still flagged after the cap, needs_human_review
        # is set instead.
        judge_flags = []
        if self.judge_panel:
            for round_num in range(1, self.judge_max_rounds + 1):
                logger.info(f"=== Judge round {round_num}/{self.judge_max_rounds}: reviewing output ===")
                flags = self.judge_panel.review(redacted)
                if not flags:
                    logger.info("Judge panel: document is clean")
                    judge_flags = []
                    break
                logger.info(f"=== {len(flags)} issue(s) flagged — running targeted retry ===")
                retry_entities = self.llm.detect(
                    text,
                    existing=all_entities,
                    detect_direct=True,
                    enable_thinking=self.llm_thinking,
                    scan_text=redacted,
                    target_quotes=[f.quote for f in flags],
                )
                retry_entities = self._deduplicate(retry_entities, all_entities)
                all_entities.extend(retry_entities)
                logger.info(f"Targeted retry pass found {len(retry_entities)} new entities")
                redacted = self._propagate_and_redact(text, all_entities)
                judge_flags = flags
            else:
                # Exhausted every round without a clean verdict — get the
                # final state's actual verdict rather than assuming the last
                # round's (now stale) flags still apply.
                logger.info("=== Final judge review after exhausting all rounds ===")
                judge_flags = self.judge_panel.review(redacted)

        needs_human_review = bool(judge_flags)
        if needs_human_review:
            logger.warning(
                f"Judge panel still flags {len(judge_flags)} issue(s) after "
                f"{self.judge_max_rounds} round(s) — needs human review"
            )

        audit = self._build_audit(text, all_entities)

        return PipelineResult(
            original_text=text,
            redacted_text=redacted,
            entities=all_entities,
            audit_log=audit,
            needs_human_review=needs_human_review,
            judge_flags=[
                {"quote": f.quote, "reason": f.reason, "judge": f.judge_name}
                for f in judge_flags
            ],
        )

    def _propagate_and_redact(self, text: str, all_entities: list[Entity]) -> str:
        """
        Coreference propagation (extends already-found entities to their
        other mentions — repeated phrases, short-form names) followed by
        redaction. Mutates all_entities in place with the propagated finds.
        """
        propagated_entities = propagate_entities(text, all_entities)
        all_entities.extend(propagated_entities)
        logger.info(f"Coreference propagation found {len(propagated_entities)} additional entities")
        return redact_document(text, all_entities)

    def _deduplicate(self, new: list[Entity], existing: list[Entity]) -> list[Entity]:
        """
        Remove entities that overlap with already-found spans. Also resolves
        overlaps within `new` itself first — a single detection call can
        return overlapping/nested entities (e.g. a name plus name+title),
        which would otherwise corrupt redaction's reverse-order offsets.
        """
        new = remove_overlapping_entities(new)
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
                "generalized_to": resolve_replacement(e),
            }
            for e in sorted(entities, key=lambda x: x.start)
        ]
