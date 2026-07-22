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

config["quasi_only"] (default False): forces the main LLM detection call to
  quasi-identifiers only, overriding the mode-based detect_direct default
  below. Meant for already-deidentified input (e.g. MIMIC-style data with
  its own bracket redaction already applied) where there's nothing left for
  rules/BERT/LLM to find as a direct identifier, and asking the LLM to hunt
  for direct identifiers anyway just risks it misreading the existing
  bracket placeholders as something to redact further.

config["no_generalize"] (default False): when True, every quasi-identifier
  falls back to its category placeholder instead of trusting the LLM's
  suggested `generalized` text — same treatment direct identifiers already
  get unconditionally. Trades away the informativeness a correct
  generalization gives (e.g. "65-70 år" instead of a blunt
  [DEMOGRAFISK-UPPGIFT]) in exchange for ruling out a generalization being
  factually wrong in a way no other check catches (e.g. hypothyroidism —
  a thyroid condition — generalized to a made-up description of a
  completely different organ). See redaction.py's resolve_replacement.

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

config["llm_configs"]: a list of {"llm_backend": ..., "llm_model_path": ...,
  "approx_params_b": ...} dicts — one entry runs a single model as usual;
  more than one runs an ensemble. Each configured backend independently
  runs its own Stage 3 detection pass over the same text (and, in the judge
  retry loop, the same targeted re-detection pass); every model's findings
  are extended into the same all_entities list and merged by the existing
  overlap-resolution in _propagate_and_redact — no separate merge logic
  needed, the same mechanism that already reconciles rules/BERT/LLM/
  gazetteer entities also reconciles multiple LLMs' entities. The point is
  recall: two models miss different things, so their union catches more
  quasi-identifiers than either alone — at roughly Nx the LLM inference
  cost per document for N backends.

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
from datetime import datetime, timezone

from entities import Entity, build_redaction_plan
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
    # Populated only when config["llm_thinking"] (or a judge's own
    # enable_thinking) is on — the model's own <think> reasoning for each
    # detect/judge call that produced one, as an explainability trail for
    # quasi-identifier decisions. Empty when thinking was never enabled.
    reasoning_log: list[dict] = field(default_factory=list)


class PIIPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "full")
        self.llm_backstop = config.get("llm_backstop", False)
        self.quasi_only = config.get("quasi_only", False)
        self.no_generalize = config.get("no_generalize", False)
        self.llm_thinking = config.get("llm_thinking", False)
        self.judge_max_rounds = config.get("judge_max_rounds", 2)

        self.rule_agent = RuleAgent() if self.mode != "llm_only" else None
        gazetteer_path = config.get("gazetteer_path")
        self.gazetteer_agent = (
            GazetteerAgent(gazetteer_path) if gazetteer_path and self.mode != "llm_only" else None
        )
        self.bert_agent = BERTAgent(config["bert_model_path"]) if self.mode == "full" else None
        self.llms: list[LLMBackend] = [
            load_llm(c["llm_backend"], c["llm_model_path"], c.get("approx_params_b", 9))
            for c in config["llm_configs"]
        ]

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

        # A backend is cached and can be reused across multiple documents
        # in one process (e.g. the CLI's judge test runs, or a long-lived
        # server using this pipeline as a library) — clear each backend's
        # reasoning_log now so PipelineResult.reasoning_log below reflects
        # only this call, not leftovers from a previous document.
        for llm in self.llms:
            llm.reasoning_log.clear()
        if self.judge_panel:
            for judge in self.judge_panel.judges:
                judge.backend.reasoning_log.clear()

        # --- Stage 1: Rule-based (skipped in llm_only mode) ---
        if self.rule_agent:
            rule_entities = self.rule_agent.detect(text)
            all_entities.extend(rule_entities)
            logger.info(f"Rule agent found {len(rule_entities)} entities")

        # --- Stage 2: BERT NER (full mode only) ---
        if self.bert_agent:
            bert_entities = self.bert_agent.detect(text)
            all_entities.extend(bert_entities)
            logger.info(f"BERT agent found {len(bert_entities)} entities")

        # --- Stage 3: LLM (one or more backends — see config["llm_configs"]) ---
        # In full mode:               LLM handles quasi-identifiers only (BERT covers direct)
        #                              — unless llm_backstop is on, then it scans the
        #                              text already redacted so far instead, catching
        #                              anything rules/BERT missed.
        # In no_bert/llm_only mode:   LLM handles direct + quasi-identifiers (no BERT to cover direct)
        # quasi_only forces quasi-identifiers only regardless of the above —
        # see config["quasi_only"] docstring at the top of this file.
        # Each configured backend runs its own independent pass over the same
        # text; every backend's findings are extended into all_entities and
        # reconciled by the same overlap-resolution that already merges
        # rules/BERT/gazetteer entities, in _propagate_and_redact below.
        # Redact through build_redaction_plan (not all_entities directly) for
        # the same reason the final redaction does: rules and BERT can flag
        # overlapping spans (e.g. both tag the same date), and redact_document
        # replaces in reverse assuming spans never overlap — feeding it the raw,
        # un-reconciled list would corrupt the very text the backstop LLM reads.
        scan_text = (
            redact_document(text, build_redaction_plan(text, all_entities), self.no_generalize)
            if self.llm_backstop else None
        )
        for llm in self.llms:
            llm_entities = llm.detect(
                text,
                existing=all_entities,
                detect_direct=(not self.quasi_only) and ((self.mode in ("llm_only", "no_bert")) or self.llm_backstop),
                enable_thinking=self.llm_thinking,
                scan_text=scan_text,
            )
            all_entities.extend(llm_entities)
            logger.info(f"LLM agent ({llm.backend_name}) found {len(llm_entities)} entities")

        # --- Stage 4: Gazetteer (skipped in llm_only mode, off unless configured) ---
        # Runs last among detection stages, purely as a gap-filler: it has
        # no context awareness (just exact string matching), so it must
        # never claim a span before a context-aware stage gets to it —
        # e.g. "Lindberg" is both a real Wikidata locality and an ordinary
        # Swedish surname; only BERT/LLM's context can tell "Maria Lindberg"
        # is a person, not a place.
        if self.gazetteer_agent:
            gazetteer_entities = self.gazetteer_agent.detect(text)
            all_entities.extend(gazetteer_entities)
            logger.info(f"Gazetteer agent found {len(gazetteer_entities)} entities")

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
                for llm in self.llms:
                    retry_entities = llm.detect(
                        text,
                        existing=all_entities,
                        detect_direct=True,
                        enable_thinking=self.llm_thinking,
                        scan_text=redacted,
                        target_quotes=[f.quote for f in flags],
                    )
                    all_entities.extend(retry_entities)
                    logger.info(f"Targeted retry pass ({llm.backend_name}) found {len(retry_entities)} new entities")
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

        # Combine every ensemble LLM's reasoning with every judge's — a judge
        # reusing a backend that's also in the ensemble (e.g. --judges mistral
        # when mistral is also one of the --llm backends) would otherwise get
        # counted twice, since they're the same object.
        reasoning_log = []
        for llm in self.llms:
            reasoning_log.extend(llm.reasoning_log)
        if self.judge_panel:
            for judge in self.judge_panel.judges:
                if judge.backend not in self.llms:
                    reasoning_log.extend(judge.backend.reasoning_log)

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
            reasoning_log=reasoning_log,
        )

    def _propagate_and_redact(self, text: str, all_entities: list[Entity]) -> str:
        """
        Coreference propagation (extends already-found entities to their
        other mentions — repeated phrases, short-form names) over every
        detection stage's raw output combined, then redaction. Mutates
        all_entities in place by appending propagated entities — but,
        deliberately, never removes or overwrites anything already in it.

        Overlapping entities across stages (e.g. a gazetteer's bare
        institution match inside a wider LLM occupation-phrase entity) are
        NOT resolved down to one survivor here — the system's job is
        finding as many quasi-identifiers as possible, so a distinct
        finding should never silently disappear from all_entities (and so
        from the audit/Label Studio export) just because it overlaps a
        different one. Conflict resolution only happens for the actual
        redacted text, via build_redaction_plan, which is computed
        separately and doesn't touch all_entities itself.
        """
        propagated_entities = propagate_entities(text, all_entities)
        all_entities.extend(propagated_entities)
        logger.info(f"Coreference propagation found {len(propagated_entities)} additional entities")
        redaction_plan = build_redaction_plan(text, all_entities)
        return redact_document(text, redaction_plan, self.no_generalize)

    def _build_audit(self, text: str, entities: list[Entity]) -> list[dict]:
        return [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "original": e.text,
                "label": e.label,
                "start": e.start,
                "end": e.end,
                "source": e.source,
                "risk": e.risk,
                "generalized_to": resolve_replacement(e, self.no_generalize),
            }
            for e in sorted(entities, key=lambda x: x.start)
        ]
