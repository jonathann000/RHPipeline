"""
Entry point for the PHI de-identification pipeline.

Usage:
    # Full pipeline (rules + BERT + LLM)
    python run.py --input notes.txt --output redacted.txt --audit audit.json

    # No BERT (rules + LLM only)
    python run.py --input notes.txt --output redacted.txt --audit audit.json --mode no_bert

    # LLM only (no rules, no BERT)
    python run.py --input notes.txt --output redacted.txt --audit audit.json --mode llm_only

    # Swap LLM backend
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm gemma

    # Ensemble two LLM backends — each runs its own detection pass, findings
    # are merged before redaction (more recall, ~Nx the LLM inference cost)
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm mistral qwen

    # LLM also backstops direct identifiers rules/BERT missed (any mode)
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm-backstop

    # Qwen3 with native thinking mode enabled — its <think> reasoning is
    # saved to --reasoning-output (default data/reasoning.json) as an
    # explainability trail for the quasi-identifier decisions it made
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm qwen --llm-thinking

    # Judge panel audits the output and retries if it flags residual PII
    python run.py --input notes.txt --output redacted.txt --audit audit.json --judges mistral qwen

    # Gazetteer stage (fast exact-match lookup against known Swedish places)
    # runs by default off sweden_entities_deid.csv in the working directory —
    # disable it, or point it at a different CSV, if needed
    python run.py --input notes.txt --output redacted.txt --audit audit.json --no-gazetteer
    python run.py --input notes.txt --output redacted.txt --audit audit.json --gazetteer other_entities.csv

    # Already-deidentified input (e.g. MIMIC-style data): LLM only, quasi-IDs only
    python run.py --input notes.txt --output redacted.txt --audit audit.json --mode llm_only --quasi-only

Modes:
    full      Rules → BERT → LLM quasi-IDs        (default)
    no_bert   Rules → LLM (direct + quasi-IDs)
    llm_only  LLM handles everything from scratch

--llm-backstop is independent of --mode: it tells the LLM which specific
spans rules/BERT already found and asks it to catch anything else,
including direct identifiers those stages missed, instead of only doing
the job normally assigned to its mode. Off by default so both strategies
can be compared against each other.

--llm-thinking asks the model to reason in a <think> block before
answering. Only meaningful on backends that support it (currently just
Qwen3) — ignored by everything else. Uses significantly more output
tokens per call, so it's off by default.

--judges lists which backends form the judge panel (off/empty by default).
Each judge reviews the redacted output; if any judge flags residual PII,
another detection pass runs and the document is re-redacted, up to
--judge-max-rounds times (default 2). A backend already loaded as the main
--llm is reused rather than loaded twice. With only two judges, any single
flag counts (real voting needs more judges to be meaningful) — see
judge.py. Loading extra models for judges is memory-heavy; pick backends
that together fit your available RAM alongside the main --llm.

--gazetteer points at a CSV of known Swedish place/institution names (see
wikidata_script.py to generate one). Defaults to sweden_entities_deid.csv
in the working directory — on by default, since it's a committed asset,
not something generated per-machine. Falls back to off (with a warning) if
that file isn't found, or pass --no-gazetteer to disable it explicitly.
Skipped in llm_only mode, like rules.

--quasi-only forces the LLM to detect quasi-identifiers only, regardless of
--mode. Meant for input that's already had direct identifiers stripped by
some other process (e.g. MIMIC's own bracket redaction) — pair with
--mode llm_only to skip rules/BERT/gazetteer entirely and avoid the LLM
hunting for direct identifiers that aren't there.

--no-generalize skips trusting the LLM's suggested generalization for any
quasi-identifier — every one falls back to its category placeholder, same
as direct identifiers already get unconditionally. Less informative output,
but rules out a generalization being factually wrong (not just insufficiently
abstracted) in a way no other check catches.

--label-studio-output writes a Label Studio pre-annotation task for this
document to the given path (overwriting it, by default — matches rerunning
the same note over and over during development), so every detected span
shows up in Label Studio's UI pre-highlighted by label, with risk and
source as per-region choices for filtering/inspection or correction. Also
(re)writes a labeling_config.xml next to it — import that once into the
Label Studio project. Off by default.
    python run.py --input notes.txt --output redacted.txt --audit audit.json --label-studio-output data/label_studio_tasks.json

--label-studio-append accumulates onto --label-studio-output's existing
task list instead of overwriting it — for batching several distinct
documents into one file before a single bulk import (e.g. building an
annotation corpus), rather than inspecting one document at a time.
"""

import argparse
import json
import os
from pipeline import PIIPipeline
from label_studio_export import write_label_studio_export, build_labeling_config

# -----------------------------------------------------------------------
# LLM backend options
# -----------------------------------------------------------------------
LLM_BACKENDS = {
    "llama": {
        "llm_backend":     "llama",
        "llm_model_path":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "approx_params_b": 8,
    },
    "mistral": {
        "llm_backend":     "mistral",
        "llm_model_path":  "mistralai/Mistral-7B-Instruct-v0.3",
        "approx_params_b": 7,
    },
    "qwen": {
        "llm_backend":     "qwen",
        "llm_model_path":  "Qwen/Qwen3-8B",
        "approx_params_b": 8,
    },
    "qwen-32b": {
        "llm_backend":     "qwen",
        "llm_model_path":  "Qwen/Qwen3-32B",
        "approx_params_b": 32,  # needs 8-bit on a 40GB card — auto-detected, see device.py
    },
    "gemma": {
        "llm_backend":     "gemma",
        "llm_model_path":  "google/gemma-2-9b-it",
        "approx_params_b": 9,
    },
    "gemma-27b": {
        "llm_backend":     "gemma",
        "llm_model_path":  "google/gemma-2-27b-it",
        "approx_params_b": 27,  # needs 4-bit on a 40GB card — auto-detected, see device.py
    },
    "gemma4-12b": {
        "llm_backend":     "gemma",
        "llm_model_path":  "google/gemma-4-12B-it",
        "approx_params_b": 12,  # fits full bf16 on a 40GB card (~29GB w/ safety margin) — auto-detected, see device.py
    },
    "gemma4-31b": {
        "llm_backend":     "gemma",
        "llm_model_path":  "google/gemma-4-31B-it",
        "approx_params_b": 31,  # needs 4-bit on a 40GB card — auto-detected, see device.py
    },
}

import os

BASE_CONFIG = {
    "bert_model_path": os.environ.get("BERT_MODEL_PATH", "./models/ModelBERTF"),
}

# A committed repo asset (see wikidata_script.py), not something that needs
# generating per-machine — on by default so a bare `sweden_entities_deid.csv`
# in the working directory is picked up without remembering to pass
# --gazetteer every time. --no-gazetteer opts back out for benchmarking.
DEFAULT_GAZETTEER_PATH = os.environ.get("GAZETTEER_PATH", "sweden_entities_deid.csv")


def main():
    parser = argparse.ArgumentParser(description="PHI De-identification Pipeline")
    parser.add_argument("--input",  default="data/notes.txt",  help="Path to input text file")
    parser.add_argument("--output", default="data/redacted.txt",  help="Path to write redacted text")
    parser.add_argument("--audit",  default="data/audit.json",  help="Path to write audit log (JSON)")
    parser.add_argument(
        "--reasoning-output",
        default="data/reasoning.json",
        help="Path to write the model's <think> reasoning trail, when --llm-thinking (or a judge's own thinking) produced any (default: data/reasoning.json). Nothing is written if there's no reasoning to save."
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "no_bert", "llm_only"],
        help=(
            "full:     Rules + BERT + LLM (default)\n"
            "no_bert:  Rules + LLM only\n"
            "llm_only: LLM handles everything"
        )
    )
    parser.add_argument(
        "--llm",
        nargs="+",
        default=["llama"],
        choices=list(LLM_BACKENDS.keys()),
        help="LLM backend(s) to use — pass more than one (e.g. --llm mistral qwen) to run an ensemble: each backend runs its own detection pass and every model's findings are merged via the pipeline's existing overlap-resolution before redaction (default: llama)"
    )
    parser.add_argument(
        "--llm-backstop",
        action="store_true",
        help="LLM also catches direct identifiers rules/BERT missed, independent of --mode (default: off)"
    )
    parser.add_argument(
        "--llm-thinking",
        action="store_true",
        help="Ask the LLM to reason in a <think> block before answering (Qwen3 only, ignored elsewhere; default: off)"
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=[],
        choices=list(LLM_BACKENDS.keys()),
        help="Backends forming the judge panel that audits the redacted output (default: none — panel off)"
    )
    parser.add_argument(
        "--judge-max-rounds",
        type=int,
        default=2,
        help="Max detect-then-rejudge rounds before flagging for human review (default: 2)"
    )
    parser.add_argument(
        "--gazetteer",
        default=DEFAULT_GAZETTEER_PATH,
        help=f"Path to a CSV of known Swedish place/institution names (default: {DEFAULT_GAZETTEER_PATH} — see wikidata_script.py)"
    )
    parser.add_argument(
        "--no-gazetteer",
        action="store_true",
        help="Disable the gazetteer stage, overriding --gazetteer's default (default: off, i.e. gazetteer runs)"
    )
    parser.add_argument(
        "--quasi-only",
        action="store_true",
        help="LLM detects quasi-identifiers only, regardless of --mode (default: off) — for input already stripped of direct identifiers"
    )
    parser.add_argument(
        "--no-generalize",
        action="store_true",
        help="Never trust the LLM's suggested generalization for a quasi-identifier — always use its category placeholder instead (default: off)"
    )
    parser.add_argument(
        "--label-studio-output",
        default=None,
        help="Write this document's detections as a Label Studio pre-annotation task to this path (overwriting it), plus (re)write a labeling_config.xml alongside it (default: off)"
    )
    parser.add_argument(
        "--label-studio-append",
        action="store_true",
        help="Accumulate onto --label-studio-output's existing tasks instead of overwriting (default: off — for batching multiple documents into one file)"
    )
    args = parser.parse_args()

    judge_configs = [
        {"name": name, **LLM_BACKENDS[name]}
        for name in args.judges
    ]

    gazetteer_path = None if args.no_gazetteer else args.gazetteer
    if gazetteer_path and not os.path.exists(gazetteer_path):
        print(f"Gazetteer file not found at '{gazetteer_path}' — running without it. Pass --gazetteer to point at a real path, or --no-gazetteer to silence this.")
        gazetteer_path = None

    config = {
        **BASE_CONFIG,
        "llm_configs": [LLM_BACKENDS[name] for name in args.llm],
        "mode": args.mode,
        "llm_backstop": args.llm_backstop,
        "llm_thinking": args.llm_thinking,
        "judges": judge_configs,
        "judge_max_rounds": args.judge_max_rounds,
        "gazetteer_path": gazetteer_path,
        "quasi_only": args.quasi_only,
        "no_generalize": args.no_generalize,
    }
    print(
        f"Mode: {args.mode} | LLM: {'+'.join(args.llm)} | Backstop: {args.llm_backstop} | "
        f"Thinking: {args.llm_thinking} | Judges: {args.judges or 'none'} | "
        f"Gazetteer: {gazetteer_path or 'off'} | Quasi-only: {args.quasi_only} | "
        f"No-generalize: {args.no_generalize}"
    )

    pipe = PIIPipeline(config)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    result = pipe.run(text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result.redacted_text)

    with open(args.audit, "w", encoding="utf-8") as f:
        json.dump(result.audit_log, f, ensure_ascii=False, indent=2)

    if result.reasoning_log:
        with open(args.reasoning_output, "w", encoding="utf-8") as f:
            json.dump(result.reasoning_log, f, ensure_ascii=False, indent=2)

    if args.label_studio_output:
        write_label_studio_export(
            args.label_studio_output,
            text,
            result.entities,
            # output_file (not just source_file) distinguishes reruns on the
            # same input document (e.g. before/after a code fix) — otherwise
            # two such tasks look identical in Label Studio's Data Manager,
            # with nothing but list order to tell them apart.
            task_data_extra={"source_file": args.input, "output_file": args.output},
            append=args.label_studio_append,
        )
        config_path = os.path.join(os.path.dirname(args.label_studio_output) or ".", "label_studio_config.xml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(build_labeling_config())

    print(f"Done. {len(result.entities)} entities found and redacted.")
    print(f"Redacted text : {args.output}")
    print(f"Audit log     : {args.audit}")
    if result.reasoning_log:
        print(f"Reasoning log : {args.reasoning_output} ({len(result.reasoning_log)} entries)")
    if args.label_studio_output:
        print(f"Label Studio  : {args.label_studio_output} (config: {config_path})")

    print("\n" + "=" * 70)
    print("REDACTED TEXT")
    print("=" * 70)
    print(result.redacted_text)

    if result.needs_human_review:
        print("\n" + "=" * 70)
        print(f"NEEDS HUMAN REVIEW — judge panel still flags {len(result.judge_flags)} issue(s) after {args.judge_max_rounds} round(s):")
        print("=" * 70)
        for flag in result.judge_flags:
            print(f"  [{flag['judge']}] \"{flag['quote']}\" — {flag['reason']}")
    elif args.judges:
        print("\nJudge panel: document is clean.")


if __name__ == "__main__":
    main()
