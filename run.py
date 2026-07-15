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

    # LLM also backstops direct identifiers rules/BERT missed (any mode)
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm-backstop

    # Qwen3 with native thinking mode enabled
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm qwen --llm-thinking

    # Judge panel audits the output and retries if it flags residual PII
    python run.py --input notes.txt --output redacted.txt --audit audit.json --judges mistral qwen

    # Gazetteer stage: fast exact-match lookup against known Swedish places
    python run.py --input notes.txt --output redacted.txt --audit audit.json --gazetteer sweden_entities_deid.csv

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
wikidata_script.py to generate one). Off by default. Skipped in llm_only
mode, like rules.

--quasi-only forces the LLM to detect quasi-identifiers only, regardless of
--mode. Meant for input that's already had direct identifiers stripped by
some other process (e.g. MIMIC's own bracket redaction) — pair with
--mode llm_only to skip rules/BERT/gazetteer entirely and avoid the LLM
hunting for direct identifiers that aren't there.
"""

import argparse
import json
from pipeline import PIIPipeline

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
        "approx_params_b": 27,  # needs 8-bit on a 40GB card — auto-detected, see device.py
    },
    "mixtral": {
        "llm_backend":     "mixtral",
        "llm_model_path":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # Mixture-of-experts: ~47B total params (all resident in memory
        # regardless of routing, so this must reflect the total, not the
        # ~13B active per token) vs an 8x7B config's ~56B naive estimate —
        # shared attention/embedding layers across experts bring it down.
        "approx_params_b": 47,
    },
}

import os

BASE_CONFIG = {
    "bert_model_path": os.environ.get("BERT_MODEL_PATH", "./models/ModelBERTF"),
}


def main():
    parser = argparse.ArgumentParser(description="PHI De-identification Pipeline")
    parser.add_argument("--input",  default="data/notes.txt",  help="Path to input text file")
    parser.add_argument("--output", default="data/redacted.txt",  help="Path to write redacted text")
    parser.add_argument("--audit",  default="data/audit.json",  help="Path to write audit log (JSON)")
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
        default="llama",
        choices=list(LLM_BACKENDS.keys()),
        help="LLM backend to use (default: llama)"
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
        default=None,
        help="Path to a CSV of known Swedish place/institution names (default: off — see wikidata_script.py)"
    )
    parser.add_argument(
        "--quasi-only",
        action="store_true",
        help="LLM detects quasi-identifiers only, regardless of --mode (default: off) — for input already stripped of direct identifiers"
    )
    args = parser.parse_args()

    judge_configs = [
        {"name": name, **LLM_BACKENDS[name]}
        for name in args.judges
    ]

    config = {
        **BASE_CONFIG,
        **LLM_BACKENDS[args.llm],
        "mode": args.mode,
        "llm_backstop": args.llm_backstop,
        "llm_thinking": args.llm_thinking,
        "judges": judge_configs,
        "judge_max_rounds": args.judge_max_rounds,
        "gazetteer_path": args.gazetteer,
        "quasi_only": args.quasi_only,
    }
    print(
        f"Mode: {args.mode} | LLM: {args.llm} | Backstop: {args.llm_backstop} | "
        f"Thinking: {args.llm_thinking} | Judges: {args.judges or 'none'} | "
        f"Gazetteer: {args.gazetteer or 'off'} | Quasi-only: {args.quasi_only}"
    )

    pipe = PIIPipeline(config)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    result = pipe.run(text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result.redacted_text)

    with open(args.audit, "w", encoding="utf-8") as f:
        json.dump(result.audit_log, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(result.entities)} entities found and redacted.")
    print(f"Redacted text : {args.output}")
    print(f"Audit log     : {args.audit}")

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
