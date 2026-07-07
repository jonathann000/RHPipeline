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
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm eurollm

    # LLM also backstops direct identifiers rules/BERT missed (any mode)
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm-backstop

    # Qwen3 with native thinking mode enabled
    python run.py --input notes.txt --output redacted.txt --audit audit.json --llm qwen --llm-thinking

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
"""

import argparse
import json
from pipeline import PIIPipeline

# -----------------------------------------------------------------------
# LLM backend options
# -----------------------------------------------------------------------
LLM_BACKENDS = {
    "llama": {
        "llm_backend":    "llama",
        "llm_model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    "mistral": {
        "llm_backend":    "mistral",
        "llm_model_path": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "qwen": {
        "llm_backend":    "qwen",
        "llm_model_path": "Qwen/Qwen3-8B",
    },
    "gemma": {
        "llm_backend":    "gemma",
        "llm_model_path": "google/gemma-2-9b-it",
    },
    "gpt-sw3": {
        "llm_backend":    "gpt-sw3",
        "llm_model_path": "AI-Sweden-Models/gpt-sw3-20b-instruct",
    },
    "eurollm": {
        "llm_backend":    "eurollm",
        "llm_model_path": "utter-project/EuroLLM-9B-Instruct",
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
    args = parser.parse_args()

    config = {
        **BASE_CONFIG,
        **LLM_BACKENDS[args.llm],
        "mode": args.mode,
        "llm_backstop": args.llm_backstop,
        "llm_thinking": args.llm_thinking,
    }
    print(f"Mode: {args.mode} | LLM: {args.llm} | Backstop: {args.llm_backstop} | Thinking: {args.llm_thinking}")

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


if __name__ == "__main__":
    main()
