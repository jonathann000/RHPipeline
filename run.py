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

Modes:
    full      Rules → BERT → LLM quasi-IDs        (default)
    no_bert   Rules → LLM (direct + quasi-IDs)
    llm_only  LLM handles everything from scratch
"""

import argparse
import json
from pipeline import PIIPipeline

# -----------------------------------------------------------------------
# LLM backend options
# -----------------------------------------------------------------------
LLM_BACKENDS = {
    "gpt-sw3": {
        "llm_backend":    "gpt-sw3",
        "llm_model_path": "AI-Sweden-Models/gpt-sw3-20b-instruct",
    },
    "eurollm": {
        "llm_backend":    "eurollm",
        "llm_model_path": "utter-project/EuroLLM-9B-Instruct",
    },
    "llama": {
        "llm_backend":    "llama",
        "llm_model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
}

import os

BASE_CONFIG = {
    "bert_model_path": os.environ.get("BERT_MODEL_PATH", "./models/ModelBERTF"),
}


def main():
    parser = argparse.ArgumentParser(description="PHI De-identification Pipeline")
    parser.add_argument("--input",  required=True,  help="Path to input text file")
    parser.add_argument("--output", required=True,  help="Path to write redacted text")
    parser.add_argument("--audit",  required=True,  help="Path to write audit log (JSON)")
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
        default="gpt-sw3",
        choices=list(LLM_BACKENDS.keys()),
        help="LLM backend to use (default: gpt-sw3)"
    )
    args = parser.parse_args()

    config = {**BASE_CONFIG, **LLM_BACKENDS[args.llm], "mode": args.mode}
    print(f"Mode: {args.mode} | LLM: {args.llm}")

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
