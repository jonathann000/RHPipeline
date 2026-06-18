"""
Local smoke test — runs the pipeline with mock LLM and no BERT.
No GPU, no model downloads, no dependencies beyond stdlib needed.

Usage:
    python test_local.py
"""

from pipeline import PIIPipeline

CONFIG = {
    "mode": "no_bert",
    "llm_backend": "mock",
    "llm_model_path": "",       # unused by mock
    "bert_model_path": "",      # unused in no_bert mode
}

SAMPLE_TEXT = (
    "Patienten Erik Svensson, personnummer 850312-1234, "
    "tel 070-123 45 67, e-post erik.svensson@example.com. "
    "67-årig ensamstående snickare bosatt i 412 63 Göteborg. "
    "Diagnostiserad med Huntingtons sjukdom 2024-03-15."
)


def main():
    pipe = PIIPipeline(CONFIG)
    result = pipe.run(SAMPLE_TEXT)

    print("=== ORIGINAL ===")
    print(result.original_text)
    print("\n=== REDACTED ===")
    print(result.redacted_text)
    print(f"\n=== ENTITIES ({len(result.entities)}) ===")
    for e in sorted(result.entities, key=lambda x: x.start):
        print(f"  [{e.source:4}] {e.label:20} | \"{e.text}\" -> {e.generalized or '[REDACTED]'}")
    print("\n=== AUDIT LOG ===")
    for entry in result.audit_log:
        print(f"  {entry}")


if __name__ == "__main__":
    main()
