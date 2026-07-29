"""
Local BERT-backbone comparison — score each token-classification NER checkpoint
on DIRECT-identifier coverage in ISOLATION (no LLM, rules, or gazetteer), so you
can compare BERT models on CPU locally before committing to a full GPU run.

For each model it runs BERTAgent.detect() over the synthetic notes and reports,
against data/synthetic_notes_key.json:
  - direct-recall : fraction of the key's DIRECT identifiers a BERT span covers
  - decoy-keep    : fraction of decoys BERT leaves untouched (over-detection proxy)
  - detections    : total spans produced (context for the two rates above)

BERT only handles direct identifiers (plus Age), so quasi-identifiers and
medications are deliberately not scored here — this isolates the direct-ID
detector so the numbers reflect the checkpoint alone.

Usage:
    python compare_bert.py                    # all of saved_models/* + models/MBERTHIPAA
    python compare_bert.py --models saved_models/bert-base-cased models/MBERTHIPAA
"""

import argparse
import gc
import glob
import json
import os

# Force CPU so this is portable and avoids MPS op-support surprises on the small
# encoders — patched before BERTAgent's __init__ imports resolve_device_map.
import device
device.resolve_device_map = lambda: {"": "cpu"}

from bert_agent import BERTAgent
from score import overlaps


def main():
    ap = argparse.ArgumentParser(description="Compare BERT NER checkpoints on direct-ID coverage (local, CPU).")
    ap.add_argument("--key", default="data/synthetic_notes_key.json")
    ap.add_argument("--notes-dir", default="data")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Model paths to compare (default: saved_models/* + models/MBERTHIPAA)")
    args = ap.parse_args()

    key = json.load(open(args.key, encoding="utf-8"))
    notes = [n for n in key if not n.startswith("_")]
    texts = {n: open(os.path.join(args.notes_dir, n), encoding="utf-8").read() for n in notes}

    models = args.models
    if not models:
        models = sorted(glob.glob("saved_models/*"))
        if os.path.exists("models/MBERTHIPAA/config.json"):
            models.append("models/MBERTHIPAA")

    header = f"{'model':46} {'direct-recall':>16} {'decoy-keep':>11} {'detections':>11}"
    print(header)
    print("-" * len(header))

    for mpath in models:
        label = os.path.basename(mpath.rstrip("/"))
        try:
            agent = BERTAgent(mpath)
        except Exception as e:
            print(f"{label:46}  LOAD FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue

        try:
            caught = total_direct = kept = total_decoy = det_total = 0
            for n in notes:
                ent_texts = [e.text for e in agent.detect(texts[n])]
                det_total += len(ent_texts)
                for item in key[n]["should_redact"]:
                    if item["type"] != "direct":
                        continue
                    total_direct += 1
                    if any(overlaps(item["text"], t) for t in ent_texts):
                        caught += 1
                for d in key[n]["should_not_flag"]:
                    total_decoy += 1
                    if not any(overlaps(d, t) for t in ent_texts):
                        kept += 1
            dr = f"{100 * caught / total_direct:5.1f}% ({caught}/{total_direct})"
            dk = f"{100 * kept / total_decoy:5.1f}%"
            print(f"{label:46} {dr:>16} {dk:>11} {det_total:>11}")
        except Exception as e:
            print(f"{label:46}  DETECT FAILED: {type(e).__name__}: {str(e)[:60]}")
        finally:
            del agent
            gc.collect()

    print("\ndirect-recall = key direct identifiers a BERT span covers; decoy-keep = decoys "
          "left untouched (higher = less over-detection). Isolated BERT stage — no LLM/rules/gazetteer.")


if __name__ == "__main__":
    main()
