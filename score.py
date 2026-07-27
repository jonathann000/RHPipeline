"""
Score a pipeline run against the synthetic gold-standard key.

Measures, per note and in aggregate:
  - Recall on direct identifiers   (fraction actually removed from the output)
  - Recall on quasi-identifiers     (fraction actually removed from the output)
  - Decoy preservation              (precision proxy — fraction of things that
                                     should NOT be touched that survived intact)
  - Medication handling             (flagged in the audit AND kept verbatim)

The de-identification goal is that sensitive text is *gone from the redacted
output*, so recall here is measured against redacted.txt (did the span actually
disappear?), not merely against the audit log (was it detected?). Detection is
reported separately as a secondary number, since a span can be redacted via
coreference/overlap even when labeled differently than the key expects.

Matching is on normalized surface text (casefold + whitespace-collapsed),
not character offsets — see data/synthetic_notes_key.json's _about. This is
deliberately approximate (substring match; inflected forms like a genitive
"-s" are caught as leaks, short tokens can in principle false-match inside a
longer word) — it's an evaluation aid, not a certified metric.

Usage:
    # Batch: score every note whose <stem>.redacted.txt + <stem>.audit.json
    # exist under --runs-dir (stem = note filename without .txt).
    python score.py --key data/synthetic_notes_key.json --runs-dir data/out

    # Single note:
    python score.py --note synthetic_note2.txt \
        --redacted data/out/synthetic_note2.redacted.txt \
        --audit data/out/synthetic_note2.audit.json

    # List every miss / over-redaction for inspection:
    python score.py --runs-dir data/out --verbose
"""

import argparse
import json
import os
import re


def norm(s: str) -> str:
    """Casefold + collapse whitespace — the canonical form all matching uses."""
    return re.sub(r"\s+", " ", (s or "").strip().casefold())


def present(needle: str, haystack_norm: str) -> bool:
    """True if `needle` (normalized) appears as a substring of already-normalized text."""
    n = norm(needle)
    return bool(n) and n in haystack_norm


def overlaps(a: str, b: str) -> bool:
    """True if either normalized string contains the other — a loose span overlap."""
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a in b or b in a)


def score_note(key_entry: dict, audit: list, redacted_text: str) -> dict:
    red_norm = norm(redacted_text)
    audit_originals = [a.get("original", "") for a in audit]
    audit_meds = [a.get("original", "") for a in audit if a.get("label") == "medication"]

    def detected(text: str) -> bool:
        return any(overlaps(text, o) for o in audit_originals)

    buckets = {"direct": {"total": 0, "removed": 0, "detected": 0, "misses": []},
               "quasi":  {"total": 0, "removed": 0, "detected": 0, "misses": []}}

    for item in key_entry.get("should_redact", []):
        b = buckets[item["type"]]
        b["total"] += 1
        # "removed" = the sensitive surface text no longer appears verbatim in
        # the output (it was replaced by a placeholder/generalization, or a
        # higher-priority span consumed it).
        if not present(item["text"], red_norm):
            b["removed"] += 1
        else:
            b["misses"].append(item)
        if detected(item["text"]):
            b["detected"] += 1

    # Decoys: should still be present verbatim (never flagged/redacted).
    decoys = key_entry.get("should_not_flag", [])
    preserved, over_redacted = 0, []
    for d in decoys:
        if present(d, red_norm):
            preserved += 1
        else:
            over_redacted.append(d)

    # Medications: should be flagged (label=medication in audit) AND kept verbatim.
    meds = key_entry.get("medications_flag_keep", [])
    med_ok, med_missing_flag, med_wrongly_redacted = 0, [], []
    for m in meds:
        flagged = any(overlaps(m, o) for o in audit_meds)
        kept = present(m, red_norm)
        if flagged and kept:
            med_ok += 1
        if not flagged:
            med_missing_flag.append(m)
        if not kept:
            med_wrongly_redacted.append(m)

    return {
        "direct": buckets["direct"],
        "quasi": buckets["quasi"],
        "decoys": {"total": len(decoys), "preserved": preserved, "over_redacted": over_redacted},
        "meds": {"total": len(meds), "ok": med_ok,
                 "missing_flag": med_missing_flag, "wrongly_redacted": med_wrongly_redacted},
    }


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "   n/a"


def main():
    ap = argparse.ArgumentParser(description="Score a pipeline run against the synthetic gold key.")
    ap.add_argument("--key", default="data/synthetic_notes_key.json")
    ap.add_argument("--runs-dir", help="Directory holding <stem>.redacted.txt and <stem>.audit.json per note")
    ap.add_argument("--note", help="Single-note mode: the note's key (e.g. synthetic_note2.txt)")
    ap.add_argument("--redacted", help="Single-note mode: path to that note's redacted.txt")
    ap.add_argument("--audit", help="Single-note mode: path to that note's audit.json")
    ap.add_argument("--verbose", action="store_true", help="List every miss / over-redaction")
    args = ap.parse_args()

    key = json.load(open(args.key, encoding="utf-8"))
    note_names = [n for n in key if not n.startswith("_")]

    # Resolve which (note -> redacted_path, audit_path) pairs to score.
    jobs = []
    if args.note:
        if not (args.redacted and args.audit):
            ap.error("--note requires --redacted and --audit")
        jobs.append((args.note, args.redacted, args.audit))
    elif args.runs_dir:
        for n in note_names:
            stem = n[:-4] if n.endswith(".txt") else n
            red = os.path.join(args.runs_dir, f"{stem}.redacted.txt")
            aud = os.path.join(args.runs_dir, f"{stem}.audit.json")
            if os.path.exists(red) and os.path.exists(aud):
                jobs.append((n, red, aud))
            else:
                print(f"(skipping {n}: expected {stem}.redacted.txt + {stem}.audit.json in {args.runs_dir})")
    else:
        ap.error("provide either --runs-dir (batch) or --note/--redacted/--audit (single)")

    if not jobs:
        print("Nothing to score — no matching output files found.")
        return

    header = f"{'note':22} {'direct-recall':>13} {'quasi-recall':>13} {'decoy-keep':>11} {'med-ok':>8}"
    print(header)
    print("-" * len(header))

    agg = {"direct": [0, 0], "quasi": [0, 0], "decoys": [0, 0], "meds": [0, 0]}
    all_misses = []

    for note, red_path, aud_path in jobs:
        redacted = open(red_path, encoding="utf-8").read()
        audit = json.load(open(aud_path, encoding="utf-8"))
        r = score_note(key[note], audit, redacted)

        agg["direct"][0] += r["direct"]["removed"];  agg["direct"][1] += r["direct"]["total"]
        agg["quasi"][0]  += r["quasi"]["removed"];   agg["quasi"][1]  += r["quasi"]["total"]
        agg["decoys"][0] += r["decoys"]["preserved"]; agg["decoys"][1] += r["decoys"]["total"]
        agg["meds"][0]   += r["meds"]["ok"];          agg["meds"][1]   += r["meds"]["total"]

        print(f"{note:22} "
              f"{pct(r['direct']['removed'], r['direct']['total'])} ({r['direct']['removed']:2}/{r['direct']['total']:<2}) "
              f"{pct(r['quasi']['removed'], r['quasi']['total'])} ({r['quasi']['removed']:2}/{r['quasi']['total']:<2}) "
              f"{pct(r['decoys']['preserved'], r['decoys']['total'])} "
              f"{pct(r['meds']['ok'], r['meds']['total'])}")

        if args.verbose:
            all_misses.append((note, r))

    print("-" * len(header))
    print(f"{'TOTAL':22} "
          f"{pct(*agg['direct'])} ({agg['direct'][0]}/{agg['direct'][1]}) "
          f"{pct(*agg['quasi'])} ({agg['quasi'][0]}/{agg['quasi'][1]}) "
          f"{pct(*agg['decoys'])} "
          f"{pct(*agg['meds'])}")

    print("\nLegend: direct/quasi-recall = fraction of expected identifiers actually "
          "removed from the output. decoy-keep = fraction of decoys left intact "
          "(precision proxy). med-ok = medications both flagged in the audit AND kept verbatim.")

    if args.verbose:
        for note, r in all_misses:
            lines = []
            for kind in ("direct", "quasi"):
                for m in r[kind]["misses"]:
                    lines.append(f"    MISS  [{kind}/{m['label']}] {m['text']!r}")
            for d in r["decoys"]["over_redacted"]:
                lines.append(f"    OVER-REDACTED (decoy) {d!r}")
            for m in r["meds"]["wrongly_redacted"]:
                lines.append(f"    MED redacted (should keep) {m!r}")
            for m in r["meds"]["missing_flag"]:
                lines.append(f"    MED not flagged in audit {m!r}")
            if lines:
                print(f"\n{note}:")
                print("\n".join(lines))


if __name__ == "__main__":
    main()
