"""
Rebuild redacted text from a Label Studio export — the human-reviewed layer.

Workflow this closes:
    pipeline  ->  Label Studio pre-annotations  ->  human verifies/edits/adds
              ->  export JSON  ->  THIS SCRIPT  ->  final redacted document(s)

For each task it takes the human **annotations** (falling back to the model
**predictions** for any task with no annotation yet), reconstructs Entity
objects from the label/risk/source regions — plus the `generalized` text
preserved in each region's `meta` — and then runs the SAME redaction machinery
the pipeline uses (build_redaction_plan + redact_document). So the reviewed
output is rendered byte-for-byte the way the pipeline would render those spans:
direct identifiers -> placeholders, quasi-identifiers -> their generalization
(or a placeholder if none / if --no-generalize), medications kept verbatim,
and overlapping spans resolved by the same priority rules.

Usage:
    python label_studio_rebuild.py --input data/out/qwen32b-synthetic_note1.json
    python label_studio_rebuild.py --input export.json --out-dir data/out/reviewed --print
    python label_studio_rebuild.py --input export.json --no-generalize   # placeholders only
"""

import argparse
import json
import os
from datetime import datetime, timezone

from entities import Entity, build_redaction_plan, ALWAYS_DIRECT_LABELS, NEVER_REDACT_LABELS
from redaction import redact_document, resolve_replacement, PLACEHOLDERS


def _parse_generalized(meta: dict | None) -> str | None:
    """Recover the generalization the pipeline stored in a region's meta.text
    (a line like 'generalized: mindre ort i ...'). None if absent."""
    for line in (meta or {}).get("text", []):
        if isinstance(line, str) and line.startswith("generalized:"):
            return line[len("generalized:"):].strip() or None
    return None


def entities_from_result(result: list) -> list[Entity]:
    """Turn one annotation/prediction 'result' list into Entity objects.
    The label/risk/source items for one span share an `id`; group by it."""
    regions: dict[str, dict] = {}
    for item in result:
        regions.setdefault(item.get("id"), {})[item.get("from_name")] = item

    entities = []
    for parts in regions.values():
        label_item = parts.get("label")
        if not label_item:
            continue  # a stray region with no Labels control — nothing to redact by
        v = label_item["value"]
        labels = v.get("labels") or []
        if not labels:
            continue
        risk = "low"
        if "risk" in parts:
            choices = parts["risk"]["value"].get("choices") or []
            if choices:
                risk = choices[0]
        source = "manual"  # human-added spans have no source choice — mark them
        if "source" in parts:
            choices = parts["source"]["value"].get("choices") or []
            if choices:
                source = choices[0]
        entities.append(Entity(
            text=v.get("text", ""),
            label=labels[0],
            start=v["start"],
            end=v["end"],
            source=source,
            generalized=_parse_generalized(label_item.get("meta")),
            risk=risk,
        ))
    return entities


def _leak_variants(t: str) -> list[str]:
    """The span's text plus a Swedish-genitive-trimmed form (mirrors
    entities.remove_overlapping_entities' own leak check)."""
    t = t.strip()
    variants = [t]
    if len(t) > 3 and t.lower().endswith("s"):
        variants.append(t[:-1])
    return variants


def render_merged(text: str, entities: list[Entity], no_generalize: bool) -> str:
    """
    Collapse every cluster of overlapping spans into ONE readable redaction
    over the cluster's full extent, instead of build_redaction_plan's safe-but-
    fragmented `[A][B][C]`. Safety is preserved because the replacement always
    covers the entire extent — nothing nested can survive:

      - single-span clusters render exactly as the pipeline would (placeholder,
        generalization, or verbatim for medication) via resolve_replacement.
      - if a direct-identifier label (personnummer, name, date, ...) owns or
        ties for the largest span, the whole extent gets that hard placeholder
        (so exact dates stay [DATUM], not "mars 2024").
      - otherwise the LARGEST span's generalization is used, but only if it
        covers the whole extent and does NOT textually re-introduce any nested
        span (leak check, incl. genitive -s). This safely subsumes a nested
        name ("make med specialistyrke..." doesn't contain "Anders") while
        rejecting a leaky one ("adress i Uppsala" contains the nested
        "Uppsala") -> placeholder. Either way the full extent is replaced, so a
        nested identifier is always removed.
    """
    ents = [e for e in entities if e.end > e.start and e.text.strip()]
    if not ents:
        return text
    ents.sort(key=lambda e: (e.start, -(e.end - e.start)))

    # Group into maximal connected components of overlapping spans.
    components: list[list[Entity]] = []
    current = [ents[0]]
    current_end = ents[0].end
    for e in ents[1:]:
        if e.start < current_end:            # overlaps the running component
            current.append(e)
            current_end = max(current_end, e.end)
        else:
            components.append(current)
            current = [e]
            current_end = e.end
    components.append(current)

    spans: list[tuple[int, int, str]] = []
    for comp in components:
        lo = min(e.start for e in comp)
        hi = max(e.end for e in comp)
        if len(comp) == 1:
            spans.append((lo, hi, resolve_replacement(comp[0], no_generalize)))
            continue

        # The largest span owns the cluster. Medications never "own" a mixed
        # cluster (they'd be kept verbatim over something sensitive); a direct
        # label wins ties over a co-extensive quasi so exact dates/names/IDs
        # still hard-redact.
        candidates = [e for e in comp if e.label not in NEVER_REDACT_LABELS] or comp
        dominant = max(
            candidates,
            key=lambda e: (e.end - e.start, e.label in ALWAYS_DIRECT_LABELS),
        )
        others = [e for e in comp if e is not dominant]

        if dominant.label in ALWAYS_DIRECT_LABELS:
            replacement = PLACEHOLDERS.get(dominant.label, "[REDAKTERAD]")
        else:
            gen = resolve_replacement(dominant, no_generalize)
            covers = dominant.start == lo and dominant.end == hi
            is_generalization = not gen.startswith("[")
            leaks = any(
                v.lower() in gen.lower()
                for m in others for v in _leak_variants(m.text)
            )
            if covers and is_generalization and not leaks:
                replacement = gen
            else:
                replacement = PLACEHOLDERS.get(dominant.label, "[REDAKTERAD]")
        spans.append((lo, hi, replacement))

    # Merge touching spans that resolved to the same text (same as redact_document).
    spans.sort()
    merged: list[tuple[int, int, str]] = []
    for lo, hi, rep in spans:
        if merged and merged[-1][1] == lo and merged[-1][2] == rep:
            merged[-1] = (merged[-1][0], hi, rep)
        else:
            merged.append((lo, hi, rep))

    result = text
    for lo, hi, rep in sorted(merged, key=lambda m: m[0], reverse=True):
        result = result[:lo] + rep + result[hi:]
    return result


def pick_result(task: dict, layer: str) -> tuple[list, str]:
    """Return (result_items, which_layer). 'auto' prefers a human annotation,
    falling back to the model prediction for tasks not yet reviewed."""
    if layer in ("auto", "annotations"):
        for a in task.get("annotations", []):
            if not a.get("was_cancelled") and a.get("result") is not None:
                return a["result"], "annotations"
    if layer in ("auto", "predictions") and task.get("predictions"):
        return task["predictions"][0].get("result", []), "predictions"
    return [], "none"


def stem_for(task: dict, idx: int) -> str:
    src = task.get("data", {}).get("source_file")
    if src:
        base = os.path.basename(src)
        return base[:-4] if base.endswith(".txt") else base
    return f"task{idx}"


def build_audit(entities: list[Entity], no_generalize: bool) -> list[dict]:
    return [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original": e.text,
            "label": e.label,
            "start": e.start,
            "end": e.end,
            "source": e.source,
            "risk": e.risk,
            "generalized_to": resolve_replacement(e, no_generalize),
        }
        for e in sorted(entities, key=lambda x: x.start)
    ]


def main():
    ap = argparse.ArgumentParser(description="Rebuild redacted text from a Label Studio export.")
    ap.add_argument("--input", required=True, help="Label Studio export JSON (single task or a list)")
    ap.add_argument("--out-dir", default="data/out/reviewed", help="Where to write <stem>.reviewed.{redacted.txt,audit.json}")
    ap.add_argument("--layer", choices=["auto", "annotations", "predictions"], default="auto",
                    help="Which layer to rebuild from (default auto: human annotation, else prediction)")
    ap.add_argument("--no-generalize", action="store_true",
                    help="Never trust generalizations — placeholder every quasi-identifier")
    ap.add_argument("--split-overlaps", action="store_true",
                    help="Render overlapping spans as separate fragmented placeholders "
                         "(the pipeline's build_redaction_plan behavior) instead of the "
                         "default readable collapse into one span per overlap cluster")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Also print each reconstructed redaction")
    args = ap.parse_args()

    data = json.load(open(args.input, encoding="utf-8"))
    tasks = data if isinstance(data, list) else [data]
    os.makedirs(args.out_dir, exist_ok=True)

    for i, task in enumerate(tasks):
        text = task.get("data", {}).get("text", "")
        result, used = pick_result(task, args.layer)
        entities = entities_from_result(result)

        if args.split_overlaps:
            plan = build_redaction_plan(text, entities)
            redacted = redact_document(text, plan, args.no_generalize)
        else:
            redacted = render_merged(text, entities, args.no_generalize)

        stem = stem_for(task, i)
        red_path = os.path.join(args.out_dir, f"{stem}.reviewed.redacted.txt")
        aud_path = os.path.join(args.out_dir, f"{stem}.reviewed.audit.json")
        with open(red_path, "w", encoding="utf-8") as f:
            f.write(redacted)
        with open(aud_path, "w", encoding="utf-8") as f:
            json.dump(build_audit(entities, args.no_generalize), f, ensure_ascii=False, indent=2)

        gens = sum(1 for e in entities if e.generalized)
        print(f"[{stem}] {len(entities)} regions from '{used}' layer, {gens} with a generalization -> {red_path}")
        if args.do_print:
            print("=" * 70)
            print(redacted)
            print("=" * 70)


if __name__ == "__main__":
    main()
