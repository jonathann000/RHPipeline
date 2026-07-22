"""
Label Studio export — turns pipeline entities into Label Studio
pre-annotation tasks, so the same span/label/risk/source data already
produced for redaction can be visualized and corrected in Label Studio
instead of only inspected via the audit JSON.

Label/risk/source are three separate facets of the same span, but a single
Label Studio "Labels" control only carries one taxonomy per region. So
`label` is the primary Labels control (drives the highlight color), while
`risk` and `source` are each a `perRegion` Choices control targeting the
same text object. All three result items for one entity share the same
`id` — that shared id is what tells Label Studio they're one annotated
region wearing three hats, not three unrelated regions.
"""

import json
import os
from entities import Entity, NEVER_REDACT_LABELS
from redaction import PLACEHOLDERS

# Deterministic, visually distinct categorical palette, assigned in
# sorted-label order so colors stay stable across runs regardless of which
# labels happen to appear in a given document.
_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#000075",
]

LABELS = sorted(set(PLACEHOLDERS) | NEVER_REDACT_LABELS)
LABEL_COLORS = {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(LABELS)}

# Traffic-light coloring for risk — intuitive independent of label color.
RISK_COLORS = {"high": "#e74c3c", "medium": "#f39c12", "low": "#2ecc71"}

SOURCES = ["rule", "bert", "gazetteer", "llm", "propagated"]


def build_labeling_config() -> str:
    """
    The Label Studio XML labeling interface matching LABELS/risk/source.
    Import this once when setting up the Label Studio project; regenerate
    and re-import it if a new entity label is ever added to the taxonomy.
    """
    label_tags = "\n".join(
        f'    <Label value="{label}" background="{color}"/>'
        for label, color in LABEL_COLORS.items()
    )
    risk_tags = "\n".join(
        f'    <Choice value="{risk}" background="{color}"/>'
        for risk, color in RISK_COLORS.items()
    )
    source_tags = "\n".join(f'    <Choice value="{source}"/>' for source in SOURCES)
    return f"""<View>
  <Text name="text" value="$text"/>
  <Labels name="label" toName="text">
{label_tags}
  </Labels>
  <Choices name="risk" toName="text" perRegion="true" required="false" showInline="true">
{risk_tags}
  </Choices>
  <Choices name="source" toName="text" perRegion="true" required="false" showInline="true">
{source_tags}
  </Choices>
</View>
"""


def _entity_to_results(entity: Entity, region_id: str) -> list[dict]:
    """The label/risk/source result items for one entity, sharing region_id."""
    value_base = {"start": entity.start, "end": entity.end, "text": entity.text}
    meta_text = [f"source: {entity.source}", f"confidence: {entity.confidence:.2f}"]
    if entity.generalized:
        meta_text.append(f"generalized: {entity.generalized}")

    results = [
        {
            "id": region_id,
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {**value_base, "labels": [entity.label]},
            "meta": {"text": meta_text},
        },
        {
            "id": region_id,
            "from_name": "risk",
            "to_name": "text",
            "type": "choices",
            "value": {**value_base, "choices": [entity.risk]},
        },
    ]
    if entity.source in SOURCES:
        results.append({
            "id": region_id,
            "from_name": "source",
            "to_name": "text",
            "type": "choices",
            "value": {**value_base, "choices": [entity.source]},
        })
    return results


def build_task(text: str, entities: list[Entity], task_data_extra: dict | None = None) -> dict:
    """One Label Studio task (one document) with all entities pre-annotated as a single prediction."""
    result = []
    for i, entity in enumerate(sorted(entities, key=lambda e: e.start)):
        result.extend(_entity_to_results(entity, region_id=f"ent_{i}"))
    return {
        "data": {"text": text, **(task_data_extra or {})},
        "predictions": [{"model_version": "rhpipeline", "result": result}],
    }


def write_label_studio_export(
    path: str,
    text: str,
    entities: list[Entity],
    task_data_extra: dict | None = None,
    append: bool = False,
) -> None:
    """
    Write this document's task to `path`, replacing whatever was there
    before by default — matches the common single-document dev loop (run
    the pipeline, look at just this note in Label Studio, rerun after a
    prompt tweak without stale tasks piling up in the file).

    append=True instead accumulates this task onto `path`'s existing task
    list, for the different use case of batching many distinct documents
    into one file for a single bulk import (e.g. building an annotation
    corpus) rather than inspecting one document at a time.
    """
    task = build_task(text, entities, task_data_extra)
    tasks = []
    if append and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                tasks = json.load(f)
            except json.JSONDecodeError:
                tasks = []
    tasks.append(task)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
