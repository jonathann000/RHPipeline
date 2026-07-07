"""
BERT NER agent — wraps your fine-tuned ModelBERTF checkpoint.
Handles direct identifier detection (names, structured PII the rules miss).
"""

from entities import Entity


class BERTAgent:
    def __init__(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from device import resolve_device_map

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            device_map=resolve_device_map(),
        )
        self.model.eval()
        self.device = self.model.device
        # Checkpoints vary wildly in real context length (512 for classic
        # BERT/RoBERTa, 128k for this MoE-based OAI model) — read it from the
        # loaded model instead of hardcoding, so we never truncate a document
        # the model could actually have handled in one pass.
        self.max_length = getattr(self.model.config, "max_position_embeddings", 512)

    def detect(self, text: str) -> list[Entity]:
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        import torch
        with torch.no_grad():
            logits = self.model(**encoding).logits[0]

        predicted_ids = logits.argmax(dim=-1).tolist()
        id2label = self.model.config.id2label

        # Collapse B/I spans into single entities
        entities = []
        current: dict | None = None

        for idx, (pred_id, (char_start, char_end)) in enumerate(
            zip(predicted_ids, offset_mapping)
        ):
            if char_start == 0 and char_end == 0:
                # Special token ([CLS], [SEP], [PAD])
                if current:
                    entities.append(self._finalize(current, text))
                    current = None
                continue

            label = id2label[pred_id]

            if label == "O":
                if current:
                    entities.append(self._finalize(current, text))
                    current = None
                continue

            prefix, cat = label.split("-", 1)

            if prefix == "B" or (prefix == "S"):
                if current:
                    entities.append(self._finalize(current, text))
                current = {"label": cat, "start": char_start, "end": char_end}

            elif prefix in ("I", "E") and current and current["label"] == cat:
                current["end"] = char_end

            else:
                # Label mismatch — close current, start new
                if current:
                    entities.append(self._finalize(current, text))
                current = {"label": cat, "start": char_start, "end": char_end}

        if current:
            entities.append(self._finalize(current, text))

        return self._merge_fragments(text, entities)

    def _finalize(self, span: dict, text: str) -> Entity:
        return Entity(
            text=text[span["start"]:span["end"]],
            label=span["label"],
            start=span["start"],
            end=span["end"],
            source="bert",
            confidence=0.95,
        )

    def _merge_fragments(self, text: str, entities: list[Entity]) -> list[Entity]:
        """
        Some checkpoints emit inconsistent IOB tags for what should be one
        entity — back-to-back B- tags for a first + last name, or a tag that
        only covers part of a word's subword pieces. Snap each span out to
        whole-word boundaries, then merge adjacent same-label entities that
        are separated only by whitespace into a single entity.
        """
        snapped = sorted(
            (self._fix_numeric_person(self._snap_to_word_boundary(text, e)) for e in entities),
            key=lambda e: e.start,
        )

        merged: list[Entity] = []
        for entity in snapped:
            if (
                merged
                and merged[-1].label == entity.label
                and _is_mergeable_gap(text[merged[-1].end:entity.start])
            ):
                prev = merged[-1]
                merged[-1] = Entity(
                    text=text[prev.start:entity.end],
                    label=prev.label,
                    start=prev.start,
                    end=entity.end,
                    source="bert",
                    confidence=min(prev.confidence, entity.confidence),
                )
            else:
                merged.append(entity)

        return merged

    def _fix_numeric_person(self, entity: Entity) -> Entity:
        """
        A real person's name never contains a digit — a digit-containing
        span labeled private_person is a misclassified number phrase (age,
        etc.), not a boundary issue. Relabel rather than drop, so it still
        gets redacted under the correct category.
        """
        if entity.label == "private_person" and any(c.isdigit() for c in entity.text):
            return Entity(
                text=entity.text,
                label="demographics",
                start=entity.start,
                end=entity.end,
                source=entity.source,
                confidence=entity.confidence,
            )
        return entity

    def _snap_to_word_boundary(self, text: str, entity: Entity) -> Entity:
        start, end = entity.start, entity.end
        # Trim whitespace the offset mapping may have included at the edges.
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        # Expand outward to cover the whole word if the span starts/ends mid-word.
        while start > 0 and _is_word_char(text[start - 1]):
            start -= 1
        while end < len(text) and _is_word_char(text[end]):
            end += 1
        return Entity(
            text=text[start:end],
            label=entity.label,
            start=start,
            end=end,
            source=entity.source,
            confidence=entity.confidence,
        )


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "-"


def _is_mergeable_gap(gap: str) -> bool:
    """
    Whitespace-only gaps are always mergeable (e.g. first + last name).
    Also tolerate a single trailing period (e.g. "Dr. Namn") — abbreviated
    titles are common right before a name and belong to the same entity.
    A comma or other punctuation is left unmerged, since that more often
    separates genuinely distinct list items.
    """
    stripped = gap.strip()
    return stripped == "" or stripped == "."
