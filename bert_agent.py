"""
BERT NER agent — wraps your fine-tuned ModelBERTF checkpoint.
Handles direct identifier detection (names, structured PII the rules miss).
"""

import logging
from entities import Entity, remove_overlapping_entities
from chunking import chunk_by_sentences

logger = logging.getLogger(__name__)

# Minimum softmax confidence to trust a non-O tag prediction — see its use
# in _detect_chunk for the empirical measurement behind this value.
_MIN_TAG_CONFIDENCE = 0.75

# Maps a checkpoint's own raw category names to this pipeline's internal
# label taxonomy, so redaction.py/entities.py never need to know which BERT
# model produced an entity. The older ModelBERTF/Roberta checkpoint already
# emits our internal names directly (private_person, private_email, ...) —
# those aren't in this dict, so .get(cat, cat) passes them through
# unchanged. The newer MBERTHIPAA checkpoint (HIPAA Safe Harbor's 18
# identifier categories) uses its own vocabulary, mapped here:
# - SSN/Med_Num/HPV_Num/Account_Num/Liscense_Num all collapse to
#   account_number — each is a different flavor of "unique ID number", the
#   same bucket rule_agent.py already uses for personnummer/passport
#   numbers, so this is consistent with existing precedent, not a new idea.
# - Fax collapses to private_phone — a fax number is redacted for the same
#   reason and the same way as a phone number; a separate category buys
#   nothing.
# - Vehicle/Device_Num/URL/IP/Bio/Face are genuinely new concepts this
#   pipeline had no bucket for at all — given dedicated categories (see
#   ALWAYS_DIRECT_LABELS/PLACEHOLDERS) rather than being crammed into an
#   unrelated existing one.
# - "etc" is HIPAA Safe Harbor's own catch-all ("any other unique
#   identifying number, characteristic, or code") — mapped to a matching
#   catch-all here rather than reusing "secret", which already means
#   something more specific (passwords/PINs) in this pipeline.
_BERT_LABEL_MAP = {
    "Name": "private_person",
    "Address": "private_address",
    "Dates": "private_date",
    "Phone": "private_phone",
    "Fax": "private_phone",
    "Email": "private_email",
    "SSN": "account_number",
    "Med_Num": "account_number",
    "HPV_Num": "account_number",
    "Account_Num": "account_number",
    "Liscense_Num": "account_number",
    "Vehicle": "private_vehicle",
    "Device_Num": "private_device",
    "URL": "private_url",
    "IP": "private_ip",
    "Bio": "private_biometric",
    "Face": "private_photo",
    "etc": "private_other",
}


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

    def detect(
        self,
        text: str,
        chunk_size: int | None = None,
        chunk_overlap: int = 50,
        chunk_by: str = "sentences",
    ) -> list[Entity]:
        """
        chunk_size: max characters per chunk. None (default) auto-decides —
                    if the document's real token count fits within
                    self.max_length it's processed in a single pass;
                    otherwise chunks are sized as large as safely fits,
                    computed from this tokenizer's own token->character
                    mapping on the actual text rather than a generic
                    chars-per-token guess (which varies by language/model).
                    Pass an explicit value to override.
        chunk_overlap: characters shared between consecutive windows in
                    "chars" mode, so an entity sitting on a window boundary
                    is fully visible in at least one window instead of being
                    cut in half. Ignored for "sentences" mode.
        chunk_by: "sentences" (default) — group whole lines/sentences up to
                                the chunk size, never cutting one in half
                                (see chunk_by_sentences) — guarantees no
                                mid-entity cuts by construction.
                  "chars"     — fixed-size sliding character windows.
                                Benchmarked statistically indistinguishable
                                from "sentences" on real accuracy at a large
                                enough chunk size, but offers no structural
                                guarantee against cutting mid-entity.
        """
        if chunk_size is None:
            full = self.tokenizer(text, truncation=False, return_offsets_mapping=True)
            token_count = len(full["input_ids"])
            if token_count <= self.max_length:
                return self._merge_fragments(text, self._detect_chunk(text, offset=0))

            # Chunk as large as safely fits under max_length, with an 85%
            # margin to absorb re-tokenization drift at chunk cut points
            # (a chunk's own [CLS]/[SEP] and slightly different subword
            # splits near its edges vs. the full-document tokenization).
            safe_token_budget = max(int(self.max_length * 0.85), 1)
            offsets = full["offset_mapping"]
            cutoff_idx = min(safe_token_budget, len(offsets) - 1)
            chunk_size = offsets[cutoff_idx][1] or len(text)
            logger.info(
                f"Document has {token_count} tokens (limit {self.max_length}) — "
                f"chunking at ~{chunk_size} chars per chunk ({chunk_by})"
            )

        if len(text) <= chunk_size:
            entities = self._detect_chunk(text, offset=0)
        elif chunk_by == "sentences":
            entities = []
            for start, end in chunk_by_sentences(text, chunk_size):
                entities.extend(self._detect_chunk(text[start:end], offset=start))
            entities = remove_overlapping_entities(entities)
        else:
            entities = []
            start = 0
            step = max(chunk_size - chunk_overlap, 1)
            while start < len(text):
                end = min(start + chunk_size, len(text))
                entities.extend(self._detect_chunk(text[start:end], offset=start))
                if end == len(text):
                    break
                start += step
            # Overlap regions can produce the same entity twice (once from
            # each neighboring chunk) — resolve like any other overlap.
            entities = remove_overlapping_entities(entities)

        return self._merge_fragments(text, entities)

    def _detect_chunk(self, text: str, offset: int) -> list[Entity]:
        """Run one forward pass over `text`, offsetting spans by `offset` into the full document."""
        token_count = len(self.tokenizer(text, truncation=False)["input_ids"])
        if token_count > self.max_length:
            logger.warning(
                f"BERT input has {token_count} tokens, exceeding this model's "
                f"{self.max_length}-token limit — {token_count - self.max_length} tokens "
                f"will be silently dropped by truncation. Pass chunk_size to "
                f"BERTAgent.detect() to process the whole document instead."
            )

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
        confidences = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
        id2label = self.model.config.id2label

        # Collapse B/I spans into single entities
        entities = []
        current: dict | None = None

        for idx, (pred_id, conf, (char_start, char_end)) in enumerate(
            zip(predicted_ids, confidences, offset_mapping)
        ):
            if char_start == 0 and char_end == 0:
                # Special token ([CLS], [SEP], [PAD])
                if current:
                    entities.append(self._finalize(current, text, offset))
                    current = None
                continue

            label = id2label[pred_id]

            # A prediction only barely above chance shouldn't be trusted as
            # a real tag — measured empirically on MBERTHIPAA: genuine
            # entities (names, phone numbers) sit at 0.95-1.0 confidence,
            # while observed false positives (plain numbers in a lab-value
            # list misread as Address) sat at 0.57-0.65. Treating a
            # low-confidence non-O prediction as O closes that specific,
            # measured gap instead of guessing at a threshold.
            if label != "O" and conf < _MIN_TAG_CONFIDENCE:
                label = "O"

            if label == "O":
                if current:
                    entities.append(self._finalize(current, text, offset))
                    current = None
                continue

            prefix, cat = label.split("-", 1)
            cat = _BERT_LABEL_MAP.get(cat, cat)

            if prefix == "B" or (prefix == "S"):
                if current:
                    entities.append(self._finalize(current, text, offset))
                current = {"label": cat, "start": char_start, "end": char_end}

            elif prefix in ("I", "E") and current and current["label"] == cat:
                current["end"] = char_end

            else:
                # Label mismatch — close current, start new
                if current:
                    entities.append(self._finalize(current, text, offset))
                current = {"label": cat, "start": char_start, "end": char_end}

        if current:
            entities.append(self._finalize(current, text, offset))

        return entities

    def _finalize(self, span: dict, text: str, offset: int = 0) -> Entity:
        return Entity(
            text=text[span["start"]:span["end"]],
            label=span["label"],
            start=span["start"] + offset,
            end=span["end"] + offset,
            source="bert",
            confidence=0.95,
            # Always high — a NER hit on one of these categories is a
            # direct-identifier finding, not a graded judgment call like the
            # LLM's quasi-identifier risk. See rule_agent.py's identical
            # reasoning for why this can't be left at Entity's "low" default.
            risk="high",
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
        snapped = [e for e in snapped if not _is_bogus_email(e)]

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
                    risk="high",
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
                risk="high",
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
            risk=entity.risk,
        )


def _is_word_char(c: str) -> bool:
    return c.isalnum() or c == "-"


def _is_bogus_email(entity: Entity) -> bool:
    """
    A real email address always contains '@' — a private_email-labeled span
    without one is a misclassification, not a legitimate finding under any
    category. Chunking without full-document context has been observed to
    trigger this specifically on ordinary period-ending words near a chunk
    boundary (e.g. "utredning.", "kyrkokören."), silently mangling readable
    text into a wrong-category placeholder.
    """
    return entity.label == "private_email" and "@" not in entity.text


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
