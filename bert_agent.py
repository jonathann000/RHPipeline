"""
BERT NER agent — wraps your fine-tuned ModelBERTF checkpoint.
Handles direct identifier detection (names, structured PII the rules miss).
"""

from entities import Entity


class BERTAgent:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detect(self, text: str) -> list[Entity]:
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
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

        return entities

    def _finalize(self, span: dict, text: str) -> Entity:
        return Entity(
            text=text[span["start"]:span["end"]],
            label=span["label"],
            start=span["start"],
            end=span["end"],
            source="bert",
            confidence=0.95,
        )
