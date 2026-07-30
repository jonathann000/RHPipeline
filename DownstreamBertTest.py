import json
import numpy as np
import torch
from collections import Counter
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# =====================================================================
# 1. Label Studio JSON Parser Bridge
# =====================================================================
import re

def parse_label_studio_export_chunked(json_path_or_data):
    """
    Parses a Label Studio JSON export and splits large documents into smaller chunks
    (by newline or sentence boundary), recalculating annotation offsets for each chunk.
    """
    if isinstance(json_path_or_data, str):
        with open(json_path_or_data, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        raw_data = json_path_or_data

    source_texts = []
    privacy_masks = []

    for entry in raw_data:
        full_text = entry.get("data", {}).get("text", "")
        full_spans = []
        
        # Pull all annotations for the full page
        if entry.get("predictions") and len(entry["predictions"]) > 0:
            results = entry["predictions"][0].get("result", [])
            for res in results:
                val = res.get("value", {})
                if "start" in val and "end" in val and "labels" in val:
                    full_spans.append({
                        "start": val["start"],
                        "end": val["end"],
                        "label": val["labels"][0]
                    })
        
        # --- CHUNKING LOGIC ---
        # Find boundaries to split on: newlines, or periods followed by a space
        split_points = [0]
        for match in re.finditer(r'\n|(?<=\.)\s+', full_text):
            split_points.append(match.end())
        split_points.append(len(full_text))
        
        # Slice the text and distribute the annotations
        for i in range(len(split_points) - 1):
            c_start = split_points[i]
            c_end = split_points[i + 1]
            chunk_text = full_text[c_start:c_end]
            
            # Skip empty or whitespace-only chunks to save compute
            if not chunk_text.strip():
                continue
                
            chunk_spans = []
            for span in full_spans:
                # Check if the annotation sits entirely within this chunk's boundaries
                if span["start"] >= c_start and span["end"] <= c_end:
                    chunk_spans.append({
                        # Recalculate offsets relative to the start of the chunk
                        "start": span["start"] - c_start, 
                        "end": span["end"] - c_start,
                        "label": span["label"]
                    })
            
            source_texts.append(chunk_text)
            privacy_masks.append(chunk_spans)

    return Dataset.from_dict({
        "source_text": source_texts,
        "privacy_mask": privacy_masks
    })

# Initialize your dataset with the new chunked parser
raw_dataset = parse_label_studio_export_chunked("label_studio_gemma4.json")

# For demonstration: split into train/val (In production, use actual separate exports or .train_test_split())
split_ds = raw_dataset.train_test_split(test_size=0.5 if len(raw_dataset) == 1 else 0.2)
train_dataset = split_ds["train"]
val_dataset = split_ds["test"]

# =====================================================================
# 2. Schema and Label Mapping
# =====================================================================
Hippa_categories = [
    "Name",
    "Address",
    "Dates",
    "Phone",
    "Email",
    "Account_Num",
    "Vehicle",
    "Device_Num",
    "URL",
    "IP",
    "Bio",
    "Face",
    "Age",
    "etc"
]

labels = ["O"] + [
    f"{prefix}-{cat}"
    for cat in Hippa_categories
    for prefix in ["B", "I"]
]

openai_label2id = {label: i for i, label in enumerate(labels)}
openai_id2label = {i: label for label, i in openai_label2id.items()}
outside_id = openai_label2id["O"]

# Mapping Label Studio explicit raw labels to standard HIPAA targets
category_mapping = {
    "private_person": "Name",
    "private_address": "Address",
    "private_date": "Dates",
    "private_phone": "Phone",
    "private_email": "Email",
    "account_number": "Account_Num",
    "private_vehicle": "Vehicle",
    "private_device": "Device_Num",
    "private_url": "URL",
    "private_ip": "IP",
    "private_biometric": "Bio",
    "private_photo": "Face",
    "private_other": "etc"
}

# --- Sanity check ---
seen = Counter()
for row in train_dataset:
    for span in row["privacy_mask"]:
        seen[span["label"]] += 1

mapped_keys   = set(category_mapping) & set(seen)
unmapped_keys = set(seen) - set(category_mapping)
print("Raw label counts:", dict(seen))
print("Mapped (will train):", sorted(mapped_keys))
print("Dropped -> 'O':", sorted(unmapped_keys))

# =====================================================================
# 3. Model Iteration & Subword Alignment Training
# =====================================================================
#Change to local model can do multiple at onece or just one
model_ids = [
    "./models/saved_models/KBLab_bert-base-swedish-cased",
]

all_results = {}
detailed_reports = {}

for model_id in model_ids:
    print(f"\n{'='*50}\nTraining model: {model_id}\n{'='*50}")
    
    safe_model_name = model_id.replace("/", "_")
    output_dir = f"./checkpoints_{safe_model_name}"
    save_dir = f"./saved_models/{safe_model_name}"

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(openai_label2id),
        id2label=openai_id2label,
        label2id=openai_label2id,
        ignore_mismatched_sizes=True 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def preprocess(examples):
        tok = tokenizer(
            examples["source_text"],
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        all_labels = []

        for offsets, spans in zip(tok["offset_mapping"], examples["privacy_mask"]):
            tok_starts = [s for s, _ in offsets]
            tok_ends   = [e for _, e in offsets]

            labels_row = [
                -100 if (s == 0 and e == 0) else outside_id
                for s, e in offsets
            ]

            for span in spans:
                start_char = span["start"]
                end_char = span["end"]
                raw_label = span["label"]

                mapped = category_mapping.get(raw_label, "O")
                if mapped == "O":
                    continue 

                idxs = []
                for i, (t_start, t_end) in enumerate(zip(tok_starts, tok_ends)):
                    if t_start == 0 and t_end == 0:
                        continue
                    # Check character offset overlap
                    if max(start_char, t_start) < min(end_char, t_end):
                        idxs.append(i)

                if not idxs:
                    continue

                labels_row[idxs[0]] = openai_label2id[f"B-{mapped}"]
                for mid in idxs[1:]:
                    labels_row[mid] = openai_label2id[f"I-{mapped}"]

            all_labels.append(labels_row)

        tok.pop("offset_mapping")
        tok["labels"] = all_labels
        return tok

    print(f"Tokenizing datasets for {model_id}...")
    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1,
        bf16=torch.cuda.is_available(), 
        load_best_model_at_end=False
    )

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=2)
        labels = pred.label_ids

        true_labels = []
        true_preds = []

        for l, p in zip(labels, preds):
            l_clean = []
            p_clean = []
            for li, pi in zip(l, p):
                if li == -100:
                    continue
                l_clean.append(model.config.id2label[li])
                p_clean.append(model.config.id2label[pi])
            true_labels.append(l_clean)
            true_preds.append(p_clean)

        p = precision_score(true_labels, true_preds)
        r = recall_score(true_labels, true_preds)
        f1 = f1_score(true_labels, true_preds)
        f2 = 5 * (p * r) / ((4 * p) + r) if (p + r) > 0 else 0.0

        return {"precision": p, "recall": r, "f1": f1, "f2": f2}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print(f"Starting training for {model_id}...")
    trainer.train()

    print(f"Saving {model_id} to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

print("\nProcessing complete!")