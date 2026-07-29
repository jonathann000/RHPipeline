import os
from datasets import load_from_disk
import numpy as np
import torch
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# 0. Load dataset
ds = load_from_disk("SV_PURE3")

# HIPAA-style categories (BIO)
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

# 1. Load base model with correct label set
model_id = "answerdotai/ModernBERT-base"

model = AutoModelForTokenClassification.from_pretrained(
    model_id,
    num_labels=len(openai_label2id),
    id2label=openai_id2label,
    label2id=openai_label2id
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

category_mapping = {
    "EMAIL": "Email",
    "TELEPHONENUM": "Phone",
    "DRIVERLICENSENUM": "Account_Num",
    "IDCARDNUM": "Account_Num",
    "SOCIALNUM": "Account_Num",
    "PASSPORTNUM": "Account_Num",
    "CREDITCARDNUMBER": "Account_Num",
    "TAXNUM": "Account_Num",
    "BUILDINGNUM": "Address",
    "STREET": "Address",
    "CITY": "Address",
    "ZIPCODE": "Address",
    "COUNTRY": "Address",
    "GIVENNAME": "Name",
    "SURNAME": "Name",
    "TITLE": "Name",
    "AGE": "Age",
    "DATE": "Dates"
}

dataset = ds
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
print("Train rows:", train_dataset.num_rows)
print("Val rows:", val_dataset.num_rows)

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
assert mapped_keys, "No dataset labels matched category_mapping - check label strings!"

# =====================================================================
# 1. Define the models to cycle through
# =====================================================================
model_ids = [
    "answerdotai/ModernBERT-base",
    "albert-base-v2",
    "distilbert-base-uncased",
    "bert-base-cased",
    "KBLab/megatron-bert-large-swedish-cased-165k",
    "KBLab/bert-base-swedish-cased"
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
    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names, load_from_cache_file=False)
    tokenized_val = val_dataset.map(preprocess, batched=True, remove_columns=val_dataset.column_names, load_from_cache_file=False)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=100,
        bf16=True, 
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
        
        if p + r == 0:
            f2 = 0.0
        else:
            f2 = 5 * (p * r) / ((4 * p) + r)

        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "f2": f2
        }

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
    
    print(f"Evaluating {model_id}...")
    final_metrics = trainer.evaluate()
    all_results[model_id] = final_metrics

    # --- Generate per-label classification report ---
    predictions_output = trainer.predict(tokenized_val)
    preds_idx = np.argmax(predictions_output.predictions, axis=2)
    eval_labels = predictions_output.label_ids

    true_labels_report = []
    true_preds_report = []
    for l, p in zip(eval_labels, preds_idx):
        l_clean = [model.config.id2label[li] for li, pi in zip(l, p) if li != -100]
        p_clean = [model.config.id2label[pi] for li, pi in zip(l, p) if li != -100]
        true_labels_report.append(l_clean)
        true_preds_report.append(p_clean)

    report_str = classification_report(true_labels_report, true_preds_report, digits=4)
    detailed_reports[model_id] = report_str

    print(f"Saving {model_id} to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

# =====================================================================
# 2. Save Summary & Per-Label Metrics to a .txt File
# =====================================================================
summary_file_path = "model_evaluation_summary.txt"
print(f"\nWriting evaluation summary and per-label reports to {summary_file_path}...")

with open(summary_file_path, "w", encoding="utf-8") as f:
    f.write("="*70 + "\n")
    f.write("MULTI-MODEL PIPELINE EVALUATION SUMMARY\n")
    f.write("="*70 + "\n\n")

    for m_id, metrics in all_results.items():
        f.write(f"Model: {m_id}\n")
        f.write(f"  Precision: {metrics.get('eval_precision', 0):.4f}\n")
        f.write(f"  Recall:    {metrics.get('eval_recall', 0):.4f}\n")
        f.write(f"  F1 Score:  {metrics.get('eval_f1', 0):.4f}\n")
        f.write(f"  F2 Score:  {metrics.get('eval_f2', 0):.4f}\n")
        f.write(f"  Loss:      {metrics.get('eval_loss', 0):.4f}\n")
        f.write("-" * 40 + "\n")
        f.write("Per-Label Classification Report (seqeval):\n")
        f.write(detailed_reports[m_id])
        f.write("\n" + "="*70 + "\n\n")

print("Done! All results and classification reports have been successfully exported.")
