# RHPipeline — Swedish Clinical PHI De-identification

A modular pipeline for removing personal and health-identifying information
(PHI/PII) from Swedish clinical notes. It combines fast deterministic detectors
with LLM-based reasoning to catch both **direct identifiers** (names, personnummer,
phone numbers, dates) and **quasi-identifiers** (a rare diagnosis, an unusual
occupation, a small-town detail that re-identifies someone in combination).

Each detection stage is an independent, swappable module behind a uniform
`detect()` interface; the pipeline merges their findings, resolves overlaps,
propagates coreferences, and produces redacted text plus an audit trail.

## How it works

Detection runs as a sequence of stages (which stages run depends on `--mode`):

| Stage | Module | Role |
|-------|--------|------|
| 1. Rules | [rule_agent.py](rule_agent.py) | Regex for structured PII (personnummer, phone, email, dates, zip codes) |
| 2. BERT NER | [bert_agent.py](bert_agent.py) | Token classification for direct identifiers (names, etc.) |
| 3. LLM | [llm_backend.py](llm_backend.py) | Quasi-identifiers, and direct identifiers when no BERT |
| 4. Gazetteer | [gazetteer_agent.py](gazetteer_agent.py) | Exact-match lookup of known Swedish places/institutions (Wikidata) |
| 5. Coreference | [coreference.py](coreference.py) | Propagates each found entity to its other mentions |
| 6. Redaction | [redaction.py](redaction.py) | Generalizes or placeholders each span; writes the audit log |
| 7. Judge panel | [judge.py](judge.py) | (Optional) audits the output for residual PII and retries if flagged |

The shared data model and all overlap/conflict resolution live in
[entities.py](entities.py). [pipeline.py](pipeline.py) orchestrates the stages;
[run.py](run.py) is the CLI wrapping it.

## Requirements

- Python 3.10+
- A BERT NER checkpoint (for `full` mode) — see [Models](#models)
- A HuggingFace token (`HF_TOKEN`) for gated LLM/BERT downloads
- GPU recommended for the LLM stage; quantization is auto-selected to fit the
  detected card (see [device.py](device.py))

Install into an isolated virtualenv:

```bash
export HF_TOKEN=hf_...        # required for gated model downloads
bash setup.sh                 # creates .venv and installs requirements.txt
```

## Quickstart

**1. Smoke test (no GPU, no model downloads)** — verifies the pipeline wiring
end-to-end with a mock LLM and rules only:

```bash
.venv/bin/python test_local.py
```

**2. A real run.** Point `BERT_MODEL_PATH` at your NER checkpoint, then:

```bash
export BERT_MODEL_PATH=./models/MBERTHIPAA     # or ./models/Roberta
bash run_cluster.sh --input data/notes.txt --llm mistral
```

`run_cluster.sh` is a thin wrapper that uses `.venv/bin/python` and passes flags
through to `run.py`. To call the CLI directly:

```bash
.venv/bin/python run.py --input data/notes.txt \
    --output data/out/redacted.txt --audit data/out/audit.json --llm mistral
```

Outputs default to **`data/out/`** (git-ignored — see [Data layout](#data-layout)).

## Usage

### Modes (`--mode`)

| Mode | Stages | Use for |
|------|--------|---------|
| `full` (default) | Rules → BERT → LLM (quasi) → Gazetteer | Highest coverage |
| `no_bert` | Rules → LLM (direct + quasi) → Gazetteer | Benchmark without BERT |
| `llm_only` | LLM only (no rules/BERT/gazetteer) | Pure LLM baseline |

### Common flags

| Flag | Effect |
|------|--------|
| `--llm NAME [NAME ...]` | LLM backend(s). Multiple = ensemble (union of findings). Options: `llama`, `mistral`, `qwen`, `qwen-32b`, `gemma`, `gemma-27b`, `gemma4-12b`, `gemma4-31b` |
| `--llm-backstop` | LLM also catches direct identifiers rules/BERT missed |
| `--llm-thinking` | Ask the model to reason before answering (Qwen3 only); saved to `--reasoning-output` |
| `--judges NAME [NAME ...]` | Judge panel that audits the output and retries if it flags residual PII |
| `--judge-max-rounds N` | Max detect-then-rejudge rounds (default 2) |
| `--gazetteer PATH` / `--no-gazetteer` | Point at / disable the gazetteer CSV (default `sweden_entities_deid.csv`) |
| `--quasi-only` | LLM detects quasi-identifiers only (for already-deidentified input, e.g. MIMIC) |
| `--no-generalize` | Always use category placeholders instead of trusting LLM generalizations |
| `--label-studio-output PATH` | Export detections as a Label Studio pre-annotation task |

The full, authoritative flag reference (with the reasoning behind each) is the
module docstring at the top of [run.py](run.py).

## Data layout

```
data/
  notes.txt, notes2*.txt, synthetic_note.txt   # tracked sample INPUTS (fixtures)
  out/                                          # git-ignored — all generated OUTPUT
    redacted.txt, audit.json, reasoning.json, label_studio_*.…
```

**Inputs go in `data/` root and are tracked. Everything the pipeline generates
goes under `data/out/` and is git-ignored** (`.gitignore` only lists `data/out/`).
This keeps redacted texts, audit logs, and experiment runs from piling up in the
repo, while still versioning the sample notes. New input files added to `data/`
are tracked automatically; new outputs written to `data/out/` are ignored
automatically — no per-file `.gitignore` edits needed.

## Models

Place NER checkpoints under `models/` (git-ignored). This repo has been used with:

- `models/MBERTHIPAA` — HIPAA Safe Harbor 18-category NER (current default)
- `models/Roberta` — Swedish RoBERTa NER
- `models/ModelOAI`

Select one via `BERT_MODEL_PATH` (defaults to `./models/MBERTHIPAA`). LLM
checkpoints are downloaded from HuggingFace on first use and cached in
`.model_cache/`.

## Regenerating the gazetteer

`sweden_entities_deid.csv` (committed) is a Wikidata export of Swedish place and
name entities. To rebuild it:

```bash
.venv/bin/python wikidata_script.py     # writes sweden_entities_deid.csv
```

## Label Studio

Pass `--label-studio-output data/out/label_studio_tasks.json` to export detected
spans as a Label Studio pre-annotation task (label + risk + source per region).
A matching `label_studio_config.xml` is written alongside it — import that once
into your Label Studio project. Use `--label-studio-append` to batch several
documents into one file. See [label_studio_export.py](label_studio_export.py).

## Module reference

| Module | Purpose |
|--------|---------|
| [run.py](run.py) | CLI entry point (argument parsing, I/O) |
| [pipeline.py](pipeline.py) | Stage orchestration → `PipelineResult` |
| [entities.py](entities.py) | `Entity` data model + overlap/conflict resolution |
| [rule_agent.py](rule_agent.py) | Regex detectors for structured PII |
| [bert_agent.py](bert_agent.py) | BERT NER detector (chunking-aware) |
| [gazetteer_agent.py](gazetteer_agent.py) | Aho-Corasick place/name matcher |
| [llm_backend.py](llm_backend.py) | Swappable LLM detector + judge, prompts, JSON parsing |
| [coreference.py](coreference.py) | Propagates entities to their other mentions |
| [redaction.py](redaction.py) | Applies generalizations/placeholders |
| [judge.py](judge.py) | Judge panel over redacted output |
| [chunking.py](chunking.py) | Sentence-aware chunking (shared by BERT + LLM) |
| [device.py](device.py) | GPU/MPS/CPU device + quantization selection |
| [label_studio_export.py](label_studio_export.py) | Label Studio task/config export |
| [wikidata_script.py](wikidata_script.py) | Builds the gazetteer CSV from Wikidata |
| [test_local.py](test_local.py) | No-GPU smoke test (mock LLM) |
