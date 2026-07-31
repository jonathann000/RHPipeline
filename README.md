# RHPipeline — Swedish Clinical PHI De-identification

A modular pipeline for removing personal and health-identifying information
(PHI/PII) from Swedish clinical notes. It combines fast deterministic detectors
with LLM-based reasoning to catch both **direct identifiers** (names, personnummer,
phone numbers, dates) and **quasi-identifiers** (a rare diagnosis, an unusual
occupation, a small-town detail that re-identifies someone in combination).

Each detection stage is an independent, swappable module behind a uniform
`detect()` interface; the pipeline merges their findings, resolves overlaps,
propagates coreferences, and produces redacted text plus an audit trail. It also
ships an **evaluation harness** (gold-key scoring across models) and a
**human-in-the-loop review flow** (Label Studio pre-annotation → correction →
rebuild).

## How it works

Detection runs as a sequence of stages (which stages run depends on `--mode`):

| Stage | Module | Role |
|-------|--------|------|
| 1. Rules | [rule_agent.py](rule_agent.py) | Regex for structured PII (personnummer, phone, email, dates, zip codes) |
| 2. BERT NER | [bert_agent.py](bert_agent.py) | Token classification for direct identifiers (names, etc.) |
| 3. LLM | [llm_backend.py](llm_backend.py) | Quasi-identifiers, and direct identifiers when no BERT |
| 3.5 Generalization *(optional)* | [generalize_agent.py](generalize_agent.py) | Proposes generalizations for quasi-IDs as a **separate** step (see `--generalize-backend`); off by default, in which case the detection LLM generalizes inline |
| 4. Gazetteer | [gazetteer_agent.py](gazetteer_agent.py) | Exact-match lookup of known Swedish places/institutions (Wikidata) |
| 5. Coreference | [coreference.py](coreference.py) | Propagates each found entity to its other mentions |
| 6. Redaction | [redaction.py](redaction.py) | Generalizes or placeholders each span; writes the audit log |
| 7. Judge panel *(optional)* | [judge.py](judge.py) | Audits the output for residual PII and retries if flagged |

The shared data model and all overlap/conflict resolution live in
[entities.py](entities.py). [pipeline.py](pipeline.py) orchestrates the stages;
[run.py](run.py) is the CLI wrapping it.

## Requirements

- Python 3.10+
- A BERT NER checkpoint (for `full` mode) — see [Models](#models)
- A HuggingFace token (`HF_TOKEN`) for gated LLM/BERT downloads
- GPU recommended for the LLM stage; quantization (bf16 → 8-bit → 4-bit) is
  auto-selected to fit the detected card (see [device.py](device.py))

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
export BERT_MODEL_PATH=jtamondo/RH-BEHRT_Swedish_Hippa_BERT_BASE    # or jtamondo/RH-BEHRT_KBLAB_Megatron
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
| `--llm NAME [NAME ...]` | LLM backend(s). Multiple = ensemble (union of findings). See [LLM backends](#llm-backends). |
| `--generalize-backend NAME` | Run generalization as a **separate** stage with this backend (after detection, before export) instead of coupled into the detection LLM. Pass the same name as `--llm` to reuse that loaded model. Default: off. |
| `--llm-backstop` | LLM also catches direct identifiers rules/BERT missed |
| `--llm-thinking` | Ask the model to reason before answering (Qwen3 only); saved to `--reasoning-output` |
| `--judges NAME [NAME ...]` | Judge panel that audits the output and retries if it flags residual PII |
| `--judge-max-rounds N` | Max detect-then-rejudge rounds (default 2) |
| `--gazetteer PATH` / `--no-gazetteer` | Point at / disable the gazetteer CSV (default `sweden_entities_deid.csv`) |
| `--quasi-only` | LLM detects quasi-identifiers only (for already-deidentified input, e.g. MIMIC) |
| `--no-generalize` | Always use category placeholders instead of trusting LLM generalizations |
| `--label-studio-output PATH` | Export detections as a Label Studio pre-annotation task (see [Human review](#human-review--rebuild)) |

The full, authoritative flag reference (with the reasoning behind each) is the
module docstring at the top of [run.py](run.py).

### LLM backends

Selectable by name via `--llm` (and `--generalize-backend` / `--judges`):

| Name | Model | Size |
|------|-------|------|
| `llama` | meta-llama/Meta-Llama-3.1-8B-Instruct | ~8B |
| `mistral` | mistralai/Mistral-7B-Instruct-v0.3 | ~7B |
| `mistral-small-24b` | mistralai/Mistral-Small-24B-Instruct-2501 | ~24B |
| `ministral8b` | mistralai/Ministral-8B-Instruct-2410 | ~8B |
| `qwen` | Qwen/Qwen3-8B | ~8B |
| `qwen-32b` | Qwen/Qwen3-32B | ~32B |
| `qwen3.6-27b` | Qwen/Qwen3.6-27B | ~27B |
| `gemma` | google/gemma-2-9b-it | ~9B |
| `gemma-27b` | google/gemma-2-27b-it | ~27B |
| `gemma4-12b` | google/gemma-4-12B-it | ~12B |
| `gemma4-31b` | google/gemma-4-31B-it | ~31B |

Add a new backend by appending an entry to `LLM_BACKENDS` in [run.py](run.py):
`{"llm_backend": "<family>", "llm_model_path": "<HF id or local dir>",
"approx_params_b": <billions>}`. The backend must be a **causal LM** — the
generic loader uses `AutoModelForCausalLM` and the tokenizer's own chat template,
so most instruct-tuned text models work with no code change. Multimodal/MoE
models whose config isn't a causal LM (e.g. the Mistral-3 vision line) won't
load. Set `approx_params_b` to **total** parameters (matters for MoE memory
sizing), and confirm compatibility first with:
`AutoConfig.from_pretrained("<id>").model_type` / `architectures`.

## Evaluation

A small gold-standard harness for scoring de-identification quality and
comparing models, built around fictional synthetic notes.

- `data/synthetic_note*.txt` — fictional Swedish clinical notes (tracked fixtures).
- `data/synthetic_notes_key.json` — the gold key: for each note, the spans that
  **must be redacted** (direct + quasi, with risk), the **medications** to flag
  but keep verbatim, and the **decoys** (normal vitals, common conditions) that
  must be left intact.

Run every note through the pipeline and score it in one command with
[eval_synthetic.sh](eval_synthetic.sh) — `RUN_LABEL` namespaces the output
directory so runs from different models don't overwrite each other:

```bash
RUN_LABEL=qwen32b   bash eval_synthetic.sh --llm qwen-32b   --mode full
RUN_LABEL=gemma4    bash eval_synthetic.sh --llm gemma4-31b --mode full
```

Then score / compare with [score.py](score.py):

```bash
.venv/bin/python score.py --runs-dir data/out/qwen32b --verbose     # per-note + misses
.venv/bin/python score.py --compare data/out/qwen32b data/out/gemma4  # models side by side
```

Metrics (recall is measured against the **redacted output** — did the text
actually disappear — not just the audit log):

| Metric | Meaning |
|--------|---------|
| direct-recall | fraction of direct identifiers actually removed |
| quasi-recall | fraction of quasi-identifiers actually removed |
| decoy-keep | fraction of decoys left intact (precision proxy — higher is better) |
| med-ok | medications both flagged in the audit **and** kept verbatim |

## Human review & rebuild

The intended production loop is **pipeline → human review in Label Studio →
rebuild the final redaction from the reviewed spans**:

1. **Pre-annotate.** Add `--label-studio-output data/out/tasks.json` to a run
   ([label_studio_export.py](label_studio_export.py)) to write a Label Studio
   task with every detected span pre-highlighted by label, plus risk and source
   as per-region choices, and the proposed generalization in each region's meta.
   A matching `label_studio_config.xml` is written alongside — import it once
   into the project. `--label-studio-append` accumulates several documents into
   one file.
2. **Review.** A human verifies/corrects/adds spans, labels, and generalizations
   in Label Studio, then exports the annotated JSON.
3. **Rebuild.** [label_studio_rebuild.py](label_studio_rebuild.py) turns that
   export back into a final redacted document, using the same redaction
   machinery as the pipeline:

   ```bash
   .venv/bin/python label_studio_rebuild.py --input export.json --out-dir data/out/reviewed
   ```

   It reads the human **annotation** layer (falling back to model predictions
   for un-reviewed tasks). By default it collapses overlapping spans into one
   readable redaction per cluster; `--split-overlaps` reverts to the pipeline's
   fragmenting behavior, and `--no-generalize` forces placeholders everywhere.

## Data layout

```
data/
  notes.txt, synthetic_note*.txt        # tracked sample INPUTS (fixtures)
  synthetic_notes_key.json              # tracked gold key for the evaluation harness
  out/                                  # git-ignored — all generated OUTPUT
    <run>/redacted.txt, audit.json, label_studio_tasks.json, ...
    reviewed/<note>.reviewed.redacted.txt   # rebuilt from a Label Studio export
```

**Inputs go in `data/` root and are tracked. Everything the pipeline generates
goes under `data/out/` and is git-ignored** (`.gitignore` only lists `data/out/`).
New input files added to `data/` are tracked automatically; new outputs written
to `data/out/` are ignored automatically — no per-file `.gitignore` edits needed.

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

## Module reference

| Module | Purpose |
|--------|---------|
| [run.py](run.py) | CLI entry point (argument parsing, I/O) |
| [pipeline.py](pipeline.py) | Stage orchestration → `PipelineResult` |
| [entities.py](entities.py) | `Entity` data model + overlap/conflict resolution |
| [rule_agent.py](rule_agent.py) | Regex detectors for structured PII |
| [bert_agent.py](bert_agent.py) | BERT NER detector (chunking-aware) |
| [gazetteer_agent.py](gazetteer_agent.py) | Aho-Corasick place/name matcher |
| [llm_backend.py](llm_backend.py) | Swappable LLM detector + judge + generalizer, prompts, JSON parsing |
| [generalize_agent.py](generalize_agent.py) | Optional separate generalization stage |
| [coreference.py](coreference.py) | Propagates entities to their other mentions |
| [redaction.py](redaction.py) | Applies generalizations/placeholders |
| [judge.py](judge.py) | Judge panel over redacted output |
| [chunking.py](chunking.py) | Sentence-aware chunking (shared by BERT + LLM) |
| [device.py](device.py) | GPU/MPS/CPU device + quantization selection |
| [label_studio_export.py](label_studio_export.py) | Label Studio task/config export |
| [label_studio_rebuild.py](label_studio_rebuild.py) | Rebuild redacted text from a reviewed Label Studio export |
| [wikidata_script.py](wikidata_script.py) | Builds the gazetteer CSV from Wikidata |
| [score.py](score.py) | Score a run against the synthetic gold key |
| [eval_synthetic.sh](eval_synthetic.sh) | Run all synthetic notes through the pipeline and score them |
| [test_local.py](test_local.py) | No-GPU smoke test (mock LLM) |
```
