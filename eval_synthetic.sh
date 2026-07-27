#!/bin/bash
# Run the pipeline over every synthetic note in the gold key, then score it.
#
# Usage:
#   bash eval_synthetic.sh [pipeline flags...]
#   RUN_LABEL=<name> bash eval_synthetic.sh [pipeline flags...]
# e.g.
#   RUN_LABEL=qwen32b  bash eval_synthetic.sh --llm qwen-32b --mode full
#   RUN_LABEL=mistral  bash eval_synthetic.sh --llm mistral  --mode full
#   bash eval_synthetic.sh --llm mistral qwen --judges qwen        # ensemble + judge
#
# RUN_LABEL namespaces the output directory (data/out/<label>/) so runs from
# different models don't overwrite each other — set one per model, then compare
# them with:  python score.py --compare data/out/qwen32b data/out/mistral
#
# Any flags you pass are forwarded to run_cluster.sh (--llm, --mode, --judges,
# --llm-backstop, ...). Do NOT pass --input/--output/--audit — this script sets
# those per note so the scorer can find the results by name.
#
# Outputs land in <out>/<stem>.redacted.txt and <stem>.audit.json, then
# score.py compares them against data/synthetic_notes_key.json.

set -e

KEY="data/synthetic_notes_key.json"
# RUN_LABEL (optional) puts this run in its own subdir so models don't clobber.
OUT="data/out${RUN_LABEL:+/$RUN_LABEL}"
LS_TASKS="$OUT/label_studio_tasks.json"
mkdir -p "$OUT"

# Accumulate every note's detections into ONE Label Studio task file so the
# whole batch can be inspected in a single project. Start clean each run, then
# append per note (append on a not-yet-existing file just creates it). The
# matching label_studio_config.xml is (re)written alongside it automatically.
rm -f "$LS_TASKS"

# Derive the note stems (filenames without .txt) straight from the key, so the
# set stays in sync with whatever notes the key actually covers.
STEMS=$(.venv/bin/python -c "import json; print(' '.join(n[:-4] for n in json.load(open('$KEY')) if not n.startswith('_')))")

for stem in $STEMS; do
    echo "=================================================================="
    echo "  Running pipeline on data/${stem}.txt"
    echo "=================================================================="
    bash run_cluster.sh \
        --input "data/${stem}.txt" \
        --output "${OUT}/${stem}.redacted.txt" \
        --audit "${OUT}/${stem}.audit.json" \
        --label-studio-output "$LS_TASKS" \
        --label-studio-append \
        "$@"
done

echo ""
echo "Label Studio tasks: ${LS_TASKS}"
echo "  Import ${OUT}/label_studio_config.xml into your Label Studio project once,"
echo "  then import ${LS_TASKS} to see all ${OUT} notes pre-annotated by label/risk/source."

echo ""
echo "=================================================================="
echo "  SCORING"
echo "=================================================================="
.venv/bin/python score.py --key "$KEY" --runs-dir "$OUT"
