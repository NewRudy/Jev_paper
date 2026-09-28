#!/usr/bin/env bash
# One-command reproduction: data -> verification -> training -> tables -> figures.
#
#   bash run_all.sh            # data + verify + tables (no GPU needed)
#   bash run_all.sh --train    # also runs the training arms (needs a GPU, ~3 h on T4)
#
# Training itself lives in kaggle/cell_v2.py because it is executed on Kaggle's
# free T4. This script does everything local: regenerate every dataset from
# seeds, re-verify them independently, then rebuild tables and figures from
# whatever runs/ the training produced.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/gen"

echo "=== 1/5  generate datasets (deterministic seeds) ==="
python3 gen/make_dataset_v2.py     --seed 11
python3 gen/make_ambiguity_v2.py   --seed 31
python3 gen/make_dataset_deep.py   --seed 23
python3 gen/make_arft_dataset.py   --seed 42

echo "=== 2/5  adopt public benchmarks (StepGame, CLUTRR) ==="
python3 gen/adopt_benchmarks.py --which both --per-k 100

echo "=== 3/5  independent verification (labels re-derived from rendered text) ==="
python3 gen/verify_v2.py data/chain_train_k12.jsonl data/chain_train_deep_k14.jsonl \
    data/chain_test_k5.jsonl data/chain_test_k6.jsonl data/ambiguous.jsonl
python3 gen/verify_ambiguity.py data

echo "=== 4/5  training ==="
if [ "${1:-}" = "--train" ]; then
  echo "Training runs on Kaggle: paste kaggle/cell_v2.py into the notebook"
  echo "(GPU T4 x2, free tier) and Save & Run All. Results land in runs/."
else
  echo "skipped (pass --train to remind yourself; the GPU work runs on Kaggle)"
fi

echo "=== 5/5  tables and figures ==="
if compgen -G "runs/*/rows.json" > /dev/null; then
  python3 gen/make_results.py --runs runs --out results \
      --compare arft:std ls01:std anchorkl:std ctrlrand:std || true
  python3 gen/make_figures.py --runs runs --out results/figures \
      --arms base std arft --tags amb_a1_stated amb_a1_unstated amb_a4 amb_a5 || true
else
  echo "no runs/*/rows.json yet — train first, then re-run this script"
fi

echo
echo "done. tables: results/tables.md   figures: results/figures/"
