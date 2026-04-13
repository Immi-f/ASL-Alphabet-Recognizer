#!/usr/bin/env bash
# Train both models on both datasets sequentially.
# Usage: bash scripts/train_all.sh [extra train.py flags]
set -euo pipefail

cd "$(dirname "$0")/.."

run() {
    local model=$1
    local dataset=$2
    local splits_dir=$3
    shift 3
    echo
    echo "=============================================================="
    echo "  Training ${model} on ${dataset}"
    echo "=============================================================="
    python src/train.py \
        --model "${model}" \
        --splits-dir "${splits_dir}" \
        --checkpoint-dir "checkpoints/${dataset}" \
        --runs-dir "runs/${dataset}" \
        "$@"
}

run mobilenet_v3_small kaggle  data/splits_kaggle  "$@"
run resnet50           kaggle  data/splits_kaggle  "$@"
run mobilenet_v3_small archive data/splits_archive "$@"
run resnet50           archive data/splits_archive "$@"

echo
echo "All four training runs finished."
echo "Checkpoints: checkpoints/{kaggle,archive}/{mobilenet_v3_small,resnet50}/best.pth"
echo "Curves:      runs/{kaggle,archive}/{mobilenet_v3_small,resnet50}/training_curves.png"
