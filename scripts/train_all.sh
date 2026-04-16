#!/usr/bin/env bash
# Train both models on the combined Kaggle + archive dataset.
# Usage: bash scripts/train_all.sh [extra train.py flags]
set -euo pipefail

cd "$(dirname "$0")/.."

run() {
    local model=$1
    shift
    echo
    echo "=============================================================="
    echo "  Training ${model} on combined dataset"
    echo "=============================================================="
    python src/train.py \
        --model "${model}" \
        --splits-dir data/splits \
        --checkpoint-dir checkpoints \
        --runs-dir runs \
        "$@"
}

run mobilenet_v3_small "$@"
run resnet50           "$@"

echo
echo "Both training runs finished."
echo "Checkpoints: checkpoints/{mobilenet_v3_small,resnet50}/best.pth"
echo "Curves:      runs/{mobilenet_v3_small,resnet50}/training_curves.png"
