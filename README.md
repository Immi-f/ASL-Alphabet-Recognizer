# ASL Alphabet Recognizer

Real-time American Sign Language alphabet recognition using transfer learning (MobileNetV3-Small or ResNet-50) and OpenCV.

Classifies 29 classes: A–Z, *space*, *delete*, and *nothing*.

## Datasets

The project supports combining multiple dataset sources via a registry in [src/dataset.py](src/dataset.py). Place any subset under `data/` — missing sources are silently skipped.

| Source key | Layout under `data/` | Where to get it |
|---|---|---|
| `kaggle_asl_alphabet` | `raw/asl_alphabet_train/<A..Z,del,nothing,space>/*.jpg` | Kaggle: *ASL Alphabet* by Akash Nagaraj |
| `kaggle_asl_alphabet_test` | `raw/asl_alphabet_test/<A_test.jpg, ...>` | Ships with the same Kaggle bundle |
| `massey` | `raw/massey/<a..z>/*.png` | Massey University Gesture Dataset |
| `archive_synthetic_train` | `archive/Train_Alphabet/<A..Z,Blank>/*.png` | Synthetic ASL alphabet (A–Z + Blank → `nothing`) |
| `archive_synthetic_test` | `archive/Test_Alphabet/<A..Z,Blank>/*.png` | Same synthetic set, held-out split |

To add another dataset, append an entry to `DATASET_SOURCES` in [src/dataset.py](src/dataset.py) — each source is a callable that yields `(relative_path, canonical_label)` tuples. Label overrides handle naming mismatches (e.g. lowercase folders).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Generate train/val/test splits

This runs automatically the first time you train, or you can run it manually:

```bash
# Use all dataset sources present under data/
python src/dataset.py --data-dir data --splits-dir data/splits

# Or restrict to specific sources
python src/dataset.py --sources kaggle_asl_alphabet archive_synthetic_train
```

### 2. Train

Two backbones are available via `--model`: `mobilenet_v3_small` (default, fast) and `resnet50` (larger, more capacity). Checkpoints and plots are written to `checkpoints/<model>/` and `runs/<model>/`, so you can train both architectures in parallel without clobbering each other.

```bash
# MobileNetV3-Small (default)
python src/train.py \
    --data-dir data \
    --splits-dir data/splits \
    --epochs 15 \
    --batch-size 64 \
    --lr 0.001 \
    --freeze-epochs 3

# ResNet-50 — in a second terminal
python src/train.py --model resnet50 --batch-size 32 --epochs 15
```

Resume from a checkpoint:

```bash
python src/train.py --resume checkpoints/mobilenet_v3_small/best.pth --epochs 25
```

### 3. Evaluate

```bash
python src/evaluate.py \
    --checkpoint checkpoints/mobilenet_v3_small/best.pth \
    --data-dir data \
    --splits-dir data/splits \
    --runs-dir runs
```

This prints per-class accuracy and saves a confusion matrix to `runs/confusion_matrix.png`. The architecture is read from the checkpoint, so the same command works for any trained model.

### 4. Live inference

```bash
python src/inference.py --checkpoint checkpoints/mobilenet_v3_small/best.pth --camera 0 --roi-size 300
```

A green rectangle marks the ROI — hold your hand sign inside it. Press **q** to quit.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
├── data/
│   ├── raw/              # Kaggle + Massey datasets (gitignored)
│   ├── archive/          # Synthetic ASL alphabet (gitignored)
│   ├── processed/        # Optional preprocessed data (gitignored)
│   └── splits/           # train.csv, val.csv, test.csv
├── src/
│   ├── dataset.py        # Dataset class, transforms, multi-source split generation
│   ├── model.py          # MobileNetV3-Small / ResNet-50 with custom heads
│   ├── train.py          # Training loop with two-phase fine-tuning
│   ├── evaluate.py       # Test metrics and confusion matrix
│   └── inference.py      # Real-time webcam inference
├── checkpoints/          # Saved model weights (gitignored)
├── runs/                 # Training plots, confusion matrices (gitignored)
├── tests/
│   └── test_dataset.py
├── requirements.txt
└── README.md
```
