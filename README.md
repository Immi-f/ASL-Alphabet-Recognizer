# ASL Alphabet Recognizer

Real-time American Sign Language alphabet recognition using MobileNetV3 transfer learning and OpenCV.

Classifies 29 classes: A–Z, *space*, *delete*, and *nothing*.

## Dataset

Download the **ASL Alphabet** dataset from Kaggle:

**https://www.kaggle.com/datasets/grassknoted/asl-alphabet**

After downloading, extract so the folder structure looks like:

```
data/raw/asl_alphabet_train/
├── A/
├── B/
├── ...
├── Z/
├── del/
├── nothing/
└── space/
```

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
python src/dataset.py --raw-dir data/raw --splits-dir data/splits
```

### 2. Train

```bash
python src/train.py \
    --data-dir data/raw \
    --splits-dir data/splits \
    --epochs 15 \
    --batch-size 64 \
    --lr 0.001 \
    --freeze-epochs 3 \
    --checkpoint-dir checkpoints \
    --runs-dir runs
```

Resume from a checkpoint:

```bash
python src/train.py --resume checkpoints/best.pth --epochs 25
```

### 3. Evaluate

```bash
python src/evaluate.py \
    --checkpoint checkpoints/best.pth \
    --data-dir data/raw \
    --splits-dir data/splits \
    --runs-dir runs
```

This prints per-class accuracy and saves a confusion matrix to `runs/confusion_matrix.png`.

### 4. Live inference

```bash
python src/inference.py --checkpoint checkpoints/best.pth --camera 0 --roi-size 300
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
│   ├── raw/              # Dataset images (gitignored)
│   ├── processed/        # Optional preprocessed data (gitignored)
│   └── splits/           # train.csv, val.csv, test.csv
├── src/
│   ├── dataset.py        # Dataset class, transforms, split generation
│   ├── model.py          # MobileNetV3-Small with custom head
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
