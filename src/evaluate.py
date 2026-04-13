"""Evaluate a trained model on the test split and produce metrics."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ASLDataset, LABEL_MAP, val_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASL model on test set.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--data-dir", default="data", help="Root of data (contains raw/, archive/).")
    parser.add_argument("--splits-dir", default="data/splits", help="Directory with split CSVs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--runs-dir", default="runs", help="Directory to save confusion matrix.")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix (normalized)")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model import build_model

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "mobilenet_v3_small")
    model = build_model(num_classes=len(LABEL_MAP), pretrained=False, arch=arch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    test_csv = os.path.join(args.splits_dir, "test.csv")
    test_ds = ASLDataset(test_csv, args.data_dir, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    preds, labels = evaluate(model, test_loader, device)

    idx_to_label = {v: k for k, v in LABEL_MAP.items()}
    class_names = [idx_to_label[i] for i in range(len(LABEL_MAP))]

    print("\nPer-class accuracy:")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    overall_acc = (preds == labels).mean()
    print(f"Overall test accuracy: {overall_acc:.4f}")

    os.makedirs(args.runs_dir, exist_ok=True)
    cm_path = os.path.join(args.runs_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, class_names, cm_path)


if __name__ == "__main__":
    main()
