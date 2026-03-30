"""Training loop for the ASL Alphabet Recognizer."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ASLDataset, LABEL_MAP, train_transform, val_transform
from model import build_model, freeze_backbone, unfreeze_backbone


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASL Alphabet Recognizer.")
    parser.add_argument("--data-dir", default="data/raw", help="Root of raw image data.")
    parser.add_argument("--splits-dir", default="data/splits", help="Directory with split CSVs.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Where to save checkpoints.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--runs-dir", default="runs", help="Directory for training plots.")
    parser.add_argument("--freeze-epochs", type=int, default=3,
                        help="Epochs to train with frozen backbone before unfreezing.")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def save_plots(history, runs_dir):
    os.makedirs(runs_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(runs_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {runs_dir}/training_curves.png")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = os.path.join(args.splits_dir, "train.csv")
    val_csv = os.path.join(args.splits_dir, "val.csv")

    if not os.path.isfile(train_csv):
        print("Split CSVs not found — generating splits …")
        from dataset import generate_splits
        generate_splits(args.data_dir, args.splits_dir)

    train_ds = ASLDataset(train_csv, args.data_dir, transform=train_transform)
    val_ds = ASLDataset(val_csv, args.data_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = build_model(num_classes=len(LABEL_MAP)).to(device)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from epoch {start_epoch} (val_acc={ckpt['val_acc']:.4f})")

    # Phase 1: frozen backbone
    freeze_backbone(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if epoch == args.freeze_epochs:
            print("Unfreezing backbone for full fine-tuning …")
            unfreeze_backbone(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        print(f"  val_loss={val_loss:.4f}    val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, "best.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "label_map": LABEL_MAP,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f})")

    save_plots(history, args.runs_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
