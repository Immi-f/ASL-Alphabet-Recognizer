"""ASL Alphabet Dataset — loading, transforms, and split generation."""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
    "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15,
    "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23,
    "Y": 24, "Z": 25, "del": 26, "nothing": 27, "space": 28,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ASLDataset(Dataset):
    """Reads image paths and labels from a split manifest CSV.

    The CSV must have columns: ``path`` (relative to *data_dir*) and ``label``
    (folder name matching a key in :data:`LABEL_MAP`).
    """

    def __init__(self, csv_path: str, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["path"])
        image = Image.open(img_path).convert("RGB")
        label = LABEL_MAP[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


def generate_splits(raw_dir: str, splits_dir: str, train_ratio=0.8, val_ratio=0.1):
    """Walk *raw_dir*, collect image paths, and write train/val/test CSVs.

    Images are expected at ``raw_dir/asl_alphabet_train/<CLASS>/*.jpg``.
    """
    from sklearn.model_selection import train_test_split

    base = os.path.join(raw_dir, "asl_alphabet_train")
    if not os.path.isdir(base):
        raise FileNotFoundError(
            f"Expected dataset directory at {base}. "
            "Download the ASL Alphabet Dataset from Kaggle and extract it into data/raw/."
        )

    records = []
    for class_name in sorted(os.listdir(base)):
        class_dir = os.path.join(base, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_path = os.path.join("asl_alphabet_train", class_name, fname)
                records.append({"path": rel_path, "label": class_name})

    if not records:
        raise RuntimeError(f"No images found under {base}")

    df = pd.DataFrame(records)

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), stratify=df["label"], random_state=42
    )
    relative_val = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), stratify=temp_df["label"], random_state=42
    )

    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)

    print(f"Splits written to {splits_dir}:")
    print(f"  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate train/val/test split CSVs.")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw dataset root.")
    parser.add_argument("--splits-dir", default="data/splits", help="Output directory for CSVs.")
    args = parser.parse_args()

    generate_splits(args.raw_dir, args.splits_dir)
