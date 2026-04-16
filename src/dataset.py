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


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _walk_class_folders(raw_dir, subdir, label_overrides=None):
    """Yield (rel_path, canonical_label) for raw_dir/subdir/<class>/<file>."""
    base = os.path.join(raw_dir, subdir)
    if not os.path.isdir(base):
        return
    overrides = label_overrides or {}
    for class_name in sorted(os.listdir(base)):
        class_dir = os.path.join(base, class_name)
        if not os.path.isdir(class_dir):
            continue
        canonical = overrides.get(class_name, class_name)
        if canonical not in LABEL_MAP:
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(IMAGE_EXTS):
                yield os.path.join(subdir, class_name, fname), canonical


def _walk_flat_files(raw_dir, subdir, label_from_filename):
    """Yield (rel_path, canonical_label) for raw_dir/subdir/<file>."""
    base = os.path.join(raw_dir, subdir)
    if not os.path.isdir(base):
        return
    for fname in sorted(os.listdir(base)):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        label = label_from_filename(fname)
        if label not in LABEL_MAP:
            continue
        yield os.path.join(subdir, fname), label


# Registry of dataset sources. Add a new entry to pull in another dataset.
# Each value is a callable raw_dir -> iterable of (rel_path, canonical_label).
_LETTER_UPPER = {c.lower(): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

DATASET_SOURCES = {
    # Kaggle "ASL Alphabet" (Akash Nagaraj) — the main training set.
    "kaggle_asl_alphabet": lambda d: _walk_class_folders(d, "raw/asl_alphabet_train"),
    # Ships with the same Kaggle bundle: one flat folder with files like "A_test.jpg".
    "kaggle_asl_alphabet_test": lambda d: _walk_flat_files(
        d, "raw/asl_alphabet_test",
        label_from_filename=lambda f: f.split("_")[0],
    ),
    # Massey University gesture dataset — lowercase class folders, A–Z only.
    "massey": lambda d: _walk_class_folders(
        d, "raw/massey", label_overrides=_LETTER_UPPER,
    ),
    # Synthetic ASL alphabet (archive/) — A–Z class folders plus "Blank" → "nothing".
    "archive_synthetic_train": lambda d: _walk_class_folders(
        d, "archive/Train_Alphabet", label_overrides={"Blank": "nothing"},
    ),
    "archive_synthetic_test": lambda d: _walk_class_folders(
        d, "archive/Test_Alphabet", label_overrides={"Blank": "nothing"},
    ),
}


def generate_splits(data_dir: str, splits_dir: str, sources=None,
                    train_ratio=0.8, val_ratio=0.1):
    """Walk selected dataset sources and write train/val/test CSVs.

    Sources default to every entry in :data:`DATASET_SOURCES` that exists on disk.
    """
    from sklearn.model_selection import train_test_split

    if sources is None:
        sources = list(DATASET_SOURCES.keys())

    records = []
    for src in sources:
        if src not in DATASET_SOURCES:
            raise ValueError(f"Unknown source '{src}'. Known: {list(DATASET_SOURCES)}")
        src_records = list(DATASET_SOURCES[src](data_dir))
        if not src_records:
            print(f"  [skip] source '{src}' produced 0 samples (directory missing?)")
            continue
        print(f"  source '{src}': {len(src_records)} images")
        for rel_path, label in src_records:
            records.append({"path": rel_path, "label": label, "source": src})

    if not records:
        raise RuntimeError(
            "No images found from any configured source. "
            f"Expected at least one of {list(DATASET_SOURCES)} under {raw_dir}."
        )

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
    parser.add_argument("--data-dir", default="data", help="Path to data root (contains raw/, archive/).")
    parser.add_argument("--splits-dir", default="data/splits", help="Output directory for CSVs.")
    parser.add_argument("--sources", nargs="+", default=None,
                        choices=list(DATASET_SOURCES.keys()),
                        help="Which dataset sources to include. Default: all available.")
    args = parser.parse_args()

    generate_splits(args.data_dir, args.splits_dir, sources=args.sources)
