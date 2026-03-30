"""Unit tests for the ASL dataset module."""

import os
import sys
import tempfile

import pandas as pd
import torch
from PIL import Image

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import ASLDataset, LABEL_MAP, train_transform, val_transform


def _create_dummy_dataset(tmp_dir, n_images=3):
    """Create a minimal fake dataset and CSV for testing."""
    csv_rows = []
    for label in list(LABEL_MAP.keys())[:5]:  # use 5 classes for speed
        class_dir = os.path.join(tmp_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n_images):
            fname = f"{label}_{i}.jpg"
            img = Image.new("RGB", (200, 200), color=(i * 30, 100, 200))
            img.save(os.path.join(class_dir, fname))
            csv_rows.append({"path": os.path.join(label, fname), "label": label})

    csv_path = os.path.join(tmp_dir, "test_split.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    return csv_path


class TestLabelMap:
    def test_label_count_is_29(self):
        assert len(LABEL_MAP) == 29, f"Expected 29 labels, got {len(LABEL_MAP)}"

    def test_indices_are_contiguous(self):
        indices = sorted(LABEL_MAP.values())
        assert indices == list(range(29))


class TestASLDataset:
    def test_output_tensor_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = _create_dummy_dataset(tmp_dir)
            ds = ASLDataset(csv_path, tmp_dir, transform=val_transform)
            img, label = ds[0]
            assert img.shape == (3, 224, 224), f"Expected (3,224,224), got {img.shape}"
            assert isinstance(label, int)

    def test_dataset_length(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = _create_dummy_dataset(tmp_dir, n_images=4)
            ds = ASLDataset(csv_path, tmp_dir, transform=val_transform)
            assert len(ds) == 5 * 4  # 5 classes × 4 images


class TestTransforms:
    def test_train_val_transforms_differ(self):
        assert train_transform is not val_transform
        assert str(train_transform) != str(val_transform), (
            "train_transform and val_transform must have different compositions"
        )

    def test_train_transform_output_shape(self):
        img = Image.new("RGB", (400, 400))
        tensor = train_transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        img = Image.new("RGB", (400, 400))
        tensor = val_transform(img)
        assert tensor.shape == (3, 224, 224)
