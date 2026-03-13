"""
fer2013_dataset.py
==================
Dataset loader for FER2013 — an alternative / pre-training source.

FER2013 contains 48×48 grayscale face images across 7 emotion classes:
  0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

Two format variants are supported:
  A) Folder layout (Kaggle download):
       data/FER2013/train/<class>/<img>.png
       data/FER2013/test/<class>/<img>.png

  B) CSV format (original Kaggle challenge):
       data/FER2013/fer2013.csv
       Columns: emotion, pixels, Usage (Training / PublicTest / PrivateTest)

Paper reference: Baselines in Table 5 were originally evaluated on FER2013.
This loader is provided so those baselines can be retrained on FANE-format data,
or so users can pre-train the Swin-FANE backbone on FER2013 before fine-tuning.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data.augmentations import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# FER2013 class mapping (matches FANE ordering where possible)
# ---------------------------------------------------------------------------
# FER2013 native order: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
FER2013_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# Remapping: FER2013 class index → our unified emotion index
# (Both orderings happen to be identical, so this is an identity mapping.)
FER2013_TO_UNIFIED = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


class FER2013FolderDataset(Dataset):
    """
    FER2013 dataset loaded from a folder structure (post-Kaggle extraction).

    Expected layout:
        root_dir/
        ├── train/
        │   ├── angry/
        │   ├── disgust/
        │   └── ...
        └── test/
            ├── angry/
            └── ...

    Args:
        root_dir:    Path to FER2013 root directory.
        split:       'train' or 'test'.
        image_size:  Target resolution (images are upsampled from 48×48).
        transform:   Optional augmentation pipeline.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 224,
        transform=None,
    ):
        self.root_dir   = Path(root_dir)
        self.split      = split
        self.image_size = image_size
        self.transform  = transform or (
            get_train_transforms(image_size) if split == 'train'
            else get_val_transforms(image_size)
        )

        self.samples: List[Dict] = self._scan()
        print(f"[FER2013FolderDataset] Split='{split}' | Samples={len(self.samples)}")

    def _scan(self) -> List[Dict]:
        """Scans folder structure and collects all image paths with labels."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples = []
        for cls_name in FER2013_CLASSES:
            # Try both 'angry' and 'anger' naming conventions
            for variant in [cls_name, cls_name.rstrip('e') + 'y',
                            cls_name.capitalize()]:
                cls_dir = split_dir / variant
                if cls_dir.exists():
                    label = FER2013_CLASSES.index(cls_name)
                    for img_path in sorted(cls_dir.iterdir()):
                        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                            samples.append({
                                'path':  str(img_path),
                                'label': label,
                            })
                    break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load as grayscale (FER2013 native), convert to 3-channel RGB
        img_gray = cv2.imread(sample['path'], cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            img_gray = np.zeros((48, 48), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        augmented = self.transform(image=img_rgb)
        return {
            'image': augmented['image'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'mask':  None,
            'path':  sample['path'],
        }


class FER2013CSVDataset(Dataset):
    """
    FER2013 dataset loaded from the original CSV file.

    CSV format:
        emotion, pixels, Usage
        0, "70 80 82 ...", Training

    Args:
        csv_path:    Path to fer2013.csv.
        split:       'train', 'val', or 'test'.
                     Maps to CSV Usage: Training / PublicTest / PrivateTest.
        image_size:  Target spatial resolution.
        transform:   Optional augmentation pipeline.
    """

    USAGE_MAP = {
        'train': 'Training',
        'val':   'PublicTest',
        'test':  'PrivateTest',
    }

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        image_size: int = 224,
        transform=None,
    ):
        self.csv_path   = csv_path
        self.split      = split
        self.image_size = image_size
        self.transform  = transform or (
            get_train_transforms(image_size) if split == 'train'
            else get_val_transforms(image_size)
        )

        self.samples: List[Tuple[np.ndarray, int]] = self._load_csv()
        print(f"[FER2013CSVDataset] Split='{split}' | Samples={len(self.samples)}")

    def _load_csv(self) -> List[Tuple[np.ndarray, int]]:
        """
        Parses the CSV file and loads pixel arrays into memory.

        Returns:
            List of (pixel_array, label) tuples.
        """
        usage_key = self.USAGE_MAP.get(self.split, 'Training')
        samples   = []

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Usage', 'Training') != usage_key:
                    continue

                label  = FER2013_TO_UNIFIED[int(row['emotion'])]
                pixels = np.array(row['pixels'].split(), dtype=np.uint8)
                img    = pixels.reshape(48, 48)          # 48×48 grayscale
                samples.append((img, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_gray, label = self.samples[idx]

        # Convert grayscale → RGB (3-channel required by Swin Transformer)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        augmented = self.transform(image=img_rgb)
        return {
            'image': augmented['image'],
            'label': torch.tensor(label, dtype=torch.long),
            'mask':  None,
            'path':  f"fer2013_csv_{idx}",
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_fer2013_dataloaders(
    root_dir: str,
    csv_path: Optional[str] = None,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train/val/test DataLoaders for FER2013.

    Automatically detects whether to use folder or CSV format.

    Args:
        root_dir:    Path to FER2013 dataset directory.
        csv_path:    Path to fer2013.csv (if using CSV format).
        image_size:  Target resolution.
        batch_size:  Mini-batch size.
        num_workers: Number of DataLoader worker threads.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    if csv_path and os.path.exists(csv_path):
        # CSV format
        print(f"[FER2013] Using CSV format: {csv_path}")
        DatasetClass = FER2013CSVDataset
        train_ds = DatasetClass(csv_path, split='train', image_size=image_size)
        val_ds   = DatasetClass(csv_path, split='val',   image_size=image_size)
        test_ds  = DatasetClass(csv_path, split='test',  image_size=image_size)
    else:
        # Folder format
        print(f"[FER2013] Using folder format: {root_dir}")
        DatasetClass = FER2013FolderDataset
        train_ds = DatasetClass(root_dir, split='train', image_size=image_size)
        # FER2013 folder format has no separate val set; reuse test
        val_ds   = DatasetClass(root_dir, split='test',  image_size=image_size)
        test_ds  = DatasetClass(root_dir, split='test',  image_size=image_size)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
