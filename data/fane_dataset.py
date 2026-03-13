"""
fane_dataset.py
===============
Dataset loader for the FANE (Facial Attention Network Embedding) dataset.

Dataset structure expected on disk:
    data/FANE/
    ├── images/
    │   ├── anger/          (2,150 images)
    │   ├── disgust/        (1,280 images)
    │   ├── fear/           (1,820 images)
    │   ├── happiness/      (4,520 images)
    │   ├── sadness/        (2,310 images)
    │   ├── surprise/       (1,705 images)
    │   └── neutral/        (3,128 images)
    └── masks/              (optional expressive-region masks, same structure)
        ├── anger/
        ├── ...
        └── neutral/

Total: ~16,913 images across 7 emotion categories
       ~1,240 subjects (subject IDs extracted from filename convention)

Filename convention (for subject-independent split):
    <subject_id>_<session>_<frame>.jpg
    e.g.  sub042_session03_frame0017.jpg

If your filenames do not embed subject IDs, the loader falls back to a
random stratified split.

Paper reference: Section 3.1 – 3.3
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# MTCNN for face localisation (paper Section 3.3)
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("[WARNING] facenet-pytorch not installed. "
          "MTCNN face detection will be skipped.")

from data.augmentations import (
    get_train_transforms,
    get_val_transforms,
    get_robustness_transform,
)


# ---------------------------------------------------------------------------
# Emotion class definitions (paper Equation 1)
# ---------------------------------------------------------------------------
EMOTION_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
CLASS_TO_IDX    = {cls: idx for idx, cls in enumerate(EMOTION_CLASSES)}

# Negative emotion indices used in stress formulation (Equation 4)
NEGATIVE_EMOTION_INDICES = [0, 1, 2, 4]  # anger, disgust, fear, sadness


# ---------------------------------------------------------------------------
# Helper: extract subject ID from filename
# ---------------------------------------------------------------------------

def extract_subject_id(filename: str) -> Optional[str]:
    """
    Extracts subject identifier from filename using common naming conventions.

    Supports patterns:
      - sub042_session03_frame0017.jpg  →  "sub042"
      - S042_001.jpg                    →  "S042"
      - 042_17.jpg                      →  "042"

    Returns None if no subject ID pattern is detected.
    """
    # Pattern 1: sub<id>_...
    m = re.match(r'^(sub\d+)', filename, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Pattern 2: S<id>_...
    m = re.match(r'^(S\d+)', filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 3: numeric ID at start
    m = re.match(r'^(\d{3,5})', filename)
    if m:
        return m.group(1)

    return None


# ---------------------------------------------------------------------------
# Core dataset class: FANEDataset
# ---------------------------------------------------------------------------

class FANEDataset(Dataset):
    """
    PyTorch Dataset for the FANE facial emotion recognition dataset.

    Supports:
      - Frame-level emotion classification (Phase I)
      - Temporal sequence construction for stress estimation (Phase II)
      - Subject-independent splits (paper Section 5.1)
      - Optional MTCNN face localisation
      - Optional mask loading (expressive region annotations)

    Args:
        root_dir:        Path to dataset root (contains 'images/' and optionally 'masks/').
        split:           One of 'train', 'val', 'test'.
        split_file:      Optional path to pre-saved JSON split file.
                         If None, splits are computed automatically.
        image_size:      Target spatial resolution.
        sequence_length: Number of consecutive frames per temporal sequence.
                         Set to 1 for frame-level dataset (no temporal grouping).
        use_mtcnn:       Whether to apply MTCNN face cropping.
        use_masks:       Whether to load expressive-region mask annotations.
        transform:       Albumentations transform pipeline (overrides default).
        subject_independent: Enforce subject-independent split.
        seed:            Random seed for reproducible splitting.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        split_file: Optional[str] = None,
        image_size: int = 224,
        sequence_length: int = 1,
        use_mtcnn: bool = False,
        use_masks: bool = False,
        transform=None,
        subject_independent: bool = True,
        seed: int = 42,
    ):
        self.root_dir         = Path(root_dir)
        self.split            = split
        self.image_size       = image_size
        self.sequence_length  = sequence_length
        self.use_masks        = use_masks
        self.subject_independent = subject_independent
        self.seed             = seed

        # ---- Set up default transforms if not provided ----
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        # ---- Initialise MTCNN (optional) ----
        self.mtcnn = None
        if use_mtcnn and MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                image_size=image_size,
                keep_all=False,          # Detect single largest face
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                device='cpu'
            )

        # ---- Load or compute split ----
        if split_file and os.path.exists(split_file):
            self.samples = self._load_split_from_file(split_file, split)
        else:
            all_samples = self._scan_dataset()
            splits      = self._create_splits(all_samples, subject_independent)
            self.samples = splits[split]

            # Optionally save the split for reproducibility
            if split_file:
                self._save_splits(splits, split_file)

        # ---- Build temporal sequences if needed ----
        if sequence_length > 1:
            self.sequences = self._build_sequences()
        else:
            self.sequences = None

        print(f"[FANEDataset] Split='{split}' | "
              f"Samples={len(self.samples)} | "
              f"Sequences={len(self.sequences) if self.sequences else 'N/A'}")

    # ------------------------------------------------------------------
    # Dataset scanning
    # ------------------------------------------------------------------

    def _scan_dataset(self) -> List[Dict]:
        """
        Recursively scans the dataset directory and builds a list of sample dicts.

        Returns:
            List of dicts: {path, label, label_name, subject_id, mask_path}
        """
        images_dir = self.root_dir / "images"
        masks_dir  = self.root_dir / "masks"

        if not images_dir.exists():
            # Fallback: assume root_dir directly contains class folders
            images_dir = self.root_dir
            print(f"[FANEDataset] 'images/' subdir not found; "
                  f"scanning {self.root_dir} directly.")

        samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        for class_name in EMOTION_CLASSES:
            class_dir = images_dir / class_name
            if not class_dir.exists():
                print(f"[WARNING] Class folder not found: {class_dir}")
                continue

            label = CLASS_TO_IDX[class_name]

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in valid_extensions:
                    continue

                # Determine mask path (same filename, masks/ subdirectory)
                mask_path = None
                if self.use_masks:
                    candidate = masks_dir / class_name / img_path.name
                    if candidate.exists():
                        mask_path = str(candidate)

                # Extract subject ID for subject-independent split
                subject_id = extract_subject_id(img_path.stem)

                samples.append({
                    'path':       str(img_path),
                    'label':      label,
                    'label_name': class_name,
                    'subject_id': subject_id,
                    'mask_path':  mask_path,
                })

        print(f"[FANEDataset] Found {len(samples)} images across "
              f"{len(EMOTION_CLASSES)} classes.")
        return samples

    # ------------------------------------------------------------------
    # Split creation (subject-independent)
    # ------------------------------------------------------------------

    def _create_splits(
        self,
        samples: List[Dict],
        subject_independent: bool,
        ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    ) -> Dict[str, List[Dict]]:
        """
        Creates train/val/test splits.

        If subject_independent=True and subject IDs are available:
            Splits by subject so no subject appears in multiple folds.
        Otherwise:
            Stratified random split by class.

        Args:
            samples:  All dataset samples.
            subject_independent: Whether to enforce subject-based splitting.
            ratios:   (train_ratio, val_ratio, test_ratio). Must sum to 1.

        Returns:
            Dict with keys 'train', 'val', 'test'.
        """
        rng = random.Random(self.seed)

        # Check if subject IDs are available
        subjects = [s['subject_id'] for s in samples if s['subject_id'] is not None]
        has_subjects = len(subjects) > len(samples) * 0.5  # >50% have subject IDs

        if subject_independent and has_subjects:
            return self._subject_split(samples, ratios, rng)
        else:
            print("[FANEDataset] Subject IDs not detected; "
                  "using stratified random split.")
            return self._stratified_split(samples, ratios, rng)

    def _subject_split(
        self,
        samples: List[Dict],
        ratios: Tuple[float, float, float],
        rng: random.Random
    ) -> Dict[str, List[Dict]]:
        """
        Splits by subject ID (subject-independent evaluation, paper Section 5.1).
        """
        # Gather unique subjects
        subject_set = sorted({s['subject_id'] for s in samples
                               if s['subject_id'] is not None})
        rng.shuffle(subject_set)

        n = len(subject_set)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])

        train_subjects = set(subject_set[:n_train])
        val_subjects   = set(subject_set[n_train:n_train + n_val])
        test_subjects  = set(subject_set[n_train + n_val:])

        splits = {'train': [], 'val': [], 'test': []}
        for s in samples:
            sid = s['subject_id']
            if sid in train_subjects:
                splits['train'].append(s)
            elif sid in val_subjects:
                splits['val'].append(s)
            elif sid in test_subjects:
                splits['test'].append(s)
            else:
                # Samples without subject ID go to training
                splits['train'].append(s)

        print(f"[FANEDataset] Subject-independent split: "
              f"train={len(splits['train'])}, "
              f"val={len(splits['val'])}, "
              f"test={len(splits['test'])}")
        return splits

    def _stratified_split(
        self,
        samples: List[Dict],
        ratios: Tuple[float, float, float],
        rng: random.Random
    ) -> Dict[str, List[Dict]]:
        """
        Stratified random split maintaining class proportions.
        """
        # Group by class
        class_buckets: Dict[int, List[Dict]] = {i: [] for i in range(len(EMOTION_CLASSES))}
        for s in samples:
            class_buckets[s['label']].append(s)

        splits = {'train': [], 'val': [], 'test': []}

        for label, bucket in class_buckets.items():
            rng.shuffle(bucket)
            n = len(bucket)
            n_train = int(n * ratios[0])
            n_val   = int(n * ratios[1])

            splits['train'].extend(bucket[:n_train])
            splits['val'].extend(bucket[n_train:n_train + n_val])
            splits['test'].extend(bucket[n_train + n_val:])

        # Shuffle each split
        for key in splits:
            rng.shuffle(splits[key])

        return splits

    # ------------------------------------------------------------------
    # Temporal sequence construction
    # ------------------------------------------------------------------

    def _build_sequences(self) -> List[List[Dict]]:
        """
        Groups consecutive same-subject frames into temporal sequences.

        Strategy (paper Section 3.1):
          - Group samples by subject_id and label_name (recording session)
          - Create non-overlapping windows of length T
          - If no subject IDs, group by class with random shuffling

        Returns:
            List of sequences, where each sequence is a list of T sample dicts.
        """
        T = self.sequence_length

        # Group by (subject_id, label) to create pseudo-sessions
        session_map: Dict[str, List[Dict]] = {}
        for s in self.samples:
            key = f"{s.get('subject_id', 'unknown')}_{s['label']}"
            if key not in session_map:
                session_map[key] = []
            session_map[key].append(s)

        sequences = []
        for key, session_samples in session_map.items():
            # Slide a non-overlapping window of length T
            for i in range(0, len(session_samples) - T + 1, T):
                seq = session_samples[i:i + T]
                if len(seq) == T:
                    sequences.append(seq)

        print(f"[FANEDataset] Built {len(sequences)} temporal sequences "
              f"of length T={T}.")
        return sequences

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_splits(self, splits: Dict[str, List[Dict]], path: str):
        """Saves split dictionaries to a JSON file for reproducibility."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"[FANEDataset] Splits saved to {path}")

    def _load_split_from_file(self, split_file: str, split: str) -> List[Dict]:
        """Loads a previously saved split from a JSON file."""
        with open(split_file) as f:
            splits = json.load(f)
        print(f"[FANEDataset] Loaded '{split}' split from {split_file} "
              f"({len(splits[split])} samples).")
        return splits[split]

    # ------------------------------------------------------------------
    # Image loading helpers
    # ------------------------------------------------------------------

    def _load_image(self, img_path: str) -> np.ndarray:
        """
        Loads an image as a BGR numpy array and converts to RGB.

        If MTCNN is available, applies face detection and crops the face region.
        If no face is detected, falls back to the full image.

        Args:
            img_path: Absolute path to the image file.

        Returns:
            RGB numpy array of shape (H, W, 3) in uint8.
        """
        # Load with OpenCV (BGR)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback: try Pillow for uncommon formats
            pil_img = Image.open(img_path).convert('RGB')
            return np.array(pil_img)

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Optional MTCNN face localisation (paper Section 3.3)
        if self.mtcnn is not None:
            pil_img = Image.fromarray(img_rgb)
            cropped = self.mtcnn(pil_img)  # Returns tensor or None
            if cropped is not None:
                # Convert tensor (C, H, W) back to numpy uint8
                cropped_np = (cropped.permute(1, 2, 0).numpy() * 128 + 127.5)
                img_rgb = cropped_np.clip(0, 255).astype(np.uint8)

        return img_rgb

    def _load_mask(self, mask_path: str) -> Optional[np.ndarray]:
        """
        Loads an expressive-region mask as a binary numpy array.

        Args:
            mask_path: Absolute path to the mask image.

        Returns:
            Binary mask of shape (H, W) with values in {0, 1},
            or None if the path is invalid.
        """
        if mask_path is None or not os.path.exists(mask_path):
            return None

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None

        # Binarise: any non-zero pixel = expressive region
        return (mask > 127).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.sequences is not None:
            return len(self.sequences)
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves a single sample or temporal sequence.

        Returns a dict with:
          - 'image':    Transformed image tensor  (C, H, W)         [frame mode]
                     or sequence tensor           (T, C, H, W)      [sequence mode]
          - 'label':    Integer emotion class label
          - 'mask':     Expressive region mask     (1, H, W) or None
          - 'path':     Image file path(s)
        """
        if self.sequences is not None:
            return self._get_sequence(idx)
        else:
            return self._get_frame(idx)

    def _get_frame(self, idx: int) -> Dict:
        """Returns a single frame sample."""
        sample = self.samples[idx]

        img   = self._load_image(sample['path'])
        mask  = self._load_mask(sample.get('mask_path'))

        # Apply augmentation transform
        augmented = self.transform(image=img)
        img_tensor = augmented['image']   # (C, H, W) float32

        # Resize mask to match image size and convert to tensor
        mask_tensor = None
        if mask is not None:
            mask_resized = cv2.resize(mask, (self.image_size, self.image_size))
            mask_tensor  = torch.from_numpy(mask_resized).unsqueeze(0)  # (1, H, W)

        return {
            'image': img_tensor,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'mask':  mask_tensor,
            'path':  sample['path'],
        }

    def _get_sequence(self, idx: int) -> Dict:
        """Returns a temporal sequence of T frames."""
        seq_samples = self.sequences[idx]

        frames     = []
        masks      = []
        paths      = []
        # Use the label of the majority class in the sequence
        labels     = [s['label'] for s in seq_samples]
        seq_label  = max(set(labels), key=labels.count)

        for sample in seq_samples:
            img  = self._load_image(sample['path'])
            mask = self._load_mask(sample.get('mask_path'))

            augmented  = self.transform(image=img)
            img_tensor = augmented['image']   # (C, H, W)
            frames.append(img_tensor)

            if mask is not None:
                mask_resized = cv2.resize(mask, (self.image_size, self.image_size))
                masks.append(torch.from_numpy(mask_resized).unsqueeze(0))

            paths.append(sample['path'])

        # Stack into (T, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        masks_tensor  = torch.stack(masks, dim=0) if masks else None

        return {
            'image': frames_tensor,                              # (T, C, H, W)
            'label': torch.tensor(seq_label, dtype=torch.long),
            'mask':  masks_tensor,
            'path':  paths,
        }

    # ------------------------------------------------------------------
    # Utility: class distribution
    # ------------------------------------------------------------------

    def get_class_distribution(self) -> Dict[str, int]:
        """Returns the count of samples per emotion class."""
        dist = {cls: 0 for cls in EMOTION_CLASSES}
        samples = self.samples
        for s in samples:
            dist[EMOTION_CLASSES[s['label']]] += 1
        return dist

    def get_class_weights(self) -> torch.Tensor:
        """
        Computes inverse-frequency class weights for weighted cross-entropy loss.

        Returns:
            Float tensor of shape (num_classes,).
        """
        dist   = self.get_class_distribution()
        counts = torch.tensor([dist[cls] for cls in EMOTION_CLASSES], dtype=torch.float)
        weights = 1.0 / (counts + 1e-6)
        return weights / weights.sum() * len(EMOTION_CLASSES)


# ---------------------------------------------------------------------------
# Dataset factory function
# ---------------------------------------------------------------------------

def build_fane_dataloaders(
    root_dir: str,
    image_size: int = 224,
    sequence_length: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    use_mtcnn: bool = False,
    use_masks: bool = False,
    subject_independent: bool = True,
    seed: int = 42,
    split_file: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to build train/val/test DataLoaders for the FANE dataset.

    Args:
        root_dir:           Path to FANE dataset root.
        image_size:         Spatial resolution.
        sequence_length:    Temporal sequence length (1 = frame-level).
        batch_size:         Mini-batch size.
        num_workers:        Number of DataLoader worker processes.
        use_mtcnn:          Enable MTCNN face detection.
        use_masks:          Load expressive-region mask annotations.
        subject_independent: Enforce subject-independent splitting.
        seed:               Random seed.
        split_file:         Optional path to save/load split JSON.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    shared_kwargs = dict(
        root_dir=root_dir,
        image_size=image_size,
        sequence_length=sequence_length,
        use_mtcnn=use_mtcnn,
        use_masks=use_masks,
        subject_independent=subject_independent,
        seed=seed,
        split_file=split_file,
    )

    train_dataset = FANEDataset(split='train', **shared_kwargs)
    val_dataset   = FANEDataset(split='val',   **shared_kwargs)
    test_dataset  = FANEDataset(split='test',  **shared_kwargs)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
