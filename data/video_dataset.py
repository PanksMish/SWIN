"""
video_dataset.py
================
Dataset loader for real-world video files.

Used for:
  - Inference on arbitrary video files (scripts/infer_video.py)
  - Robustness evaluation under frame-drop and temporal jitter (Section 5.6)
  - Temporal stress trajectory visualisation

Supports any video format readable by OpenCV (MP4, AVI, MOV, etc.).

Face detection is performed per-frame using MTCNN. Frames where no face is
detected are either skipped or filled with a black placeholder, depending
on the `skip_no_face` parameter.

Paper reference: Section 5.6 (robustness under frame-drop and temporal jitter)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from torch.utils.data import Dataset, IterableDataset

from data.augmentations import get_val_transforms

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


class VideoFrameDataset(Dataset):
    """
    Loads a video file and exposes individual frames as a Dataset.

    Frames are decoded once at construction time and stored in memory.
    For long videos, use VideoFrameIterator instead.

    Args:
        video_path:     Path to the video file.
        image_size:     Target frame resolution.
        sequence_length: Number of frames per temporal window (T).
        frame_stride:   Step between consecutive frames (1 = every frame).
        max_frames:     Maximum number of frames to load (None = all).
        use_mtcnn:      Apply MTCNN face detection on each frame.
        skip_no_face:   Skip frames where MTCNN detects no face.
        transform:      Augmentation pipeline (defaults to val transform).
    """

    def __init__(
        self,
        video_path: str,
        image_size: int = 224,
        sequence_length: int = 16,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        use_mtcnn: bool = True,
        skip_no_face: bool = False,
        transform=None,
    ):
        self.video_path      = Path(video_path)
        self.image_size      = image_size
        self.sequence_length = sequence_length
        self.frame_stride    = frame_stride
        self.skip_no_face    = skip_no_face
        self.transform       = transform or get_val_transforms(image_size)

        # Initialise MTCNN
        self.mtcnn = None
        if use_mtcnn and MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                image_size=image_size,
                keep_all=False,
                min_face_size=40,
                device='cpu'
            )

        # Load and preprocess all frames
        self.frames: List[torch.Tensor] = self._load_frames(max_frames)
        # Build temporal windows
        self.windows: List[torch.Tensor] = self._build_windows()

        print(f"[VideoFrameDataset] '{self.video_path.name}': "
              f"{len(self.frames)} frames → {len(self.windows)} sequences "
              f"of T={sequence_length}.")

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _load_frames(self, max_frames: Optional[int]) -> List[torch.Tensor]:
        """
        Decodes video frames, applies face detection, and returns tensors.

        Returns:
            List of (C, H, W) float32 tensors.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        frames   = []
        frame_no = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Apply stride (only process every `frame_stride`-th frame)
            if frame_no % self.frame_stride != 0:
                frame_no += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # MTCNN face detection
            if self.mtcnn is not None:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(frame_rgb)
                cropped = self.mtcnn(pil_img)
                if cropped is not None:
                    # Convert tensor (C, H, W) → numpy uint8
                    img_np = (cropped.permute(1, 2, 0).numpy() * 128 + 127.5)
                    frame_rgb = img_np.clip(0, 255).astype(np.uint8)
                elif self.skip_no_face:
                    frame_no += 1
                    continue
                # else: keep full frame if no face detected

            # Apply transforms (resize + normalise)
            augmented = self.transform(image=frame_rgb)
            frames.append(augmented['image'])   # (C, H, W)

            frame_no += 1
            if max_frames and len(frames) >= max_frames:
                break

        cap.release()
        return frames

    def _build_windows(self) -> List[torch.Tensor]:
        """
        Slides a non-overlapping window of size T over the frame list.

        Returns:
            List of (T, C, H, W) tensors.
        """
        T       = self.sequence_length
        windows = []
        for i in range(0, len(self.frames) - T + 1, T):
            window = torch.stack(self.frames[i:i + T], dim=0)  # (T, C, H, W)
            windows.append(window)
        return windows

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return {
            'image': self.windows[idx],   # (T, C, H, W)
            'label': torch.tensor(-1, dtype=torch.long),   # Unknown for inference
            'path':  str(self.video_path),
            'window_index': idx,
        }

    # ------------------------------------------------------------------
    # Robustness helpers (Section 5.6)
    # ------------------------------------------------------------------

    def apply_frame_drop(self, drop_rate: float) -> 'VideoFrameDataset':
        """
        Simulates unstable video acquisition by randomly dropping frames.

        Args:
            drop_rate: Fraction of frames to drop (e.g., 0.2 → drop 20%).

        Returns:
            A new VideoFrameDataset with dropped frames (in-place modification).
        """
        import random
        n_keep    = int(len(self.frames) * (1.0 - drop_rate))
        keep_idx  = sorted(random.sample(range(len(self.frames)), n_keep))
        self.frames  = [self.frames[i] for i in keep_idx]
        self.windows = self._build_windows()
        return self

    def apply_temporal_jitter(self, max_jitter: int = 3) -> 'VideoFrameDataset':
        """
        Simulates temporal jitter by randomly shuffling frames within a small window.

        Args:
            max_jitter: Maximum number of positions a frame can shift.
        """
        import random
        jittered = []
        for i, f in enumerate(self.frames):
            j = i + random.randint(-max_jitter, max_jitter)
            j = max(0, min(j, len(self.frames) - 1))
            jittered.append(self.frames[j])
        self.frames  = jittered
        self.windows = self._build_windows()
        return self


# ---------------------------------------------------------------------------
# Utility: get video metadata
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    """
    Returns basic metadata about a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: fps, total_frames, duration_sec, width, height.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return {
        'fps':          fps,
        'total_frames': total_frames,
        'duration_sec': duration_sec,
        'width':        width,
        'height':       height,
    }
