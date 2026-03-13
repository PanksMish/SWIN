"""
augmentations.py
================
Custom augmentation pipeline for the Swin-FANE framework.

Implements the augmentation strategy described in Section 3.3 of the paper:
  - Horizontal flip
  - Colour jittering
  - Random cropping
  - Rotation by ±15°
  - Gaussian blur (motion simulation)
  - Photometric normalisation with ImageNet statistics

Additionally includes perturbation transforms used in the robustness evaluation
(Section 5.6): occlusion, low-light, motion blur, and pose jitter.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random


# ---------------------------------------------------------------------------
# ImageNet normalisation statistics
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Returns the training augmentation pipeline.

    Strategy (paper Section 3.3):
    - Face already localised & cropped by MTCNN (done in dataset loader)
    - Resize to 224×224
    - Random horizontal flip (simulates left/right head turn)
    - Colour jitter (illumination variation)
    - Random crop with padding (slight spatial shift)
    - Small rotation ±15° (mild pose change)
    - Gaussian blur (motion blur simulation)
    - Normalise with ImageNet statistics

    Args:
        image_size: Target spatial resolution (default 224).

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        # Ensure correct size after MTCNN crop (may not be exactly 224)
        A.Resize(height=image_size, width=image_size, p=1.0),

        # Random horizontal flip — simulates head orientation variation
        A.HorizontalFlip(p=0.5),

        # Colour jitter — simulates varying illumination and camera settings
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
            p=0.7
        ),

        # Random crop with reflective padding (±10% spatial jitter)
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.85, 1.0),   # Crop at least 85% of the image
            ratio=(0.9, 1.1),
            p=0.5
        ),

        # Rotation ±15° as specified in paper Section 3.3
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),

        # Gaussian blur — simulates camera motion blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # Slight random brightness/contrast as extra photometric augmentation
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4
        ),

        # Normalise with ImageNet statistics (paper Section 3.3)
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        # Convert HWC numpy array to CHW torch tensor
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Returns the validation / test augmentation pipeline (no random augmentation).

    Args:
        image_size: Target spatial resolution.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


# ---------------------------------------------------------------------------
# Robustness perturbation transforms (Section 5.6)
# ---------------------------------------------------------------------------

class OcclusionTransform:
    """
    Randomly occludes a rectangular region of a facial image to simulate
    partial face obstruction (e.g., hand, glasses, mask).

    Args:
        occlusion_fraction: Fraction of image area to occlude (e.g., 0.2 → 20%).
        fill_value: Pixel value used to fill the occluded region (default: 0 = black).
    """

    def __init__(self, occlusion_fraction: float = 0.2, fill_value: int = 0):
        self.occlusion_fraction = occlusion_fraction
        self.fill_value = fill_value

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: HxWxC numpy array in [0, 255] uint8.

        Returns:
            Occluded image.
        """
        h, w = image.shape[:2]
        # Determine occlusion patch size
        patch_area = int(h * w * self.occlusion_fraction)
        patch_h = int(np.sqrt(patch_area))
        patch_w = patch_h

        # Random top-left corner
        y0 = random.randint(0, max(0, h - patch_h))
        x0 = random.randint(0, max(0, w - patch_w))

        result = image.copy()
        result[y0:y0 + patch_h, x0:x0 + patch_w] = self.fill_value
        return result


class LowLightTransform:
    """
    Simulates low-light acquisition by darkening the image.

    Args:
        brightness_factor: Multiplicative factor in [0, 1]. 
                           Values < 1 darken the image.
    """

    def __init__(self, brightness_factor: float = 0.5):
        self.brightness_factor = brightness_factor

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: HxWxC numpy array in [0, 255] uint8.

        Returns:
            Darkened image.
        """
        result = (image.astype(np.float32) * self.brightness_factor).clip(0, 255)
        return result.astype(np.uint8)


class MotionBlurTransform:
    """
    Applies motion blur to simulate camera shake or fast head movement.

    Args:
        blur_sigma: Standard deviation for Gaussian blur kernel.
    """

    def __init__(self, blur_sigma: float = 2.0):
        self.blur_sigma = blur_sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: HxWxC numpy array.

        Returns:
            Blurred image.
        """
        kernel_size = int(6 * self.blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), self.blur_sigma)
        return blurred


class JPEGCompressionTransform:
    """
    Simulates JPEG compression artefacts by encoding and decoding the image.

    Args:
        quality: JPEG quality level in [1, 100]. Lower values = more artefacts.
    """

    def __init__(self, quality: int = 50):
        self.quality = quality

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: HxWxC numpy array in [0, 255] uint8.

        Returns:
            JPEG-compressed image.
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded


def get_robustness_transform(perturbation: str, level: float, image_size: int = 224):
    """
    Returns a full robustness evaluation pipeline for a given perturbation type.

    Args:
        perturbation: One of ['occlusion', 'low_light', 'motion_blur', 'jpeg'].
        level: Perturbation level (interpretation depends on perturbation type).
        image_size: Target image resolution.

    Returns:
        Callable that accepts a numpy image and returns a torch tensor.
    """
    # Build the appropriate perturbation transform
    if perturbation == 'occlusion':
        perturb_fn = OcclusionTransform(occlusion_fraction=level)
    elif perturbation == 'low_light':
        perturb_fn = LowLightTransform(brightness_factor=level)
    elif perturbation == 'motion_blur':
        perturb_fn = MotionBlurTransform(blur_sigma=level)
    elif perturbation == 'jpeg':
        perturb_fn = JPEGCompressionTransform(quality=int(level))
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation}")

    # Standard post-processing pipeline (resize → normalise → to tensor)
    post_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    def combined(image: np.ndarray) -> torch.Tensor:
        """Apply perturbation followed by standard normalisation."""
        perturbed = perturb_fn(image)
        result = post_transform(image=perturbed)
        return result['image']

    return combined


# ---------------------------------------------------------------------------
# Utility: Denormalise tensor for visualisation
# ---------------------------------------------------------------------------

def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses ImageNet normalisation to produce a displayable image.

    Args:
        tensor: CHW torch tensor.

    Returns:
        HWC numpy array in [0, 255] uint8.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor * std + mean                  # Undo normalisation
    img  = img.clamp(0, 1).permute(1, 2, 0)    # CHW → HWC, clip to [0,1]
    return (img.numpy() * 255).astype(np.uint8)
