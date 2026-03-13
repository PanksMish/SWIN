"""
swin_fane.py
============
Swin-FANE Spatial Encoder — Phase I of the two-phase framework.

Combines:
  1. SwinTransformerBackbone — hierarchical feature extraction
  2. FANEModule              — region-aware attention over facial areas
  3. EmotionClassifierHead   — maps embedding to emotion probability distribution

Algorithm 1 and Algorithm 2 from the paper are implemented here.

Forward pass:
    x_t (B, C, H, W)
      ↓  Swin Transformer
    F   (B, N_tokens, feature_dim)     ← intermediate feature map
      ↓  FANE Module
    z_t (B, embedding_dim)             ← frame-level embedding
      ↓  Linear + Softmax
    p_t (B, num_classes)               ← emotion probability vector

Paper reference: Sections 4.1–4.2, Algorithms 1–2, Equations 6–10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.swin_transformer import build_swin_backbone
from models.fane_module import FANEModule


class EmotionClassifierHead(nn.Module):
    """
    Linear classification head for frame-level emotion prediction.

    Maps embedding z_t → logits → emotion probabilities p_t.
    (Paper Equations 8–10)

    Args:
        embedding_dim: Dimension of the input embedding z_t.
        num_classes:   Number of emotion categories (7 for FANE dataset).
        dropout:       Dropout probability before the linear layer.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout    = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Linear classifier W_k, b_k (paper Equation 8)
        self.fc = nn.Linear(embedding_dim, num_classes)

        # Weight initialisation
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Embedding tensor (B, embedding_dim).

        Returns:
            logits: Raw class logits (B, num_classes).
            probs:  Softmax probabilities (B, num_classes) — paper Equation 9.
        """
        z      = self.layer_norm(z)
        z      = self.dropout(z)
        logits = self.fc(z)                      # Equation 8: ℓ_k = W_k z + b_k
        probs  = F.softmax(logits, dim=-1)       # Equation 9: softmax normalisation
        return logits, probs


class SwinFANEEncoder(nn.Module):
    """
    Phase I spatial encoder: Swin Transformer + FANE + Emotion Classifier.

    Implements Algorithm 1 (Swin-FANE Spatial Encoding) and
    Algorithm 2 (Frame-Level Emotion Classification) from the paper.

    Args:
        swin_model_name:   timm model identifier for the Swin backbone.
        pretrained:        Load ImageNet pretrained Swin weights.
        embedding_dim:     Target embedding dimension (z_t dimension).
        num_classes:       Number of emotion categories.
        fane_hidden_dim:   FANE bottleneck projection size.
        use_mask_guidance: Enable mask-supervised FANE attention.
        classifier_dropout: Dropout in the classification head.
        drop_path_rate:    Swin stochastic depth rate.
    """

    def __init__(
        self,
        swin_model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        embedding_dim: int = 1024,
        num_classes: int = 7,
        fane_hidden_dim: Optional[int] = 256,
        use_mask_guidance: bool = False,
        classifier_dropout: float = 0.3,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes

        # ---- 1. Swin Transformer backbone ----
        self.backbone = build_swin_backbone(
            model_name=swin_model_name,
            pretrained=pretrained,
            output_dim=embedding_dim,
            drop_path_rate=drop_path_rate,
        )

        # Native feature dim of the backbone (before our projection)
        native_dim = self.backbone.native_dim if hasattr(self.backbone, 'native_dim') \
                     else embedding_dim

        # ---- 2. FANE attention module ----
        self.fane = FANEModule(
            feature_dim=native_dim,
            hidden_dim=fane_hidden_dim,
            use_mask_guidance=use_mask_guidance,
        )

        # ---- Projection from FANE output (native_dim) → embedding_dim ----
        # Only needed if FANE output dim != embedding_dim
        if native_dim != embedding_dim:
            self.fane_proj = nn.Sequential(
                nn.LayerNorm(native_dim),
                nn.Linear(native_dim, embedding_dim),
                nn.GELU(),
            )
        else:
            self.fane_proj = nn.Identity()

        # ---- 3. Emotion classifier head ----
        self.classifier = EmotionClassifierHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of the Swin-FANE spatial encoder.

        Implements Algorithm 1 + Algorithm 2 from the paper.

        Args:
            x:    Input facial frame batch. Shape: (B, C, H, W).
            mask: Optional binary expressive-region mask.
                  Shape: (B, 1, H, W). Used for FANE mask guidance.

        Returns:
            Dict containing:
              'embedding': Frame embedding z_t    (B, embedding_dim)
              'logits':    Class logits            (B, num_classes)
              'probs':     Emotion probabilities   (B, num_classes)
              'attention': FANE attention map      (B, N_tokens, feature_dim)
              'pred_label': Predicted emotion class (B,)  — Equation 10
        """
        # Step 1: Swin Transformer forward
        # Returns final embedding AND intermediate feature map
        z_swin, F = self.backbone(x)   # z_swin: (B, embedding_dim), F: (B, N, C)

        # Step 2: FANE module — region-aware attention refinement
        # z_fane is GlobalAvgPool(F'), A is the attention mask
        z_fane, A = self.fane(F, mask=mask)   # (B, native_dim), (B, N, C)

        # Project FANE embedding to target dimension
        z_fane = self.fane_proj(z_fane)        # (B, embedding_dim)

        # Combine backbone embedding with FANE embedding (additive fusion)
        z = z_swin + z_fane                    # (B, embedding_dim)

        # Step 3: Emotion classification (Equations 8–10)
        logits, probs = self.classifier(z)     # (B, num_classes) each

        # Predicted class label (Equation 10): ŷ_t = argmax_k p_t^k
        pred_label = torch.argmax(probs, dim=-1)   # (B,)

        return {
            'embedding':  z,
            'logits':     logits,
            'probs':      probs,
            'attention':  A,
            'pred_label': pred_label,
        }

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts frame-level embeddings without computing classification output.
        Used for building the temporal embedding sequence Z in Phase II.

        Args:
            x: Input frames (B, C, H, W).

        Returns:
            Embeddings z_t (B, embedding_dim).
        """
        with torch.no_grad():
            output = self.forward(x)
        return output['embedding']

    def get_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
