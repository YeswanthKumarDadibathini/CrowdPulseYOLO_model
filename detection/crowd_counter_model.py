"""
CrowdPulse - Pretrained Deep Learning Crowd Counter (Transfer Learning)
Project: CrowdPulse - Crowd Density Management System (SIH25165)

Beginner-Friendly Explanation:
Standard object detectors like YOLO draw boxes around individual visible bodies,
which works great for sparse scenes (5-40 people) but undercounts in dense crowds
(100-3000 people) where only tiny heads are visible.

To achieve next-level headcount accuracy on dense crowds, this module uses
Transfer Learning on a pretrained deep vision backbone (ConvNeXt-Tiny, ResNet-50,
or EfficientNet-B0 pretrained on ImageNet) paired with a Multi-Scale Spatial
Pyramid Pooling (SPP) regression head trained directly on your 1,198-image
Dataset_CrowdPulse benchmark.

Key Mathematical Safeguards (Zero Mistakes):
1. Predicts log(1 + count) so small crowds (15 people) and massive crowds (2,500 people)
   receive balanced accuracy without exploding gradients.
2. Converts predictions back via exp(y) - 1 clamped at >= 0 (never predicts negative people).
3. Uses Test-Time Augmentation (average of original + horizontally flipped frame)
   at inference time to boost stability and reduce count error.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

import config


# Standard ImageNet normalization statistics expected by pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MultiScaleCrowdHead(nn.Module):
    """Multi-Scale Spatial Pyramid Pooling (SPP) Regression Head.

    Pools convolutional feature maps at multiple spatial scales:
    - 1x1 Average Pool (overall crowd density across the whole frame)
    - 1x1 Max Pool (peak congestion intensity)
    - 2x2 Average Pool (quadrant-level crowd distribution)
    Total pooled dimension = in_channels * (1 + 1 + 4) = 6 * in_channels.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool_1 = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool_2 = nn.AdaptiveAvgPool2d((2, 2))

        pooled_dim = in_channels * 6

        self.regressor = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        p1 = torch.flatten(self.avg_pool_1(feat_map), 1)
        p2 = torch.flatten(self.max_pool_1(feat_map), 1)
        p3 = torch.flatten(self.avg_pool_2(feat_map), 1)

        combined = torch.cat([p1, p2, p3], dim=1)
        log_count = self.regressor(combined).squeeze(-1)
        return log_count


class CrowdPulseCountNet(nn.Module):
    """Pretrained CNN Backbone + Multi-Scale Crowd Count Regressor."""

    SUPPORTED_BACKBONES = ("convnext_tiny", "resnet50", "efficientnet_b0")

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone_name

        if backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            self.features = base.features
            feature_channels = 768

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            # Keep all layers up to layer4 (exclude avgpool & fc)
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
            )
            feature_channels = 2048

        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            feature_channels = 1280

        self.head = MultiScaleCrowdHead(
            in_channels=feature_channels,
            hidden_dim=256,
            dropout=dropout,
        )

    def freeze_backbone(self):
        """Freezes pretrained feature extractor for Stage 1 head warmup."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes all layers for Stage 2 full end-to-end fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning predicted log1p(crowd_count)."""
        feat_map = self.features(x)
        return self.head(feat_map)

    @torch.no_grad()
    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        """Converts log1p output back into non-negative headcount."""
        log_pred = self.forward(x)
        return torch.expm1(log_pred.clamp(min=0.0, max=10.0))


class CrowdCountPredictor:
    """High-level inference wrapper for loading the trained CrowdPulseCountNet checkpoint."""

    DEFAULT_CHECKPOINT = config.MODELS_DIR / "crowdpulse_count_model.pth"
    METRICS_FILE = config.MODELS_DIR / "training_metrics.json"

    def __init__(self, checkpoint_path: Optional[Union[str, Path]] = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self.DEFAULT_CHECKPOINT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CrowdPulseCountNet] = None
        self.backbone_name: str = "convnext_tiny"
        self.img_size: int = 384
        self.metadata: Dict = {}

        if self.checkpoint_path.exists():
            self._load_checkpoint()

    @property
    def is_ready(self) -> bool:
        """Returns True if trained weights are loaded and ready for inference."""
        return self.model is not None

    def _load_checkpoint(self):
        """Loads saved model weights and preprocessing metadata."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.backbone_name = ckpt.get("backbone_name", "convnext_tiny")
        self.img_size = int(ckpt.get("img_size", 384))
        self.metadata = ckpt.get("metrics", {})

        # Instantiate architecture without re-downloading ImageNet weights
        self.model = CrowdPulseCountNet(
            backbone_name=self.backbone_name,
            pretrained=False,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.eval_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray, use_tta: bool = True) -> Dict:
        """Predicts exact crowd headcount from an RGB NumPy array.

        Parameters:
            image_rgb (np.ndarray): Standard optical CCTV RGB frame (H, W, 3)
            use_tta (bool): If True, averages predictions of original + horizontally
                            flipped image (Test-Time Augmentation) for higher accuracy.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Trained CrowdPulseCountNet checkpoint not found at {self.checkpoint_path}. "
                "Run `python training/train_crowd_model.py` first."
            )

        pil_img = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        tensor_orig = self.eval_transform(pil_img).unsqueeze(0).to(self.device)

        if use_tta:
            pil_flip = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            tensor_flip = self.eval_transform(pil_flip).unsqueeze(0).to(self.device)
            batch = torch.cat([tensor_orig, tensor_flip], dim=0)
            preds = self.model.predict_count(batch)
            raw_count = float(preds.mean().item())
        else:
            raw_count = float(self.model.predict_count(tensor_orig).item())

        predicted_int = max(0, int(round(raw_count)))

        return {
            "predicted_count": predicted_int,
            "raw_float_count": round(raw_count, 2),
            "backbone": self.backbone_name,
            "img_size": self.img_size,
            "device": str(self.device).upper(),
            "tta_used": use_tta,
        }

    @classmethod
    def load_training_metrics(cls) -> Optional[Dict]:
        """Loads saved training & evaluation metrics JSON if available."""
        if cls.METRICS_FILE.exists():
            with open(cls.METRICS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

