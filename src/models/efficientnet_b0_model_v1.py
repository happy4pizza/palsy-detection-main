from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class MultiPoseEfficientNetB0(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, dropout=0.3):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.encoder = backbone.features
        self.feature_dim = backbone.classifier[1].in_features  # 1280
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, P, C, H, W] -> [B * P, C, H, W]
        """
        B, P, C, H, W = x.shape

        x = x.view(B * P, C, H, W)
        features = self.encoder(x)               # [B*P, 1280, h, w]
        features = self.pool(features)           # [B*P, 1280, 1, 1]
        features = features.view(B, P, -1)       # [B, P, 1280]

        patient_features = features.mean(dim=1)  # [B, 1280]
        logits = self.classifier(patient_features)

        return logits
