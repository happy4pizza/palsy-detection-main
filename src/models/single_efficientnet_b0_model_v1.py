from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.models.base import BasePalsyModel
from src.training.config import ModelConfig


class SingleImageEfficientNetB0(BasePalsyModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(self.feature_dim, config.num_classes),
        )

    def freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.backbone_frozen = True
        self._frozen_backbone_modules = [self.encoder]
        self._trainable_backbone_modules = []
        self._apply_backbone_train_policy(self.training)

    def unfreeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
        self._frozen_backbone_modules = []
        self._trainable_backbone_modules = [self.encoder]
        self._apply_backbone_train_policy(self.training)

    def unfreeze_last_n_blocks(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"n must be >= 0. Got {n}.")
        if n == 0:
            return

        blocks = list(self.encoder.children())
        if n > len(blocks):
            raise ValueError(
                f"Requested unfreeze_last_n_blocks={n}, but encoder only has {len(blocks)} blocks."
            )

        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False

        trainable_blocks = blocks[-n:]
        frozen_blocks = blocks[:-n]

        for block in trainable_blocks:
            for param in block.parameters():
                param.requires_grad = True
        self.backbone_frozen = False
        self._frozen_backbone_modules = frozen_blocks
        self._trainable_backbone_modules = trainable_blocks
        self._apply_backbone_train_policy(self.training)

    def get_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "head": [
                parameter
                for parameter in self.classifier.parameters()
                if parameter.requires_grad
            ],
            "backbone": [
                parameter
                for parameter in self.encoder.parameters()
                if parameter.requires_grad
            ],
        }

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected x to have shape [B, C, H, W], but got {tuple(x.shape)}")

        features = self.encoder(x)
        features = self.pool(features)
        features = features.reshape(features.size(0), self.feature_dim)

        logits = self.classifier(features)
        return {
            "logits": logits,
            "features": features,
        }
