from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.models.aggregators import build_aggregator
from src.models.base import BasePalsyModel
from src.training.config import ModelConfig


class MultiImageEfficientNetB0(BasePalsyModel):
    """
    Multi-image / multi-pose EfficientNet-B0 model for patient-level HB grading.

    Input:
        x: [B, P, C, H, W]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.encoder = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = backbone.classifier[1].in_features
        self.aggregator = build_aggregator(
            config.aggregation,
            feature_dim=self.feature_dim,
            num_poses=config.num_poses,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(self.aggregator.output_dim, config.num_classes),
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
            ] + [
                parameter
                for parameter in self.aggregator.parameters()
                if parameter.requires_grad
            ],
            "backbone": [
                parameter
                for parameter in self.encoder.parameters()
                if parameter.requires_grad
            ],
        }

    def extract_per_image_features(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected x to have shape [B, P, C, H, W], but got {tuple(x.shape)}"
            )

        batch_size, num_poses, channels, height, width = x.shape
        x = x.reshape(batch_size * num_poses, channels, height, width)

        features = self.encoder(x)
        features = self.pool(features)
        features = features.flatten(1)
        return features.reshape(batch_size, num_poses, self.feature_dim)

    def aggregate_features(self, per_image_features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return self.aggregator(per_image_features)

    def classify_features(self, patient_features: Tensor) -> Tensor:
        return self.classifier(patient_features)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        per_image_features = self.extract_per_image_features(x)
        patient_features, aggregation_outputs = self.aggregate_features(per_image_features)
        logits = self.classify_features(patient_features)

        return {
            "logits": logits,
            "patient_features": patient_features,
            "per_image_features": per_image_features,
            **aggregation_outputs,
        }
