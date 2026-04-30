from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BaseAggregator(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, per_image_features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError


class MeanAggregator(BaseAggregator):
    def __init__(self, feature_dim: int) -> None:
        super().__init__(output_dim=feature_dim)

    def forward(self, per_image_features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        return per_image_features.mean(dim=1), {}


class ConcatAggregator(BaseAggregator):
    def __init__(self, feature_dim: int, num_poses: int) -> None:
        if num_poses <= 0:
            raise ValueError(f"num_poses must be > 0 for concat aggregation. Got {num_poses}.")
        self.num_poses = num_poses
        super().__init__(output_dim=feature_dim * num_poses)

    def forward(self, per_image_features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if per_image_features.size(1) != self.num_poses:
            raise ValueError(
                f"ConcatAggregator expected {self.num_poses} poses but got "
                f"{per_image_features.size(1)}."
            )
        return per_image_features.flatten(start_dim=1), {}


class AttentionAggregator(BaseAggregator):
    def __init__(self, feature_dim: int) -> None:
        super().__init__(output_dim=feature_dim)
        self.scorer = nn.Linear(feature_dim, 1)

    def forward(self, per_image_features: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        scores = self.scorer(per_image_features)
        attention_weights = torch.softmax(scores, dim=1)
        patient_features = (attention_weights * per_image_features).sum(dim=1)
        return patient_features, {"attention_weights": attention_weights.squeeze(-1)}


AGGREGATOR_REGISTRY = {
    "mean": MeanAggregator,
    "concat": ConcatAggregator,
    "attention": AttentionAggregator,
}


def build_aggregator(
    aggregation: str,
    *,
    feature_dim: int,
    num_poses: int | None,
) -> BaseAggregator:
    if aggregation == "mean":
        return MeanAggregator(feature_dim)
    if aggregation == "concat":
        if num_poses is None:
            raise ValueError("num_poses must be set when aggregation='concat'.")
        return ConcatAggregator(feature_dim, num_poses=num_poses)
    if aggregation == "attention":
        return AttentionAggregator(feature_dim)
    raise ValueError(
        f"Unknown aggregation '{aggregation}'. "
        f"Expected one of {sorted(AGGREGATOR_REGISTRY)}."
    )
