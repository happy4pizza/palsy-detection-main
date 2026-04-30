from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from src.training.config import ModelConfig


class BasePalsyModel(nn.Module, ABC):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone_frozen = False
        self._frozen_backbone_modules: list[nn.Module] = []
        self._trainable_backbone_modules: list[nn.Module] = []

    @abstractmethod
    def forward(self, x: Tensor):
        raise NotImplementedError

    @abstractmethod
    def freeze_backbone(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def unfreeze_backbone(self) -> None:
        raise NotImplementedError

    def unfreeze_last_n_blocks(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"n must be >= 0. Got {n}.")
        if n == 0:
            return
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement unfreeze_last_n_blocks()."
        )

    def get_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "default": [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ]
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_backbone_train_policy(mode)
        return self

    def _apply_backbone_train_policy(self, mode: bool) -> None:
        if not mode:
            return

        for module in self._trainable_backbone_modules:
            module.train()
        for module in self._frozen_backbone_modules:
            module.eval()
