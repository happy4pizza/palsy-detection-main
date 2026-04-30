from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BasePalsyModel
from src.models.efficientnet_b0_model_v1 import MultiImageEfficientNetB0
from src.models.single_efficientnet_b0_model_v1 import SingleImageEfficientNetB0
from src.training.config import ModelConfig, OptimizerConfig, SchedulerConfig


MODEL_REGISTRY = {
    "multi_image_efficientnet_b0": MultiImageEfficientNetB0,
    "single_image_efficientnet_b0": SingleImageEfficientNetB0,
}


def build_model(config: ModelConfig) -> nn.Module:
    try:
        model_class = MODEL_REGISTRY[config.name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model name '{config.name}'. "
            f"Currently supported: {', '.join(sorted(MODEL_REGISTRY))}"
        ) from exc

    model = model_class(config)
    apply_freezing_policy(model, config)
    return model


def apply_freezing_policy(model: nn.Module, config: ModelConfig) -> None:
    if not isinstance(model, BasePalsyModel):
        return

    if config.freeze_backbone:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    if config.unfreeze_last_n_blocks > 0:
        model.unfreeze_last_n_blocks(config.unfreeze_last_n_blocks)


def make_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    parameter_groups = _build_parameter_groups(model, config)
    return torch.optim.AdamW(
        parameter_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if not config.enabled:
        return None
    if config.name != "reduce_on_plateau":
        raise ValueError(
            f"Unsupported scheduler name '{config.name}'. "
            "Currently supported: reduce_on_plateau"
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.mode,
        factor=config.factor,
        patience=config.patience,
        min_lr=config.min_lr,
    )


def summarize_model(model: nn.Module) -> tuple[int, int]:
    num_total_params = sum(parameter.numel() for parameter in model.parameters())
    num_trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return num_total_params, num_trainable_params


def _build_parameter_groups(
    model: nn.Module,
    config: OptimizerConfig,
) -> list[dict[str, object]]:
    if isinstance(model, BasePalsyModel):
        raw_groups = model.get_parameter_groups()
    else:
        raw_groups = {
            "default": [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
        }

    parameter_groups: list[dict[str, object]] = []
    for group_name, params in raw_groups.items():
        trainable_params = [parameter for parameter in params if parameter.requires_grad]
        if not trainable_params:
            continue

        group_lr = (
            config.backbone_lr
            if group_name == "backbone" and config.backbone_lr is not None
            else config.lr
        )
        parameter_groups.append(
            {
                "name": group_name,
                "params": trainable_params,
                "lr": group_lr,
            }
        )

    if not parameter_groups:
        raise ValueError("No trainable parameters were found for the optimizer.")

    return parameter_groups
