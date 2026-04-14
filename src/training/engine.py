from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.efficientnet_b0_model_v1 import MultiPoseEfficientNetB0
from src.models.single_efficientnet_b0_model_v1 import SingleImageEfficientNetB0
from src.training.config import TrainConfig


MODEL_INPUT_KEYS = {
    "multi_pose": "images",
    "single_image": "image",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested device 'mps' is not available.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device 'cuda' is not available.")
    return torch.device(device_name)


def build_model(config: TrainConfig) -> nn.Module:
    if config.model_name == "multi_pose":
        return MultiPoseEfficientNetB0(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
    if config.model_name == "single_image":
        return SingleImageEfficientNetB0(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model_name '{config.model_name}'.")


def configure_trainable_parameters(
    model: nn.Module,
    freeze_backbone: bool,
    unfreeze_last_n_blocks: int,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

    if freeze_backbone or unfreeze_last_n_blocks > 0:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    if unfreeze_last_n_blocks > 0:
        encoder_blocks = list(model.encoder.children())
        if unfreeze_last_n_blocks > len(encoder_blocks):
            raise ValueError(
                f"Requested unfreeze_last_n_blocks={unfreeze_last_n_blocks}, "
                f"but encoder only has {len(encoder_blocks)} blocks."
            )

        for block in encoder_blocks[-unfreeze_last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> dict[str, int]:
    trainable_tensors = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_scalars = sum(parameter.numel() for parameter in trainable_tensors)
    return {
        "trainable_parameter_tensors": len(trainable_tensors),
        "trainable_parameter_scalars": trainable_scalars,
    }


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    parameter_groups: list[dict[str, Any]] = []

    backbone_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    head_params = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]

    if backbone_params:
        parameter_groups.append(
            {
                "name": "backbone",
                "params": backbone_params,
                "lr": config.backbone_lr,
            }
        )
    if head_params:
        parameter_groups.append(
            {
                "name": "head",
                "params": head_params,
                "lr": config.head_lr,
            }
        )

    if not parameter_groups:
        raise ValueError("No trainable parameters were found.")

    return torch.optim.Adam(parameter_groups, weight_decay=config.weight_decay)


def build_loss_function(
    labels: list[int],
    num_classes: int,
    use_class_weights: bool,
    device: torch.device,
) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()

    label_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=num_classes).float()

    weights = torch.ones(num_classes, dtype=torch.float32)
    present_mask = class_counts > 0
    weights[present_mask] = 1.0 / torch.sqrt(class_counts[present_mask])

    weights = weights / weights.mean()

    return nn.CrossEntropyLoss(weight=weights.to(device))


def unpack_batch(batch: dict[str, Any], model_name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch[MODEL_INPUT_KEYS[model_name]].to(device)
    labels = batch["label"].to(device).long()
    return inputs, labels


def compute_mae_sum(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.abs(predictions - labels).float().sum().item()


def run_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    model_name: str,
    track_mae: bool,
    limit_batches: int | None,
    epoch_index: int,
    total_epochs: int,
    phase_name: str,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_mae_sum = 0.0

    progress = tqdm(dataloader, desc=f"{phase_name} {epoch_index:02d}/{total_epochs}", leave=False)

    for batch_index, batch in enumerate(progress, start=1):
        if limit_batches is not None and batch_index > limit_batches:
            break

        inputs, labels = unpack_batch(batch, model_name=model_name, device=device)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        if track_mae:
            total_mae_sum += compute_mae_sum(predictions, labels)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        postfix = {
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.4f}",
        }
        if track_mae:
            postfix["mae"] = f"{(total_mae_sum / total_samples):.4f}"
        progress.set_postfix(postfix)

    if total_samples == 0:
        raise ValueError(f"{phase_name} epoch processed zero samples.")

    metrics = {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }
    if track_mae:
        metrics["mae"] = total_mae_sum / total_samples
    return metrics


def get_optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    learning_rates: dict[str, float] = {}
    for index, parameter_group in enumerate(optimizer.param_groups):
        name = str(parameter_group.get("name", f"group_{index}"))
        learning_rates[f"{name}_lr"] = float(parameter_group["lr"])
    return learning_rates


def is_improved(current_value: float, best_value: float | None, mode: str) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return current_value < best_value
    return current_value > best_value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def fit(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
    config: TrainConfig,
    run_dir: Path,
) -> dict[str, Any]:
    history: list[dict[str, float | int]] = []
    best_metric_value: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint_path = run_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            model_name=config.model_name,
            track_mae=config.track_mae,
            limit_batches=config.limit_train_batches,
            epoch_index=epoch,
            total_epochs=config.epochs,
            phase_name="Train",
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                model_name=config.model_name,
                track_mae=config.track_mae,
                limit_batches=config.limit_val_batches,
                epoch_index=epoch,
                total_epochs=config.epochs,
                phase_name="Val",
            )

        scheduler.step(val_metrics["loss"])
        learning_rates = get_optimizer_lrs(optimizer)

        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            **learning_rates,
        }

        if config.track_mae:
            epoch_record["train_mae"] = train_metrics["mae"]
            epoch_record["val_mae"] = val_metrics["mae"]

        history.append(epoch_record)

        monitor_value = float(epoch_record[config.monitor_metric])
        if is_improved(monitor_value, best_metric_value, config.monitor_mode):
            best_metric_value = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        history_payload = {
            "config": config.to_dict(),
            "epochs": history,
        }
        save_json(run_dir / "history.json", history_payload)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        metrics_message = (
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.4f}"
        )
        if config.track_mae:
            metrics_message += (
                f" | train_mae={train_metrics['mae']:.4f} | val_mae={val_metrics['mae']:.4f}"
            )
        for name, value in learning_rates.items():
            metrics_message += f" | {name}={value:.6f}"
        print(metrics_message, flush=True)

        if epoch >= config.min_epochs and config.early_stopping_patience >= 0:
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch:02d} after "
                    f"{epochs_without_improvement} epochs without improvement.",
                    flush=True,
                )
                break

    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_monitor_metric": config.monitor_metric,
        "best_monitor_value": best_metric_value,
        "epochs_ran": len(history),
        "last_epoch": history[-1] if history else None,
        "best_epoch_record": next((record for record in history if record["epoch"] == best_epoch), None),
    }
    save_json(run_dir / "summary.json", summary)
    return summary
