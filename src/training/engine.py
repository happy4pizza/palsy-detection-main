from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.training.artifacts import (
    append_metrics_row,
    build_training_paths,
    get_epoch_evaluation_summary_path,
    get_epoch_predictions_path,
    save_checkpoint,
    save_config_snapshot,
    save_json,
    save_prediction_rows,
)
from src.training.config import LossConfig, TrainConfig
from src.training.metrics import EpochMetrics, compute_evaluation_metrics
from src.training.model_factory import build_model, make_optimizer, make_scheduler, summarize_model
from src.training.train_loader import build_dataloaders


METRIC_MODES = {
    "val_loss": "min",
    "val_mae_hb": "min",
    "val_acc": "max",
    "val_macro_f1": "max",
    "val_balanced_acc": "max",
    "val_within_1_grade_acc": "max",
}


def get_device(preferred_device: str | None = None) -> torch.device:
    if preferred_device is not None:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    num_epochs: int,
    log_every: int,
) -> EpochMetrics:
    if log_every <= 0:
        raise ValueError(f"log_every must be >= 1. Got {log_every}.")

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    use_non_blocking = device.type == "cuda"

    for batch_idx, batch in enumerate(train_loader, start=1):
        inputs = batch["inputs"].to(device, non_blocking=use_non_blocking)
        labels = batch["label"].to(device, non_blocking=use_non_blocking)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        logits, _ = _extract_logits(outputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += batch_size

        avg_loss = running_loss / running_total
        avg_acc = running_correct / running_total

        if batch_idx % log_every == 0 or batch_idx == len(train_loader):
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Batch {batch_idx:03d}/{len(train_loader):03d} | "
                f"loss={loss.item():.4f} | "
                f"avg_loss={avg_loss:.4f} | "
                f"avg_acc={avg_acc:.3f}"
            )

    return EpochMetrics(
        loss=running_loss / running_total,
        accuracy=running_correct / running_total,
        sample_count=running_total,
    )


def evaluate(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    epoch: int,
    split: str,
) -> tuple[EpochMetrics, list[dict[str, object]]]:
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    raw_rows: list[dict[str, object]] = []
    all_labels: list[int] = []
    all_preds: list[int] = []
    num_classes: int | None = None
    use_non_blocking = device.type == "cuda"

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["inputs"].to(device, non_blocking=use_non_blocking)
            labels = batch["label"].to(device, non_blocking=use_non_blocking)
            metadata_rows = batch["metadata"]

            outputs = model(inputs)
            logits, output_dict = _extract_logits(outputs)
            probabilities = torch.softmax(logits, dim=1)
            num_classes = logits.size(1)

            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == labels).sum().item()
            running_total += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            attention_weights = output_dict.get("attention_weights")
            attention_rows = (
                attention_weights.detach().cpu().tolist()
                if attention_weights is not None
                else [None] * batch_size
            )

            for metadata, label, pred, probs, attention_row in zip(
                metadata_rows,
                labels.cpu().tolist(),
                preds.cpu().tolist(),
                probabilities.cpu().tolist(),
                attention_rows,
            ):
                row = {
                    "patient_id": str(metadata.get("patient_id", "")),
                    "pose_indices": json.dumps(metadata.get("pose_indices", [])),
                    "filepaths": json.dumps(metadata.get("filepaths", [])),
                    "true_class_index": label,
                    "pred_class_index": pred,
                    "true_hb": label + 1,
                    "pred_hb": pred + 1,
                    "correct": int(pred == label),
                    "abs_error": abs(pred - label),
                    "within_1_grade": int(abs(pred - label) <= 1),
                    "confidence": max(probs),
                }
                if attention_row is not None:
                    row["attention_weights"] = json.dumps(attention_row)
                for class_idx, probability in enumerate(probs, start=1):
                    row[f"prob_hb_{class_idx}"] = probability
                raw_rows.append(row)

    metrics = compute_evaluation_metrics(
        loss=running_loss / running_total,
        labels=all_labels,
        preds=all_preds,
        num_classes=num_classes or 0,
    )

    export_rows = [
        {
            "epoch": epoch,
            "split": split,
            **row,
            f"{split}_loss": metrics.loss,
            f"{split}_acc": metrics.accuracy,
            f"{split}_mae_hb": metrics.mae_hb,
            f"{split}_macro_f1": metrics.macro_f1,
            f"{split}_balanced_acc": metrics.balanced_accuracy,
            f"{split}_within_1_grade_acc": metrics.within_one_grade_accuracy,
        }
        for row in raw_rows
    ]

    return metrics, export_rows


def run_training(config: TrainConfig, *, config_path: Path | None = None) -> None:
    if config.runtime.num_epochs <= 0:
        raise ValueError(f"num_epochs must be > 0. Got {config.runtime.num_epochs}.")
    if config.runtime.early_stopping_patience is not None and config.runtime.early_stopping_patience <= 0:
        raise ValueError(
            "early_stopping_patience must be > 0 when provided. "
            f"Got {config.runtime.early_stopping_patience}."
        )
    if config.runtime.early_stopping_min_delta < 0:
        raise ValueError(
            "early_stopping_min_delta must be >= 0. "
            f"Got {config.runtime.early_stopping_min_delta}."
        )
    if config.model.unfreeze_last_n_blocks < 0:
        raise ValueError(
            "unfreeze_last_n_blocks must be >= 0. "
            f"Got {config.model.unfreeze_last_n_blocks}."
        )
    _get_metric_mode(config.runtime.checkpoint_metric)
    if config.scheduler.enabled:
        expected_scheduler_mode = _get_metric_mode(config.scheduler.monitor)
        if config.scheduler.mode != expected_scheduler_mode:
            raise ValueError(
                f"scheduler.mode must be '{expected_scheduler_mode}' when monitor="
                f"'{config.scheduler.monitor}'. Got '{config.scheduler.mode}'."
            )

    device = get_device(config.runtime.device)
    set_seed(config.runtime.seed)

    paths = build_training_paths(
        config.export,
        model_name=config.model.name,
        freeze_backbone=config.model.freeze_backbone,
        lr=config.optimizer.lr,
        seed=config.runtime.seed,
        aggregation=config.model.aggregation,
        unfreeze_last_n_blocks=config.model.unfreeze_last_n_blocks,
    )
    save_config_snapshot(config, paths.config_path, source_config_path=config_path)
    if config_path is not None:
        config_path = Path(config_path)
        print(f"Config file: {config_path}")

    data_config = replace(
        config.data,
        pin_memory=(
            config.data.pin_memory
            if config.data.pin_memory is not None
            else device.type == "cuda"
        ),
        persistent_workers=(
            config.data.persistent_workers
            if config.data.persistent_workers is not None
            else config.data.num_workers > 0
        ),
    )
    train_loader, val_loader = build_dataloaders(data_config, model_config=config.model)

    model = build_model(config.model).to(device)
    criterion, class_weights = build_criterion(
        config.loss,
        labels=train_loader.dataset.get_labels(),
        num_classes=config.model.num_classes,
        device=device,
    )
    optimizer = make_optimizer(model, config.optimizer)
    scheduler = make_scheduler(optimizer, config.scheduler)

    num_total_params, num_trainable_params = summarize_model(model)
    initial_lrs = get_learning_rates(optimizer)
    print(f"Using device: {device}")
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")
    print(f"Total parameters: {num_total_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")
    print(f"Model aggregation: {config.model.aggregation}")
    print(f"Head LR: {initial_lrs.get('head', initial_lrs.get('default', config.optimizer.lr)):.2e}")
    if "backbone" in initial_lrs:
        print(f"Backbone LR: {initial_lrs['backbone']:.2e}")
    if class_weights is not None:
        print(f"Class weights: {[round(weight, 6) for weight in class_weights.cpu().tolist()]}")
    if scheduler is not None:
        print(
            "Scheduler: "
            f"{config.scheduler.name} "
            f"(monitor={config.scheduler.monitor}, mode={config.scheduler.mode})"
        )
    if config.runtime.early_stopping_patience is not None:
        print(
            "Early stopping: "
            f"monitor={config.runtime.checkpoint_metric} "
            f"(patience={config.runtime.early_stopping_patience}, "
            f"min_delta={config.runtime.early_stopping_min_delta})"
        )
    print(f"Run directory: {paths.output_dir}")

    best_score = _initial_best_score(config.runtime.checkpoint_metric)
    best_epoch = 0
    best_metrics: dict[str, float | int | None] | None = None
    last_metrics_row: dict[str, float | int | None] | None = None
    early_stopping_best_score = _initial_best_score(config.runtime.checkpoint_metric)
    early_stopping_wait = 0
    stopped_early = False

    for epoch in range(1, config.runtime.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.runtime.num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=config.runtime.num_epochs,
            log_every=config.runtime.log_every,
        )
        val_metrics, val_rows = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            split="val",
        )
        if scheduler is not None:
            scheduler.step(_get_validation_metric(config.scheduler.monitor, val_metrics))

        current_lrs = get_learning_rates(optimizer)

        metrics_row = {
            "epoch": epoch,
            **_format_learning_rate_metrics(current_lrs),
            **train_metrics.to_dict("train_"),
            **val_metrics.to_dict("val_"),
        }
        append_metrics_row(paths.metrics_path, metrics_row)
        last_metrics_row = metrics_row

        save_json(
            get_epoch_evaluation_summary_path(paths, split="val", epoch=epoch),
            {
                "epoch": epoch,
                "split": "val",
                **val_metrics.to_summary_dict(),
            },
        )
        if config.export.export_predictions_every_epoch:
            save_prediction_rows(
                get_epoch_predictions_path(paths, split="val", epoch=epoch),
                val_rows,
            )

        checkpoint_metrics = {
            **_format_learning_rate_metrics(current_lrs),
            **train_metrics.to_dict("train_"),
            **val_metrics.to_dict("val_"),
        }
        save_checkpoint(
            output_path=paths.last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch,
            metrics=checkpoint_metrics,
        )

        current_score = _get_validation_metric(config.runtime.checkpoint_metric, val_metrics)
        if _is_strictly_better(config.runtime.checkpoint_metric, current_score, best_score):
            best_score = current_score
            best_epoch = epoch
            best_metrics = checkpoint_metrics
            save_checkpoint(
                output_path=paths.best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                metrics=checkpoint_metrics,
            )
            save_prediction_rows(paths.best_predictions_path, val_rows)
            save_json(
                paths.best_evaluation_summary_path,
                {
                    "epoch": epoch,
                    "split": "val",
                    **val_metrics.to_summary_dict(),
                },
            )
            print(f"Saved new best checkpoint: {paths.best_checkpoint_path}")

        mae_hb_text = (
            f"{val_metrics.mae_hb:.3f}"
            if val_metrics.mae_hb is not None
            else "n/a"
        )
        within_one_text = (
            f"{val_metrics.within_one_grade_accuracy:.3f}"
            if val_metrics.within_one_grade_accuracy is not None
            else "n/a"
        )
        macro_f1_text = (
            f"{val_metrics.macro_f1:.3f}"
            if val_metrics.macro_f1 is not None
            else "n/a"
        )
        balanced_acc_text = (
            f"{val_metrics.balanced_accuracy:.3f}"
            if val_metrics.balanced_accuracy is not None
            else "n/a"
        )
        head_lr_text = _format_learning_rate_text(current_lrs.get("head", current_lrs.get("default")))
        backbone_lr_text = _format_learning_rate_text(current_lrs.get("backbone"))
        backbone_lr_part = f" | lr_backbone={backbone_lr_text}" if backbone_lr_text != "n/a" else ""
        print(
            f"Epoch {epoch:02d} summary | "
            f"lr_head={head_lr_text} | "
            f"train_loss={train_metrics.loss:.4f} | "
            f"train_acc={train_metrics.accuracy:.3f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_acc={val_metrics.accuracy:.3f} | "
            f"val_macro_f1={macro_f1_text} | "
            f"val_balanced_acc={balanced_acc_text} | "
            f"val_mae_hb={mae_hb_text} | "
            f"val_within_1_grade_acc={within_one_text}"
            f"{backbone_lr_part}"
        )

        if config.runtime.early_stopping_patience is not None:
            if _is_improvement(
                config.runtime.checkpoint_metric,
                current_score,
                early_stopping_best_score,
                config.runtime.early_stopping_min_delta,
            ):
                early_stopping_best_score = current_score
                early_stopping_wait = 0
            else:
                early_stopping_wait += 1
                print(
                    "Early stopping wait: "
                    f"{early_stopping_wait}/{config.runtime.early_stopping_patience}"
                )
                if early_stopping_wait >= config.runtime.early_stopping_patience:
                    stopped_early = True
                    print(
                        "Early stopping triggered "
                        f"on epoch {epoch} using {config.runtime.checkpoint_metric}."
                    )
                    break

    save_json(
        paths.summary_path,
        {
            "run_name": paths.run_name,
            "output_dir": str(paths.output_dir),
            "source_config_path": str(config_path) if config_path is not None else None,
            "best_epoch": best_epoch,
            "best_metric_name": config.runtime.checkpoint_metric,
            "best_metric_value": best_score,
            "best_metrics": best_metrics,
            "last_epoch_metrics": last_metrics_row,
            "stopped_early": stopped_early,
            "early_stopping_patience": config.runtime.early_stopping_patience,
            "early_stopping_min_delta": config.runtime.early_stopping_min_delta,
        },
    )
    print(f"Training complete. Best epoch: {best_epoch}")


def build_criterion(
    config: LossConfig,
    *,
    labels: list[int],
    num_classes: int,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor | None]:
    if config.name != "cross_entropy":
        raise ValueError(
            f"Unsupported loss '{config.name}'. Currently supported: cross_entropy"
        )

    class_weights = _resolve_class_weights(
        config,
        labels=labels,
        num_classes=num_classes,
        device=device,
    )
    return nn.CrossEntropyLoss(weight=class_weights), class_weights


def _resolve_class_weights(
    config: LossConfig,
    *,
    labels: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    if config.class_weights is not None and config.use_balanced_class_weights:
        raise ValueError("Specify either class_weights or use_balanced_class_weights, not both.")

    if config.class_weights is not None:
        if len(config.class_weights) != num_classes:
            raise ValueError(
                f"class_weights must have length {num_classes}. "
                f"Got {len(config.class_weights)}."
            )
        if any(weight <= 0 for weight in config.class_weights):
            raise ValueError("class_weights must all be > 0.")
        return torch.tensor(config.class_weights, dtype=torch.float32, device=device)

    if not config.use_balanced_class_weights:
        return None

    counts = [0 for _ in range(num_classes)]
    for label in labels:
        if not 0 <= label < num_classes:
            raise ValueError(f"Invalid training label {label}. Expected 0 <= label < {num_classes}.")
        counts[label] += 1

    if any(count == 0 for count in counts):
        raise ValueError(
            "Cannot compute balanced class weights because at least one class has zero samples: "
            f"{counts}"
        )

    total_samples = sum(counts)
    weights = [
        total_samples / (num_classes * class_count)
        for class_count in counts
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _extract_logits(outputs: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if torch.is_tensor(outputs):
        return outputs, {"logits": outputs}
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model output dict must contain a 'logits' key.")
        if not torch.is_tensor(logits):
            raise TypeError("Model output 'logits' must be a torch.Tensor.")
        return logits, outputs
    raise TypeError(
        "Model outputs must be either a torch.Tensor or a dict containing 'logits'."
    )


def get_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        group.get("name", f"group_{idx}"): group["lr"]
        for idx, group in enumerate(optimizer.param_groups)
    }


def _format_learning_rate_metrics(learning_rates: dict[str, float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for group_name, lr in learning_rates.items():
        metrics[f"lr_{group_name}"] = lr
    return metrics


def _format_learning_rate_text(lr: float | None) -> str:
    return f"{lr:.2e}" if lr is not None else "n/a"


def _get_validation_metric(metric_name: str, val_metrics: EpochMetrics) -> float:
    if metric_name == "val_loss":
        return val_metrics.loss
    if metric_name == "val_mae_hb":
        if val_metrics.mae_hb is None:
            raise ValueError("val_mae_hb is not available for metric selection.")
        return val_metrics.mae_hb
    if metric_name == "val_acc":
        return val_metrics.accuracy
    if metric_name == "val_macro_f1":
        if val_metrics.macro_f1 is None:
            raise ValueError("val_macro_f1 is not available for metric selection.")
        return val_metrics.macro_f1
    if metric_name == "val_balanced_acc":
        if val_metrics.balanced_accuracy is None:
            raise ValueError("val_balanced_acc is not available for metric selection.")
        return val_metrics.balanced_accuracy
    if metric_name == "val_within_1_grade_acc":
        if val_metrics.within_one_grade_accuracy is None:
            raise ValueError("val_within_1_grade_acc is not available for metric selection.")
        return val_metrics.within_one_grade_accuracy
    raise ValueError(f"Unsupported metric: {metric_name}")


def _initial_best_score(metric_name: str) -> float:
    mode = _get_metric_mode(metric_name)
    return float("inf") if mode == "min" else float("-inf")


def _get_metric_mode(metric_name: str) -> str:
    try:
        return METRIC_MODES[metric_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported metric: {metric_name}") from exc


def _is_strictly_better(metric_name: str, current_score: float, best_score: float) -> bool:
    mode = _get_metric_mode(metric_name)
    if mode == "min":
        return current_score < best_score
    return current_score > best_score


def _is_improvement(
    metric_name: str,
    current_score: float,
    best_score: float,
    min_delta: float,
) -> bool:
    mode = _get_metric_mode(metric_name)
    if mode == "min":
        return current_score < (best_score - min_delta)
    return current_score > (best_score + min_delta)
