from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.training.config import ExportConfig, TrainConfig, config_to_dict


@dataclass(slots=True)
class TrainingPaths:
    run_name: str
    output_dir: Path
    config_path: Path
    summary_path: Path
    metrics_path: Path
    evaluations_dir: Path
    best_predictions_path: Path
    best_evaluation_summary_path: Path
    checkpoints_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path


def build_training_paths(
    config: ExportConfig,
    *,
    model_name: str,
    freeze_backbone: bool,
    lr: float,
    seed: int,
    aggregation: str | None = None,
    unfreeze_last_n_blocks: int = 0,
) -> TrainingPaths:
    run_name = config.run_name or build_run_name(
        model_name=model_name,
        freeze_backbone=freeze_backbone,
        lr=lr,
        seed=seed,
        aggregation=aggregation,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    output_dir = config.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    evaluations_dir = output_dir / config.evaluations_dirname
    checkpoints_dir = output_dir / config.checkpoint_dirname
    return TrainingPaths(
        run_name=run_name,
        output_dir=output_dir,
        config_path=output_dir / config.config_filename,
        summary_path=output_dir / config.summary_filename,
        metrics_path=output_dir / config.metrics_filename,
        evaluations_dir=evaluations_dir,
        best_predictions_path=evaluations_dir / config.best_predictions_filename,
        best_evaluation_summary_path=evaluations_dir / "best_val_summary.json",
        checkpoints_dir=checkpoints_dir,
        best_checkpoint_path=checkpoints_dir / config.best_checkpoint_filename,
        last_checkpoint_path=checkpoints_dir / config.last_checkpoint_filename,
    )


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def save_config_snapshot(
    config: TrainConfig,
    output_path: Path,
    *,
    source_config_path: Path | None = None,
) -> None:
    payload = config_to_dict(config)
    payload["resolved_output_dir"] = str(output_path.parent)
    payload["run_name"] = output_path.parent.name
    payload["source_config_path"] = str(source_config_path) if source_config_path is not None else None
    save_json(output_path, payload)


def append_metrics_row(output_path: Path, row: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_prediction_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, float | int | None],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "config": config_to_dict(config),
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, output_path)


def get_epoch_predictions_path(paths: TrainingPaths, split: str, epoch: int) -> Path:
    return paths.evaluations_dir / f"{split}_epoch_{epoch:03d}_predictions.csv"


def get_epoch_evaluation_summary_path(paths: TrainingPaths, split: str, epoch: int) -> Path:
    return paths.evaluations_dir / f"{split}_epoch_{epoch:03d}_summary.json"


def build_run_name(
    *,
    model_name: str,
    freeze_backbone: bool,
    lr: float,
    seed: int,
    aggregation: str | None = None,
    unfreeze_last_n_blocks: int = 0,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    model_signature = model_name if aggregation in (None, "") else f"{model_name}-{aggregation}"
    model_slug = _slugify(model_signature)
    backbone_mode = "freeze" if freeze_backbone else "finetune"
    if unfreeze_last_n_blocks > 0:
        backbone_mode = f"{backbone_mode}-u{unfreeze_last_n_blocks}"
    lr_text = _format_learning_rate(lr)
    return f"{timestamp}_{model_slug}_{backbone_mode}_lr{lr_text}_seed{seed}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return slug.strip("-") or "run"


def _format_learning_rate(value: float) -> str:
    base, exponent = f"{value:.0e}".split("e")
    return f"{base}e{int(exponent)}"
