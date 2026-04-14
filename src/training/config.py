from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
RUNS_DIR = DATA_DIR / "runs"

VALID_MODELS = {"multi_pose", "single_image"}
VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
VALID_MONITOR_METRICS = {"train_loss", "train_acc", "train_mae", "val_loss", "val_acc", "val_mae"}
VALID_MONITOR_MODES = {"min", "max"}
VALID_AUGMENTATION = {"none", "light"}
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    output_root: Path
    manifest_path: Path
    model_name: str
    num_classes: int
    train_split: str
    val_split: str
    batch_size: int
    epochs: int
    early_stopping_patience: int
    min_epochs: int
    device: str
    seed: int
    num_workers: int
    dropout: float
    pretrained: bool
    freeze_backbone: bool
    unfreeze_last_n_blocks: int
    head_lr: float
    backbone_lr: float
    weight_decay: float
    use_class_weights: bool
    track_mae: bool
    train_augmentation: str
    pose_indices: tuple[int, ...] | None
    overfit_patients: int | None
    max_train_patients: int | None
    max_val_patients: int | None
    limit_train_batches: int | None
    limit_val_batches: int | None
    monitor_metric: str
    monitor_mode: str
    scheduler_patience: int
    scheduler_factor: float

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    def to_dict(self) -> dict[str, object]:
        config_dict = asdict(self)
        config_dict["output_root"] = str(self.output_root)
        config_dict["manifest_path"] = str(self.manifest_path)
        config_dict["run_dir"] = str(self.run_dir)
        return config_dict


def default_manifest_for_model(model_name: str) -> Path:
    if model_name == "multi_pose":
        return MANIFEST_DIR / "manifest_face224.parquet"
    if model_name == "single_image":
        return MANIFEST_DIR / "manifest_single_image_face224.parquet"
    raise ValueError(f"Unsupported model_name '{model_name}'. Expected one of {sorted(VALID_MODELS)}.")


def parse_pose_indices(raw_value: str | None) -> tuple[int, ...] | None:
    if raw_value is None:
        return None

    stripped = raw_value.strip()
    if not stripped:
        return None

    pose_indices: list[int] = []
    for token in stripped.split(","):
        token = token.strip()
        if not token:
            continue
        pose_indices.append(int(token))

    if not pose_indices:
        return None

    return tuple(sorted(set(pose_indices)))


def _timestamped_run_name(model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{timestamp}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fast-iteration facial palsy baseline.")

    parser.add_argument("--run-name")
    parser.add_argument("--output-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--model", dest="model_name", default="multi_pose", choices=sorted(VALID_MODELS))
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=sorted(VALID_DEVICES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-augmentation", default="none", choices=sorted(VALID_AUGMENTATION))
    parser.add_argument("--pose-indices", type=parse_pose_indices)
    parser.add_argument("--overfit-patients", type=int)
    parser.add_argument("--max-train-patients", type=int)
    parser.add_argument("--max-val-patients", type=int)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--monitor-metric", default="val_loss", choices=sorted(VALID_MONITOR_METRICS))
    parser.add_argument("--monitor-mode", default="min", choices=sorted(VALID_MONITOR_MODES))
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)

    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)

    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)

    parser.add_argument("--use-class-weights", dest="use_class_weights", action="store_true")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.set_defaults(use_class_weights=True)

    parser.add_argument("--track-mae", dest="track_mae", action="store_true")
    parser.add_argument("--no-track-mae", dest="track_mae", action="store_false")
    parser.set_defaults(track_mae=True)

    return parser


def _validate_config(config: TrainConfig) -> None:
    if config.num_classes <= 0:
        raise ValueError(f"num_classes must be > 0. Got {config.num_classes}.")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0. Got {config.batch_size}.")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be > 0. Got {config.epochs}.")
    if config.min_epochs <= 0:
        raise ValueError(f"min_epochs must be > 0. Got {config.min_epochs}.")
    if config.min_epochs > config.epochs:
        raise ValueError(f"min_epochs ({config.min_epochs}) cannot exceed epochs ({config.epochs}).")
    if config.early_stopping_patience < 0:
        raise ValueError(
            f"early_stopping_patience must be >= 0. Got {config.early_stopping_patience}."
        )
    if config.scheduler_patience < 0:
        raise ValueError(f"scheduler_patience must be >= 0. Got {config.scheduler_patience}.")
    if not 0.0 < config.scheduler_factor < 1.0:
        raise ValueError(f"scheduler_factor must be in (0, 1). Got {config.scheduler_factor}.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1). Got {config.dropout}.")
    if config.head_lr <= 0:
        raise ValueError(f"head_lr must be > 0. Got {config.head_lr}.")
    if config.backbone_lr <= 0:
        raise ValueError(f"backbone_lr must be > 0. Got {config.backbone_lr}.")
    if config.weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0. Got {config.weight_decay}.")
    if config.train_split not in VALID_SPLITS:
        raise ValueError(f"train_split must be one of {sorted(VALID_SPLITS)}. Got {config.train_split}.")
    if config.val_split not in VALID_SPLITS:
        raise ValueError(f"val_split must be one of {sorted(VALID_SPLITS)}. Got {config.val_split}.")
    if config.freeze_backbone and config.unfreeze_last_n_blocks > 0:
        raise ValueError("freeze_backbone=true cannot be combined with unfreeze_last_n_blocks > 0.")
    if config.unfreeze_last_n_blocks < 0:
        raise ValueError(
            f"unfreeze_last_n_blocks must be >= 0. Got {config.unfreeze_last_n_blocks}."
        )
    if config.overfit_patients is not None and config.overfit_patients <= 0:
        raise ValueError(f"overfit_patients must be > 0. Got {config.overfit_patients}.")
    if config.max_train_patients is not None and config.max_train_patients <= 0:
        raise ValueError(
            f"max_train_patients must be > 0. Got {config.max_train_patients}."
        )
    if config.max_val_patients is not None and config.max_val_patients <= 0:
        raise ValueError(f"max_val_patients must be > 0. Got {config.max_val_patients}.")
    if config.limit_train_batches is not None and config.limit_train_batches <= 0:
        raise ValueError(
            f"limit_train_batches must be > 0. Got {config.limit_train_batches}."
        )
    if config.limit_val_batches is not None and config.limit_val_batches <= 0:
        raise ValueError(f"limit_val_batches must be > 0. Got {config.limit_val_batches}.")
    if config.overfit_patients is not None and config.max_train_patients is not None:
        raise ValueError("Use either overfit_patients or max_train_patients, not both.")

    expected_modes = {
        "train_loss": "min",
        "train_mae": "min",
        "val_loss": "min",
        "val_mae": "min",
        "train_acc": "max",
        "val_acc": "max",
    }
    expected_mode = expected_modes[config.monitor_metric]
    if config.monitor_mode != expected_mode:
        raise ValueError(
            f"monitor_metric='{config.monitor_metric}' requires monitor_mode='{expected_mode}', "
            f"got '{config.monitor_mode}'."
        )
    if config.monitor_metric.endswith("_mae") and not config.track_mae:
        raise ValueError("Monitoring MAE requires track_mae=true.")


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    manifest_path = args.manifest_path or default_manifest_for_model(args.model_name)
    run_name = args.run_name or _timestamped_run_name(args.model_name)

    config = TrainConfig(
        run_name=run_name,
        output_root=args.output_root.resolve(),
        manifest_path=manifest_path.resolve(),
        model_name=args.model_name,
        num_classes=args.num_classes,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        dropout=args.dropout,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        use_class_weights=args.use_class_weights,
        track_mae=args.track_mae,
        train_augmentation=args.train_augmentation,
        pose_indices=args.pose_indices,
        overfit_patients=args.overfit_patients,
        max_train_patients=args.max_train_patients,
        max_val_patients=args.max_val_patients,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        monitor_metric=args.monitor_metric,
        monitor_mode=args.monitor_mode,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
    )
    _validate_config(config)
    return config
