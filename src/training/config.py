from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
import tomllib


@dataclass(slots=True)
class DataConfig:
    manifest_path: Path = Path("data/manifests/manifest_face224.parquet")
    batch_size: int = 4
    num_workers: int = 0
    max_patients: int | None = None
    pose_indices: list[int] | None = None
    expected_pose_indices: list[int] | None = None
    project_root: Path | None = None
    subset_seed: int = 42
    use_augmentation: bool = False
    validate_filepaths: bool = True
    pin_memory: bool | None = None
    persistent_workers: bool | None = None


@dataclass(slots=True)
class ModelConfig:
    name: str = "multi_image_efficientnet_b0"
    num_classes: int = 6
    pretrained: bool = True
    dropout: float = 0.4
    aggregation: str = "mean"
    num_poses: int | None = 8
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 1e-3
    backbone_lr: float | None = None
    weight_decay: float = 1e-4


@dataclass(slots=True)
class LossConfig:
    name: str = "cross_entropy"
    class_weights: list[float] | None = None
    use_balanced_class_weights: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    num_epochs: int = 40
    log_every: int = 1
    seed: int = 42
    checkpoint_metric: str = "val_loss"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    device: str | None = None


@dataclass(slots=True)
class SchedulerConfig:
    enabled: bool = True
    name: str = "reduce_on_plateau"
    monitor: str = "val_loss"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-6


@dataclass(slots=True)
class ExportConfig:
    output_dir: Path = Path("artifacts/training")
    run_name: str | None = None
    metrics_filename: str = "metrics.csv"
    config_filename: str = "config.json"
    summary_filename: str = "summary.json"
    checkpoint_dirname: str = "checkpoints"
    evaluations_dirname: str = "evaluations"
    best_checkpoint_filename: str = "best.pt"
    last_checkpoint_filename: str = "last.pt"
    best_predictions_filename: str = "best_val_predictions.csv"
    export_predictions_every_epoch: bool = True


@dataclass(slots=True)
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


DEFAULT_CONFIG_PATH = Path("config.toml")


def load_config(path: Path) -> TrainConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    valid_sections = {"data", "model", "optimizer", "loss", "runtime", "scheduler", "export"}
    unknown_sections = set(raw) - valid_sections
    if unknown_sections:
        raise ValueError(f"Unknown config sections in {config_path}: {sorted(unknown_sections)}")

    return TrainConfig(
        data=_load_section(
            DataConfig,
            raw.get("data"),
            path_fields={"manifest_path", "project_root"},
        ),
        model=_load_section(ModelConfig, raw.get("model")),
        optimizer=_load_section(OptimizerConfig, raw.get("optimizer")),
        loss=_load_section(LossConfig, raw.get("loss")),
        runtime=_load_section(RuntimeConfig, raw.get("runtime")),
        scheduler=_load_section(SchedulerConfig, raw.get("scheduler")),
        export=_load_section(ExportConfig, raw.get("export"), path_fields={"output_dir"}),
    )


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    return _serialize_paths(asdict(config))


def _load_section(
    config_class,
    raw_values: dict[str, Any] | None,
    *,
    path_fields: set[str] | None = None,
):
    path_fields = path_fields or set()
    raw_values = raw_values or {}

    default_instance = config_class()
    valid_field_names = {field_info.name for field_info in fields(config_class)}
    unknown_fields = set(raw_values) - valid_field_names
    if unknown_fields:
        raise ValueError(
            f"Unknown fields for {config_class.__name__}: {sorted(unknown_fields)}"
        )

    kwargs: dict[str, Any] = {}
    for field_info in fields(config_class):
        value = raw_values.get(field_info.name, getattr(default_instance, field_info.name))
        if value is not None and field_info.name in path_fields:
            value = Path(value)
        kwargs[field_info.name] = value

    return config_class(**kwargs)


def _serialize_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_paths(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_serialize_paths(inner_value) for inner_value in value]
    return value
