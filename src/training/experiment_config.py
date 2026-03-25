from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TaskType = Literal["multi_pose", "single_image"]
TransformName = Literal["basic", "express"]
ModelName = Literal[
    "efficientnet_b0_multi_pose",
    "efficientnet_b0_single_image",
]


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    task_type: TaskType
    manifest_path: Path
    model_name: ModelName | None = None
    transform_name: TransformName = "basic"
    output_dir: Path = PROJECT_ROOT / "data" / "runs"
    batch_size: int = 4
    num_epochs: int = 15
    learning_rate: float = 1e-4
    dropout: float = 0.3
    pretrained: bool = True
    freeze_backbone: bool = True
    use_class_weights: bool = True
    track_mae: bool = True
    num_workers: int = 0
    shuffle_train: bool = True
    num_classes: int = 6
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    device: str = "auto"
    seed: int | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / f"{self.run_name}.pt"

    @property
    def history_path(self) -> Path:
        return self.output_dir / f"{self.run_name}_history.json"

    def to_serializable(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest_path"] = str(self.manifest_path)
        data["output_dir"] = str(self.output_dir)
        data["checkpoint_path"] = str(self.checkpoint_path)
        data["history_path"] = str(self.history_path)
        return data


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def config_from_dict(
    raw: dict[str, Any],
    base_dir: Path,
    defaults: dict[str, Any] | None = None,
) -> ExperimentConfig:
    merged = dict(defaults or {})
    merged.update(raw)

    if "run_name" not in merged:
        raise ValueError("Each experiment requires 'run_name'.")
    if "task_type" not in merged:
        raise ValueError(f"Experiment '{merged['run_name']}' requires 'task_type'.")
    if "manifest_path" not in merged:
        raise ValueError(f"Experiment '{merged['run_name']}' requires 'manifest_path'.")

    merged["manifest_path"] = _resolve_path(merged["manifest_path"], base_dir)

    output_dir = merged.get("output_dir", PROJECT_ROOT / "data" / "runs")
    merged["output_dir"] = _resolve_path(output_dir, base_dir)

    return ExperimentConfig(**merged)
