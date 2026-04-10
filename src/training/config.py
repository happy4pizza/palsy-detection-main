from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TOML_PATH = BASE_DIR / "config.toml"
VALID_SPLITS = {"train", "val", "test"}
VALID_DEVICES = {"cpu", "cuda", "mps", "auto"}
VALID_MONITOR_MODES = {"min", "max"}
VALID_MONITOR_METRICS = {"val_loss", "val_acc", "val_mae"}
VALID_MODELS = {"efficientnet_b0", "single_efficientnet_b0"}


@dataclass(frozen=True)
class StudyConfig:
    name: str
    output_root: Path
    repeats: int


@dataclass(frozen=True)
class DataConfig:
    manifest_path: Path
    batch_size: int
    num_workers: int
    num_classes: int
    train_split: str
    val_split: str
    test_split: str


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    device: str
    early_stopping_patience: int
    base_seed: int
    monitor_metric: str
    monitor_mode: str


@dataclass(frozen=True)
class LossConfig:
    use_class_weights: bool
    track_mae: bool


@dataclass(frozen=True)
class EvaluationConfig:
    save_predictions_csv: bool
    save_confusion_matrix_csv: bool
    save_confusion_matrix_png: bool
    save_normalized_confusion_matrix_png: bool


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    pretrained: bool
    dropout: float
    freeze_backbone: bool
    unfreeze_last_n_blocks: int
    head_lr: float
    backbone_lr: float
    weight_decay: float


@dataclass(frozen=True)
class AppConfig:
    study: StudyConfig
    data: DataConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    experiments: list[ExperimentConfig]


def _expect_key(section: dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required key '{section_name}.{key}'.")
    return section[key]


def _expect_str(section: dict[str, Any], key: str, section_name: str) -> str:
    value = _expect_key(section, key, section_name)
    if not isinstance(value, str):
        raise ValueError(
            f"Invalid type for '{section_name}.{key}': expected str, got {type(value).__name__}."
        )
    return value


def _expect_bool(section: dict[str, Any], key: str, section_name: str) -> bool:
    value = _expect_key(section, key, section_name)
    if not isinstance(value, bool):
        raise ValueError(
            f"Invalid type for '{section_name}.{key}': expected bool, got {type(value).__name__}."
        )
    return value


def _expect_positive_int(section: dict[str, Any], key: str, section_name: str) -> int:
    value = _expect_key(section, key, section_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Invalid type for '{section_name}.{key}': expected int, got {type(value).__name__}."
        )
    if value <= 0:
        raise ValueError(f"'{section_name}.{key}' must be > 0. Got {value}.")
    return value


def _expect_non_negative_int(section: dict[str, Any], key: str, section_name: str) -> int:
    value = _expect_key(section, key, section_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Invalid type for '{section_name}.{key}': expected int, got {type(value).__name__}."
        )
    if value < 0:
        raise ValueError(f"'{section_name}.{key}' must be >= 0. Got {value}.")
    return value


def _expect_non_negative_float(section: dict[str, Any], key: str, section_name: str) -> float:
    value = _expect_key(section, key, section_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Invalid type for '{section_name}.{key}': expected float, got {type(value).__name__}."
        )
    value = float(value)
    if value < 0:
        raise ValueError(f"'{section_name}.{key}' must be >= 0. Got {value}.")
    return value


def _normalize_config_values(config: dict[str, Any]) -> None:
    """Normalize configurable string enum values in-place."""
    data = config.get("data")
    if isinstance(data, dict):
        for key in ("train_split", "val_split", "test_split"):
            value = data.get(key)
            if isinstance(value, str):
                data[key] = value.strip().lower()

    training = config.get("training")
    if isinstance(training, dict):
        for key in ("device", "monitor_metric", "monitor_mode"):
            value = training.get(key)
            if isinstance(value, str):
                training[key] = value.strip().lower()

    experiments = config.get("experiments")
    if isinstance(experiments, list):
        for exp in experiments:
            if isinstance(exp, dict):
                model = exp.get("model")
                if isinstance(model, str):
                    exp["model"] = model.strip().lower()


def _validate_config(config: dict[str, Any]) -> None:
    required_sections = ["study", "data", "training", "loss", "evaluation", "experiments"]
    for section_name in required_sections:
        if section_name not in config:
            raise ValueError(f"Missing required config section: '{section_name}'.")

    study = config["study"]
    data = config["data"]
    training = config["training"]
    loss = config["loss"]
    evaluation = config["evaluation"]
    experiments = config["experiments"]

    if not isinstance(study, dict):
        raise ValueError("Section 'study' must be a table/object.")
    if not isinstance(data, dict):
        raise ValueError("Section 'data' must be a table/object.")
    if not isinstance(training, dict):
        raise ValueError("Section 'training' must be a table/object.")
    if not isinstance(loss, dict):
        raise ValueError("Section 'loss' must be a table/object.")
    if not isinstance(evaluation, dict):
        raise ValueError("Section 'evaluation' must be a table/object.")
    if not isinstance(experiments, list):
        raise ValueError("Section 'experiments' must be an array of tables.")
    if not experiments:
        raise ValueError("Section 'experiments' must contain at least one experiment.")

    _expect_str(study, "name", "study")
    _expect_str(study, "output_root", "study")
    _expect_positive_int(study, "repeats", "study")

    _expect_str(data, "manifest_path", "data")
    _expect_positive_int(data, "batch_size", "data")
    _expect_non_negative_int(data, "num_workers", "data")
    _expect_positive_int(data, "num_classes", "data")

    train_split = _expect_str(data, "train_split", "data")
    val_split = _expect_str(data, "val_split", "data")
    test_split = _expect_str(data, "test_split", "data")
    for split_key, split_value in [
        ("train_split", train_split),
        ("val_split", val_split),
        ("test_split", test_split),
    ]:
        if split_value not in VALID_SPLITS:
            raise ValueError(
                f"Invalid 'data.{split_key}': {split_value}. Expected one of {sorted(VALID_SPLITS)}."
            )

    _expect_positive_int(training, "epochs", "training")
    device = _expect_str(training, "device", "training")
    if device not in VALID_DEVICES:
        raise ValueError(
            f"Invalid 'training.device': {device}. Expected one of {sorted(VALID_DEVICES)}."
        )
    _expect_non_negative_int(training, "early_stopping_patience", "training")
    _expect_non_negative_int(training, "base_seed", "training")

    monitor_metric = _expect_str(training, "monitor_metric", "training")
    if monitor_metric not in VALID_MONITOR_METRICS:
        raise ValueError(
            f"Invalid 'training.monitor_metric': {monitor_metric}. "
            f"Expected one of {sorted(VALID_MONITOR_METRICS)}."
        )

    monitor_mode = _expect_str(training, "monitor_mode", "training")
    if monitor_mode not in VALID_MONITOR_MODES:
        raise ValueError(
            f"Invalid 'training.monitor_mode': {monitor_mode}. "
            f"Expected one of {sorted(VALID_MONITOR_MODES)}."
        )

    expected_modes = {"val_loss": "min", "val_mae": "min", "val_acc": "max"}
    expected_mode = expected_modes[monitor_metric]
    if monitor_mode != expected_mode:
        raise ValueError(
            f"Incompatible monitor settings: training.monitor_metric='{monitor_metric}' "
            f"requires training.monitor_mode='{expected_mode}', got '{monitor_mode}'."
        )

    track_mae = _expect_bool(loss, "track_mae", "loss")
    _expect_bool(loss, "use_class_weights", "loss")

    if monitor_metric == "val_mae" and not track_mae:
        raise ValueError(
            "'training.monitor_metric=val_mae' requires 'loss.track_mae=true'."
        )

    _expect_bool(evaluation, "save_predictions_csv", "evaluation")
    _expect_bool(evaluation, "save_confusion_matrix_csv", "evaluation")
    _expect_bool(evaluation, "save_confusion_matrix_png", "evaluation")
    _expect_bool(evaluation, "save_normalized_confusion_matrix_png", "evaluation")

    required_exp_keys = [
        "name",
        "model",
        "pretrained",
        "dropout",
        "freeze_backbone",
        "unfreeze_last_n_blocks",
        "head_lr",
        "backbone_lr",
        "weight_decay",
    ]

    seen_experiment_names: set[str] = set()
    for i, exp in enumerate(experiments):
        section_name = f"experiments[{i}]"
        if not isinstance(exp, dict):
            raise ValueError(f"'{section_name}' must be a table/object.")
        for key in required_exp_keys:
            if key not in exp:
                raise ValueError(f"Missing required key '{section_name}.{key}'.")

        exp_name = _expect_str(exp, "name", section_name)
        if exp_name in seen_experiment_names:
            raise ValueError(f"Duplicate experiment name found: '{exp_name}'.")
        seen_experiment_names.add(exp_name)

        model = _expect_str(exp, "model", section_name)
        if model not in VALID_MODELS:
            raise ValueError(
                f"Invalid '{section_name}.model': {model}. Expected one of {sorted(VALID_MODELS)}."
            )

        _expect_bool(exp, "pretrained", section_name)
        freeze_backbone = _expect_bool(exp, "freeze_backbone", section_name)
        unfreeze_last_n_blocks = _expect_non_negative_int(exp, "unfreeze_last_n_blocks", section_name)

        dropout = _expect_non_negative_float(exp, "dropout", section_name)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"'{section_name}.dropout' must be in [0, 1). Got {dropout}.")

        head_lr = _expect_non_negative_float(exp, "head_lr", section_name)
        backbone_lr = _expect_non_negative_float(exp, "backbone_lr", section_name)
        _expect_non_negative_float(exp, "weight_decay", section_name)

        if head_lr <= 0:
            raise ValueError(f"'{section_name}.head_lr' must be > 0. Got {head_lr}.")
        if backbone_lr <= 0:
            raise ValueError(f"'{section_name}.backbone_lr' must be > 0. Got {backbone_lr}.")
        if freeze_backbone and unfreeze_last_n_blocks > 0:
            raise ValueError(
                f"'{section_name}' has conflicting settings: "
                "freeze_backbone=true with unfreeze_last_n_blocks>0."
            )


def _to_dataclass_config(config: dict[str, Any], config_file: Path) -> AppConfig:
    study = config["study"]
    data = config["data"]
    training = config["training"]
    loss = config["loss"]
    evaluation = config["evaluation"]
    experiments = config["experiments"]

    def resolve_path(raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (config_file.parent / path).resolve()
        return path

    study_cfg = StudyConfig(
        name=study["name"],
        output_root=resolve_path(study["output_root"]),
        repeats=study["repeats"],
    )
    data_cfg = DataConfig(
        manifest_path=resolve_path(data["manifest_path"]),
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        num_classes=data["num_classes"],
        train_split=data["train_split"],
        val_split=data["val_split"],
        test_split=data["test_split"],
    )
    training_cfg = TrainingConfig(
        epochs=training["epochs"],
        device=training["device"],
        early_stopping_patience=training["early_stopping_patience"],
        base_seed=training["base_seed"],
        monitor_metric=training["monitor_metric"],
        monitor_mode=training["monitor_mode"],
    )
    loss_cfg = LossConfig(
        use_class_weights=loss["use_class_weights"],
        track_mae=loss["track_mae"],
    )
    evaluation_cfg = EvaluationConfig(
        save_predictions_csv=evaluation["save_predictions_csv"],
        save_confusion_matrix_csv=evaluation["save_confusion_matrix_csv"],
        save_confusion_matrix_png=evaluation["save_confusion_matrix_png"],
        save_normalized_confusion_matrix_png=evaluation["save_normalized_confusion_matrix_png"],
    )

    exp_cfgs = [
        ExperimentConfig(
            name=exp["name"],
            model=exp["model"],
            pretrained=exp["pretrained"],
            dropout=float(exp["dropout"]),
            freeze_backbone=exp["freeze_backbone"],
            unfreeze_last_n_blocks=exp["unfreeze_last_n_blocks"],
            head_lr=float(exp["head_lr"]),
            backbone_lr=float(exp["backbone_lr"]),
            weight_decay=float(exp["weight_decay"]),
        )
        for exp in experiments
    ]

    return AppConfig(
        study=study_cfg,
        data=data_cfg,
        training=training_cfg,
        loss=loss_cfg,
        evaluation=evaluation_cfg,
        experiments=exp_cfgs,
    )


def load_config(config_path: str | Path | None = None) -> AppConfig:
    config_file = Path(config_path) if config_path is not None else DEFAULT_TOML_PATH
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not config_file.is_file():
        raise ValueError(f"Config path is not a file: {config_file}")

    try:
        with config_file.open("rb") as f:
            config = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in config file '{config_file}': {e}") from e

    _normalize_config_values(config)
    _validate_config(config)
    return _to_dataclass_config(config, config_file.resolve())
