from __future__ import annotations

from pathlib import Path

from src.training.engine import run_experiment
from src.training.experiment_config import ExperimentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "training_manifest_single_images.parquet"


def main() -> None:
    config = ExperimentConfig(
        run_name="efficientnet_b0_single_v1",
        task_type="single_image",
        model_name="efficientnet_b0_single_image",
        manifest_path=MANIFEST_PATH,
        transform_name="express",
        batch_size=4,
        num_epochs=15,
        learning_rate=1e-5,
        dropout=0.3,
        pretrained=True,
        freeze_backbone=True,
        use_class_weights=False,
        track_mae=True,
        num_workers=0,
        shuffle_train=True,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
