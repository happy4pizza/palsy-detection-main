from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from src.data_pipeline.dataset import (
    MEEIDataset,
    MEEISingleImageDataset,
    filter_manifest_dataframe,
    load_manifest_dataframe,
    sample_patient_ids,
)
from src.data_pipeline.transforms import get_eval_transform, get_train_transform
from src.training.config import TrainConfig, parse_args
from src.training.engine import (
    build_loss_function,
    build_model,
    build_optimizer,
    configure_trainable_parameters,
    count_trainable_parameters,
    fit,
    get_device,
    seed_everything,
)


DATASET_CLASSES = {
    "multi_pose": MEEIDataset,
    "single_image": MEEISingleImageDataset,
}


def summarize_labels(labels: list[int]) -> dict[int, int]:
    label_counts = Counter(labels)
    return {int(label): int(label_counts[label]) for label in sorted(label_counts)}


def build_datasets(config: TrainConfig):
    manifest_df = load_manifest_dataframe(manifest_path=config.manifest_path)
    dataset_class = DATASET_CLASSES[config.model_name]

    train_transform = get_train_transform(use_augmentation=config.train_augmentation == "light")
    eval_transform = get_eval_transform()

    train_patient_ids: list[str] | None = None
    val_patient_ids: list[str] | None = None
    effective_val_split = config.val_split

    if config.overfit_patients is not None:
        candidate_df = filter_manifest_dataframe(
            manifest_df,
            split=config.train_split,
            pose_indices=config.pose_indices,
        )
        selected_patient_ids = sample_patient_ids(
            candidate_df,
            max_patients=config.overfit_patients,
            subset_seed=config.seed,
        )
        train_patient_ids = selected_patient_ids
        val_patient_ids = selected_patient_ids
        effective_val_split = config.train_split
    else:
        if config.max_train_patients is not None:
            train_df = filter_manifest_dataframe(
                manifest_df,
                split=config.train_split,
                pose_indices=config.pose_indices,
            )
            train_patient_ids = sample_patient_ids(
                train_df,
                max_patients=config.max_train_patients,
                subset_seed=config.seed,
            )
        if config.max_val_patients is not None:
            val_df = filter_manifest_dataframe(
                manifest_df,
                split=config.val_split,
                pose_indices=config.pose_indices,
            )
            val_patient_ids = sample_patient_ids(
                val_df,
                max_patients=config.max_val_patients,
                subset_seed=config.seed,
            )

    train_dataset = dataset_class(
        manifest_df=manifest_df,
        split=config.train_split,
        transform=train_transform,
        patient_ids=train_patient_ids,
        pose_indices=config.pose_indices,
    )
    val_dataset = dataset_class(
        manifest_df=manifest_df,
        split=effective_val_split,
        transform=eval_transform,
        patient_ids=val_patient_ids,
        pose_indices=config.pose_indices,
    )

    selected_patients = {
        "train_patient_ids": train_dataset.patient_ids,
        "val_patient_ids": val_dataset.patient_ids,
        "overfit_mode": config.overfit_patients is not None,
    }
    return train_dataset, val_dataset, selected_patients


def build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    use_pin_memory = device.type == "cuda"
    use_persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    return train_loader, val_loader


def save_run_metadata(run_dir: Path, config: TrainConfig, selected_patients: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=2)

    with (run_dir / "selected_patients.json").open("w", encoding="utf-8") as file:
        json.dump(selected_patients, file, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    seed_everything(config.seed)

    device = get_device(config.device)
    train_dataset, val_dataset, selected_patients = build_datasets(config)
    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
    )

    run_dir = config.run_dir
    save_run_metadata(run_dir, config, selected_patients)

    model = build_model(config).to(device)
    configure_trainable_parameters(
        model,
        freeze_backbone=config.freeze_backbone,
        unfreeze_last_n_blocks=config.unfreeze_last_n_blocks,
    )
    trainable_counts = count_trainable_parameters(model)

    criterion = build_loss_function(
        labels=train_dataset.get_labels(),
        num_classes=config.num_classes,
        use_class_weights=config.use_class_weights,
        device=device,
    )
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    print(f"Run name: {config.run_name}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Using device: {device}", flush=True)
    print(f"Model: {config.model_name}", flush=True)
    print(f"Manifest: {config.manifest_path}", flush=True)
    print(f"Train patients/samples: {len(train_dataset)}", flush=True)
    print(f"Val patients/samples: {len(val_dataset)}", flush=True)
    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Val batches: {len(val_loader)}", flush=True)
    print(f"Pose subset: {config.pose_indices}", flush=True)
    print(f"Train class counts: {summarize_labels(train_dataset.get_labels())}", flush=True)
    print(f"Val class counts: {summarize_labels(val_dataset.get_labels())}", flush=True)
    print(
        f"Trainable params: {trainable_counts['trainable_parameter_scalars']} "
        f"across {trainable_counts['trainable_parameter_tensors']} tensors",
        flush=True,
    )

    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        run_dir=run_dir,
    )

    print("Training complete.", flush=True)
    print(f"Best checkpoint: {summary['best_checkpoint_path']}", flush=True)
    print(
        f"Best {summary['best_monitor_metric']}: {summary['best_monitor_value']} "
        f"at epoch {summary['best_epoch']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
