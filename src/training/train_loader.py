from __future__ import annotations

from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from src.data_pipeline.dataset import MEEIDataset, MEEISingleImageDataset
from src.data_pipeline.transforms import get_eval_transform, get_train_transform
from src.training.config import DataConfig, ModelConfig


DATASET_REGISTRY = {
    "multi_image_efficientnet_b0": MEEIDataset,
    "single_image_efficientnet_b0": MEEISingleImageDataset,
}


def build_datasets(config: DataConfig, *, model_config: ModelConfig):
    try:
        dataset_class = DATASET_REGISTRY[model_config.name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model name '{model_config.name}'. "
            f"Expected one of {sorted(DATASET_REGISTRY)}."
        ) from exc

    if dataset_class is MEEIDataset:
        _validate_multi_image_pose_config(config, model_config)

    dataset_kwargs = {
        "manifest_path": config.manifest_path,
        "max_patients": config.max_patients,
        "pose_indices": config.pose_indices,
        "expected_pose_indices": config.expected_pose_indices,
        "project_root": config.project_root,
        "subset_seed": config.subset_seed,
        "validate_filepaths": config.validate_filepaths,
    }

    train_dataset = dataset_class(
        split="train",
        transform=get_train_transform(use_augmentation=config.use_augmentation),
        **dataset_kwargs,
    )

    val_dataset = dataset_class(
        split="val",
        transform=get_eval_transform(),
        **dataset_kwargs,
    )

    return train_dataset, val_dataset


def build_dataloaders(
    config: DataConfig,
    *,
    model_config: ModelConfig,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets(config, model_config=model_config)

    pin_memory = config.pin_memory if config.pin_memory is not None else False
    persistent_workers = (
        config.persistent_workers
        if config.persistent_workers is not None
        else config.num_workers > 0
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": palsy_collate_fn,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader


def palsy_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "inputs": default_collate([sample["inputs"] for sample in batch]),
        "label": default_collate([sample["label"] for sample in batch]),
        "metadata": [sample["metadata"] for sample in batch],
    }


def _validate_multi_image_pose_config(
    data_config: DataConfig,
    model_config: ModelConfig,
) -> None:
    if data_config.pose_indices is None:
        raise ValueError(
            f"{model_config.name} requires data.pose_indices to be set so the experiment "
            "declares exactly which poses are used."
        )
    if data_config.expected_pose_indices is None:
        raise ValueError(
            f"{model_config.name} requires data.expected_pose_indices to be set so pose "
            "coverage is enforced instead of assumed from the manifest."
        )

    pose_indices = [int(pose_index) for pose_index in data_config.pose_indices]
    expected_pose_indices = [int(pose_index) for pose_index in data_config.expected_pose_indices]

    if pose_indices != expected_pose_indices:
        raise ValueError(
            "For multi-image training, data.pose_indices and data.expected_pose_indices "
            f"must match exactly. Got {pose_indices} and {expected_pose_indices}."
        )

    if model_config.num_poses is not None and len(expected_pose_indices) != model_config.num_poses:
        raise ValueError(
            f"{model_config.name} expects model.num_poses={model_config.num_poses}, but "
            f"data.expected_pose_indices has length {len(expected_pose_indices)}."
        )
