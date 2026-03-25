from __future__ import annotations

import json
import random
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.efficientnet_b0_model_v1 import MultiPoseEfficientNetB0
from src.models.single_efficientnet_b0_model_v1 import SingleImageEfficientNetB0
from src.preprocessing.meei_dataset import MEEIDataset
from src.preprocessing.meei_single_dataset import MEEISingleImageDataset
from src.preprocessing.transforms import get_basic_transform, get_express_transform
from src.training.experiment_config import ExperimentConfig


TRANSFORM_MAP = {
    "basic": get_basic_transform,
    "express": get_express_transform,
}


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, mps.")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device 'cuda' is unavailable.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested device 'mps' is unavailable.")

    return torch.device(device)


def freeze_backbone_if_needed(model: nn.Module, task_type: str, freeze_backbone: bool) -> None:
    if not freeze_backbone:
        return

    for param in model.encoder.parameters():
        param.requires_grad = False

    if task_type == "multi_pose":
        for param in model.encoder[-2].parameters():
            param.requires_grad = True
    for param in model.encoder[-1].parameters():
        param.requires_grad = True


def _build_transform(name: str):
    if name not in TRANSFORM_MAP:
        raise ValueError(f"Unknown transform '{name}'. Available: {sorted(TRANSFORM_MAP)}")
    return TRANSFORM_MAP[name]()


def _build_datasets(config: ExperimentConfig) -> tuple[Dataset, Dataset, str]:
    transform = _build_transform(config.transform_name)

    if config.task_type == "multi_pose":
        train_dataset = MEEIDataset(config.manifest_path, "train", transform)
        val_dataset = MEEIDataset(config.manifest_path, "val", transform)
        return train_dataset, val_dataset, "images"

    if config.task_type == "single_image":
        train_dataset = MEEISingleImageDataset(config.manifest_path, "train", transform)
        val_dataset = MEEISingleImageDataset(config.manifest_path, "val", transform)
        return train_dataset, val_dataset, "image"

    raise ValueError(f"Unknown task_type '{config.task_type}'.")


def _build_model(config: ExperimentConfig) -> nn.Module:
    if config.model_name is None and config.task_type == "multi_pose":
        return MultiPoseEfficientNetB0(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )

    if config.model_name is None and config.task_type == "single_image":
        return SingleImageEfficientNetB0(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )

    if config.model_name == "efficientnet_b0_multi_pose":
        if config.task_type != "multi_pose":
            raise ValueError(
                "model_name='efficientnet_b0_multi_pose' requires task_type='multi_pose'."
            )
        return MultiPoseEfficientNetB0(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )

    if config.model_name == "efficientnet_b0_single_image":
        if config.task_type != "single_image":
            raise ValueError(
                "model_name='efficientnet_b0_single_image' requires task_type='single_image'."
            )
        return SingleImageEfficientNetB0(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout=config.dropout,
        )

    raise ValueError(
        f"Unsupported model/task combination: model_name={config.model_name}, "
        f"task_type={config.task_type}"
    )


def set_seed_if_needed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_counts(dataset: Dataset, num_classes: int = 6) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)

    for i in range(len(dataset)):
        label = int(dataset[i]["label"])
        counts[label] += 1

    return counts


def get_loss_function(
    use_class_weights: bool,
    train_dataset: Dataset,
    device: torch.device,
    num_classes: int = 6,
) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()

    class_counts = get_class_counts(train_dataset, num_classes=num_classes).float()
    if torch.any(class_counts == 0):
        raise ValueError(
            "Cannot compute class weights because at least one class has zero samples. "
            f"Class counts: {class_counts.tolist()}"
        )

    weights = 1.0 / torch.sqrt(class_counts)
    weights = weights / weights.sum() * num_classes
    print(f"Using class weights: {weights.tolist()}", flush=True)
    return nn.CrossEntropyLoss(weight=weights.to(device))


def unpack_batch(batch: dict[str, Any], image_key: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch[image_key].to(device)
    labels = batch["label"].to(device).long()
    return images, labels


def compute_batch_mae(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.abs(preds - labels).float().sum().item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    image_key: str,
    track_mae: bool = True,
) -> tuple[float, float, float | None]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_mae_sum = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        images, labels = unpack_batch(batch, image_key, device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        if track_mae:
            total_mae_sum += compute_batch_mae(preds, labels)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        postfix = {
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "avg_acc": f"{avg_acc:.4f}",
        }
        if track_mae:
            postfix["avg_mae"] = f"{(total_mae_sum / total_samples):.4f}"
        progress_bar.set_postfix(postfix)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    epoch_mae = total_mae_sum / total_samples if track_mae else None
    return epoch_loss, epoch_acc, epoch_mae


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    image_key: str,
    track_mae: bool = True,
) -> tuple[float, float, float | None]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_mae_sum = 0.0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    for batch in progress_bar:
        images, labels = unpack_batch(batch, image_key, device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        if track_mae:
            total_mae_sum += compute_batch_mae(preds, labels)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        postfix = {
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "avg_acc": f"{avg_acc:.4f}",
        }
        if track_mae:
            postfix["avg_mae"] = f"{(total_mae_sum / total_samples):.4f}"
        progress_bar.set_postfix(postfix)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    epoch_mae = total_mae_sum / total_samples if track_mae else None
    return epoch_loss, epoch_acc, epoch_mae


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed_if_needed(config.seed)
    device = get_device(config.device)
    print(f"Running '{config.run_name}' on device: {device}", flush=True)
    if config.seed is not None:
        print(f"Seed: {config.seed}", flush=True)

    train_dataset, val_dataset, image_key = _build_datasets(config)
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True)

    model = _build_model(config).to(device)
    freeze_backbone_if_needed(model, config.task_type, config.freeze_backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameter tensors: {len(trainable_params)}", flush=True)

    criterion = get_loss_function(
        use_class_weights=config.use_class_weights,
        train_dataset=train_dataset,
        device=device,
        num_classes=config.num_classes,
    )

    optimizer = torch.optim.Adam(trainable_params, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    batch = next(iter(train_loader))
    images, labels = unpack_batch(batch, image_key, device)
    outputs = model(images)
    print(f"Sanity check images shape: {tuple(images.shape)}", flush=True)
    print(f"Sanity check labels min/max: {labels.min().item()} / {labels.max().item()}", flush=True)
    print(f"Sanity check outputs shape: {tuple(outputs.shape)}", flush=True)

    history: dict[str, Any] = {
        "config": config.to_serializable(),
        "epochs": [],
    }
    best_val_loss = float("inf")

    print("Starting training...", flush=True)
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1:02d}/{config.num_epochs}", flush=True)
        train_loss, train_acc, train_mae = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            image_key=image_key,
            track_mae=config.track_mae,
        )
        val_loss, val_acc, val_mae = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            image_key=image_key,
            track_mae=config.track_mae,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        if config.track_mae:
            epoch_record["train_mae"] = train_mae
            epoch_record["val_mae"] = val_mae
        history["epochs"].append(epoch_record)

        metrics = (
            f"Epoch {epoch + 1:02d}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        if config.track_mae:
            metrics += f" | train_mae={train_mae:.4f} | val_mae={val_mae:.4f}"
        metrics += f" | lr={current_lr:.6f}"
        print(metrics, flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.checkpoint_path)
            print(f"Saved best model to {config.checkpoint_path}", flush=True)

        with config.history_path.open("w") as f:
            json.dump(history, f, indent=2)

    print(f"Training complete for '{config.run_name}'.", flush=True)
    print(f"Best checkpoint: {config.checkpoint_path}", flush=True)
    print(f"History saved to: {config.history_path}", flush=True)

    return {
        "run_name": config.run_name,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(config.checkpoint_path),
        "history_path": str(config.history_path),
    }
