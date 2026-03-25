from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocessing.meei_single_dataset import MEEISingleImageDataset
from src.preprocessing.transforms import get_express_transform
from src.models.single_efficientnet_b0_model_v1 import SingleImageEfficientNetB0

# =========================================================
# Config
# =========================================================
BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 1e-5
DROPOUT = 0.3

FREEZE_BACKBONE = True
USE_CLASS_WEIGHTS = False
TRACK_MAE = True

NUM_WORKERS = 0
SHUFFLE_TRAIN = True

RUN_NAME = "efficientnet_b0_baseline_v1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MAN_DIR = DATA_DIR / "manifests"
RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

PAR_PATH = MAN_DIR / "training_manifest_single_images.parquet"

CHECKPOINT_PATH = RUNS_DIR / f"{RUN_NAME}.pt"
HISTORY_PATH = RUNS_DIR / f"{RUN_NAME}_history.json"


# =========================================================
# Device
# =========================================================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================================================
# Helpers
# =========================================================
def freeze_backbone_if_needed(model: nn.Module, freeze_backbone: bool) -> None:
    if not freeze_backbone:
        return

    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.encoder[-1].parameters():
        param.requires_grad = True


def get_class_counts(dataset: MEEISingleImageDataset, num_classes: int = 6) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)

    for i in range(len(dataset)):
        sample = dataset[i]
        label = int(sample["label"])
        counts[label] += 1

    return counts


def get_loss_function(
    use_class_weights: bool,
    train_dataset: MEEISingleImageDataset,
    device: torch.device,
    num_classes: int = 6
) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()

    class_counts = get_class_counts(train_dataset, num_classes=num_classes).float()

    if torch.any(class_counts == 0):
        raise ValueError(
            f"Cannot compute class weights because at least one class has zero samples. "
            f"Class counts: {class_counts.tolist()}"
        )

    weights = 1.0 / torch.sqrt(class_counts)
    weights = weights / weights.sum() * num_classes

    print(f"Using class weights: {weights.tolist()}", flush=True)
    return nn.CrossEntropyLoss(weight=weights.to(device))


def unpack_batch(batch: dict, device: torch.device):
    images = batch["image"].to(device)
    labels = batch["label"].to(device).long()
    return images, labels


def compute_batch_mae(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.abs(preds - labels).float().sum().item()


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    track_mae: bool = True
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_mae_sum = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        images, labels = unpack_batch(batch, device)

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
            avg_mae = total_mae_sum / total_samples
            postfix["avg_mae"] = f"{avg_mae:.4f}"

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
    track_mae: bool = True
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_mae_sum = 0.0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:
        images, labels = unpack_batch(batch, device)

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
            avg_mae = total_mae_sum / total_samples
            postfix["avg_mae"] = f"{avg_mae:.4f}"

        progress_bar.set_postfix(postfix)

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    epoch_mae = total_mae_sum / total_samples if track_mae else None

    return epoch_loss, epoch_acc, epoch_mae


# =========================================================
# Main
# =========================================================
def main():
    device = get_device()
    print(f"Using device: {device}", flush=True)

    transform = get_express_transform()

    train_dataset = MEEISingleImageDataset(PAR_PATH, "train", transform)
    val_dataset = MEEISingleImageDataset(PAR_PATH, "val", transform)

    print(f"Train size: {len(train_dataset)}", flush=True)
    print(f"Val size: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_TRAIN,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Val batches: {len(val_loader)}", flush=True)

    model = SingleImageEfficientNetB0(
        num_classes=6,
        pretrained=True,
        dropout=DROPOUT
    ).to(device)

    freeze_backbone_if_needed(model, FREEZE_BACKBONE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameter tensors: {len(trainable_params)}", flush=True)

    criterion = get_loss_function(
        use_class_weights=USE_CLASS_WEIGHTS,
        train_dataset=train_dataset,
        device=device,
        num_classes=6
    )

    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # Sanity check
    batch = next(iter(train_loader))
    images, labels = unpack_batch(batch, device)
    outputs = model(images)

    print(f"Sanity check images shape: {images.shape}", flush=True)
    print(f"Sanity check labels shape: {labels.shape}", flush=True)
    print(f"Sanity check labels min/max: {labels.min().item()} / {labels.max().item()}", flush=True)
    print(f"Sanity check outputs shape: {outputs.shape}", flush=True)

    history = {
        "config": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "freeze_backbone": FREEZE_BACKBONE,
            "use_class_weights": USE_CLASS_WEIGHTS,
            "track_mae": TRACK_MAE,
            "num_workers": NUM_WORKERS,
            "run_name": RUN_NAME,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "history_path": str(HISTORY_PATH),
        },
        "epochs": []
    }

    best_val_loss = float("inf")

    print("Starting training...", flush=True)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1:02d}/{NUM_EPOCHS}", flush=True)

        train_loss, train_acc, train_mae = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            track_mae=TRACK_MAE
        )

        val_loss, val_acc, val_mae = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            track_mae=TRACK_MAE
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }

        if TRACK_MAE:
            epoch_record["train_mae"] = train_mae
            epoch_record["val_mae"] = val_mae

        history["epochs"].append(epoch_record)

        metrics_str = (
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if TRACK_MAE:
            metrics_str += (
                f" | train_mae={train_mae:.4f} | val_mae={val_mae:.4f}"
            )

        metrics_str += f" | lr={current_lr:.6f}"

        print(metrics_str, flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved best model to {CHECKPOINT_PATH}", flush=True)

        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    print("Training complete.", flush=True)
    print(f"Best checkpoint: {CHECKPOINT_PATH}", flush=True)
    print(f"History saved to: {HISTORY_PATH}", flush=True)


if __name__ == "__main__":
    main()
