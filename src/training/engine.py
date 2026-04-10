from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_pipeline.dataset import MEEIDataset


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: MEEIDataset,
        val_dataset: MEEIDataset,
        device: torch.device,
        batch_size: int = 32,
        num_epochs: int = 20,
        learning_rate: float = 1e-3,
        use_class_weights: bool = False,
        track_mae: bool = True,
        num_classes: int = 6,
        num_workers: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.use_class_weights = use_class_weights
        self.track_mae = track_mae
        self.num_classes = num_classes

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.criterion = self._build_loss_function()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_class_counts(self) -> torch.Tensor:
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for i in range(len(self.train_dataset)):
            label = int(self.train_dataset[i]["label"])
            counts[label] += 1
        return counts

    def _build_loss_function(self) -> nn.Module:
        if not self.use_class_weights:
            return nn.CrossEntropyLoss()

        class_counts = self._get_class_counts().float()
        if torch.any(class_counts == 0):
            raise ValueError(
                "Cannot compute class weights because at least one class has zero samples. "
                f"Class counts: {class_counts.tolist()}"
            )

        weights = 1.0 / torch.sqrt(class_counts)
        weights = weights / weights.sum() * self.num_classes
        print(f"Using class weights: {weights.tolist()}", flush=True)
        return nn.CrossEntropyLoss(weight=weights.to(self.device))

    def _unpack_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["images"].to(self.device)
        labels = batch["label"].to(self.device).long()
        return images, labels

    @staticmethod
    def _compute_batch_mae(preds: torch.Tensor, labels: torch.Tensor) -> float:
        return torch.abs(preds - labels).float().sum().item()

    def train_one_epoch(self) -> tuple[float, float, float | None]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_mae_sum = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            images, labels = self._unpack_batch(batch)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            preds = outputs.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            if self.track_mae:
                total_mae_sum += self._compute_batch_mae(preds, labels)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            postfix = {
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "avg_acc": f"{avg_acc:.4f}",
            }
            if self.track_mae:
                postfix["avg_mae"] = f"{(total_mae_sum / total_samples):.4f}"
            progress_bar.set_postfix(postfix)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        epoch_mae = total_mae_sum / total_samples if self.track_mae else None
        return epoch_loss, epoch_acc, epoch_mae

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float, float | None]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_mae_sum = 0.0

        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        for batch in progress_bar:
            images, labels = self._unpack_batch(batch)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            if self.track_mae:
                total_mae_sum += self._compute_batch_mae(preds, labels)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            postfix = {
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "avg_acc": f"{avg_acc:.4f}",
            }
            if self.track_mae:
                postfix["avg_mae"] = f"{(total_mae_sum / total_samples):.4f}"
            progress_bar.set_postfix(postfix)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        epoch_mae = total_mae_sum / total_samples if self.track_mae else None
        return epoch_loss, epoch_acc, epoch_mae

    def fit(self) -> list[dict[str, float | int]]:
        history: list[dict[str, float | int]] = []
        for epoch in range(self.num_epochs):
            train_loss, train_acc, train_mae = self.train_one_epoch()
            val_loss, val_acc, val_mae = self.evaluate()

            record: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            if self.track_mae and train_mae is not None and val_mae is not None:
                record["train_mae"] = train_mae
                record["val_mae"] = val_mae

            history.append(record)
        return history
