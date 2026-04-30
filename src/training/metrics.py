from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    sample_count: int
    mae_hb: float | None = None
    within_one_grade_accuracy: float | None = None
    macro_f1: float | None = None
    balanced_accuracy: float | None = None
    per_class_accuracy: dict[str, float | None] | None = None
    per_class_support: dict[str, int] | None = None
    confusion_matrix: list[list[int]] | None = None
    class_labels: list[int] | None = None

    def to_dict(self, prefix: str) -> dict[str, float | int | None]:
        row: dict[str, float | int | None] = {
            f"{prefix}loss": self.loss,
            f"{prefix}acc": self.accuracy,
            f"{prefix}sample_count": self.sample_count,
        }
        if self.mae_hb is not None:
            row[f"{prefix}mae_hb"] = self.mae_hb
        if self.within_one_grade_accuracy is not None:
            row[f"{prefix}within_1_grade_acc"] = self.within_one_grade_accuracy
        if self.macro_f1 is not None:
            row[f"{prefix}macro_f1"] = self.macro_f1
        if self.balanced_accuracy is not None:
            row[f"{prefix}balanced_acc"] = self.balanced_accuracy
        if self.per_class_accuracy is not None:
            for hb_grade, class_accuracy in self.per_class_accuracy.items():
                row[f"{prefix}acc_hb_{hb_grade}"] = class_accuracy
        return row

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "sample_count": self.sample_count,
            "mae_hb": self.mae_hb,
            "within_one_grade_accuracy": self.within_one_grade_accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "per_class_accuracy": self.per_class_accuracy,
            "per_class_support": self.per_class_support,
            "class_labels_hb": self.class_labels,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_evaluation_metrics(
    *,
    loss: float,
    labels: list[int],
    preds: list[int],
    num_classes: int,
) -> EpochMetrics:
    sample_count = len(labels)
    if sample_count == 0:
        raise ValueError("Cannot compute evaluation metrics with zero samples.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0. Got {num_classes}.")
    if len(preds) != sample_count:
        raise ValueError(
            f"labels and preds must be the same length. Got {sample_count} and {len(preds)}."
        )

    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    exact_matches = 0
    within_one_matches = 0
    abs_errors: list[int] = []

    for label, pred in zip(labels, preds):
        if not 0 <= label < num_classes:
            raise ValueError(f"Invalid label {label}. Expected 0 <= label < {num_classes}.")
        if not 0 <= pred < num_classes:
            raise ValueError(f"Invalid prediction {pred}. Expected 0 <= pred < {num_classes}.")

        confusion_matrix[label][pred] += 1
        exact_matches += int(pred == label)
        abs_error = abs(pred - label)
        abs_errors.append(abs_error)
        within_one_matches += int(abs_error <= 1)

    class_labels = list(range(1, num_classes + 1))
    per_class_accuracy: dict[str, float | None] = {}
    per_class_support: dict[str, int] = {}
    recalls_for_average: list[float] = []
    f1_for_average: list[float] = []

    for class_idx, hb_grade in enumerate(class_labels):
        tp = confusion_matrix[class_idx][class_idx]
        support = sum(confusion_matrix[class_idx])
        predicted_count = sum(row[class_idx] for row in confusion_matrix)

        recall = tp / support if support > 0 else None
        precision = tp / predicted_count if predicted_count > 0 else None

        if recall is not None:
            recalls_for_average.append(recall)

        if recall is not None and precision is not None and (precision + recall) > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        elif support > 0 or predicted_count > 0:
            f1_score = 0.0
        else:
            f1_score = None

        if f1_score is not None:
            f1_for_average.append(f1_score)

        per_class_accuracy[str(hb_grade)] = recall
        per_class_support[str(hb_grade)] = support

    mae_hb = sum(abs_errors) / sample_count
    balanced_accuracy = (
        sum(recalls_for_average) / len(recalls_for_average)
        if recalls_for_average
        else None
    )
    macro_f1 = sum(f1_for_average) / len(f1_for_average) if f1_for_average else None

    return EpochMetrics(
        loss=loss,
        accuracy=exact_matches / sample_count,
        sample_count=sample_count,
        mae_hb=mae_hb,
        within_one_grade_accuracy=within_one_matches / sample_count,
        macro_f1=macro_f1,
        balanced_accuracy=balanced_accuracy,
        per_class_accuracy=per_class_accuracy,
        per_class_support=per_class_support,
        confusion_matrix=confusion_matrix,
        class_labels=class_labels,
    )
