"""Patient-level image dataset for MEEI facial palsy classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

try:
    from .transforms import get_basic_transform
except ImportError:  # pragma: no cover - fallback for script-style imports
    from transforms import get_basic_transform


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MAN_DIR = DATA_DIR / "manifests"
PAR_PATH = MAN_DIR / "image_manifest_images_only.parquet"

VALID_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {"patient_id", "pose_index", "filepath", "hb_grade", "split"}


class MEEIDataset(Dataset):
    """Patient-level dataset that returns all poses for one patient.

    Required manifest columns:
    - `patient_id`
    - `pose_index`
    - `filepath`
    - `hb_grade`
    - `split`

    Args:
        manifest_path: Path to image-only manifest parquet file.
        split: One of `train`, `val`, or `test`.
        transform: Optional torchvision-compatible callable applied to each image.
            If `None`, a default baseline transform is used.
    """

    def __init__(self, manifest_path: Path = PAR_PATH, split: str = "train", transform=None) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform if transform is not None else get_basic_transform()

        df = self._load_manifest(self.manifest_path)
        self.samples = self._build_patient_samples(df=df, split=split)

    @staticmethod
    def _load_manifest(manifest_path: Path) -> pd.DataFrame:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_parquet(manifest_path)
        if df.empty:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        return df

    @staticmethod
    def _build_patient_samples(df: pd.DataFrame, split: str) -> list[dict[str, Any]]:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {sorted(VALID_SPLITS)}")

        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            raise ValueError(f"No rows found for split '{split}'.")

        split_df = split_df.sort_values(["patient_id", "pose_index"]).reset_index(drop=True)
        samples: list[dict[str, Any]] = []

        for patient_id, group in split_df.groupby("patient_id", sort=True):
            group = group.sort_values("pose_index")
            filepaths = group["filepath"].tolist()
            pose_indices = group["pose_index"].tolist()

            labels = group["hb_grade"].unique()
            if len(labels) != 1:
                raise ValueError(f"Patient {patient_id} has multiple hb_grade values: {labels}")

            if not filepaths:
                raise ValueError(f"Patient {patient_id} has no image rows in split '{split}'.")

            samples.append(
                {
                    "patient_id": patient_id,
                    "filepaths": filepaths,
                    "pose_indices": pose_indices,
                    "hb_grade": int(labels[0]),
                }
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        images = []
        for filepath in sample["filepaths"]:
            with Image.open(filepath) as image:
                image = image.convert("RGB")
                image = self.transform(image)
            images.append(image)

        stacked_images = torch.stack(images, dim=0)
        label = torch.tensor(sample["hb_grade"], dtype=torch.long)

        return {
            "images": stacked_images,
            "label": label,
            "patient_id": sample["patient_id"],
            "pose_indices": sample["pose_indices"],
        }
