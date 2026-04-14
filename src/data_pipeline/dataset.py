from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data_pipeline.transforms import get_basic_transform


VALID_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {"patient_id", "pose_index", "filepath", "hb_grade", "split"}


def load_manifest_dataframe(
    manifest_path: Path | None = None,
    manifest_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if manifest_df is not None:
        df = manifest_df.copy()
    elif manifest_path is not None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        df = pd.read_parquet(manifest_path)
    else:
        raise ValueError("Either manifest_path or manifest_df must be provided.")

    if df.empty:
        raise ValueError("Manifest is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    return df


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    patient_df = df[["patient_id", "hb_grade"]].drop_duplicates().sort_values("patient_id").reset_index(drop=True)

    label_counts = patient_df.groupby("patient_id")["hb_grade"].nunique()
    bad_patients = label_counts[label_counts != 1]
    if not bad_patients.empty:
        raise ValueError(
            "Some patients have multiple hb_grade values:\n" + bad_patients.to_string()
        )

    return patient_df


def sample_patient_ids(df: pd.DataFrame, max_patients: int, subset_seed: int = 42) -> list[str]:
    patient_ids = build_patient_table(df)["patient_id"].tolist()
    if max_patients <= 0:
        raise ValueError(f"max_patients must be > 0. Got {max_patients}.")
    if max_patients >= len(patient_ids):
        return patient_ids

    rng = np.random.default_rng(subset_seed)
    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)
    return sorted(shuffled[:max_patients])


def filter_manifest_dataframe(
    df: pd.DataFrame,
    split: str | None = None,
    patient_ids: Sequence[str] | None = None,
    pose_indices: Sequence[int] | None = None,
    max_patients: int | None = None,
    subset_seed: int = 42,
) -> pd.DataFrame:
    filtered = df.copy()

    if split is not None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {sorted(VALID_SPLITS)}")
        filtered = filtered[filtered["split"] == split].copy()

    if patient_ids is not None:
        patient_id_set = {str(patient_id) for patient_id in patient_ids}
        filtered = filtered[filtered["patient_id"].astype(str).isin(patient_id_set)].copy()

    if pose_indices is not None:
        pose_index_set = {int(pose_index) for pose_index in pose_indices}
        filtered = filtered[filtered["pose_index"].isin(pose_index_set)].copy()

    if max_patients is not None:
        selected_patient_ids = sample_patient_ids(filtered, max_patients=max_patients, subset_seed=subset_seed)
        filtered = filtered[filtered["patient_id"].astype(str).isin(set(selected_patient_ids))].copy()

    if filtered.empty:
        raise ValueError("No rows remain after applying dataset filters.")

    return filtered.sort_values(["patient_id", "pose_index"]).reset_index(drop=True)


class MEEIDataset(Dataset):
    """Patient-level dataset that returns all requested poses for one patient."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        split: str = "train",
        transform=None,
        *,
        manifest_df: pd.DataFrame | None = None,
        patient_ids: Sequence[str] | None = None,
        pose_indices: Sequence[int] | None = None,
        max_patients: int | None = None,
        subset_seed: int = 42,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.split = split
        self.transform = transform if transform is not None else get_basic_transform()

        df = load_manifest_dataframe(manifest_path=self.manifest_path, manifest_df=manifest_df)
        filtered_df = filter_manifest_dataframe(
            df=df,
            split=split,
            patient_ids=patient_ids,
            pose_indices=pose_indices,
            max_patients=max_patients,
            subset_seed=subset_seed,
        )

        self.samples = self._build_patient_samples(filtered_df)
        self.labels = [int(sample["hb_grade"]) for sample in self.samples]
        self.patient_ids = [str(sample["patient_id"]) for sample in self.samples]

    @staticmethod
    def _build_patient_samples(df: pd.DataFrame) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        for patient_id, group in df.groupby("patient_id", sort=True):
            group = group.sort_values("pose_index")
            filepaths = group["filepath"].tolist()
            pose_indices = group["pose_index"].tolist()

            labels = group["hb_grade"].unique()
            if len(labels) != 1:
                raise ValueError(f"Patient {patient_id} has multiple hb_grade values: {labels}")

            if not filepaths:
                raise ValueError(f"Patient {patient_id} has no image rows in split '{group['split'].iloc[0]}'.")

            samples.append(
                {
                    "patient_id": str(patient_id),
                    "filepaths": filepaths,
                    "pose_indices": pose_indices,
                    "hb_grade": int(labels[0]),
                }
            )

        if not samples:
            raise ValueError("No patient samples were created from the manifest.")

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

    def get_labels(self) -> list[int]:
        return self.labels.copy()


class MEEISingleImageDataset(Dataset):
    """Dataset that returns one image per sample for a single-image baseline."""

    def __init__(
        self,
        manifest_path: Path | None = None,
        split: str = "train",
        transform=None,
        *,
        manifest_df: pd.DataFrame | None = None,
        patient_ids: Sequence[str] | None = None,
        pose_indices: Sequence[int] | None = None,
        max_patients: int | None = None,
        subset_seed: int = 42,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.split = split
        self.transform = transform if transform is not None else get_basic_transform()

        df = load_manifest_dataframe(manifest_path=self.manifest_path, manifest_df=manifest_df)
        self.df = filter_manifest_dataframe(
            df=df,
            split=split,
            patient_ids=patient_ids,
            pose_indices=pose_indices,
            max_patients=max_patients,
            subset_seed=subset_seed,
        )

        self.labels = self.df["hb_grade"].astype(int).tolist()
        self.patient_ids = self.df["patient_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        img_path = row["filepath"]
        label = torch.tensor(int(row["hb_grade"]), dtype=torch.long)

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "patient_id": str(row["patient_id"]),
            "pose_index": int(row["pose_index"]),
        }

    def get_labels(self) -> list[int]:
        return self.labels.copy()

