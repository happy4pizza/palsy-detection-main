from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data_pipeline.path_utils import PROJECT_ROOT, resolve_manifest_filepath
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
    patient_df = (
        df[["patient_id", "hb_grade"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )

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
        selected_patient_ids = sample_patient_ids(
            filtered,
            max_patients=max_patients,
            subset_seed=subset_seed,
        )
        filtered = filtered[filtered["patient_id"].astype(str).isin(set(selected_patient_ids))].copy()

    if filtered.empty:
        raise ValueError("No rows remain after applying dataset filters.")

    return filtered.sort_values(["patient_id", "pose_index"]).reset_index(drop=True)


class BasePalsyDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | None = None,
        split: str = "train",
        transform=None,
        *,
        manifest_df: pd.DataFrame | None = None,
        patient_ids: Sequence[str] | None = None,
        pose_indices: Sequence[int] | None = None,
        expected_pose_indices: Sequence[int] | None = None,
        max_patients: int | None = None,
        project_root: Path | None = None,
        subset_seed: int = 42,
        validate_filepaths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.split = split
        self.transform = transform if transform is not None else get_basic_transform()
        self.project_root = (Path(project_root) if project_root is not None else PROJECT_ROOT).resolve()
        self.expected_pose_indices = self._normalize_pose_indices(expected_pose_indices)

        df = load_manifest_dataframe(manifest_path=self.manifest_path, manifest_df=manifest_df)
        filtered_df = filter_manifest_dataframe(
            df=df,
            split=split,
            patient_ids=patient_ids,
            pose_indices=pose_indices,
            max_patients=max_patients,
            subset_seed=subset_seed,
        )

        self.samples = self._build_samples(filtered_df, pose_indices=pose_indices)
        if validate_filepaths:
            self._validate_sample_filepaths(self.samples)
        self.labels = [int(sample["hb_grade"]) for sample in self.samples]
        self.patient_ids = [str(sample["patient_id"]) for sample in self.samples]

    @staticmethod
    def _normalize_pose_indices(pose_indices: Sequence[int] | None) -> list[int] | None:
        if pose_indices is None:
            return None
        return [int(pose_index) for pose_index in pose_indices]

    def _build_samples(
        self,
        df: pd.DataFrame,
        *,
        pose_indices: Sequence[int] | None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _validate_sample_filepaths(self, samples: list[dict[str, Any]]) -> None:
        missing: list[str] = []
        for sample in samples:
            for filepath, resolved_filepath in zip(
                sample["filepaths"],
                sample["resolved_filepaths"],
            ):
                if not Path(resolved_filepath).exists():
                    missing.append(filepath)
        if missing:
            raise FileNotFoundError(f"Missing image files: {missing[:5]}")

    @staticmethod
    def _resolve_filepaths(filepaths: list[str], *, project_root: Path) -> list[str]:
        return [
            str(resolve_manifest_filepath(filepath, project_root=project_root))
            for filepath in filepaths
        ]

    def get_labels(self) -> list[int]:
        return self.labels.copy()

    def __len__(self) -> int:
        return len(self.samples)


class MEEIDataset(BasePalsyDataset):
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
        expected_pose_indices: Sequence[int] | None = None,
        max_patients: int | None = None,
        project_root: Path | None = None,
        subset_seed: int = 42,
        validate_filepaths: bool = True,
    ) -> None:
        normalized_expected_pose_indices = self._normalize_pose_indices(expected_pose_indices)
        if normalized_expected_pose_indices is None and pose_indices is not None:
            normalized_expected_pose_indices = self._normalize_pose_indices(pose_indices)

        super().__init__(
            manifest_path=manifest_path,
            split=split,
            transform=transform,
            manifest_df=manifest_df,
            patient_ids=patient_ids,
            pose_indices=pose_indices,
            expected_pose_indices=normalized_expected_pose_indices,
            max_patients=max_patients,
            project_root=project_root,
            subset_seed=subset_seed,
            validate_filepaths=validate_filepaths,
        )

    def _build_samples(
        self,
        df: pd.DataFrame,
        *,
        pose_indices: Sequence[int] | None,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        for patient_id, group in df.groupby("patient_id", sort=True):
            group = group.sort_values("pose_index")
            filepaths = [str(filepath) for filepath in group["filepath"].tolist()]
            patient_pose_indices = [int(pose_index) for pose_index in group["pose_index"].tolist()]

            labels = group["hb_grade"].unique()
            if len(labels) != 1:
                raise ValueError(f"Patient {patient_id} has multiple hb_grade values: {labels}")

            if not filepaths:
                raise ValueError(
                    f"Patient {patient_id} has no image rows in split '{group['split'].iloc[0]}'."
                )

            if len(set(patient_pose_indices)) != len(patient_pose_indices):
                raise ValueError(f"Patient {patient_id} has duplicate pose indices: {patient_pose_indices}")

            if self.expected_pose_indices is not None and patient_pose_indices != self.expected_pose_indices:
                raise ValueError(
                    f"Patient {patient_id} pose indices {patient_pose_indices} do not match expected "
                    f"{self.expected_pose_indices}."
                )

            samples.append(
                {
                    "patient_id": str(patient_id),
                    "filepaths": filepaths,
                    "resolved_filepaths": self._resolve_filepaths(
                        filepaths,
                        project_root=self.project_root,
                    ),
                    "pose_indices": patient_pose_indices,
                    "hb_grade": int(labels[0]),
                }
            )

        if not samples:
            raise ValueError("No patient samples were created from the manifest.")

        return samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        images = []
        for filepath in sample["resolved_filepaths"]:
            with Image.open(filepath) as image:
                image = image.convert("RGB")
                image = self.transform(image)
            images.append(image)

        inputs = torch.stack(images, dim=0)
        label = torch.tensor(sample["hb_grade"], dtype=torch.long)

        return {
            "inputs": inputs,
            "label": label,
            "metadata": {
                "patient_id": sample["patient_id"],
                "pose_indices": sample["pose_indices"],
                "filepaths": sample["filepaths"],
                "sample_index": idx,
            },
        }


class MEEISingleImageDataset(BasePalsyDataset):
    """Dataset that returns one image per sample for a single-image baseline."""

    def _build_samples(
        self,
        df: pd.DataFrame,
        *,
        pose_indices: Sequence[int] | None,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        expected_pose_index_set = (
            set(self.expected_pose_indices)
            if self.expected_pose_indices is not None
            else None
        )

        for row in df.itertuples(index=False):
            pose_index = int(row.pose_index)
            if expected_pose_index_set is not None and pose_index not in expected_pose_index_set:
                raise ValueError(
                    f"Pose index {pose_index} is not in expected_pose_indices "
                    f"{sorted(expected_pose_index_set)}."
                )

            filepath = str(row.filepath)
            samples.append(
                {
                    "patient_id": str(row.patient_id),
                    "filepaths": [filepath],
                    "resolved_filepaths": self._resolve_filepaths(
                        [filepath],
                        project_root=self.project_root,
                    ),
                    "pose_indices": [pose_index],
                    "hb_grade": int(row.hb_grade),
                }
            )

        if not samples:
            raise ValueError("No image samples were created from the manifest.")

        return samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        img_path = sample["resolved_filepaths"][0]
        label = torch.tensor(sample["hb_grade"], dtype=torch.long)

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "inputs": image,
            "label": label,
            "metadata": {
                "patient_id": sample["patient_id"],
                "pose_indices": sample["pose_indices"],
                "pose_index": sample["pose_indices"][0],
                "filepaths": sample["filepaths"],
                "sample_index": idx,
            },
        }
