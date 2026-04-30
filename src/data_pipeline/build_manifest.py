"""Build a unified media manifest for model training and analysis.

This script scans `data/raw` for image/video files, enriches rows with metadata from
`pat_info.xlsx`, and writes both parquet and CSV manifests under `data/manifests`.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data_pipeline.path_utils import PROJECT_ROOT, relativize_to_project_root


BASE_DIR = PROJECT_ROOT
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_XLSX_PATH = DEFAULT_RAW_DIR / "pat_info.xlsx"
DEFAULT_OUT_PARQUET = DEFAULT_DATA_DIR / "manifests" / "manifest_raw.parquet"
DEFAULT_OUT_CSV = DEFAULT_DATA_DIR / "manifests" / "manifest_raw.csv"

MEDIA_SUFFIXES = {".jpg", ".jpeg", ".mp4"}
POSE_PATTERN = re.compile(r"_(\d+)\.(?:jpe?g)$", re.IGNORECASE)

HB_GRADE_BY_FOLDER = {
    "Normal": 0,
    "NearNormalFlaccid": 1,
    "NearNormalSynkinetic": 1,
    "MildFlaccid": 2,
    "MildSynkinetic": 2,
    "ModerateFlaccid": 3,
    "ModerateSynkinetic": 3,
    "SevereFlaccid": 4,
    "SevereSynkinetic": 4,
    "CompleteFlaccid": 5,
    "CompleteSynkinetic": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset manifest from raw media.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--metadata-xlsx", type=Path, default=DEFAULT_XLSX_PATH)
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_OUT_PARQUET)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def extract_pose_index(filename: str) -> int | None:
    match = POSE_PATTERN.search(filename)
    return int(match.group(1)) if match else None


def extract_hb_grade(severity_folder: str) -> int:
    try:
        return HB_GRADE_BY_FOLDER[severity_folder.strip()]
    except KeyError as exc:
        raise ValueError(f"Unknown severity folder: {severity_folder}") from exc


def iter_media_paths(raw_dir: Path) -> Iterable[Path]:
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
            yield path


def parse_file_record(path: Path, raw_dir: Path) -> dict[str, object] | None:
    rel_parts = path.relative_to(raw_dir).parts

    # Supported layouts:
    # 1) Normal/<patient>/<file>
    # 2) <cohort>/<severity>/<patient>/<file>
    if len(rel_parts) == 3 and rel_parts[0] == "Normal":
        cohort = "Normal"
        severity_folder = "Normal"
        patient_id = rel_parts[1]
    elif len(rel_parts) >= 4:
        cohort = rel_parts[0]
        severity_folder = rel_parts[1]
        patient_id = rel_parts[2]
    else:
        logging.warning("Skipping path with unexpected structure: %s", path)
        return None

    suffix = path.suffix.lower()
    modality = "video" if suffix == ".mp4" else "image"
    pose_index = None if modality == "video" else extract_pose_index(path.name)

    return {
        "patient_id": patient_id.strip(),
        "cohort": cohort,
        "severity_folder": severity_folder,
        "modality": modality,
        "pose_index": pose_index,
        "filepath": relativize_to_project_root(path),
    }


def load_file_manifest(raw_dir: Path) -> pd.DataFrame:
    records = []
    for path in iter_media_paths(raw_dir):
        record = parse_file_record(path, raw_dir)
        if record is not None:
            record["hb_grade"] = extract_hb_grade(str(record["severity_folder"]))
            records.append(record)

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise ValueError(f"No media files found under {raw_dir}")

    return manifest


def load_patient_metadata(metadata_xlsx: Path) -> pd.DataFrame:
    meta = pd.read_excel(metadata_xlsx)

    required_columns = {"Sub-category", "Category", "#", "Side", "Gender", "Age"}
    missing = required_columns - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata file missing required columns: {sorted(missing)}")

    meta["patient_id"] = (
        meta["Sub-category"].astype(str).str.strip()
        + meta["Category"].astype(str).str.strip()
        + meta["#"].astype(str).str.strip()
    )

    meta["patient_id"] = meta["patient_id"].astype(str).str.strip()
    return meta[["patient_id", "Side", "Gender", "Age"]]


def merge_manifest_with_metadata(file_manifest: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    merged = file_manifest.merge(meta, on="patient_id", how="left")

    missing_meta = merged["Side"].isna().sum()
    if missing_meta:
        logging.warning("%s rows are missing patient metadata after merge.", missing_meta)

    columns = [
        "patient_id",
        "cohort",
        "severity_folder",
        "pose_index",
        "modality",
        "filepath",
        "hb_grade",
        "Side",
        "Gender",
        "Age",
    ]
    merged = merged[columns].copy()
    merged.sort_values(["patient_id", "modality", "filepath"], inplace=True, kind="stable")
    merged.reset_index(drop=True, inplace=True)
    return merged


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    raw_dir = args.raw_dir.resolve()
    metadata_xlsx = args.metadata_xlsx.resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    if not metadata_xlsx.exists():
        raise FileNotFoundError(f"Metadata spreadsheet not found: {metadata_xlsx}")

    file_manifest = load_file_manifest(raw_dir)
    meta = load_patient_metadata(metadata_xlsx)
    final_manifest = merge_manifest_with_metadata(file_manifest, meta)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    final_manifest.to_parquet(args.out_parquet, index=False)
    final_manifest.to_csv(args.out_csv, index=False)

    logging.info("Wrote %s rows to %s and %s", len(final_manifest), args.out_parquet, args.out_csv)


if __name__ == "__main__":
    main()
