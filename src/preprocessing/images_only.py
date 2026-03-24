"""Build an image-only manifest from a split manifest.

This script filters out non-image rows and writes image-only parquet/csv files
used by the training dataset loader.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MAN_DIR = DATA_DIR / "manifests"
DEFAULT_IN_PARQUET = MAN_DIR / "image_manifest_with_splits.parquet"
DEFAULT_OUT_PARQUET = MAN_DIR / "training_manifest_images_only.parquet"
DEFAULT_OUT_CSV = MAN_DIR / "training_manifest_images_only.csv"

REQUIRED_COLUMNS = {"patient_id", "pose_index", "filepath", "hb_grade", "split", "modality"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create image-only manifest from split manifest.")
    parser.add_argument("--in-parquet", type=Path, default=DEFAULT_IN_PARQUET)
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_OUT_PARQUET)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input manifest not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Input manifest is empty: {path}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input manifest missing required columns: {sorted(missing)}")

    return df


def build_image_manifest(df: pd.DataFrame) -> pd.DataFrame:
    image_df = df[df["modality"] == "image"].copy()
    if image_df.empty:
        raise ValueError("No image rows found in input manifest.")

    image_df = image_df[["patient_id", "pose_index", "filepath", "hb_grade", "split"]].copy()
    image_df = image_df.sort_values(by=["patient_id", "pose_index"]).reset_index(drop=True)
    return image_df


def save_outputs(image_df: pd.DataFrame, out_parquet: Path, out_csv: Path) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    image_df.to_parquet(out_parquet, index=False)
    image_df.to_csv(out_csv, index=False)

    logging.info("Saved %s rows", len(image_df))
    logging.info("Saved: %s", out_parquet)
    logging.info("Saved: %s", out_csv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    df = load_manifest(args.in_parquet.resolve())
    image_df = build_image_manifest(df)

    logging.info("Image-only table shape: %s", image_df.shape)
    logging.info("Unique patients: %s", image_df["patient_id"].nunique())
    logging.debug("Head:\n%s", image_df.head(10))

    save_outputs(
        image_df=image_df,
        out_parquet=args.out_parquet.resolve(),
        out_csv=args.out_csv.resolve(),
    )


if __name__ == "__main__":
    main()
