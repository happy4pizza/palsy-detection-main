"""Single entry point for building filtered image manifests from a split manifest.

This script supports two modes:
- ``images_only``: keep every image row and drop non-image media.
- ``single_image``: keep only one pose index per patient/image row subset.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MAN_DIR = DATA_DIR / "manifests"
DEFAULT_IN_PARQUET = MAN_DIR / "manifest_split.parquet"

MODE_ALIASES = {
    "images_only": "images_only",
    "single_image": "single_image",
    "single_images": "single_image",
}

DEFAULT_OUTPUTS = {
    "images_only": {
        "parquet": MAN_DIR / "manifest_image_only.parquet",
        "csv": MAN_DIR / "manifest_image_only.csv",
    },
    "single_image": {
        "parquet": MAN_DIR / "manifest_single_image.parquet",
        "csv": MAN_DIR / "manifest_single_image.csv",
    },
}

REQUIRED_COLUMNS = {"patient_id", "pose_index", "filepath", "hb_grade", "split", "modality"}
DEFAULT_SINGLE_POSE_INDEX = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create filtered image manifests from split manifest.")
    parser.add_argument("--in-parquet", type=Path, default=DEFAULT_IN_PARQUET)
    parser.add_argument("--mode", default="images_only", choices=sorted(MODE_ALIASES))
    parser.add_argument("--single-pose-index", type=int, default=DEFAULT_SINGLE_POSE_INDEX)
    parser.add_argument("--out-parquet", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def normalize_mode(mode: str) -> str:
    try:
        return MODE_ALIASES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of {sorted(MODE_ALIASES)}") from exc


def resolve_output_paths(
    mode: str,
    out_parquet: Path | None,
    out_csv: Path | None,
) -> tuple[Path, Path]:
    defaults = DEFAULT_OUTPUTS[mode]
    resolved_out_parquet = out_parquet or defaults["parquet"]
    resolved_out_csv = out_csv or defaults["csv"]
    return resolved_out_parquet, resolved_out_csv


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


def build_image_manifest(
    df: pd.DataFrame,
    mode: str = "images_only",
    single_pose_index: int = DEFAULT_SINGLE_POSE_INDEX,
) -> pd.DataFrame:
    normalized_mode = normalize_mode(mode)

    if normalized_mode == "images_only":
        image_df = df[df["modality"] == "image"].copy()
    else:
        image_df = df[
            (df["modality"] == "image")
            & (df["pose_index"] == single_pose_index)
        ].copy()

    if image_df.empty:
        raise ValueError(
            f"No rows found for mode '{normalized_mode}'"
            + (
                f" with pose_index={single_pose_index}."
                if normalized_mode == "single_image"
                else "."
            )
        )

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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    mode = normalize_mode(args.mode)
    df = load_manifest(args.in_parquet.resolve())
    image_df = build_image_manifest(
        df,
        mode=mode,
        single_pose_index=args.single_pose_index,
    )

    out_parquet, out_csv = resolve_output_paths(
        mode=mode,
        out_parquet=args.out_parquet,
        out_csv=args.out_csv,
    )

    logging.info("Mode: %s", mode)
    if mode == "single_image":
        logging.info("Single-image pose index: %s", args.single_pose_index)
    logging.info("Filtered image table shape: %s", image_df.shape)
    logging.info("Unique patients: %s", image_df["patient_id"].nunique())
    logging.debug("Head:\n%s", image_df.head(10))

    save_outputs(
        image_df=image_df,
        out_parquet=out_parquet.resolve(),
        out_csv=out_csv.resolve(),
    )


if __name__ == "__main__":
    main()
