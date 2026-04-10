"""Create patient-level train/val/test splits and merge back to the manifest.

This script:
1) Loads `image_manifest.parquet`
2) Validates one HB grade per patient
3) Performs stratified patient-level splits
4) Merges split labels onto every media row
5) Writes patient and row-level split outputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MAN_DIR = DATA_DIR / "manifests"
DEFAULT_IN_PARQUET = MAN_DIR / "manifest_raw.parquet"
DEFAULT_OUT_PATIENT_SPLITS_CSV = MAN_DIR / "patient_split.csv"
DEFAULT_OUT_WITH_SPLITS_PARQUET = MAN_DIR / "manifest_split.parquet"
DEFAULT_OUT_WITH_SPLITS_CSV = MAN_DIR / "manifest_split.csv"

REQUIRED_COLUMNS = {"patient_id", "hb_grade", "cohort", "severity_folder"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified patient splits.")
    parser.add_argument("--in-parquet", type=Path, default=DEFAULT_IN_PARQUET)
    parser.add_argument("--out-patient-splits-csv", type=Path, default=DEFAULT_OUT_PATIENT_SPLITS_CSV)
    parser.add_argument("--out-with-splits-parquet", type=Path, default=DEFAULT_OUT_WITH_SPLITS_PARQUET)
    parser.add_argument("--out-with-splits-csv", type=Path, default=DEFAULT_OUT_WITH_SPLITS_CSV)
    parser.add_argument("--train-size", type=float, default=0.70, help="Fraction of patients in train split.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction of patients in val split.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction of patients in test split.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    sizes = [train_size, val_size, test_size]
    if any(s <= 0 or s >= 1 for s in sizes):
        raise ValueError("Split sizes must each be between 0 and 1 (exclusive).")

    if abs(sum(sizes) - 1.0) > 1e-6:
        raise ValueError(
            f"Split sizes must sum to 1.0. Received: train={train_size}, val={val_size}, test={test_size}"
        )


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input manifest not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Input manifest is empty: {path}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input manifest missing required columns: {sorted(missing)}")

    logging.info("Loaded manifest: %s rows, %s unique patients", len(df), df["patient_id"].nunique())
    return df


def validate_one_grade_per_patient(df: pd.DataFrame) -> None:
    hb_per_patient = df.groupby("patient_id")["hb_grade"].nunique()
    bad = hb_per_patient[hb_per_patient != 1]
    if not bad.empty:
        raise ValueError("Some patients have multiple hb_grade values:\n" + bad.to_string())


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    patient_df = df[["patient_id", "hb_grade", "cohort", "severity_folder"]].drop_duplicates()
    logging.info("Patient-level table: %s rows", len(patient_df))
    logging.info("Patient counts by hb_grade:\n%s", patient_df["hb_grade"].value_counts().sort_index())
    return patient_df


def stratified_patient_split(
    patient_df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_seed: int,
) -> pd.DataFrame:
    validate_split_sizes(train_size, val_size, test_size)

    train_patients, temp_patients = train_test_split(
        patient_df,
        train_size=train_size,
        stratify=patient_df["hb_grade"],
        random_state=random_seed,
    )

    val_fraction_of_temp = val_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=val_fraction_of_temp,
        stratify=temp_patients["hb_grade"],
        random_state=random_seed,
    )

    train_patients = train_patients.copy()
    val_patients = val_patients.copy()
    test_patients = test_patients.copy()

    train_patients["split"] = "train"
    val_patients["split"] = "val"
    test_patients["split"] = "test"

    patient_splits = pd.concat([train_patients, val_patients, test_patients], ignore_index=True)
    logging.info("Patient counts by split:\n%s", patient_splits["split"].value_counts())
    logging.info("HB grade counts by split:\n%s", pd.crosstab(patient_splits["hb_grade"], patient_splits["split"]))

    return patient_splits


def validate_no_duplicate_assignments(patient_splits: pd.DataFrame) -> None:
    dup_counts = patient_splits["patient_id"].value_counts()
    bad_dups = dup_counts[dup_counts > 1]
    if not bad_dups.empty:
        raise ValueError("Some patients appear in more than one split:\n" + bad_dups.to_string())


def merge_splits(df: pd.DataFrame, patient_splits: pd.DataFrame) -> pd.DataFrame:
    df_with_splits = df.merge(patient_splits[["patient_id", "split"]], on="patient_id", how="left")
    if df_with_splits["split"].isna().any():
        missing_ids = df_with_splits.loc[df_with_splits["split"].isna(), "patient_id"].unique().tolist()
        raise ValueError(f"Some rows did not receive a split after merging. Missing patient_ids: {missing_ids}")
    return df_with_splits


def validate_no_patient_leakage(df_with_splits: pd.DataFrame) -> None:
    train_ids = set(df_with_splits.loc[df_with_splits["split"] == "train", "patient_id"])
    val_ids = set(df_with_splits.loc[df_with_splits["split"] == "val", "patient_id"])
    test_ids = set(df_with_splits.loc[df_with_splits["split"] == "test", "patient_id"])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            "Patient leakage detected:\n"
            f"train ∩ val: {sorted(train_val_overlap)}\n"
            f"train ∩ test: {sorted(train_test_overlap)}\n"
            f"val ∩ test: {sorted(val_test_overlap)}"
        )


def save_outputs(
    patient_splits: pd.DataFrame,
    df_with_splits: pd.DataFrame,
    out_patient_splits_csv: Path,
    out_with_splits_parquet: Path,
    out_with_splits_csv: Path,
) -> None:
    out_patient_splits_csv.parent.mkdir(parents=True, exist_ok=True)
    out_with_splits_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_with_splits_csv.parent.mkdir(parents=True, exist_ok=True)

    patient_splits.to_csv(out_patient_splits_csv, index=False)
    df_with_splits.to_parquet(out_with_splits_parquet, index=False)
    df_with_splits.to_csv(out_with_splits_csv, index=False)

    logging.info("Saved: %s", out_patient_splits_csv)
    logging.info("Saved: %s", out_with_splits_parquet)
    logging.info("Saved: %s", out_with_splits_csv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    df = load_manifest(args.in_parquet.resolve())
    validate_one_grade_per_patient(df)
    patient_df = build_patient_table(df)
    patient_splits = stratified_patient_split(
        patient_df=patient_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    validate_no_duplicate_assignments(patient_splits)
    df_with_splits = merge_splits(df=df, patient_splits=patient_splits)
    validate_no_patient_leakage(df_with_splits)
    save_outputs(
        patient_splits=patient_splits,
        df_with_splits=df_with_splits,
        out_patient_splits_csv=args.out_patient_splits_csv.resolve(),
        out_with_splits_parquet=args.out_with_splits_parquet.resolve(),
        out_with_splits_csv=args.out_with_splits_csv.resolve(),
    )


if __name__ == "__main__":
    main()
