from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data_pipeline.dataset import build_patient_table, load_manifest_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
DEFAULT_IN_MANIFEST = MANIFEST_DIR / "manifest_face224.parquet"
DEFAULT_OUT_DIR = MANIFEST_DIR / "eval_splits"
VALID_STRATEGIES = {"kfold", "repeated_holdout"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create patient-level evaluation manifests.")
    parser.add_argument("--in-manifest", type=Path, default=DEFAULT_IN_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--strategy", default="kfold", choices=sorted(VALID_STRATEGIES))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def merge_patient_assignments(df: pd.DataFrame, patient_split_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.drop(columns=["split"], errors="ignore").merge(
        patient_split_df[["patient_id", "split"]],
        on="patient_id",
        how="left",
    )
    if merged["split"].isna().any():
        raise ValueError("Some manifest rows did not receive a split assignment.")
    return merged


def validate_kfold_inputs(patient_df: pd.DataFrame, num_folds: int) -> None:
    if num_folds <= 1:
        raise ValueError(f"num_folds must be > 1. Got {num_folds}.")

    min_class_size = int(patient_df["hb_grade"].value_counts().min())
    if min_class_size < num_folds:
        raise ValueError(
            f"Cannot build {num_folds} stratified folds because the smallest class only has "
            f"{min_class_size} patients."
        )


def save_split_outputs(
    *,
    base_name: str,
    output_dir: Path,
    patient_split_df: pd.DataFrame,
    row_manifest_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_csv_path = output_dir / f"{base_name}_patients.csv"
    manifest_parquet_path = output_dir / f"{base_name}.parquet"
    manifest_csv_path = output_dir / f"{base_name}.csv"

    patient_split_df.to_csv(patient_csv_path, index=False)
    row_manifest_df.to_parquet(manifest_parquet_path, index=False)
    row_manifest_df.to_csv(manifest_csv_path, index=False)

    logging.info("Saved patient assignments: %s", patient_csv_path)
    logging.info("Saved row manifest: %s", manifest_parquet_path)
    logging.info("Saved row manifest CSV: %s", manifest_csv_path)


def build_kfold_manifests(
    df: pd.DataFrame,
    input_stem: str,
    output_dir: Path,
    num_folds: int,
    random_seed: int,
) -> None:
    patient_df = build_patient_table(df)
    validate_kfold_inputs(patient_df, num_folds)

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    for fold_index, (train_idx, val_idx) in enumerate(
        splitter.split(patient_df["patient_id"], patient_df["hb_grade"]),
        start=1,
    ):
        train_patients = patient_df.iloc[train_idx].copy()
        val_patients = patient_df.iloc[val_idx].copy()

        train_patients["split"] = "train"
        val_patients["split"] = "val"

        patient_split_df = pd.concat([train_patients, val_patients], ignore_index=True)
        row_manifest_df = merge_patient_assignments(df, patient_split_df)

        logging.info(
            "Fold %s patient counts by split:\n%s",
            fold_index,
            patient_split_df["split"].value_counts().to_string(),
        )
        logging.info(
            "Fold %s HB grade counts by split:\n%s",
            fold_index,
            pd.crosstab(patient_split_df["hb_grade"], patient_split_df["split"]).to_string(),
        )

        save_split_outputs(
            base_name=f"{input_stem}_fold{fold_index:02d}",
            output_dir=output_dir,
            patient_split_df=patient_split_df,
            row_manifest_df=row_manifest_df,
        )


def build_repeated_holdout_manifests(
    df: pd.DataFrame,
    input_stem: str,
    output_dir: Path,
    repeats: int,
    val_size: float,
    random_seed: int,
) -> None:
    if repeats <= 0:
        raise ValueError(f"repeats must be > 0. Got {repeats}.")
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be in (0, 1). Got {val_size}.")

    patient_df = build_patient_table(df)

    for repeat_index in range(1, repeats + 1):
        train_patients, val_patients = train_test_split(
            patient_df,
            test_size=val_size,
            stratify=patient_df["hb_grade"],
            random_state=random_seed + repeat_index - 1,
        )

        train_patients = train_patients.copy()
        val_patients = val_patients.copy()
        train_patients["split"] = "train"
        val_patients["split"] = "val"

        patient_split_df = pd.concat([train_patients, val_patients], ignore_index=True)
        row_manifest_df = merge_patient_assignments(df, patient_split_df)

        logging.info(
            "Repeat %s patient counts by split:\n%s",
            repeat_index,
            patient_split_df["split"].value_counts().to_string(),
        )
        logging.info(
            "Repeat %s HB grade counts by split:\n%s",
            repeat_index,
            pd.crosstab(patient_split_df["hb_grade"], patient_split_df["split"]).to_string(),
        )

        save_split_outputs(
            base_name=f"{input_stem}_repeat{repeat_index:02d}",
            output_dir=output_dir,
            patient_split_df=patient_split_df,
            row_manifest_df=row_manifest_df,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    input_manifest = args.in_manifest.resolve()
    output_dir = args.out_dir.resolve()
    df = load_manifest_dataframe(manifest_path=input_manifest)

    logging.info("Loaded manifest: %s rows, %s unique patients", len(df), df["patient_id"].nunique())

    if args.strategy == "kfold":
        build_kfold_manifests(
            df=df,
            input_stem=input_manifest.stem,
            output_dir=output_dir,
            num_folds=args.num_folds,
            random_seed=args.random_seed,
        )
    else:
        build_repeated_holdout_manifests(
            df=df,
            input_stem=input_manifest.stem,
            output_dir=output_dir,
            repeats=args.repeats,
            val_size=args.val_size,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
