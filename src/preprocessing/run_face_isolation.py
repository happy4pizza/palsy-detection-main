from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.preprocessing.face_isolation import isolate_face_to_224


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MAN_DIR = DATA_DIR / "manifests"

INPUT_MANIFEST = MAN_DIR / "training_manifest_images_only.parquet"
OUTPUT_MANIFEST = MAN_DIR / "training_manifest_faces224.parquet"

FACES_DIR = DATA_DIR / "faces_224"


def main() -> None:
    df = pd.read_parquet(INPUT_MANIFEST)

    new_paths = []
    success_flags = []

    for _, row in df.iterrows():
        old_path = Path(row["filepath"])

        rel_name = f"{old_path.stem}_face224.jpg"
        patient_id = str(row["patient_id"])
        new_path = FACES_DIR / patient_id / rel_name

        success = isolate_face_to_224(
            image_path=old_path,
            output_path=new_path,
            target_size=224,
        )

        new_paths.append(str(new_path) if success else None)
        success_flags.append(success)

    df["face_filepath"] = new_paths
    df["face_success"] = success_flags

    df = df[df["face_success"]].copy()
    df["filepath"] = df["face_filepath"]
    df = df.drop(columns=["face_filepath", "face_success"])

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_MANIFEST, index=False)

    print(f"Saved: {OUTPUT_MANIFEST}")
    print(f"Rows kept: {len(df)}")


if __name__ == "__main__":
    main()
