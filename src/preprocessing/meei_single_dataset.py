from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

try:
    from .transforms import get_basic_transform
except ImportError:  # pragma: no cover
    from transforms import get_basic_transform


VALID_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {"patient_id", "pose_index", "filepath", "hb_grade", "split"}


class MEEISingleImageDataset(Dataset):
    """Dataset that returns one image per sample for a single-image baseline."""

    def __init__(self, manifest_path: Path, split: str = "train", transform=None) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform if transform is not None else get_basic_transform()

        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {sorted(VALID_SPLITS)}")

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_parquet(self.manifest_path)

        if df.empty:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        self.df = df[df["split"] == split].copy().reset_index(drop=True)

        if self.df.empty:
            raise ValueError(f"No rows found for split '{split}'.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img_path = row["filepath"]
        label = torch.tensor(int(row["hb_grade"]), dtype=torch.long)

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "patient_id": row["patient_id"],
            "pose_index": row["pose_index"],
        }

