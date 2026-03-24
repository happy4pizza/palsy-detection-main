"""Image transforms for the MEEI facial palsy pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


@dataclass(frozen=True)
class FixedTopCrop:
    """Center-crop horizontally with a fixed top anchor."""

    crop_size: int = 224
    top_offset: int = 20

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if self.crop_size > width or self.crop_size > height:
            raise ValueError(
                f"Crop size {self.crop_size} exceeds image size {(width, height)}."
            )

        left = (width - self.crop_size) // 2
        top = self.top_offset
        if top < 0 or (top + self.crop_size) > height:
            raise ValueError(
                f"Invalid top_offset={self.top_offset} for crop_size={self.crop_size} "
                f"and image height={height}."
            )

        return F.crop(img, top, left, self.crop_size, self.crop_size)


def get_basic_transform() -> transforms.Compose:
    """Return the baseline preprocessing transform used in training."""
    return transforms.Compose([
        transforms.Resize((336, 224)),
        FixedTopCrop(crop_size=224, top_offset=20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
