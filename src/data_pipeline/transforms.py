"""Image transforms for the MEEI facial palsy pipeline."""

from __future__ import annotations

import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def get_train_transform(use_augmentation: bool = False) -> transforms.Compose:
    steps = []

    # Keep augmentation modest and opt-in so early baselines stay clinically conservative.
    if use_augmentation:
        steps.extend(
            [
                transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02, hue=0.01),
            ]
        )

    steps.extend(
        [
            transforms.ToTensor(),
            _normalize(),
        ]
    )
    return transforms.Compose(steps)


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            _normalize(),
        ]
    )


def get_basic_transform() -> transforms.Compose:
    return get_eval_transform()
