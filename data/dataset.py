import os
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config.config import (
    DATASET_ROOT, IMAGE_SIZE, CLASSES, CLASS_TO_IDX,
    AUG_HFLIP_P, AUG_VFLIP_P, AUG_ROTATION_DEG,
    AUG_COLOR_JITTER, AUG_NORMALIZE_MEAN, AUG_NORMALIZE_STD,
    BATCH_SIZE, SEED
)


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns the transform pipeline for a given split.
    'train' gets augmentation. 'val' and 'test' get only resize + normalize.
    Augmentation is applied only at training time to avoid data leakage into
    evaluation — if you augment val/test, your metrics no longer reflect
    real-world performance on clean images.
    """
    normalize = transforms.Normalize(
        mean=AUG_NORMALIZE_MEAN,
        std=AUG_NORMALIZE_STD
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=AUG_HFLIP_P),
            transforms.RandomVerticalFlip(p=AUG_VFLIP_P),
            transforms.RandomRotation(degrees=AUG_ROTATION_DEG),
            transforms.ColorJitter(
                brightness=AUG_COLOR_JITTER,
                contrast=AUG_COLOR_JITTER,
                saturation=AUG_COLOR_JITTER / 2,
                hue=0.0   # do not shift hue — H&E colors are diagnostically meaningful
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ])


class OralCancerDataset(Dataset):
    """
    Loads images from the train/val/test folder structure.
    Filters by magnification tag in the filename (e.g. '_100x_' or '_400x_').
    Skips augmented files (prefix 'aug_') because their magnification is unknown.

    Args:
        split      : "train", "val", or "test"
        magnification : "100x", "400x", or None (None = use all original images)
        transform  : torchvision transform pipeline
    """

    def __init__(self, split: str, magnification: str = None,
                 transform=None):
        self.transform = transform
        self.samples = []   # list of (path, label_idx)

        split_dir = DATASET_ROOT / split

        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected directory not found: {class_dir}")

            label = CLASS_TO_IDX[class_name]

            for img_path in sorted(class_dir.iterdir()):
                if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue

                # Skip augmented files — magnification unknown
                if img_path.name.startswith("aug_"):
                    continue

                # Filter by magnification if specified
                if magnification is not None:
                    if f"_{magnification}_" not in img_path.name:
                        continue

                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found for split='{split}', magnification='{magnification}'. "
                f"Check DATASET_ROOT and filename conventions."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict:
        """Returns {class_name: count} for this dataset instance."""
        counts = {cls: 0 for cls in CLASSES}
        for _, label in self.samples:
            class_name = CLASSES[label]
            counts[class_name] += 1
        return counts


def get_weighted_sampler(dataset: OralCancerDataset) -> WeightedRandomSampler:
    """
    Builds a WeightedRandomSampler so that each training batch is
    approximately class-balanced regardless of the true class ratio.

    How it works: each sample gets a weight = 1 / (count of its class).
    The sampler draws samples proportional to these weights, so the rare
    class (Normal) gets sampled more often than its raw frequency.
    """
    labels = [label for _, label in dataset.samples]
    class_counts = torch.zeros(len(CLASSES))
    for label in labels:
        class_counts[label] += 1

    # Weight per class = inverse frequency
    class_weights = 1.0 / class_counts
    # Assign each sample its class weight
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True   # sampling with replacement is standard for oversampling
    )
    return sampler


def get_class_weights_for_loss(dataset: OralCancerDataset) -> torch.Tensor:
    """
    Returns a weight tensor for nn.CrossEntropyLoss(weight=...).
    Computed as inverse normalized class frequency.
    This makes the loss penalize mistakes on the minority class more heavily.
    """
    labels = [label for _, label in dataset.samples]
    class_counts = torch.zeros(len(CLASSES))
    for label in labels:
        class_counts[label] += 1

    total = class_counts.sum()
    # Normalize so weights sum to num_classes (keeps loss scale stable)
    class_weights = total / (len(CLASSES) * class_counts)
    return class_weights


def get_dataloaders(magnification: str) -> dict:
    """
    Returns {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    for a given magnification level.

    Call this once per experiment:
        loaders = get_dataloaders("100x")
        loaders = get_dataloaders("400x")
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        tf = get_transforms(split)
        datasets[split] = OralCancerDataset(
            split=split,
            magnification=magnification,
            transform=tf
        )

    # Sampler only for training
    train_sampler = get_weighted_sampler(datasets["train"])

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            sampler=train_sampler,       # replaces shuffle=True
            num_workers=2,
            pin_memory=True
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        ),
    }

    # Print summary so you always see what went into training
    print(f"\nDataloaders ready for magnification: {magnification}")
    for split, ds in datasets.items():
        counts = ds.class_counts()
        print(f"  {split}: {counts} | total: {len(ds)}")

    loss_weights = get_class_weights_for_loss(datasets["train"])
    print(f"  Loss weights → Normal: {loss_weights[0]:.4f}, OSCC: {loss_weights[1]:.4f}")

    return loaders, get_class_weights_for_loss(datasets["train"])