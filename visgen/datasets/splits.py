from typing import Tuple

import torch


def training_subset_with_optional_ood_holdout(dataset, num_ood_val: int = 1):
    """Return the training subset used before building an in-domain validation split.

    If the dataset exposes OOD split bookkeeping, reserve OOD combinations through
    ``ood_validation_split`` and return its training subset. Otherwise, return the
    original dataset unchanged.
    """
    if hasattr(dataset, "_included_combinations"):
        train_data, _ = dataset.ood_validation_split(num_ood_val)
        return train_data
    return dataset


def make_validation_subset(
    dataset,
    val_fraction: float,
    seed: int = 0,
    num_ood_val: int = 1,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Split dataset into train/validation with optional OOD holdout support."""
    train_data = training_subset_with_optional_ood_holdout(
        dataset, num_ood_val=num_ood_val
    )
    val_size = int(val_fraction * len(train_data))
    train_size = len(train_data) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        train_data, [train_size, val_size], generator=generator
    )
    return train_subset, val_subset
