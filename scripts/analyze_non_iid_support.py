#!/usr/bin/env python3
"""Estimate how many samples can participate in non-iid 4-corner quadrants.

Usage:
  python scripts/analyze_non_iid_support.py --dataset clevr --config configs/datasets/clevr_non_iid.yml
"""
import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import random_split

from visgen.datasets import CLEVR, Shapes3D, IRAVEN
from visgen.datasets.non_iid import subset_with_four_case_support

DATASET_MAP = {
    "clevr": CLEVR,
    "shapes3d": Shapes3D,
    "iraven": IRAVEN,
}


def _build_training_dataset(cfg):
    cls = DATASET_MAP[cfg.dataset]
    return cls(**cfg)


def _supports_ood_validation_split(dataset) -> bool:
    return hasattr(dataset, "_included_combinations") and hasattr(
        dataset, "_split_attribute_indices"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASET_MAP), required=True)
    p.add_argument("--config", required=True, help="Path to dataset yml (e.g. *_non_iid.yml)")
    p.add_argument("--shared_other_attributes", action="store_true", default=False)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    train_cfg = cfg.data.training
    if train_cfg.dataset != args.dataset:
        raise ValueError(f"Dataset mismatch: config has {train_cfg.dataset}, arg has {args.dataset}")

    ds = _build_training_dataset(train_cfg)

    # Mirror the real train/validation split used by get_dataloaders so this
    # script measures support on the actual training subset (excluding val).
    base_seed = int(cfg.seed) if "seed" in cfg and cfg.seed is not None else None
    split_generator = (
        None if base_seed is None else torch.Generator().manual_seed(base_seed)
    )
    num_ood_val = train_cfg.num_ood_val if "num_ood_val" in train_cfg else 1
    train_data, _ = ds.ood_validation_split(num_ood_val)
    val_size = int(train_cfg.val_fraction * len(train_data))
    train_size = len(train_data) - val_size
    train_data, _ = random_split(
        train_data, [train_size, val_size], generator=split_generator
    )

    total = len(train_data)
    support_subset = subset_with_four_case_support(
        train_data,
        allowed_attributes=None,
        shared_other_attributes=args.shared_other_attributes,
    )
    supported = len(support_subset)
    unsupported = total - supported

    print("=== non-iid 4-corner support analysis ===")
    print(f"dataset: {args.dataset}")
    print(f"config:  {Path(args.config)}")
    print("analyzed_split: train (post ood_validation_split + val_fraction split)")
    print(f"total_samples:        {total}")
    print(f"supported_samples:    {supported}")
    print(f"unsupported_samples:  {unsupported}")
    print(f"support_ratio:        {supported / total:.6f}")
    print(f"unsupported_ratio:    {unsupported / total:.6f}")


if __name__ == "__main__":
    main()
