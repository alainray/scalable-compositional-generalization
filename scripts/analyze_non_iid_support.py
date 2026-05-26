#!/usr/bin/env python3
"""Estimate how many samples can participate in non-iid 4-corner quadrants.

Usage:
  python scripts/analyze_non_iid_support.py --dataset clevr --config configs/datasets/clevr_non_iid.yml
"""
import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

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

    total = len(ds)
    support_subset = subset_with_four_case_support(
        ds,
        allowed_attributes=None,
        shared_other_attributes=args.shared_other_attributes,
    )
    supported = len(support_subset)
    unsupported = total - supported

    print("=== non-iid 4-corner support analysis ===")
    print(f"dataset: {args.dataset}")
    print(f"config:  {Path(args.config)}")
    print(f"total_samples:        {total}")
    print(f"supported_samples:    {supported}")
    print(f"unsupported_samples:  {unsupported}")
    print(f"support_ratio:        {supported / total:.6f}")
    print(f"unsupported_ratio:    {unsupported / total:.6f}")


if __name__ == "__main__":
    main()
