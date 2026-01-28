from collections import defaultdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class NonIIDWrapper(Dataset):
    """Wrap a dataset to produce non-iid 4-sample batches.

    The wrapper samples two attributes A and B, two values for each of them
    (a, b) for A and (c, d) for B, and a random vector x for the remaining
    attributes. It returns the four samples matching:
    (A=a, B=c), (A=a, B=d), (A=b, B=c), (A=b, B=d).

    Notes:
        This wrapper assumes each sample has a single object (i.e., targets
        are shaped as (N, 1, num_attributes) or (N, num_attributes)).
    """

    def __init__(
        self,
        dataset: Dataset,
        max_resample_attempts: int = 10_000,
        seed: Optional[int] = None,
        allowed_attributes: Optional[Sequence[str]] = None,
        shared_other_attributes: bool = True,
    ) -> None:
        self.dataset = dataset
        self.max_resample_attempts = max_resample_attempts
        self.rng = np.random.default_rng(seed)
        self.shared_other_attributes = shared_other_attributes
        self._targets, self._attribute_values = self._prepare_targets(dataset)
        self._index_by_target = self._build_index(self._targets)
        self._attribute_indices = self._resolve_attribute_indices(
            allowed_attributes
        )

    @staticmethod
    def _unwrap_subset(dataset: Dataset) -> Tuple[Dataset, Optional[np.ndarray]]:
        base_dataset = dataset
        indices = None
        while isinstance(base_dataset, Subset):
            if indices is None:
                indices = np.asarray(base_dataset.indices)
            else:
                indices = np.asarray(base_dataset.indices)[indices]
            base_dataset = base_dataset.dataset
        return base_dataset, indices

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        del index
        for _ in range(self.max_resample_attempts):
            attr_a, attr_b = self.rng.choice(
                self._attribute_indices, size=2, replace=False
            )
            value_a, value_b = self._sample_two(self._attribute_values[attr_a])
            value_c, value_d = self._sample_two(self._attribute_values[attr_b])
            other_values = self._sample_other_attributes((attr_a, attr_b))
            desired_targets = self._build_target_quadrant(
                attr_a,
                attr_b,
                value_a,
                value_b,
                value_c,
                value_d,
                other_values,
            )
            indices = self._select_indices(desired_targets)
            if indices is None:
                continue
            images, targets = zip(*(self.dataset[idx] for idx in indices))
            images = self._stack_images(images)
            targets = torch.as_tensor(np.stack(targets))
            return images, targets
        raise RuntimeError(
            "Failed to sample a valid non-iid batch within the resample limit."
        )

    def _prepare_targets(self, dataset: Dataset) -> Tuple[np.ndarray, List[List]]:
        base_dataset, indices = self._unwrap_subset(dataset)
        if indices is not None:
            targets = base_dataset._dataset_targets[indices]
        else:
            targets = base_dataset._dataset_targets
        if targets.ndim == 3:
            if targets.shape[1] != 1:
                raise ValueError(
                    "NonIIDWrapper supports only single-object datasets."
                )
            targets = targets[:, 0, :]
        elif targets.ndim != 2:
            raise ValueError(
                "NonIIDWrapper supports targets shaped as (N, M) or (N, 1, M)."
            )
        if getattr(base_dataset, "_attribute_values", None) is None:
            attribute_values = [
                np.unique(targets[:, i]).tolist() for i in range(targets.shape[1])
            ]
        else:
            attribute_values = base_dataset._attribute_values
        return targets, attribute_values

    def _build_index(self, targets: np.ndarray) -> dict:
        index = defaultdict(list)
        for idx, row in enumerate(targets):
            index[tuple(row.tolist())].append(idx)
        return index

    def _resolve_attribute_indices(
        self, allowed_attributes: Optional[Sequence[str]]
    ) -> np.ndarray:
        num_attributes = self._targets.shape[1]
        if allowed_attributes is None:
            eligible = [
                idx
                for idx in range(num_attributes)
                if len(self._attribute_values[idx]) >= 2
            ]
            if len(eligible) < 2:
                raise ValueError(
                    "NonIIDWrapper requires at least two attributes with at least two "
                    "distinct values to sample non-iid batches."
                )
            return np.array(eligible)
        base_dataset, _ = self._unwrap_subset(self.dataset)
        resolved = [
            base_dataset._attribute_to_index(attr) for attr in allowed_attributes
        ]
        eligible = [
            idx
            for idx in resolved
            if len(self._attribute_values[idx]) >= 2
        ]
        if len(eligible) < 2:
            raise ValueError(
                "NonIIDWrapper requires at least two attributes to sample non-iid "
                "batches."
            )
        return np.array(eligible)

    def _sample_two(self, values: Iterable) -> Tuple[int, int]:
        if len(values) < 2:
            raise ValueError(
                "NonIIDWrapper requires at least two distinct values to sample."
            )
        choices = self.rng.choice(values, size=2, replace=False)
        return choices[0], choices[1]

    def _sample_other_attributes(self, fixed_indices: Tuple[int, int]) -> dict:
        other_values = {}
        for idx, values in enumerate(self._attribute_values):
            if idx in fixed_indices:
                continue
            other_values[idx] = self.rng.choice(values)
        return other_values

    def _build_target_quadrant(
        self,
        attr_a: int,
        attr_b: int,
        value_a: int,
        value_b: int,
        value_c: int,
        value_d: int,
        other_values: dict,
    ) -> List[Tuple[int, ...]]:
        def build_target(val_a, val_b, other_vals):
            target = []
            for idx in range(self._targets.shape[1]):
                if idx == attr_a:
                    target.append(val_a)
                elif idx == attr_b:
                    target.append(val_b)
                else:
                    target.append(other_vals[idx])
            return tuple(target)

        if not self.shared_other_attributes:
            other_values_list = [
                self._sample_other_attributes((attr_a, attr_b)) for _ in range(4)
            ]
        else:
            other_values_list = [other_values] * 4

        return [
            build_target(value_a, value_c, other_values_list[0]),
            build_target(value_a, value_d, other_values_list[1]),
            build_target(value_b, value_c, other_values_list[2]),
            build_target(value_b, value_d, other_values_list[3]),
        ]

    def _select_indices(self, targets: Sequence[Tuple[int, ...]]) -> Optional[List]:
        indices = []
        for target in targets:
            options = self._index_by_target.get(target)
            if not options:
                return None
            indices.append(self.rng.choice(options))
        return indices

    def _stack_images(self, images: Sequence) -> torch.Tensor:
        if torch.is_tensor(images[0]):
            return torch.stack(images, dim=0)
        return torch.as_tensor(np.stack(images))
