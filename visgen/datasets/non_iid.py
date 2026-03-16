from collections import defaultdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def subset_with_four_case_support(
    dataset: Dataset,
    allowed_attributes: Optional[Sequence[str]] = None,
    shared_other_attributes: bool = True,
) -> Subset:
    """Return a subset containing samples that can participate in a 4-case quadrant.

    A sample is kept iff there exists at least one compositional 2x2 pattern using two
    attributes where this sample is one of the four corners.
    """
    analyzer = _FourCaseSupportAnalyzer(dataset, allowed_attributes)
    valid_mask = analyzer.valid_sample_mask(
        shared_other_attributes=shared_other_attributes
    )
    indices = np.flatnonzero(valid_mask).tolist()
    return Subset(dataset, indices)


class _FourCaseSupportAnalyzer:
    """Analyze whether samples can form 4-case compositional quadrants."""

    def __init__(
        self,
        dataset: Dataset,
        allowed_attributes: Optional[Sequence[str]] = None,
    ) -> None:
        self.dataset = dataset
        self.targets, self.attribute_values, self.attribute_indices = (
            self._prepare(dataset, allowed_attributes)
        )
        self.num_attributes = self.targets.shape[1]

    def _prepare(self, dataset, allowed_attributes):
        targets, attribute_values = NonIIDWrapper._prepare_targets(self, dataset)
        # Reuse NonIIDWrapper attribute resolution logic, which expects these names.
        self._targets = targets
        self._attribute_values = attribute_values
        if allowed_attributes is None:
            attribute_indices = np.array(
                [
                    idx
                    for idx in range(targets.shape[1])
                    if len(attribute_values[idx]) >= 2
                ],
                dtype=int,
            )
        else:
            attribute_indices = NonIIDWrapper._resolve_attribute_indices(
                self, allowed_attributes
            )
        return targets, attribute_values, attribute_indices

    @staticmethod
    def _unwrap_subset(dataset: Dataset) -> Tuple[Dataset, Optional[np.ndarray]]:
        return NonIIDWrapper._unwrap_subset(dataset)

    def valid_sample_mask(self, shared_other_attributes: bool) -> np.ndarray:
        if len(self.attribute_indices) < 2:
            return np.zeros(len(self.targets), dtype=bool)
        valid = np.zeros(len(self.targets), dtype=bool)
        for i, attr_a in enumerate(self.attribute_indices):
            for attr_b in self.attribute_indices[i + 1 :]:
                if shared_other_attributes:
                    valid |= self._mask_for_attr_pair_shared(attr_a, attr_b)
                else:
                    valid |= self._mask_for_attr_pair_unshared(attr_a, attr_b)
        return valid

    def _mask_for_attr_pair_shared(self, attr_a: int, attr_b: int) -> np.ndarray:
        mask = np.zeros(len(self.targets), dtype=bool)
        other_indices = [
            idx for idx in range(self.num_attributes) if idx not in (attr_a, attr_b)
        ]
        groups = defaultdict(list)
        for idx, row in enumerate(self.targets):
            other_key = tuple(row[other_indices].tolist())
            groups[other_key].append(idx)
        for indices in groups.values():
            edge_to_indices = defaultdict(list)
            a_to_bs = defaultdict(set)
            b_to_as = defaultdict(set)
            for idx in indices:
                row = self.targets[idx]
                edge = (row[attr_a], row[attr_b])
                edge_to_indices[edge].append(idx)
                a_to_bs[edge[0]].add(edge[1])
                b_to_as[edge[1]].add(edge[0])
            valid_edges = _edges_with_rectangle(edge_to_indices, a_to_bs, b_to_as)
            for edge in valid_edges:
                mask[edge_to_indices[edge]] = True
        return mask

    def _mask_for_attr_pair_unshared(self, attr_a: int, attr_b: int) -> np.ndarray:
        mask = np.zeros(len(self.targets), dtype=bool)
        edge_to_indices = defaultdict(list)
        a_to_bs = defaultdict(set)
        b_to_as = defaultdict(set)
        for idx, row in enumerate(self.targets):
            edge = (row[attr_a], row[attr_b])
            edge_to_indices[edge].append(idx)
            a_to_bs[edge[0]].add(edge[1])
            b_to_as[edge[1]].add(edge[0])
        valid_edges = _edges_with_rectangle(edge_to_indices, a_to_bs, b_to_as)
        for edge in valid_edges:
            mask[edge_to_indices[edge]] = True
        return mask


def _edges_with_rectangle(edge_to_indices, a_to_bs, b_to_as):
    valid_edges = set()
    edge_set = set(edge_to_indices.keys())
    for a, b in edge_set:
        alt_as = b_to_as[b] - {a}
        alt_bs = a_to_bs[a] - {b}
        found = False
        for alt_a in alt_as:
            for alt_b in alt_bs:
                if (
                    (a, alt_b) in edge_set
                    and (alt_a, b) in edge_set
                    and (alt_a, alt_b) in edge_set
                ):
                    found = True
                    break
            if found:
                break
        if found:
            valid_edges.add((a, b))
    return valid_edges


class NonIIDWrapper(Dataset):
    """Wrap a dataset to produce non-iid 4-sample batches.

    The wrapper samples two attributes A and B, two values for each of them
    (a, b) for A and (c, d) for B, and a random vector x for the remaining
    attributes. It returns the four samples matching:
    (A=a, B=c), (A=a, B=d), (A=b, B=c), (A=b, B=d).

    Notes:
        This wrapper assumes each sample has a single object (i.e., targets
        are shaped as (N, 1, num_attributes) or (N, num_attributes)).
        __len__ reports the number of 4-sample groups to align epoch lengths.
    """
    group_size = 4

    def __init__(
        self,
        dataset: Dataset,
        max_resample_attempts: int = 10_000,
        seed: Optional[int] = None,
        allowed_attributes: Optional[Sequence[str]] = None,
        shared_other_attributes: bool = True,
        fully_iid: bool = False,
    ) -> None:
        self.dataset = dataset
        self.max_resample_attempts = max_resample_attempts
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.shared_other_attributes = shared_other_attributes
        self.fully_iid = fully_iid
        self._targets, self._attribute_values = self._prepare_targets(dataset)
        self._index_by_target = self._build_index(self._targets)
        if self.fully_iid:
            self._attribute_indices = np.array([], dtype=int)
        else:
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
        return len(self.dataset) // self.group_size

    def __getitem__(self, index: int):
        rng = self._rng_for_index(index)
        if self.fully_iid:
            sampled_indices = rng.choice(
                len(self.dataset), size=self.group_size, replace=True
            )
            images, targets = zip(
                *(self.dataset[int(idx)] for idx in sampled_indices)
            )
            images = self._stack_images(images)
            targets = torch.as_tensor(np.stack(targets))
            return images, targets
        for _ in range(self.max_resample_attempts):
            attr_a, attr_b = rng.choice(
                self._attribute_indices, size=2, replace=False
            )
            desired_targets = self._sample_valid_quadrant_targets(
                attr_a, attr_b, rng
            )
            if desired_targets is None:
                continue
            indices = self._select_indices(desired_targets, rng)
            if indices is None:
                continue
            images, targets = zip(*(self.dataset[idx] for idx in indices))
            images = self._stack_images(images)
            targets = torch.as_tensor(np.stack(targets))
            return images, targets
        raise RuntimeError(
            "Failed to sample a valid non-iid batch within the resample limit."
        )

    def _rng_for_index(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return self.rng
        return np.random.default_rng(self.seed + int(index))

    def _sample_valid_quadrant_targets(
        self,
        attr_a: int,
        attr_b: int,
        rng: np.random.Generator,
    ) -> Optional[List[Tuple[int, ...]]]:
        if self.shared_other_attributes:
            return self._sample_quadrant_targets_shared(attr_a, attr_b, rng)
        return self._sample_quadrant_targets_unshared(attr_a, attr_b, rng)

    def _sample_quadrant_targets_shared(
        self,
        attr_a: int,
        attr_b: int,
        rng: np.random.Generator,
    ) -> Optional[List[Tuple[int, ...]]]:
        other_indices = [
            idx for idx in range(self._targets.shape[1]) if idx not in (attr_a, attr_b)
        ]
        grouped_edges = defaultdict(lambda: defaultdict(list))
        for row in self._targets:
            target = tuple(row.tolist())
            other_key = tuple(row[other_indices].tolist())
            edge = (row[attr_a], row[attr_b])
            grouped_edges[other_key][edge].append(target)

        contexts = list(grouped_edges.keys())
        if not contexts:
            return None
        for context_idx in rng.permutation(len(contexts)):
            edge_to_targets = grouped_edges[contexts[int(context_idx)]]
            rectangle = self._sample_rectangle_from_edges(edge_to_targets, rng)
            if rectangle is None:
                continue
            a0, a1, b0, b1 = rectangle
            return [
                tuple(rng.choice(edge_to_targets[(a0, b0)]).tolist()),
                tuple(rng.choice(edge_to_targets[(a0, b1)]).tolist()),
                tuple(rng.choice(edge_to_targets[(a1, b0)]).tolist()),
                tuple(rng.choice(edge_to_targets[(a1, b1)]).tolist()),
            ]
        return None

    def _sample_quadrant_targets_unshared(
        self,
        attr_a: int,
        attr_b: int,
        rng: np.random.Generator,
    ) -> Optional[List[Tuple[int, ...]]]:
        edge_to_targets = defaultdict(list)
        for row in self._targets:
            target = tuple(row.tolist())
            edge = (row[attr_a], row[attr_b])
            edge_to_targets[edge].append(target)
        rectangle = self._sample_rectangle_from_edges(edge_to_targets, rng)
        if rectangle is None:
            return None
        a0, a1, b0, b1 = rectangle
        return [
            tuple(rng.choice(edge_to_targets[(a0, b0)]).tolist()),
            tuple(rng.choice(edge_to_targets[(a0, b1)]).tolist()),
            tuple(rng.choice(edge_to_targets[(a1, b0)]).tolist()),
            tuple(rng.choice(edge_to_targets[(a1, b1)]).tolist()),
        ]

    def _sample_rectangle_from_edges(
        self,
        edge_to_targets,
        rng: np.random.Generator,
    ) -> Optional[Tuple[int, int, int, int]]:
        a_to_bs = defaultdict(set)
        b_to_as = defaultdict(set)
        edge_set = set(edge_to_targets.keys())
        if len(edge_set) < 4:
            return None
        for a, b in edge_set:
            a_to_bs[a].add(b)
            b_to_as[b].add(a)

        edges = list(edge_set)
        for edge_idx in rng.permutation(len(edges)):
            a0, b0 = edges[int(edge_idx)]
            alt_as = list(b_to_as[b0] - {a0})
            alt_bs = list(a_to_bs[a0] - {b0})
            if not alt_as or not alt_bs:
                continue
            for alt_a_idx in rng.permutation(len(alt_as)):
                a1 = alt_as[int(alt_a_idx)]
                for alt_b_idx in rng.permutation(len(alt_bs)):
                    b1 = alt_bs[int(alt_b_idx)]
                    if (
                        (a0, b1) in edge_set
                        and (a1, b0) in edge_set
                        and (a1, b1) in edge_set
                    ):
                        return a0, a1, b0, b1
        return None

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
        resolved = []
        for attr in allowed_attributes:
            idx = base_dataset._attribute_to_index(attr)
            if hasattr(base_dataset, "_target_index_map"):
                idx = base_dataset._target_index_map.get(idx)
                if idx is None:
                    continue
            resolved.append(idx)
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

    def _select_indices(
        self,
        targets: Sequence[Tuple[int, ...]],
        rng: np.random.Generator,
    ) -> Optional[List]:
        indices = []
        for target in targets:
            options = self._index_by_target.get(target)
            if not options:
                return None
            indices.append(rng.choice(options))
        return indices

    def _stack_images(self, images: Sequence) -> torch.Tensor:
        if torch.is_tensor(images[0]):
            return torch.stack(images, dim=0)
        return torch.as_tensor(np.stack(images))
