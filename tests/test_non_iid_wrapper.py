import itertools

import numpy as np
import pytest
import torch

from visgen.datasets.non_iid import NonIIDWrapper


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, attribute_cardinalities=(5, 5, 5)):
        targets = list(itertools.product(*[range(n) for n in attribute_cardinalities]))
        self._dataset_targets = np.asarray(targets)
        self._attribute_values = [list(range(n)) for n in attribute_cardinalities]

    def __len__(self):
        return len(self._dataset_targets)

    def __getitem__(self, index):
        target = self._dataset_targets[index]
        image = torch.tensor([float(index)])
        return image, target


def _target_rows(batch_targets):
    return batch_targets.numpy().reshape(4, -1)


def test_adversarial_sampling_uses_three_non_iid_corners_and_disjoint_fourth():
    dataset = ToyDataset()
    wrapper = NonIIDWrapper(dataset, seed=0, sampling_mode="adversarial")

    _, targets = wrapper[0]
    rows = _target_rows(targets)
    first_three = rows[:3]
    adversarial = rows[3]

    # The first three samples keep the non-iid quadrant prefix ordering:
    # samples 0/1 share one selected attribute, samples 0/2 share the other,
    # and all non-selected attributes are shared by the first three samples.
    per_attr_unique = [len(set(first_three[:, idx])) for idx in range(first_three.shape[1])]
    assert sorted(per_attr_unique) == [1, 2, 2]

    for attr_idx in range(rows.shape[1]):
        assert adversarial[attr_idx] not in set(first_three[:, attr_idx])


def test_adversarial_deterministic_sampling_is_reproducible():
    dataset = ToyDataset()
    wrapper = NonIIDWrapper(
        dataset,
        sampling_mode="adversarial",
        deterministic=True,
        precompute_deterministic=True,
    )

    _, first_targets = wrapper[0]
    _, second_targets = wrapper[0]

    assert torch.equal(first_targets, second_targets)
    rows = _target_rows(first_targets)
    for attr_idx in range(rows.shape[1]):
        assert rows[3, attr_idx] not in set(rows[:3, attr_idx])


def test_legacy_fully_iid_flag_still_selects_iid_mode():
    dataset = ToyDataset()
    wrapper = NonIIDWrapper(dataset, seed=0, fully_iid=True)

    images, targets = wrapper[0]

    assert wrapper.sampling_mode == "iid"
    assert images.shape[0] == 4
    assert targets.shape == (4, 3)


def test_conflicting_sampling_mode_and_legacy_fully_iid_flag_fails():
    dataset = ToyDataset()

    with pytest.raises(ValueError, match="fully_iid=True"):
        NonIIDWrapper(dataset, fully_iid=True, sampling_mode="adversarial")
