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


def test_unpredictable_target_level_one_breaks_only_predicted_second_attribute():
    dataset = ToyDataset(attribute_cardinalities=(4, 4, 3))
    wrapper = NonIIDWrapper(
        dataset,
        seed=3,
        sampling_mode="unpredictable_target",
        num_unpredictable_attributes=1,
    )

    _, targets = wrapper[0]
    rows = _target_rows(targets)
    first_three = rows[:3]
    target = rows[3]

    varying_attrs = [
        idx for idx in range(rows.shape[1]) if len(set(first_three[:, idx])) == 2
    ]
    shared_attrs = [
        idx for idx in range(rows.shape[1]) if len(set(first_three[:, idx])) == 1
    ]
    assert len(varying_attrs) == 2

    predicted_second_attr = [
        idx for idx in varying_attrs if first_three[0, idx] != first_three[1, idx]
    ]
    broken_attrs = [
        idx for idx in varying_attrs if target[idx] not in set(first_three[:, idx])
    ]
    inferred_attrs = [
        idx for idx in varying_attrs if target[idx] in set(first_three[:, idx])
    ]

    assert len(predicted_second_attr) == 1
    assert broken_attrs == predicted_second_attr
    assert len(inferred_attrs) == 1
    for attr_idx in shared_attrs:
        assert target[attr_idx] == first_three[0, attr_idx]


def test_unpredictable_target_level_two_breaks_both_quadrant_attributes():
    dataset = ToyDataset(attribute_cardinalities=(4, 4, 3))
    wrapper = NonIIDWrapper(
        dataset,
        seed=5,
        sampling_mode="unpredictable_target",
        num_unpredictable_attributes=2,
    )

    _, targets = wrapper[0]
    rows = _target_rows(targets)
    first_three = rows[:3]
    target = rows[3]

    varying_attrs = [
        idx for idx in range(rows.shape[1]) if len(set(first_three[:, idx])) == 2
    ]
    shared_attrs = [
        idx for idx in range(rows.shape[1]) if len(set(first_three[:, idx])) == 1
    ]
    assert len(varying_attrs) == 2

    for attr_idx in varying_attrs:
        assert target[attr_idx] not in set(first_three[:, attr_idx])
    for attr_idx in shared_attrs:
        assert target[attr_idx] == first_three[0, attr_idx]


def test_unpredictable_target_deterministic_sampling_is_reproducible_and_limited():
    dataset = ToyDataset(attribute_cardinalities=(4, 4, 3))
    wrapper = NonIIDWrapper(
        dataset,
        sampling_mode="unpredictable_target",
        num_unpredictable_attributes=2,
        deterministic=True,
        precompute_deterministic=True,
        max_deterministic_candidates=7,
    )

    assert len(wrapper) <= 7
    _, first_targets = wrapper[0]
    _, second_targets = wrapper[0]

    assert torch.equal(first_targets, second_targets)
    rows = _target_rows(first_targets)
    first_three = rows[:3]
    varying_attrs = [
        idx for idx in range(rows.shape[1]) if len(set(first_three[:, idx])) == 2
    ]
    for attr_idx in varying_attrs:
        assert rows[3, attr_idx] not in set(first_three[:, attr_idx])


def test_unpredictable_target_rejects_invalid_level():
    dataset = ToyDataset()

    with pytest.raises(ValueError, match="num_unpredictable_attributes"):
        NonIIDWrapper(
            dataset,
            sampling_mode="unpredictable_target",
            num_unpredictable_attributes=3,
        )


def test_random_deterministic_sampling_has_base_and_attribute_pair_diversity():
    dataset = ToyDataset(attribute_cardinalities=(6, 6, 6))
    wrapper = NonIIDWrapper(
        dataset,
        sampling_mode="unpredictable_target",
        num_unpredictable_attributes=1,
        deterministic=True,
        precompute_deterministic=True,
        max_deterministic_candidates=64,
        seed=123,
    )

    targets = [wrapper[idx][1] for idx in range(len(wrapper))]
    base_targets = {tuple(_target_rows(target)[0]) for target in targets}
    assert len(base_targets) > 1

    changed_attr_pairs = set()
    for target in targets:
        rows = _target_rows(target)
        first_three = rows[:3]
        attr_a = tuple(
            idx
            for idx in range(rows.shape[1])
            if first_three[0, idx] != first_three[2, idx]
        )
        attr_b = tuple(
            idx
            for idx in range(rows.shape[1])
            if first_three[0, idx] != first_three[1, idx]
        )
        changed_attr_pairs.add((attr_a, attr_b))

    assert len(changed_attr_pairs) > 1


def test_random_deterministic_sampling_is_reproducible_with_same_seed():
    dataset = ToyDataset(attribute_cardinalities=(6, 6, 6))
    w1 = NonIIDWrapper(
        dataset,
        sampling_mode="unpredictable_target",
        num_unpredictable_attributes=2,
        deterministic=True,
        precompute_deterministic=True,
        max_deterministic_candidates=32,
        seed=123,
    )
    w2 = NonIIDWrapper(
        dataset,
        sampling_mode="unpredictable_target",
        num_unpredictable_attributes=2,
        deterministic=True,
        precompute_deterministic=True,
        max_deterministic_candidates=32,
        seed=123,
    )

    assert len(w1) == len(w2)
    for idx in range(len(w1)):
        assert torch.equal(w1[idx][1], w2[idx][1])


def test_random_deterministic_sampling_differs_with_different_seed():
    dataset = ToyDataset(attribute_cardinalities=(6, 6, 6))
    w1 = NonIIDWrapper(
        dataset,
        sampling_mode="unpredictable_target",
        num_unpredictable_attributes=2,
        deterministic=True,
        precompute_deterministic=True,
        max_deterministic_candidates=32,
        seed=123,
    )
    w2 = NonIIDWrapper(
        dataset,
        sampling_mode="unpredictable_target",
        num_unpredictable_attributes=2,
        deterministic=True,
        precompute_deterministic=True,
        max_deterministic_candidates=32,
        seed=124,
    )

    assert any(
        not torch.equal(w1[idx][1], w2[idx][1])
        for idx in range(min(len(w1), len(w2)))
    )


def test_deterministic_sampling_enumerate_strategy_remains_available():
    dataset = ToyDataset()
    wrapper = NonIIDWrapper(
        dataset,
        deterministic=True,
        precompute_deterministic=True,
        deterministic_sampling_strategy="enumerate",
    )

    assert len(wrapper) > 0


def test_deterministic_sampling_rejects_invalid_strategy():
    dataset = ToyDataset()

    with pytest.raises(ValueError, match="deterministic_sampling_strategy"):
        NonIIDWrapper(dataset, deterministic_sampling_strategy="bad_strategy")
