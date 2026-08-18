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


# --- muestreo adversarial constructivo (camino rapido) ---------------------
#
# Con el split ortotopico la pertenencia al soporte es contable
# (#{i : z_i >= umbral_i} <= c), asi que el wrapper construye el cuadrilatero y
# la cuarta etiqueta en vez de sortear a ciegas. Lo que sigue fija el contrato
# de ese camino: las cuatro etiquetas existen en train, la esquina esperada
# tambien, y cada modo viola exactamente los atributos que le tocan.

ORTHOTOPIC_CARDINALITIES = (6, 6, 5, 4)
ORTHOTOPIC_THRESHOLDS = (4, 4, 3, 3)


class OrthotopicToyDataset(ToyDataset):
    """Producto cartesiano recortado al soporte ortotopico ``n_hard <= c``."""

    def __init__(self, cardinalities=ORTHOTOPIC_CARDINALITIES,
                 thresholds=ORTHOTOPIC_THRESHOLDS, c=1):
        super().__init__(attribute_cardinalities=cardinalities)
        thresholds = np.asarray(thresholds)
        n_hard = (self._dataset_targets >= thresholds).sum(axis=1)
        self._dataset_targets = self._dataset_targets[n_hard <= c]
        self.thresholds = thresholds
        self.c = c


def _orthotopic_wrapper(dataset, seed=0, **kwargs):
    return NonIIDWrapper(
        dataset,
        seed=seed,
        split_thresholds=dataset.thresholds,
        split_c=dataset.c,
        **kwargs,
    )


ADVERSARIAL_MODES = [
    ({"sampling_mode": "adversarial"}, None),
    ({"sampling_mode": "unpredictable_target",
      "num_unpredictable_attributes": 2}, 2),
    ({"sampling_mode": "unpredictable_target",
      "num_unpredictable_attributes": 1}, 1),
]


def _expected_corner(first_three):
    """Cuarta esquina que completa el rectangulo formado por las otras tres."""
    expected = list(first_three[0])
    for attr_idx in range(first_three.shape[1]):
        values = list(first_three[:, attr_idx])
        once = [v for v in set(values) if values.count(v) == 1]
        if len(once) == 1:
            expected[attr_idx] = once[0]
    return tuple(int(v) for v in expected)


@pytest.mark.parametrize("mode_kwargs,num_violated", ADVERSARIAL_MODES)
def test_fast_adversarial_targets_stay_inside_the_train_support(
    mode_kwargs, num_violated
):
    dataset = OrthotopicToyDataset()
    wrapper = _orthotopic_wrapper(dataset, **mode_kwargs)
    assert wrapper._fast_enabled

    present = {tuple(int(v) for v in row) for row in dataset._dataset_targets}
    for index in range(120):
        rows = _target_rows(wrapper[index][1])
        for row in rows:
            target = tuple(int(v) for v in row)
            assert (target >= dataset.thresholds).sum() <= dataset.c
            assert target in present


@pytest.mark.parametrize("mode_kwargs,num_violated", ADVERSARIAL_MODES)
def test_fast_adversarial_expected_corner_exists_in_train(
    mode_kwargs, num_violated
):
    # El cuarto elemento es impredecible, pero la esquina que *deberia* ir ahi
    # tiene que existir: si no, la tarea es irresoluble por construccion y no
    # mide nada sobre composicionalidad.
    dataset = OrthotopicToyDataset()
    wrapper = _orthotopic_wrapper(dataset, **mode_kwargs)

    present = {tuple(int(v) for v in row) for row in dataset._dataset_targets}
    for index in range(120):
        rows = _target_rows(wrapper[index][1])
        assert _expected_corner(rows[:3]) in present


@pytest.mark.parametrize("mode_kwargs,num_violated", ADVERSARIAL_MODES)
def test_fast_adversarial_violates_exactly_the_designed_attributes(
    mode_kwargs, num_violated
):
    dataset = OrthotopicToyDataset()
    wrapper = _orthotopic_wrapper(dataset, **mode_kwargs)
    num_attributes = dataset._dataset_targets.shape[1]

    for index in range(120):
        rows = _target_rows(wrapper[index][1])
        first_three, target = rows[:3], rows[3]
        expected = _expected_corner(first_three)
        violated = [
            idx for idx in range(num_attributes) if target[idx] != expected[idx]
        ]
        varying = [
            idx
            for idx in range(num_attributes)
            if len(set(first_three[:, idx])) == 2
        ]
        assert len(varying) == 2

        if num_violated is None:  # adversarial: ningun atributo es predecible
            for attr_idx in range(num_attributes):
                assert target[attr_idx] not in set(first_three[:, attr_idx])
        else:
            assert len(violated) == num_violated
            assert set(violated).issubset(varying)
            for attr_idx in range(num_attributes):
                if attr_idx not in violated:
                    assert target[attr_idx] == expected[attr_idx]


@pytest.mark.parametrize("mode_kwargs,num_violated", ADVERSARIAL_MODES)
def test_fast_adversarial_consecutive_indices_share_one_quadrilateral(
    mode_kwargs, num_violated
):
    # Los indices 4k..4k+3 reusan el mismo cuadrilatero y violan una esquina
    # distinta cada uno, de modo que ninguna esquina queda sin ser objetivo.
    dataset = OrthotopicToyDataset()
    wrapper = _orthotopic_wrapper(dataset, **mode_kwargs)

    for base in range(0, 80, 4):
        rectangles, violated_corners = [], []
        for offset in range(4):
            rows = _target_rows(wrapper[base + offset][1])
            corners = [tuple(int(v) for v in row) for row in rows[:3]]
            expected = _expected_corner(rows[:3])
            rectangles.append(frozenset(corners + [expected]))
            violated_corners.append(expected)
        assert len(set(rectangles)) == 1
        assert len(set(violated_corners)) == 4


@pytest.mark.parametrize("mode_kwargs,num_violated", ADVERSARIAL_MODES)
def test_fast_adversarial_still_reaches_hard_attribute_values(
    mode_kwargs, num_violated
):
    # Sin valores dificiles el modelo nunca ve los conceptos que tiene que
    # extrapolar, asi que el camino rapido no puede quedarse solo con faciles.
    dataset = OrthotopicToyDataset()
    wrapper = _orthotopic_wrapper(dataset, **mode_kwargs)

    hard_seen = set()
    for index in range(400):
        for row in _target_rows(wrapper[index][1]):
            for attr_idx, value in enumerate(row):
                if value >= dataset.thresholds[attr_idx]:
                    hard_seen.add(attr_idx)
    assert hard_seen == set(range(dataset._dataset_targets.shape[1]))


def test_fast_adversarial_path_is_disabled_without_a_support_rule():
    dataset = OrthotopicToyDataset()
    wrapper = NonIIDWrapper(dataset, seed=0, sampling_mode="adversarial")

    assert not wrapper._fast_enabled
    rows = _target_rows(wrapper[0][1])
    for attr_idx in range(rows.shape[1]):
        assert rows[3, attr_idx] not in set(rows[:3, attr_idx])
