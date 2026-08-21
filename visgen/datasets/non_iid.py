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
        sampling_mode: Optional[str] = None,
        deterministic: bool = False,
        precompute_deterministic: bool = False,
        max_deterministic_candidates: int = 1_024,
        num_unpredictable_attributes: int = 1,
        deterministic_sampling_strategy: str = "random",
        max_deterministic_resample_attempts: Optional[int] = None,
        split_thresholds=None,
        split_c: Optional[int] = None,
        relaxed_fourth: bool = False,
    ) -> None:
        self.dataset = dataset
        self.max_resample_attempts = max_resample_attempts
        self.rng = np.random.default_rng(seed)
        self._base_seed = 0 if seed is None else int(seed)
        self.relaxed_fourth = bool(relaxed_fourth)
        self.shared_other_attributes = shared_other_attributes
        self.sampling_mode = self._resolve_sampling_mode(sampling_mode, fully_iid)
        self.fully_iid = self.sampling_mode == "iid"
        self.deterministic = deterministic
        self.precompute_deterministic = precompute_deterministic
        self.max_deterministic_candidates = max_deterministic_candidates
        self.num_unpredictable_attributes = self._resolve_num_unpredictable_attributes(
            num_unpredictable_attributes
        )
        self.deterministic_sampling_strategy = (
            self._resolve_deterministic_sampling_strategy(
                deterministic_sampling_strategy
            )
        )
        self.max_deterministic_resample_attempts = (
            max_deterministic_resample_attempts
        )
        self._targets, self._attribute_values = self._prepare_targets(dataset)
        self._index_by_target = self._build_index(self._targets)
        self._init_support_rule(split_thresholds, split_c)
        self._deterministic_candidates: Optional[List[List[int]]] = None
        if self.fully_iid:
            self._attribute_indices = np.array([], dtype=int)
        else:
            self._attribute_indices = self._resolve_attribute_indices(
                allowed_attributes
            )
        if self.deterministic and self.precompute_deterministic:
            self._deterministic_candidates = self._build_deterministic_candidates()

    @staticmethod
    def _resolve_sampling_mode(
        sampling_mode: Optional[str], fully_iid: bool
    ) -> str:
        """Normalize sampling mode while preserving the legacy fully_iid flag."""
        if sampling_mode is None:
            return "iid" if fully_iid else "non_iid"
        valid_modes = {"non_iid", "iid", "adversarial", "unpredictable_target"}
        if sampling_mode not in valid_modes:
            raise ValueError(
                "NonIIDWrapper sampling_mode must be one of "
                f"{sorted(valid_modes)}, got {sampling_mode!r}."
            )
        if fully_iid and sampling_mode != "iid":
            raise ValueError(
                "NonIIDWrapper received fully_iid=True with "
                f"sampling_mode={sampling_mode!r}. Use sampling_mode='iid' or "
                "remove fully_iid."
            )
        return sampling_mode

    @staticmethod
    def _resolve_deterministic_sampling_strategy(strategy: str) -> str:
        valid = {"random", "enumerate"}
        if strategy not in valid:
            raise ValueError(
                "NonIIDWrapper deterministic_sampling_strategy must be one of "
                f"{sorted(valid)}, got {strategy!r}."
            )
        return strategy

    def _resolve_num_unpredictable_attributes(
        self, num_unpredictable_attributes: int
    ) -> int:
        num_unpredictable_attributes = int(num_unpredictable_attributes)
        if self.sampling_mode != "unpredictable_target":
            return num_unpredictable_attributes
        if num_unpredictable_attributes not in (1, 2):
            raise ValueError(
                "NonIIDWrapper sampling_mode='unpredictable_target' requires "
                "num_unpredictable_attributes to be 1 or 2."
            )
        return num_unpredictable_attributes

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

    # ---- muestreo constructivo -------------------------------------------
    #
    # Con el split ortotopico la pertenencia al soporte es contable:
    #     z en soporte  <=>  #{i : z_i >= umbral_i} <= c
    # Eso permite construir cuadrilateros y cuartas etiquetas validas en O(m),
    # en vez de sortear a ciegas y barrer el dataset para comprobar.

    def _init_support_rule(self, thresholds, c) -> None:
        self._thr = None
        self._support_c = None
        if thresholds is None or c is None:
            return
        thr = np.asarray(thresholds)
        # Solo aplicable si los umbrales cubren exactamente las columnas de
        # _targets, y para c en {0, 1} (los que usan los runners).
        if thr.shape[0] != self._targets.shape[1] or int(c) not in (0, 1):
            return
        self._thr = thr
        self._support_c = int(c)
        self._easy = [
            [int(v) for v in self._attribute_values[k] if v < thr[k]]
            for k in range(self._targets.shape[1])
        ]
        self._hard = [
            [int(v) for v in self._attribute_values[k] if v >= thr[k]]
            for k in range(self._targets.shape[1])
        ]

    @property
    def _fast_enabled(self) -> bool:
        return self._thr is not None and self.sampling_mode in (
            "adversarial",
            "unpredictable_target",
        )

    def _n_hard(self, target) -> int:
        return int(sum(1 for k, v in enumerate(target) if v >= self._thr[k]))

    def _fast_quadrant(self, rng):
        """Cuadrilatero cuyas cuatro esquinas estan en el soporte y presentes."""
        m = self._targets.shape[1]
        for _ in range(64):
            i, j = (int(x) for x in rng.choice(self._attribute_indices, 2, replace=False))
            anchor = self._targets[int(rng.integers(len(self._targets)))]
            shared = {k: int(anchor[k]) for k in range(m) if k not in (i, j)}
            budget = self._support_c - sum(
                1 for k, v in shared.items() if v >= self._thr[k]
            )
            if budget < 0:
                continue
            # con c<=1: si queda presupuesto, a lo sumo UNO de los cuatro
            # valores del cuadrante puede ser dificil (dos crearian una esquina
            # con dos dificiles)
            pools = {}
            hard_slot = None
            if budget >= 1 and rng.random() < 0.5:
                hard_slot = i if rng.random() < 0.5 else j
            for k in (i, j):
                pool = list(self._easy[k])
                if k == hard_slot and self._hard[k]:
                    pool = pool + [int(rng.choice(self._hard[k]))]
                if len(pool) < 2:
                    pool = None
                pools[k] = pool
            if pools[i] is None or pools[j] is None:
                continue
            a, b = (int(x) for x in rng.choice(pools[i], 2, replace=False))
            c_, d = (int(x) for x in rng.choice(pools[j], 2, replace=False))
            corners = []
            ok = True
            for va in (a, b):
                for vb in (c_, d):
                    t = [0] * m
                    t[i], t[j] = va, vb
                    for k, v in shared.items():
                        t[k] = v
                    t = tuple(t)
                    if self._n_hard(t) > self._support_c or t not in self._index_by_target:
                        ok = False
                        break
                    corners.append(t)
                if not ok:
                    break
            if ok:
                return i, j, (a, b), (c_, d), shared, corners
        return None

    def _fast_fourth(self, i, j, vals_i, vals_j, shared, corner, rng):
        """Cuarta etiqueta adversarial, uniforme sobre el conjunto valido."""
        m = self._targets.shape[1]
        if self.sampling_mode == "adversarial":
            chg = list(range(m))
        elif self.num_unpredictable_attributes == 1:
            chg = [j]
        else:
            chg = [i, j]
        forb = [set() for _ in range(m)]
        ctx = None
        if self.relaxed_fourth:
            # Solo se prohibe el valor ESPERADO, no los dos del rectangulo: la
            # cuarta esquina puede reusar el otro. Hace falta donde el soporte
            # no deja un tercer valor -- en clevr, con material=1 compartido el
            # presupuesto de dificiles ya esta gastado y shape/size solo tienen
            # dos faciles, los que el rectangulo ya usa, asi que la regla
            # estricta descarta el cuadrilatero entero y material=1 nunca llega
            # a entrenamiento. A cambio hay que rechazar a mano las tuplas que
            # coincidan con una esquina del contexto, o el grupo sale repetido.
            for k in chg:
                forb[k] = {corner[k]}
            ctx = set()
            for va in vals_i:
                for vb in vals_j:
                    t = [0] * m
                    t[i], t[j] = va, vb
                    for k, v in shared.items():
                        t[k] = v
                    ctx.add(tuple(t))
        else:
            forb[i] = set(vals_i)
            forb[j] = set(vals_j)
            for k, v in shared.items():
                forb[k] = {v}
        fixed = {k: corner[k] for k in range(m) if k not in chg}
        budget = self._support_c - sum(
            1 for k, v in fixed.items() if v >= self._thr[k]
        )
        if budget < 0:
            return None
        easy = {k: [v for v in self._easy[k] if v not in forb[k]] for k in chg}
        hard = {k: [v for v in self._hard[k] if v not in forb[k]] for k in chg}
        # conteo por categoria, sin enumerar
        n0 = 1
        for k in chg:
            n0 *= len(easy[k])
        n1 = 0
        if budget >= 1:
            for k in chg:
                prod = len(hard[k])
                for l in chg:
                    if l != k:
                        prod *= len(easy[l])
                n1 += prod
        if n0 + n1 == 0:
            return None
        for _ in range(16):
            hard_at = None
            if n1 and rng.random() < n1 / (n0 + n1):
                w = np.array(
                    [
                        len(hard[k]) * np.prod([len(easy[l]) for l in chg if l != k])
                        for k in chg
                    ],
                    dtype=float,
                )
                hard_at = chg[int(rng.choice(len(chg), p=w / w.sum()))]
            t = list(fixed.get(k, 0) for k in range(m))
            for k in range(m):
                if k not in chg:
                    t[k] = fixed[k]
            bad = False
            for k in chg:
                pool = hard[k] if k == hard_at else easy[k]
                if not pool:
                    bad = True
                    break
                t[k] = int(rng.choice(pool))
            if bad:
                continue
            t = tuple(t)
            if self._n_hard(t) <= self._support_c and t in self._index_by_target:
                if ctx is not None and t in ctx:
                    continue
                return t
        return None

    def _fast_group(self, index: int, attempt: int = 0):
        """Cuadrilatero + cuarta adversarial. Cuatro indices consecutivos
        comparten cuadrilatero y violan una esquina distinta cada uno."""
        # semilla fija por cuadrilatero: los indices 4k..4k+3 comparten
        # cuadrilatero y violan una esquina distinta cada uno
        rng = np.random.default_rng([self._base_seed, index // 4, attempt])
        corner_slot = index % 4
        q = self._fast_quadrant(rng)
        if q is None:
            return None
        i, j, vals_i, vals_j, shared, _ = q
        # Reetiquetamos (a,b) y (c,d) para que la esquina a violar caiga
        # siempre en el slot 3. El rectangulo es el mismo conjunto, pero asi se
        # conserva la convencion de que el ultimo elemento es el objetivo: el
        # residuo algebraico ac-ad-bc+bd y el caso (3,[0,1,2]) del mixer
        # transformer leen las esquinas por posicion.
        if corner_slot < 2:
            vals_i = (vals_i[1], vals_i[0])
        if corner_slot % 2 == 0:
            vals_j = (vals_j[1], vals_j[0])
        m = self._targets.shape[1]
        corners = []
        for va in vals_i:
            for vb in vals_j:
                t = [0] * m
                t[i], t[j] = va, vb
                for k, v in shared.items():
                    t[k] = v
                corners.append(tuple(t))
        fourth = self._fast_fourth(i, j, vals_i, vals_j, shared, corners[3], rng)
        if fourth is None:
            return None
        targets = corners[:3] + [fourth]
        return self._select_indices(targets)

    def __len__(self) -> int:
        if self.deterministic:
            if self.precompute_deterministic:
                if self._deterministic_candidates is None:
                    self._deterministic_candidates = (
                        self._build_deterministic_candidates()
                    )
                return len(self._deterministic_candidates)
            return min(
                len(self.dataset) // self.group_size,
                self.max_deterministic_candidates,
            )
        return len(self.dataset) // self.group_size

    def __getitem__(self, index: int):
        if self.fully_iid:
            sampled_indices = self.rng.choice(
                len(self.dataset), size=self.group_size, replace=True
            )
            images, targets = zip(
                *(self.dataset[int(idx)] for idx in sampled_indices)
            )
            images = self._stack_images(images)
            targets = torch.as_tensor(np.stack(targets))
            return images, targets
        if self.deterministic:
            indices = self._get_deterministic_indices(index)
            images, targets = zip(*(self.dataset[idx] for idx in indices))
            images = self._stack_images(images)
            targets = torch.as_tensor(np.stack(targets))
            return images, targets
        if self._fast_enabled:
            for attempt in range(64):
                indices = self._fast_group(index, attempt)
                if indices is not None:
                    images, targets = zip(*(self.dataset[idx] for idx in indices))
                    images = self._stack_images(images)
                    targets = torch.as_tensor(np.stack(targets))
                    return images, targets
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
            if self.sampling_mode == "adversarial":
                desired_targets = self._build_adversarial_targets(desired_targets)
                if desired_targets is None:
                    continue
            elif self.sampling_mode == "unpredictable_target":
                desired_targets = self._build_unpredictable_targets(
                    desired_targets, attr_a, attr_b
                )
                if desired_targets is None:
                    continue
            indices = self._select_indices(desired_targets)
            if indices is None:
                continue
            images, targets = zip(*(self.dataset[idx] for idx in indices))
            images = self._stack_images(images)
            targets = torch.as_tensor(np.stack(targets))
            return images, targets
        fallback_indices = self._get_deterministic_indices(index, allow_lazy_search=True)
        if fallback_indices is not None:
            images, targets = zip(*(self.dataset[idx] for idx in fallback_indices))
            images = self._stack_images(images)
            targets = torch.as_tensor(np.stack(targets))
            return images, targets
        if self.sampling_mode == "adversarial":
            raise RuntimeError(
                "Failed to sample a valid adversarial non-iid batch within the "
                "resample limit. The fourth sample must have no attribute values "
                "in common with the first three samples; try a training split with "
                "more available values per attribute or disable adversarial sampling."
            )
        if self.sampling_mode == "unpredictable_target":
            raise RuntimeError(
                "Failed to sample a valid unpredictable_target non-iid batch within "
                "the resample limit. Try a split with more available values for the "
                "quadrant attributes or reduce num_unpredictable_attributes."
            )
        raise RuntimeError(
            "Failed to sample a valid non-iid batch within the resample limit."
        )

    def _get_deterministic_indices(
        self, index: int, allow_lazy_search: bool = False
    ) -> List[int]:
        if self._deterministic_candidates is None:
            if (not self.deterministic) and (not allow_lazy_search):
                raise RuntimeError("Deterministic non-iid sampling is disabled.")
            self._deterministic_candidates = self._build_deterministic_candidates()
        if not self._deterministic_candidates:
            if self.sampling_mode == "adversarial":
                raise RuntimeError(
                    "Failed to sample a valid adversarial non-iid batch: no "
                    "deterministic candidates were found with a fourth sample "
                    "that is feature-disjoint from the first three samples."
                )
            if self.sampling_mode == "unpredictable_target":
                raise RuntimeError(
                    "Failed to sample a valid unpredictable_target non-iid batch: "
                    "no deterministic candidates were found with the requested "
                    "number of unpredictable quadrant attributes."
                )
            raise RuntimeError(
                "Failed to sample a valid non-iid batch: no deterministic "
                "candidate quadrants were found."
            )
        return self._deterministic_candidates[index % len(self._deterministic_candidates)]

    def _build_deterministic_candidates(self) -> List[List[int]]:
        if self.fully_iid:
            return []
        if self.deterministic_sampling_strategy == "enumerate":
            return self._build_deterministic_candidates_enumerated()
        return self._build_deterministic_candidates_random()

    def _build_deterministic_candidates_enumerated(self) -> List[List[int]]:
        candidates: List[List[int]] = []
        seen = set()
        for i, attr_a in enumerate(self._attribute_indices):
            for attr_b in self._attribute_indices[i + 1 :]:
                if self.shared_other_attributes:
                    pair_candidates = self._find_rectangle_indices_shared(
                        attr_a, attr_b, collect_all=True
                    )
                else:
                    pair_candidates = self._find_rectangle_indices_unshared(
                        attr_a, attr_b, collect_all=True
                    )
                for candidate in pair_candidates:
                    if self.sampling_mode == "adversarial":
                        candidate = self._make_deterministic_adversarial_candidate(
                            candidate
                        )
                        if candidate is None:
                            continue
                    elif self.sampling_mode == "unpredictable_target":
                        candidate = self._make_deterministic_unpredictable_candidate(
                            candidate, attr_a, attr_b
                        )
                        if candidate is None:
                            continue
                    key = tuple(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(candidate)
                    if len(candidates) >= self.max_deterministic_candidates:
                        return candidates
        return candidates

    def _build_deterministic_candidates_random(self) -> List[List[int]]:
        if not self.shared_other_attributes:
            return self._build_deterministic_candidates_enumerated()

        candidates: List[List[int]] = []
        seen = set()
        max_attempts = self.max_deterministic_resample_attempts
        if max_attempts is None:
            max_attempts = max(10_000, self.max_deterministic_candidates * 100)

        for _ in range(max_attempts):
            candidate = self._sample_deterministic_candidate_from_base()
            if candidate is None:
                continue
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            if len(candidates) >= self.max_deterministic_candidates:
                break
        return candidates

    def _sample_deterministic_candidate_from_base(self) -> Optional[List[int]]:
        base_idx = int(self.rng.integers(len(self._targets)))
        base_target = tuple(self._targets[base_idx].tolist())
        attr_a, attr_b = self.rng.choice(
            self._attribute_indices, size=2, replace=False
        )
        attr_a = int(attr_a)
        attr_b = int(attr_b)

        alt_a_values = [
            value
            for value in self._attribute_values[attr_a]
            if value != base_target[attr_a]
        ]
        alt_b_values = [
            value
            for value in self._attribute_values[attr_b]
            if value != base_target[attr_b]
        ]
        if not alt_a_values or not alt_b_values:
            return None

        alt_a = self.rng.choice(alt_a_values)
        alt_b = self.rng.choice(alt_b_values)

        target_00 = list(base_target)
        target_01 = list(base_target)
        target_01[attr_b] = alt_b
        target_10 = list(base_target)
        target_10[attr_a] = alt_a
        target_11 = list(base_target)
        target_11[attr_a] = alt_a
        target_11[attr_b] = alt_b

        quadrant_targets = [
            tuple(target_00),
            tuple(target_01),
            tuple(target_10),
            tuple(target_11),
        ]

        if self.sampling_mode == "adversarial":
            sampled_targets = self._build_adversarial_targets(quadrant_targets)
        elif self.sampling_mode == "unpredictable_target":
            sampled_targets = self._build_unpredictable_targets(
                quadrant_targets, attr_a, attr_b
            )
        else:
            sampled_targets = quadrant_targets
        if sampled_targets is None:
            return None
        return self._select_deterministic_indices(sampled_targets)

    def _find_valid_indices_deterministic(self) -> Optional[List[int]]:
        candidates = self._build_deterministic_candidates()
        if not candidates:
            return None
        return candidates[0]

    def _find_rectangle_indices_shared(
        self, attr_a: int, attr_b: int, collect_all: bool = False
    ) -> List[List[int]]:
        other_indices = [
            idx for idx in range(self._targets.shape[1]) if idx not in (attr_a, attr_b)
        ]
        groups = defaultdict(list)
        for idx, row in enumerate(self._targets):
            other_key = tuple(row[other_indices].tolist())
            groups[other_key].append(idx)
        results: List[List[int]] = []
        for _, group_rows in sorted(groups.items(), key=lambda item: item[0]):
            candidates = self._rectangle_indices_from_rows(
                group_rows, attr_a, attr_b, collect_all=collect_all
            )
            if not candidates:
                continue
            results.extend(candidates)
            if not collect_all:
                break
        return results

    def _find_rectangle_indices_unshared(
        self, attr_a: int, attr_b: int, collect_all: bool = False
    ) -> List[List[int]]:
        return self._rectangle_indices_from_rows(
            list(range(len(self._targets))),
            attr_a,
            attr_b,
            collect_all=collect_all,
        )

    def _rectangle_indices_from_rows(
        self,
        row_indices: Sequence[int],
        attr_a: int,
        attr_b: int,
        collect_all: bool = False,
    ) -> List[List[int]]:
        edge_to_target = {}
        a_values = set()
        b_values = set()
        for row_idx in row_indices:
            row = self._targets[row_idx]
            key = (row[attr_a], row[attr_b])
            if key not in edge_to_target:
                edge_to_target[key] = tuple(row.tolist())
            a_values.add(key[0])
            b_values.add(key[1])
        sorted_as = sorted(a_values)
        sorted_bs = sorted(b_values)
        results: List[List[int]] = []
        for i, value_a in enumerate(sorted_as):
            for value_b in sorted_as[i + 1 :]:
                for j, value_c in enumerate(sorted_bs):
                    for value_d in sorted_bs[j + 1 :]:
                        corners = [
                            (value_a, value_c),
                            (value_a, value_d),
                            (value_b, value_c),
                            (value_b, value_d),
                        ]
                        if any(corner not in edge_to_target for corner in corners):
                            continue
                        candidate = [
                            int(self._index_by_target[edge_to_target[corner]][0])
                            for corner in corners
                        ]
                        if not collect_all:
                            return [candidate]
                        results.append(candidate)
        return results

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

    def _build_adversarial_targets(
        self, quadrant_targets: Sequence[Tuple[int, ...]]
    ) -> Optional[List[Tuple[int, ...]]]:
        """Return three non-iid corners plus a feature-disjoint fourth target.

        The adversarial mode is intentionally defined only at the data sampling
        level for analogic/mixer experiments that predict the fourth element
        from the first three. Transformer-style all-case prediction requires a
        separate objective/design because the fourth sample is no longer a true
        quadrant corner for the other three prediction cases.
        """
        first_three = list(quadrant_targets[:3])
        adversarial_target = self._sample_adversarial_target(first_three)
        if adversarial_target is None:
            return None
        return first_three + [adversarial_target]

    def _build_unpredictable_targets(
        self,
        quadrant_targets: Sequence[Tuple[int, ...]],
        attr_a: int,
        attr_b: int,
    ) -> Optional[List[Tuple[int, ...]]]:
        """Return three quadrant corners plus a fourth with broken target attrs.

        ``num_unpredictable_attributes=1`` breaks the second sampled quadrant
        attribute (the value normally inferred from the second context sample,
        e.g. color in square-red, square-blue, circle-red -> circle-blue).
        ``num_unpredictable_attributes=2`` breaks both sampled quadrant
        attributes. Non-quadrant attributes are kept as in the compositional
        fourth corner.
        """
        first_three = list(quadrant_targets[:3])
        expected_target = tuple(quadrant_targets[3])
        unpredictable_target = self._sample_unpredictable_target(
            first_three, expected_target, attr_a, attr_b
        )
        if unpredictable_target is None:
            return None
        return first_three + [unpredictable_target]

    def _sample_unpredictable_target(
        self,
        reference_targets: Sequence[Tuple[int, ...]],
        expected_target: Tuple[int, ...],
        attr_a: int,
        attr_b: int,
    ) -> Optional[Tuple[int, ...]]:
        candidate_targets = self._unpredictable_candidate_targets(
            reference_targets, expected_target, attr_a, attr_b
        )
        if not candidate_targets:
            return None
        return candidate_targets[int(self.rng.integers(len(candidate_targets)))]

    def _make_deterministic_unpredictable_candidate(
        self, quadrant_indices: Sequence[int], attr_a: int, attr_b: int
    ) -> Optional[List[int]]:
        first_three_indices = list(quadrant_indices[:3])
        first_three_targets = [
            tuple(self._targets[idx].tolist()) for idx in first_three_indices
        ]
        expected_target = tuple(self._targets[quadrant_indices[3]].tolist())
        candidate_targets = self._unpredictable_candidate_targets(
            first_three_targets, expected_target, attr_a, attr_b
        )
        if not candidate_targets:
            return None
        unpredictable_target = sorted(candidate_targets)[0]
        unpredictable_index = int(self._index_by_target[unpredictable_target][0])
        return first_three_indices + [unpredictable_index]

    def _unpredictable_candidate_targets(
        self,
        reference_targets: Sequence[Tuple[int, ...]],
        expected_target: Tuple[int, ...],
        attr_a: int,
        attr_b: int,
    ) -> List[Tuple[int, ...]]:
        if not reference_targets:
            return []
        refs = np.asarray(reference_targets)
        broken_attributes = (
            [attr_b]
            if self.num_unpredictable_attributes == 1
            else [attr_a, attr_b]
        )
        candidates = [list(expected_target)]
        for attr_idx in broken_attributes:
            forbidden_values = set(np.unique(refs[:, attr_idx]).tolist())
            alternative_values = [
                value
                for value in self._attribute_values[attr_idx]
                if value not in forbidden_values
            ]
            if not alternative_values:
                return []
            candidates = [
                [
                    alternative_value if idx == attr_idx else value
                    for idx, value in enumerate(candidate)
                ]
                for candidate in candidates
                for alternative_value in alternative_values
            ]
        candidate_targets = {tuple(candidate) for candidate in candidates}
        return sorted(
            target for target in candidate_targets if target in self._index_by_target
        )

    def _sample_adversarial_target(
        self, reference_targets: Sequence[Tuple[int, ...]]
    ) -> Optional[Tuple[int, ...]]:
        candidate_targets = self._adversarial_candidate_targets(reference_targets)
        if not candidate_targets:
            return None
        return candidate_targets[int(self.rng.integers(len(candidate_targets)))]

    def _make_deterministic_adversarial_candidate(
        self, quadrant_indices: Sequence[int]
    ) -> Optional[List[int]]:
        first_three_indices = list(quadrant_indices[:3])
        first_three_targets = [
            tuple(self._targets[idx].tolist()) for idx in first_three_indices
        ]
        candidate_targets = self._adversarial_candidate_targets(first_three_targets)
        if not candidate_targets:
            return None
        adversarial_target = sorted(candidate_targets)[0]
        adversarial_index = int(self._index_by_target[adversarial_target][0])
        return first_three_indices + [adversarial_index]

    def _adversarial_candidate_targets(
        self, reference_targets: Sequence[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        if not reference_targets:
            return []
        refs = np.asarray(reference_targets)
        candidate_mask = np.ones(len(self._targets), dtype=bool)
        for attr_idx in range(self._targets.shape[1]):
            forbidden_values = np.unique(refs[:, attr_idx])
            candidate_mask &= ~np.isin(self._targets[:, attr_idx], forbidden_values)
        candidate_rows = np.flatnonzero(candidate_mask)
        candidates = {
            tuple(self._targets[row_idx].tolist()) for row_idx in candidate_rows
        }
        return sorted(candidates)

    def _select_deterministic_indices(
        self, targets: Sequence[Tuple[int, ...]]
    ) -> Optional[List[int]]:
        indices = []
        for target in targets:
            options = self._index_by_target.get(target)
            if not options:
                return None
            indices.append(int(options[0]))
        return indices

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
