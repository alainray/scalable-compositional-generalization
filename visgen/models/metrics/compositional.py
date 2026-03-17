from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_2d_array(values: Union[np.ndarray, Sequence]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}.")
    return array


def _rankdata_average_ties(values: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values, kind="mergesort")
    sorted_values = values[sorter]
    ranks = np.empty_like(values, dtype=float)

    n = len(values)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * ((start + 1) + end)
        ranks[sorter[start:end]] = avg_rank
        start = end
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom == 0:
        return np.nan
    return float(np.dot(x_centered, y_centered) / denom)


def spearman_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Compute Spearman's rho using average ranks for ties."""
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")
    if x.size < 2:
        return np.nan
    x_rank = _rankdata_average_ties(x)
    y_rank = _rankdata_average_ties(y)
    return _pearson_corr(x_rank, y_rank)


def _pairwise_distances_numeric(values: np.ndarray, metric: str) -> np.ndarray:
    values = _as_2d_array(values).astype(float)
    if values.shape[0] < 2:
        return np.zeros((values.shape[0], values.shape[0]), dtype=float)

    if metric == "euclidean":
        sq_norms = np.sum(values**2, axis=1, keepdims=True)
        squared = sq_norms + sq_norms.T - 2 * values @ values.T
        squared = np.maximum(squared, 0.0)
        return np.sqrt(squared)

    if metric == "cosine":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        safe_values = np.divide(values, np.maximum(norms, 1e-12))
        similarity = safe_values @ safe_values.T
        return 1.0 - similarity

    if metric == "hamming":
        return (values[:, None, :] != values[None, :, :]).mean(axis=-1)

    raise ValueError(f"Unsupported numeric metric '{metric}'.")


def _levenshtein_distance(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        curr = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _pairwise_levenshtein(messages: Sequence[Sequence]) -> np.ndarray:
    n = len(messages)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _levenshtein_distance(messages[i], messages[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def pairwise_distances(
    values: Union[np.ndarray, Sequence],
    metric: str = "euclidean",
) -> np.ndarray:
    """Compute pairwise distance matrix for numeric vectors or token sequences."""
    if metric == "levenshtein":
        return _pairwise_levenshtein(values)
    return _pairwise_distances_numeric(np.asarray(values), metric=metric)


def topographic_similarity(
    semantic_representations: Union[np.ndarray, Sequence],
    observed_representations: Union[np.ndarray, Sequence],
    semantic_metric: str = "cosine",
    observed_metric: str = "euclidean",
) -> float:
    """TopSim = Spearman correlation between pairwise distances in two spaces."""
    sem_d = pairwise_distances(semantic_representations, metric=semantic_metric)
    obs_d = pairwise_distances(observed_representations, metric=observed_metric)
    if sem_d.shape != obs_d.shape:
        raise ValueError(
            "Both representations must contain the same number of examples. "
            f"Got distance matrices {sem_d.shape} and {obs_d.shape}."
        )
    upper = np.triu_indices(sem_d.shape[0], k=1)
    return spearman_correlation(sem_d[upper], obs_d[upper])


def parallelism_score(
    representations: np.ndarray,
    binary_attribute: Sequence,
    contexts: np.ndarray,
    positive_value: Optional[Union[int, str]] = None,
) -> float:
    """Parallelism score for a binary attribute across contexts."""
    z = _as_2d_array(np.asarray(representations, dtype=float))
    y = np.asarray(binary_attribute)
    c = _as_2d_array(np.asarray(contexts))
    if len(z) != len(y) or len(z) != len(c):
        raise ValueError("representations, binary_attribute and contexts must have same length.")

    unique_y = np.unique(y)
    if unique_y.size != 2:
        raise ValueError(f"binary_attribute must have exactly 2 unique values, got {unique_y}.")

    if positive_value is None:
        v0, v1 = unique_y[0], unique_y[1]
    else:
        if positive_value not in unique_y:
            raise ValueError(f"positive_value={positive_value} not present in binary_attribute.")
        v1 = positive_value
        v0 = unique_y[0] if unique_y[1] == positive_value else unique_y[1]

    vectors = []
    for ctx in np.unique(c, axis=0):
        mask = np.all(c == ctx, axis=1)
        mask0 = mask & (y == v0)
        mask1 = mask & (y == v1)
        if mask0.any() and mask1.any():
            mu0 = z[mask0].mean(axis=0)
            mu1 = z[mask1].mean(axis=0)
            delta = mu1 - mu0
            norm = np.linalg.norm(delta)
            if norm > 0:
                vectors.append(delta / norm)

    if len(vectors) < 2:
        return np.nan

    vectors = np.asarray(vectors)
    cosine = vectors @ vectors.T
    upper = np.triu_indices(len(vectors), k=1)
    return float(cosine[upper].mean())


def parallelism_score_categorical(
    representations: np.ndarray,
    attribute: Sequence,
    contexts: np.ndarray,
) -> float:
    """Average binary parallelism score over all pairs of attribute values."""
    y = np.asarray(attribute)
    values = np.unique(y)
    if values.size < 2:
        return np.nan

    scores = []
    for i, v0 in enumerate(values):
        for v1 in values[i + 1 :]:
            pair_mask = (y == v0) | (y == v1)
            score = parallelism_score(
                representations=np.asarray(representations)[pair_mask],
                binary_attribute=y[pair_mask],
                contexts=np.asarray(contexts)[pair_mask],
                positive_value=v1,
            )
            if not np.isnan(score):
                scores.append(score)
    return float(np.mean(scores)) if scores else np.nan


@dataclass
class SingularValueReport:
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance_ratio: np.ndarray


def singular_value_report(representations: np.ndarray) -> SingularValueReport:
    """SVD summary ordered by explained variance."""
    z = _as_2d_array(np.asarray(representations, dtype=float))
    z = z - z.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(z, full_matrices=False)
    explained = singular_values**2
    total = explained.sum()
    ratio = explained / total if total > 0 else np.zeros_like(explained)
    cumulative = np.cumsum(ratio)
    return SingularValueReport(
        singular_values=singular_values,
        explained_variance_ratio=ratio,
        cumulative_explained_variance_ratio=cumulative,
    )


def n_components_for_variance(
    representations: np.ndarray,
    variance_threshold: float = 0.9,
) -> int:
    """Smallest number of singular components explaining threshold variance."""
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1].")
    report = singular_value_report(representations)
    idx = np.searchsorted(report.cumulative_explained_variance_ratio, variance_threshold)
    return int(idx + 1)
