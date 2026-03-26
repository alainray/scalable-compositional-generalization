import numpy as np
import torch
from typing import Optional

from visgen.models.metrics import (
    hoyer_sparsity,
    n_components_for_variance,
    parallelism_score_categorical,
    singular_spectrum_auc,
    topographic_similarity_with_twonn,
)


def _as_semantic_targets(y: torch.Tensor) -> np.ndarray:
    if y.dim() > 2:
        y = y[:, -1, :]
    y_np = y.detach().cpu().numpy()
    if y_np.ndim == 1:
        y_np = y_np[:, None]
    return y_np


@torch.no_grad()
def _extract_embeddings(model, x: torch.Tensor) -> np.ndarray:
    if x.dim() == 5:
        x = x[:, -1]
    if not hasattr(model, "extract_representation"):
        raise AttributeError(
            f"{model.__class__.__name__} does not implement extract_representation()."
        )
    z = model.extract_representation(x)
    if isinstance(z, (list, tuple)):
        raise ValueError("extract_representation must return a single tensor.")
    if z.dim() > 2:
        z = torch.flatten(z, 1)
    return z.detach().cpu().numpy()


@torch.no_grad()
def compute_representation_metrics_on_loader(
    model,
    loader,
    device,
    max_samples: Optional[int] = None,
    pairwise_max_samples: Optional[int] = None,
    sampling_seed: int = 0,
    variance_threshold: float = 0.9,
    observed_metric: str = "cosine",
):
    embeddings = []
    semantics = []
    n_samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if max_samples is not None and n_samples >= max_samples:
            break

        if max_samples is not None:
            remaining = max_samples - n_samples
            x = x[:remaining]
            y = y[:remaining]

        z_np = _extract_embeddings(model, x)
        y_np = _as_semantic_targets(y)
        embeddings.append(z_np)
        semantics.append(y_np)
        n_samples += z_np.shape[0]

    if not embeddings:
        return None

    z = np.concatenate(embeddings, axis=0)
    y = np.concatenate(semantics, axis=0)

    z_pairwise = z
    y_pairwise = y
    if pairwise_max_samples is not None and pairwise_max_samples > 0 and len(z) > pairwise_max_samples:
        rng = np.random.default_rng(sampling_seed)
        sampled_idx = rng.choice(len(z), size=pairwise_max_samples, replace=False)
        z_pairwise = z[sampled_idx]
        y_pairwise = y[sampled_idx]

    topsim, twonn_id = topographic_similarity_with_twonn(
        semantic_representations=y_pairwise,
        observed_representations=z_pairwise,
        semantic_metric="cosine",
        observed_metric=observed_metric,
    )
    pscore_values = []
    for attr_idx in range(y_pairwise.shape[1]):
        if y_pairwise.shape[1] == 1:
            continue
        score = parallelism_score_categorical(
            representations=z_pairwise,
            attribute=y_pairwise[:, attr_idx],
            contexts=np.delete(y_pairwise, attr_idx, axis=1),
        )
        if not np.isnan(score):
            pscore_values.append(score)
    pscore_mean = float(np.mean(pscore_values)) if pscore_values else np.nan

    return {
        "embedding_dim": int(z.shape[1]),
        "topsim": float(topsim),
        "pscore_mean": float(pscore_mean),
        "twonn_id": float(twonn_id),
        "hoyer_sparsity": float(hoyer_sparsity(z)),
        "sv_auc": float(singular_spectrum_auc(z)),
        "n_components_90pct": int(
            n_components_for_variance(z, variance_threshold=variance_threshold)
        ),
    }
