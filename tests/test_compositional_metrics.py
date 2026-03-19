import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "visgen" / "models" / "metrics" / "compositional.py"
spec = importlib.util.spec_from_file_location("compositional_metrics", MODULE_PATH)
compositional_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compositional_metrics)


def test_topsim_is_one_for_identical_geometries():
    semantic = np.array([[0.0], [1.0], [3.0], [4.0]])
    observed = semantic * 10.0
    score = compositional_metrics.topographic_similarity(
        semantic, observed, semantic_metric="euclidean", observed_metric="euclidean"
    )
    assert np.isclose(score, 1.0)


def test_parallelism_score_detects_parallel_directions():
    z = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([0, 1, 0, 1])
    ctx = np.array([[0], [0], [1], [1]])
    score = compositional_metrics.parallelism_score(z, y, ctx)
    assert np.isclose(score, 1.0)


def test_n_components_for_variance():
    z = np.array(
        [
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    report = compositional_metrics.singular_value_report(z)
    assert np.isclose(report.explained_variance_ratio[0], 1.0)
    assert np.isclose(report.component_fraction[-1], 1.0)
    assert compositional_metrics.n_components_for_variance(z, variance_threshold=0.9) == 1


def test_singular_spectrum_auc_prefers_low_rank_representations():
    rng = np.random.default_rng(0)
    z_low_rank = np.array(
        [
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    z_full_rank = rng.normal(size=(200, 3))

    auc_low_rank = compositional_metrics.singular_spectrum_auc(z_low_rank)
    auc_full_rank = compositional_metrics.singular_spectrum_auc(z_full_rank)

    assert 0.0 <= auc_low_rank <= 1.0
    assert 0.0 <= auc_full_rank <= 1.0
    assert auc_low_rank > auc_full_rank


def test_singular_spectrum_auc_matches_hand_computation_for_rank_one_case():
    z_rank_one = np.array(
        [
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    # For k=3 components and all variance in the first singular value:
    # x = [0, 1/3, 2/3, 1], y = [0, 1, 1, 1]
    # AUC = (1/3)*(0+1)/2 + (1/3)*(1+1)/2 + (1/3)*(1+1)/2 = 5/6
    expected_auc = 5.0 / 6.0
    observed_auc = compositional_metrics.singular_spectrum_auc(z_rank_one)
    assert np.isclose(observed_auc, expected_auc)


def test_hoyer_sparsity_distinguishes_dense_and_sparse_vectors():
    dense = np.ones((16, 8))
    sparse = np.zeros((16, 8))
    sparse[:, 0] = 1.0

    dense_sparsity = compositional_metrics.hoyer_sparsity(dense)
    sparse_sparsity = compositional_metrics.hoyer_sparsity(sparse)

    assert np.isclose(dense_sparsity, 0.0)
    assert np.isclose(sparse_sparsity, 1.0)


def test_twonn_intrinsic_dimension_is_close_for_uniform_2d_cloud():
    rng = np.random.default_rng(42)
    z = rng.uniform(size=(600, 2))
    estimate = compositional_metrics.twonn_intrinsic_dimension(z, metric="euclidean")
    assert 1.4 <= estimate <= 2.6


def test_topographic_similarity_with_twonn_reuses_observed_distances():
    semantic = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    observed = semantic * 3.0

    topsim = compositional_metrics.topographic_similarity(
        semantic, observed, semantic_metric="euclidean", observed_metric="euclidean"
    )
    joint_topsim, twonn_id = compositional_metrics.topographic_similarity_with_twonn(
        semantic, observed, semantic_metric="euclidean", observed_metric="euclidean"
    )

    assert np.isclose(topsim, joint_topsim)
    assert np.isfinite(twonn_id)
