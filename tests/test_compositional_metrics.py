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
