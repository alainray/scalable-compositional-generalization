from .compositional import (
    hoyer_sparsity,
    n_components_for_variance,
    pairwise_distances,
    parallelism_score,
    parallelism_score_categorical,
    singular_spectrum_auc,
    singular_value_report,
    spearman_correlation,
    topographic_similarity,
    topographic_similarity_with_twonn,
    twonn_intrinsic_dimension,
    twonn_intrinsic_dimension_from_distances,
)
from .metrics import Accuracy, ModularMultiAccuracy, MultiAccuracy, RSquared

__all__ = [
    'Accuracy',
    'RSquared',
    'MultiAccuracy',
    'ModularMultiAccuracy',
    'pairwise_distances',
    'spearman_correlation',
    'topographic_similarity',
    'topographic_similarity_with_twonn',
    'parallelism_score',
    'parallelism_score_categorical',
    'singular_value_report',
    'singular_spectrum_auc',
    'n_components_for_variance',
    'hoyer_sparsity',
    'twonn_intrinsic_dimension',
    'twonn_intrinsic_dimension_from_distances',
]


def get_metrics(cfg,*args,version=None,**kwargs):
	metrics=[]
	for m in cfg:
		if m.name=='accuracy':metrics.append(Accuracy())
		elif m.name=='multi_accuracy':metrics.append(MultiAccuracy())
		elif m.name=='modular_multi_accuracy':metrics.append(ModularMultiAccuracy())
		elif m.name=='r2':metrics.append(RSquared())
	return metrics
