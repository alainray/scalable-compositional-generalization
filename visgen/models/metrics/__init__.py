from .compositional import (
    n_components_for_variance,
    pairwise_distances,
    parallelism_score,
    parallelism_score_categorical,
    singular_value_report,
    spearman_correlation,
    topographic_similarity,
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
    'parallelism_score',
    'parallelism_score_categorical',
    'singular_value_report',
    'n_components_for_variance',
]


def get_metrics(cfg,*args,version=None,**kwargs):
	metrics=[]
	for m in cfg:
		if m.name=='accuracy':metrics.append(Accuracy())
		elif m.name=='multi_accuracy':metrics.append(MultiAccuracy())
		elif m.name=='modular_multi_accuracy':metrics.append(ModularMultiAccuracy())
		elif m.name=='r2':metrics.append(RSquared())
	return metrics
