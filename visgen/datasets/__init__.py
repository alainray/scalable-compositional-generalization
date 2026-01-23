from torch.utils.data import DataLoader, random_split

from .cars3d import Cars3D
from .clevr import CLEVR
from .dsprites import DSprites
from .iraven import IRAVEN
from .mpi3d import MPI3D
from .non_iid import NonIIDWrapper
from .shapes3d import Shapes3D


def _wrap_non_iid(dataset, cfg):
	non_iid_cfg = getattr(cfg, "non_iid", None)
	if not non_iid_cfg:
		return dataset
	return NonIIDWrapper(
		dataset,
		max_resample_attempts=non_iid_cfg.get("max_resample_attempts", 10_000),
		seed=non_iid_cfg.get("seed"),
		allowed_attributes=non_iid_cfg.get("allowed_attributes"),
		shared_other_attributes=non_iid_cfg.get("shared_other_attributes", True),
	)


def get_dataloaders(data_cfg, writer=None):
	dataset_map = {
		"dsprites": DSprites,
		"iraven": IRAVEN,
		"mpi3d": MPI3D,
		"shapes3d": Shapes3D,
		"cars3d": Cars3D,
		"clevr": CLEVR,
	}
	d_dataloaders = {}
	for (key, cfg) in data_cfg.items():
		data = dataset_map[cfg.dataset](**cfg)
		if cfg.train:
			num_ood_val = cfg.num_ood_val if "num_ood_val" in cfg else 1
			train_data, ood_val_sets = data.ood_validation_split(num_ood_val)
			val_size = int(cfg.val_fraction * len(train_data))
			train_size = len(train_data) - val_size
			train_data, val_data = random_split(
				train_data, [train_size, val_size]
			)
			datasets = [(key, train_data), ("validation", val_data)]
			datasets += [
				(f"ood_validation_{i}", ood_val_data)
				for (i, ood_val_data) in enumerate(ood_val_sets)
			]
		else:
			datasets = [(key, data)]
		infos = {}
		if hasattr(data, "actual_difficulty"):
			infos[f"{key}_actual_difficulty"] = data.actual_difficulty
			infos[f"{key}_volume"] = data.volume
		writer.write(infos)
		for (name, data) in datasets:
			data = _wrap_non_iid(data, cfg)
			loader = DataLoader(
				data,
				batch_size=cfg.batch_size,
				num_workers=cfg.num_workers,
				pin_memory=True,
			)
			d_dataloaders[name] = loader
	return d_dataloaders
