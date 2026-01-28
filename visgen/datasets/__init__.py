from torch.utils.data import DataLoader, Subset, random_split

from .cars3d import Cars3D
from .clevr import CLEVR
from .dsprites import DSprites
from .iraven import IRAVEN
from .mpi3d import MPI3D
from .non_iid import NonIIDWrapper
from .shapes3d import Shapes3D


def _resolve_non_iid_cfg(cfg):
	wrapper_cfg = getattr(cfg, "wrapper", None)
	if wrapper_cfg:
		wrapper_name = (
			wrapper_cfg.get("name")
			if hasattr(wrapper_cfg, "get")
			else wrapper_cfg
		)
		if wrapper_name == "non_iid":
			return wrapper_cfg
	return getattr(cfg, "non_iid", None)


def _config_attribute_names(cfg):
	targets = getattr(cfg, "targets", None)
	if targets:
		if isinstance(targets, str):
			return targets.split("_")
		return list(targets)
	attributes = getattr(cfg, "attributes", None)
	if attributes:
		names = []
		for attr in attributes:
			if isinstance(attr, dict):
				names.append(attr.get("name"))
			else:
				names.append(getattr(attr, "name", None))
		return [name for name in names if name]
	return None


def _unwrap_subset(dataset):
	base_dataset = dataset
	while isinstance(base_dataset, Subset):
		base_dataset = base_dataset.dataset
	return base_dataset


def _filter_allowed_attributes(dataset, allowed_attributes):
	if not allowed_attributes:
		return allowed_attributes
	base_dataset = _unwrap_subset(dataset)
	if getattr(base_dataset, "_attribute_values", None) is None:
		attribute_values = base_dataset._get_attribute_values()
	else:
		attribute_values = base_dataset._attribute_values
	eligible = []
	unknown = []
	filtered = []
	for attr in allowed_attributes:
		try:
			idx = base_dataset._attribute_to_index(attr)
		except Exception:
			unknown.append(attr)
			continue
		if hasattr(base_dataset, "_target_index_map"):
			idx = base_dataset._target_index_map.get(idx)
			if idx is None:
				unknown.append(attr)
				continue
		if idx >= len(attribute_values):
			unknown.append(attr)
			continue
		if len(attribute_values[idx]) < 2:
			filtered.append(attr)
			continue
		eligible.append(attr)
	if filtered or unknown:
		message = "[data] non_iid allowed_attributes filtered"
		if filtered:
			message += f" (monovalued: {', '.join(filtered)})"
		if unknown:
			message += f" (unknown: {', '.join(unknown)})"
		print(message)
	return eligible


def _wrap_non_iid(dataset, cfg, split_name=None):
	non_iid_cfg = _resolve_non_iid_cfg(cfg)
	if not non_iid_cfg:
		return dataset
	if isinstance(non_iid_cfg, str):
		non_iid_cfg = {}
	apply_to = non_iid_cfg.get("apply_to")
	if apply_to and split_name not in apply_to:
		return dataset
	if split_name is None:
		return dataset
	allowed_attributes = non_iid_cfg.get("allowed_attributes")
	if not allowed_attributes:
		allowed_attributes = _config_attribute_names(cfg)
	allowed_attributes = _filter_allowed_attributes(dataset, allowed_attributes)
	return NonIIDWrapper(
		dataset,
		max_resample_attempts=non_iid_cfg.get("max_resample_attempts", 10_000),
		seed=non_iid_cfg.get("seed"),
		allowed_attributes=allowed_attributes,
		shared_other_attributes=non_iid_cfg.get("shared_other_attributes", True),
	)


def _attribute_names(base_dataset, cfg=None):
	if cfg is not None:
		configured = _config_attribute_names(cfg)
		if configured:
			return configured
	if hasattr(base_dataset, "_attribute_indices"):
		pairs = sorted(
			base_dataset._attribute_indices.items(), key=lambda item: item[1]
		)
		return [name for name, _ in pairs]
	if hasattr(base_dataset, "_ATTRIBUTE_INDICES"):
		pairs = sorted(
			base_dataset._ATTRIBUTE_INDICES.items(), key=lambda item: item[1]
		)
		return [name for name, _ in pairs]
	return None


def _log_attribute_values(dataset, writer, name, cfg=None):
	base_dataset = _unwrap_subset(dataset)
	if getattr(base_dataset, "_attribute_values", None) is None:
		attribute_values = base_dataset._get_attribute_values()
	else:
		attribute_values = base_dataset._attribute_values
	attribute_names = _attribute_names(base_dataset, cfg=cfg)
	infos = {}
	readable = []
	for idx, values in enumerate(attribute_values):
		label = (
			attribute_names[idx]
			if attribute_names and idx < len(attribute_names)
			else f"attribute_{idx}"
		)
		infos[f"{name}_attribute_values/{label}"] = len(values)
		readable.append(f"{label}: {len(values)}")
	print(f"[data] {name} attribute value counts -> " + ", ".join(readable))
	if writer is None:
		return
	writer.write(infos)


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
		_log_attribute_values(data, writer, key, cfg)
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
		non_iid_cfg = _resolve_non_iid_cfg(cfg)
		apply_to = None
		if non_iid_cfg and not isinstance(non_iid_cfg, str):
			apply_to = non_iid_cfg.get("apply_to")
		if apply_to:
			if cfg.train and "validation" in apply_to:
				datasets.append(("validation_raw", val_data))
			if (not cfg.train) and "testing" in apply_to:
				datasets.append(("testing_raw", data))
		infos = {}
		if hasattr(data, "actual_difficulty"):
			infos[f"{key}_actual_difficulty"] = data.actual_difficulty
			infos[f"{key}_volume"] = data.volume
		writer.write(infos)
		for (name, data) in datasets:
			data = _wrap_non_iid(data, cfg, name)
			loader = DataLoader(
				data,
				batch_size=cfg.batch_size,
				num_workers=cfg.num_workers,
				pin_memory=True,
			)
			d_dataloaders[name] = loader
	return d_dataloaders
