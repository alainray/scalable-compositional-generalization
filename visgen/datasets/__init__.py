import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from .cars3d import Cars3D
from .clevr import CLEVR
from .dsprites import DSprites
from .iraven import IRAVEN
from .mpi3d import MPI3D
from .non_iid import NonIIDWrapper, subset_with_four_case_support
from .shapes3d import Shapes3D
from .splits import make_validation_subset, training_subset_with_optional_ood_holdout


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


def _non_iid_cfg_for_split(cfg, split_name):
	non_iid_cfg = _resolve_non_iid_cfg(cfg)
	if not non_iid_cfg or isinstance(non_iid_cfg, str):
		return non_iid_cfg
	split_overrides = non_iid_cfg.get("split_overrides", {})
	if split_name and split_name in split_overrides:
		return {**non_iid_cfg, **split_overrides[split_name]}
	return non_iid_cfg


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


def _wrap_non_iid(dataset, cfg, split_name=None, fallback_seed=None):
	non_iid_cfg = _non_iid_cfg_for_split(cfg, split_name)
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
		seed=non_iid_cfg.get("seed", fallback_seed),
		allowed_attributes=allowed_attributes,
		shared_other_attributes=non_iid_cfg.get("shared_other_attributes", True),
		fully_iid=non_iid_cfg.get("fully_iid", False),
		deterministic=non_iid_cfg.get("deterministic", False),
		precompute_deterministic=non_iid_cfg.get("precompute_deterministic", False),
		max_deterministic_candidates=non_iid_cfg.get("max_deterministic_candidates", 1_024),
	)




def _val_4cases_cfg(cfg):
	default = {
		"enabled": False,
		"shared_other_attributes": None,
		"allowed_attributes": None,
		"replace_validation": False,
		"export_split": True,
	}
	v4_cfg = getattr(cfg, "val_4cases", None)
	if v4_cfg is None:
		return default
	if isinstance(v4_cfg, bool):
		default["enabled"] = v4_cfg
		return default
	for key in default.keys():
		if key in v4_cfg:
			default[key] = v4_cfg[key]
	return default

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


def _seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def get_dataloaders(data_cfg, writer=None, seed=None):
	dataset_map = {
		"dsprites": DSprites,
		"iraven": IRAVEN,
		"mpi3d": MPI3D,
		"shapes3d": Shapes3D,
		"cars3d": Cars3D,
		"clevr": CLEVR,
	}
	d_dataloaders = {}
	for data_split_idx, (key, cfg) in enumerate(data_cfg.items()):
		base_seed = None if seed is None else int(seed) + data_split_idx
		split_generator = (
			None if base_seed is None else torch.Generator().manual_seed(base_seed)
		)
		data = dataset_map[cfg.dataset](**cfg)
		_log_attribute_values(data, writer, key, cfg)
		if cfg.train:
			num_ood_val = cfg.num_ood_val if "num_ood_val" in cfg else 1
			train_data, ood_val_sets = data.ood_validation_split(num_ood_val)
			val_size = int(cfg.val_fraction * len(train_data))
			train_size = len(train_data) - val_size
			train_data, val_data = random_split(
				train_data, [train_size, val_size], generator=split_generator
			)
			non_iid_cfg = _resolve_non_iid_cfg(cfg)
			default_shared_other_attributes = True
			if non_iid_cfg and not isinstance(non_iid_cfg, str):
				default_shared_other_attributes = non_iid_cfg.get(
					"shared_other_attributes", True
				)
			v4_cfg = _val_4cases_cfg(cfg)
			datasets = [
				(key, train_data),
				("validation", val_data),
			]
			if v4_cfg["enabled"]:
				allowed_attributes = v4_cfg["allowed_attributes"]
				if not allowed_attributes:
					allowed_attributes = _config_attribute_names(cfg)
				shared_other_attributes = v4_cfg["shared_other_attributes"]
				if shared_other_attributes is None:
					shared_other_attributes = default_shared_other_attributes
				val_4cases = subset_with_four_case_support(
					val_data,
					allowed_attributes=allowed_attributes,
					shared_other_attributes=shared_other_attributes,
				)
				print(
					f"[data] val_4cases size: {len(val_4cases)}/{len(val_data)}"
				)
				if v4_cfg["replace_validation"]:
					datasets = [
						(key, train_data),
						("validation", val_4cases),
					]
				if v4_cfg["export_split"]:
					datasets.append(("val_4cases", val_4cases))
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
			if cfg.train and "val_4cases" in apply_to:
				val_4cases_data = None
				for split_name, split_data in datasets:
					if split_name == "val_4cases":
						val_4cases_data = split_data
						break
				if val_4cases_data is not None:
					datasets.append(("val_4cases_raw", val_4cases_data))
			if (not cfg.train) and "testing" in apply_to:
				datasets.append(("testing_raw", data))
		infos = {}
		if hasattr(data, "actual_difficulty"):
			infos[f"{key}_actual_difficulty"] = data.actual_difficulty
			infos[f"{key}_volume"] = data.volume
		writer.write(infos)
		for (name, data) in datasets:
			data = _wrap_non_iid(data, cfg, name, fallback_seed=base_seed)
			loader = DataLoader(
				data,
				batch_size=cfg.batch_size,
				num_workers=cfg.num_workers,
				pin_memory=True,
				worker_init_fn=_seed_worker if base_seed is not None else None,
				generator=split_generator,
			)
			d_dataloaders[name] = loader
	return d_dataloaders
