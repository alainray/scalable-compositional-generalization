"""Compositional Risk Minimization (CRM).

Reference: Ahuja et al., "Compositional Risk Minimization" (arXiv:2410.06303),
https://github.com/facebookresearch/compositional-risk-minimization

The method assumes an Additive Energy Distribution (AED)

    p(x | z) = exp(-sum_i E_i(x, z_i)) / Z(z)

for a group ``z = (z_1, ..., z_m)`` of attribute values. Every model in this
repository already emits exactly the required energies: their ``forward``
returns a list of per-attribute logit tensors, which we read as ``-E_i(x, .)``.

CRM then differs from the usual baseline in three places:

1. training uses a **single softmax over the observed group set** instead of one
   independent softmax per attribute;
2. a **learnable scalar per observed group** ``B_hat(z)`` is added to the logits;
3. after training, a closed-form pass over the training set produces
   ``B_star(z) = log Z(z)`` for *every* group (including unseen ones), which
   replaces ``B_hat`` at inference time.

A useful structural fact when debugging: with ``B_hat == 0``, a uniform prior and
the full cartesian product as group set, the joint cross-entropy factorises
exactly into the sum of the per-attribute cross-entropies, so CRM degenerates
into the existing baseline. All of the method's value comes from restricting the
denominator to the observed groups and from the non-additivity of
``B_hat`` / ``B_star``.

Every operation over the group axis is chunked, because that axis reaches
~5e5 entries on mpi3d and a dense ``(batch, num_groups)`` tensor at the test
batch sizes used here would not fit in GPU memory.
"""

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset

from .base import BaseModel

__all__ = [
    "MASKED_LOG_PRIOR",
    "CRMWrapper",
    "GroupSupport",
    "chunked_group_argmax",
    "chunked_group_logsumexp",
    "crm_group_energy",
    "crm_group_logits",
]

# Hard mask for groups that are absent from the training support. The reference
# implementation uses a soft floor (``clip(min=1e-8)`` -> -18.42) which becomes
# indistinguishable from a legitimate uniform prior once the number of groups
# exceeds ~1e5, so we use -inf instead and never a magic constant.
MASKED_LOG_PRIOR = float("-inf")

DEFAULT_GROUP_CHUNK = 65536


def _subset_indices(dataset):
    """Return the indices of ``dataset`` into its innermost base dataset."""
    indices = None
    current = dataset
    while isinstance(current, Subset):
        subset_indices = np.asarray(current.indices, dtype=np.int64)
        indices = subset_indices if indices is None else subset_indices[indices]
        current = current.dataset
    if indices is None:
        indices = np.arange(len(current), dtype=np.int64)
    return indices, current


def dataset_targets(dataset):
    """Return the ``(N, num_attributes)`` integer target matrix behind a dataset.

    Handles ``Subset`` stacks and the ``NonIIDWrapper``; the object axis added by
    every ``_load_data`` (shape ``(N, 1, A)``) is collapsed by keeping the last
    object, matching what the models do with 5-dim inputs.
    """
    if not isinstance(dataset, Subset) and hasattr(dataset, "dataset"):
        # NonIIDWrapper (or any other non-Subset wrapper)
        return dataset_targets(dataset.dataset)
    indices, base = _subset_indices(dataset)
    targets = np.asarray(base._dataset_targets)[indices]
    if targets.ndim == 3:
        targets = targets[:, -1, :]
    return targets.astype(np.int64)


class GroupSupport:
    """Bookkeeping for the group (attribute-combination) space.

    Attributes:
        attribute_sizes: per-attribute cardinalities ``[d_1, ..., d_m]``.
        train_groups: ``(G_tr, m)`` int64 tensor of observed training groups.
        log_p_hat: ``(G_tr,)`` float tensor, log empirical training frequencies.
        eval_groups: ``(G_ev, m)`` int64 tensor of groups scored at test time.
        train_to_eval: ``(G_tr,)`` int64, train group index -> eval group index.
    """

    def __init__(self, attribute_sizes, train_groups, train_counts, eval_groups):
        self.attribute_sizes = [int(s) for s in attribute_sizes]
        self.num_attributes = len(self.attribute_sizes)
        self.train_groups = torch.as_tensor(train_groups, dtype=torch.long)
        counts = torch.as_tensor(np.asarray(train_counts), dtype=torch.double)
        self.train_counts = counts
        self.log_p_hat = torch.log(counts / counts.sum()).float()
        self.eval_groups = torch.as_tensor(eval_groups, dtype=torch.long)
        self._radix = self._make_radix(self.attribute_sizes)
        train_codes = self._encode_cpu(self.train_groups)
        eval_codes = self._encode_cpu(self.eval_groups)
        total = int(np.prod(self.attribute_sizes))
        # dense code -> index lookup tables (int64; <= 4 MB for the largest
        # configuration used here, mpi3d with 518400 groups)
        self._train_lookup = torch.full((total,), -1, dtype=torch.long)
        self._train_lookup[train_codes] = torch.arange(len(train_codes))
        self._eval_lookup = torch.full((total,), -1, dtype=torch.long)
        self._eval_lookup[eval_codes] = torch.arange(len(eval_codes))
        self.train_to_eval = self._eval_lookup[train_codes]
        if bool((self.train_to_eval < 0).any()):
            raise ValueError(
                "eval_group_set does not cover the training support; "
                "B_hat could not be mapped onto the evaluation groups"
            )

    # -- construction ------------------------------------------------------

    @staticmethod
    def _make_radix(attribute_sizes):
        radix = np.ones(len(attribute_sizes), dtype=np.int64)
        for i in range(len(attribute_sizes) - 2, -1, -1):
            radix[i] = radix[i + 1] * attribute_sizes[i + 1]
        return torch.as_tensor(radix, dtype=torch.long)

    def _encode_cpu(self, groups):
        return (groups * self._radix).sum(dim=-1)

    @classmethod
    def from_targets(cls, targets, attribute_sizes, eval_group_set="full_product"):
        """Build the support from an ``(N, m)`` integer target matrix."""
        targets = np.asarray(targets, dtype=np.int64)
        if targets.ndim != 2:
            raise ValueError(f"targets must be 2-dim, got shape {targets.shape}")
        if targets.shape[1] != len(attribute_sizes):
            raise ValueError(
                f"targets have {targets.shape[1]} attributes but attribute_sizes "
                f"has {len(attribute_sizes)} entries"
            )
        for i, size in enumerate(attribute_sizes):
            hi = int(targets[:, i].max()) if targets.shape[0] else -1
            lo = int(targets[:, i].min()) if targets.shape[0] else 0
            if hi >= size or lo < 0:
                raise ValueError(
                    f"attribute {i} has values in [{lo}, {hi}], outside the "
                    f"declared cardinality {size}"
                )
        train_groups, train_counts = np.unique(targets, axis=0, return_counts=True)
        if eval_group_set == "full_product":
            eval_groups = np.array(
                list(itertools.product(*[range(s) for s in attribute_sizes])),
                dtype=np.int64,
            )
        elif eval_group_set == "train_support":
            eval_groups = train_groups
        else:
            raise ValueError(f"Unknown eval_group_set {eval_group_set!r}")
        return cls(attribute_sizes, train_groups, train_counts, eval_groups)

    @classmethod
    def from_loader(cls, loader, attribute_sizes, eval_group_set="full_product"):
        """Build the support from the dataset behind a training ``DataLoader``."""
        return cls.from_targets(
            dataset_targets(loader.dataset),
            attribute_sizes,
            eval_group_set=eval_group_set,
        )

    # -- indexing ----------------------------------------------------------

    def train_index(self, targets):
        """Map ``(N, m)`` targets to indices into ``train_groups`` (-1 if absent)."""
        codes = (targets * self._radix.to(targets.device)).sum(dim=-1)
        return self._train_lookup.to(targets.device)[codes]

    def eval_index(self, targets):
        """Map ``(N, m)`` targets to indices into ``eval_groups`` (-1 if absent)."""
        codes = (targets * self._radix.to(targets.device)).sum(dim=-1)
        return self._eval_lookup.to(targets.device)[codes]

    def to(self, device):
        self.train_groups = self.train_groups.to(device)
        self.eval_groups = self.eval_groups.to(device)
        self.log_p_hat = self.log_p_hat.to(device)
        self._radix = self._radix.to(device)
        self._train_lookup = self._train_lookup.to(device)
        self._eval_lookup = self._eval_lookup.to(device)
        self.train_to_eval = self.train_to_eval.to(device)
        return self

    @property
    def num_train_groups(self):
        return int(self.train_groups.shape[0])

    @property
    def num_eval_groups(self):
        return int(self.eval_groups.shape[0])

    def summary(self):
        counts = self.train_counts
        total = int(np.prod(self.attribute_sizes))
        return {
            "crm/num_attributes": self.num_attributes,
            "crm/num_groups_total": total,
            "crm/num_groups_train": self.num_train_groups,
            "crm/num_groups_eval": self.num_eval_groups,
            "crm/num_groups_unseen": total - self.num_train_groups,
            "crm/samples_per_group_min": int(counts.min().item()),
            "crm/samples_per_group_median": float(counts.median().item()),
            "crm/samples_per_group_max": int(counts.max().item()),
        }


def crm_group_energy(energies, groups):
    """Additive group scores ``s(x, z) = sum_i energies[i][:, z_i]``.

    Args:
        energies: list of ``m`` tensors of shape ``(B, d_i)``. These are read as
            ``-E_i(x, .)``, i.e. the per-attribute logits the models already emit.
        groups: ``(G, m)`` int64 tensor of attribute tuples.

    Returns:
        ``(B, G)`` float32 tensor.
    """
    out = energies[0].float().index_select(1, groups[:, 0])
    for i in range(1, len(energies)):
        out = out + energies[i].float().index_select(1, groups[:, i])
    return out


def crm_group_logits(energies, groups, bias=None, log_prior=None):
    """CRM logits ``s(x, z) + log p(z) - B(z)`` over a group set.

    ``bias`` and ``log_prior`` are ``(G,)`` tensors (or ``None``). Absent groups
    carry ``-inf`` in ``log_prior``; this propagates through the sum and yields
    exactly zero probability after the softmax.
    """
    logits = crm_group_energy(energies, groups)
    if log_prior is not None:
        logits = logits + log_prior.float().unsqueeze(0)
    if bias is not None:
        logits = logits - bias.float().unsqueeze(0)
    return logits


def _group_chunks(num_groups, chunk):
    chunk = num_groups if not chunk else min(chunk, num_groups)
    for start in range(0, num_groups, chunk):
        yield start, min(start + chunk, num_groups)


@torch.no_grad()
def chunked_group_argmax(energies, groups, bias=None, log_prior=None, chunk=None):
    """``argmax_z [ s(x,z) + log p(z) - B(z) ]`` without materialising ``(B, G)``."""
    best_val = None
    best_idx = None
    for start, stop in _group_chunks(groups.shape[0], chunk):
        logits = crm_group_logits(
            energies,
            groups[start:stop],
            bias=None if bias is None else bias[start:stop],
            log_prior=None if log_prior is None else log_prior[start:stop],
        )
        val, idx = logits.max(dim=1)
        idx = idx + start
        if best_val is None:
            best_val, best_idx = val, idx
        else:
            take = val > best_val
            best_val = torch.where(take, val, best_val)
            best_idx = torch.where(take, idx, best_idx)
    return best_idx


def _shifted_exp_sum(logits, shift):
    """``sum_z exp(logits - shift)``, treating ``-inf - -inf`` as 0 rather than NaN.

    A chunk can be entirely masked (all groups absent from the support), in which
    case both ``logits`` and ``shift`` are ``-inf`` and the plain difference is
    NaN, which would then poison the running accumulator.
    """
    diff = logits - shift.unsqueeze(1)
    diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("-inf")))
    return diff.exp().sum(dim=1)


@torch.no_grad()
def chunked_group_logsumexp(energies, groups, bias=None, log_prior=None, chunk=None):
    """``LSE_z [ s(x,z) + log p(z) - B(z) ]`` accumulated with a running maximum."""
    running_max = None
    running_sum = None
    for start, stop in _group_chunks(groups.shape[0], chunk):
        logits = crm_group_logits(
            energies,
            groups[start:stop],
            bias=None if bias is None else bias[start:stop],
            log_prior=None if log_prior is None else log_prior[start:stop],
        )
        chunk_max = logits.max(dim=1).values
        if running_max is None:
            running_max = chunk_max
            running_sum = _shifted_exp_sum(logits, chunk_max)
            continue
        new_max = torch.maximum(running_max, chunk_max)
        rescale = torch.where(
            torch.isfinite(new_max),
            torch.exp(running_max - new_max),
            torch.zeros_like(new_max),
        )
        running_sum = running_sum * rescale + _shifted_exp_sum(logits, new_max)
        running_max = new_max
    return torch.log(running_sum.clamp_min(1e-38)) + running_max


class CRMWrapper(BaseModel):
    """Wraps any model of this repo with the CRM group head.

    The wrapped model must return a list of per-attribute logit tensors from its
    ``forward``. The wrapper owns the learnable per-group bias ``B_hat`` and,
    after the post-hoc step, the extrapolated bias ``B_star``.

    This replaces the wrapped model's ``train_step`` entirely, so an auxiliary
    objective is only applied if the model exposes ``crm_outputs`` (see
    ``SplitResNet18Mixer``) *and* ``aux_loss_weight`` is non-zero, in which case
    the total is ``crm_group_ce + aux_loss_weight * aux``. Anything else the
    model computes inside its own ``train_step`` is dropped -- notably the
    ``exit_reg`` early-exit loss of ``SplitResNet18``.
    """

    def __init__(
        self,
        base_model,
        support: GroupSupport,
        test_prior: str = "uniform",
        report_baseline_metrics: bool = True,
        group_chunk: int = DEFAULT_GROUP_CHUNK,
        crm_metrics_on_train: bool = False,
        aux_loss_weight: float = 0.0,
    ):
        super().__init__(
            attributes=base_model.attributes,
            objective=base_model.objective,
            loss_fn=base_model.loss_fn,
            metric_fns=base_model.metric_fns,
        )
        self.base_model = base_model
        self.support = support
        self.test_prior = test_prior
        self.report_baseline_metrics = report_baseline_metrics
        self.group_chunk = group_chunk
        # Scoring the full evaluation group set costs O(batch * G_eval * m) per
        # call. On mpi3d that is ~1.6e9 gathers per batch, i.e. more than the
        # backbone itself, so it is off during training by default; the metrics
        # that matter (val/test) are unaffected.
        self.crm_metrics_on_train = crm_metrics_on_train
        # Weight of the wrapped model's compositional term (mixer / algebraic).
        # 0 disables it entirely, which is the plain-CRM behaviour.
        self.aux_loss_weight = float(aux_loss_weight)
        if self.aux_loss_weight != 0.0 and not hasattr(base_model, "crm_outputs"):
            raise ValueError(
                f"aux_loss_weight={aux_loss_weight} was requested but "
                f"{type(base_model).__name__} does not implement crm_outputs(); "
                "CRM has no auxiliary term to combine with"
            )
        self.attribute_sizes = support.attribute_sizes

        self.b_hat = nn.Parameter(torch.zeros(support.num_train_groups))
        self.register_buffer("log_p_hat", support.log_p_hat.clone())
        self.register_buffer("b_star", torch.zeros(support.num_eval_groups))
        self.register_buffer("b_star_ess", torch.zeros(support.num_eval_groups))
        self.register_buffer("has_b_star", torch.zeros((), dtype=torch.bool))

        self._logged_metrics = ["loss", "acc"]
        self._logged_metrics += [f"attributes/loss_{a}" for a in self.attributes]
        self._logged_metrics += [f"attributes/acc_{a}" for a in self.attributes]
        self._logged_metrics += ["crm_acc", "crm_naive_acc"]
        if report_baseline_metrics:
            self._logged_metrics += ["baseline_acc"]
        if self.aux_loss_weight != 0.0:
            self._logged_metrics += ["mixer_loss", "total_loss"]

    # -- plumbing ----------------------------------------------------------

    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self.support.to(self.b_hat.device)
        return out

    @property
    def preprocessing(self):
        return getattr(self.base_model, "preprocessing", None)

    def plot_debug(self, x, path, **kwargs):
        fn = getattr(self.base_model, "plot_debug", None)
        if callable(fn):
            return fn(x, path, **kwargs)
        return [], []

    @torch.no_grad()
    def extract_representation(self, x):
        return self.base_model.extract_representation(x)

    def energies_and_aux(self, x, y=None):
        """``(per-attribute logits, auxiliary loss)`` from a single encoder pass.

        If the wrapped model exposes ``crm_outputs`` (see
        ``SplitResNet18Mixer.crm_outputs``) its compositional term is returned
        alongside the logits, so CRM can be combined with it without running the
        encoder twice. Otherwise the auxiliary loss is zero.
        """
        if self.aux_loss_weight != 0.0 and hasattr(self.base_model, "crm_outputs"):
            out, aux = self.base_model.crm_outputs(x, y)
        else:
            if x.dim() == 5:
                x = x.reshape(-1, *x.shape[2:])
            out = self.base_model(x)
            aux = torch.zeros((), device=out[0].device if out else self.b_hat.device)
        if not isinstance(out, (list, tuple)):
            raise TypeError(
                f"{type(self.base_model).__name__}.forward must return a list of "
                "per-attribute logits for CRM"
            )
        return list(out), aux

    def energies(self, x):
        """Per-attribute logits of the wrapped model, as a list of ``(B, d_i)``."""
        return self.energies_and_aux(x)[0]

    def forward(self, x):
        return self.energies(x)

    @staticmethod
    def _flatten_targets(y):
        return y.reshape(-1, y.shape[-1])

    # -- CRM heads ---------------------------------------------------------

    def train_logits(self, energies):
        """Group logits over the observed training support (used for the loss)."""
        return crm_group_logits(
            energies,
            self.support.train_groups,
            bias=self.b_hat,
            log_prior=self.log_p_hat,
        )

    def _eval_bias_and_prior(self, extrapolate):
        """``(bias, log_prior)`` over the evaluation group set.

        ``extrapolate=True`` uses the post-hoc ``B_star`` with the configured test
        prior (uniform by default). ``extrapolate=False`` is the naive control
        from the paper's ablation: the learned ``B_hat`` with the training prior,
        and unseen groups hard-masked out.
        """
        device = self.b_hat.device
        num_eval = self.support.num_eval_groups
        if extrapolate and bool(self.has_b_star):
            bias = self.b_star
            if self.test_prior == "uniform":
                log_prior = torch.zeros(num_eval, device=device)
            elif self.test_prior == "empirical":
                log_prior = torch.full(
                    (num_eval,), MASKED_LOG_PRIOR, device=device
                ).index_copy(0, self.support.train_to_eval, self.log_p_hat)
            else:
                raise ValueError(f"Unknown test_prior {self.test_prior!r}")
            return bias, log_prior
        bias = torch.zeros(num_eval, device=device).index_copy(
            0, self.support.train_to_eval, self.b_hat.detach()
        )
        log_prior = torch.full((num_eval,), MASKED_LOG_PRIOR, device=device).index_copy(
            0, self.support.train_to_eval, self.log_p_hat
        )
        return bias, log_prior

    @torch.no_grad()
    def predict_groups(self, energies, extrapolate=True):
        """Joint CRM prediction: ``(B, m)`` predicted attribute tuples."""
        bias, log_prior = self._eval_bias_and_prior(extrapolate)
        idx = chunked_group_argmax(
            energies,
            self.support.eval_groups,
            bias=bias,
            log_prior=log_prior,
            chunk=self.group_chunk,
        )
        return self.support.eval_groups[idx]

    # -- steps -------------------------------------------------------------

    def _compute_loss(self, energies, y, grad=True):
        """Joint group cross-entropy over the observed support.

        Samples whose group is absent from the training support (which is the
        norm on the compositional test split) carry no signal for the group
        cross-entropy, so they are dropped rather than silently folded into
        group 0.

        When gradients are not needed the loss is computed as
        ``LSE_z logit(x,z) - logit(x, z_true)`` with the log-sum-exp chunked over
        the group axis, which keeps peak memory at ``O(batch * group_chunk)``
        instead of ``O(batch * G_train)`` -- the eval batch sizes in this repo
        (up to 8192) would otherwise not fit.
        """
        targets = self._flatten_targets(y)
        group_idx = self.support.train_index(targets)
        valid = group_idx >= 0
        with torch.amp.autocast("cuda", enabled=False):
            if grad:
                logits = self.train_logits(energies)
                if not bool(valid.all()):
                    logits = logits[valid]
                    group_idx = group_idx[valid]
                if group_idx.numel() == 0:
                    return logits.sum() * 0.0, targets
                return F.cross_entropy(logits, group_idx), targets

            if not bool(valid.any()):
                return torch.zeros((), device=targets.device), targets
            kept = [e[valid] for e in energies]
            idx = group_idx[valid]
            lse = chunked_group_logsumexp(
                kept,
                self.support.train_groups,
                bias=self.b_hat,
                log_prior=self.log_p_hat,
                chunk=self.group_chunk,
            )
            target_groups = self.support.train_groups[idx]
            target_logit = torch.zeros_like(lse)
            for i, e in enumerate(kept):
                target_logit = target_logit + e.float().gather(
                    1, target_groups[:, i].unsqueeze(1)
                ).squeeze(1)
            target_logit = target_logit + self.log_p_hat[idx] - self.b_hat[idx]
            return (lse - target_logit).mean(), targets

    def _attribute_losses(self, energies, targets):
        """Per-attribute cross-entropies, kept so the existing logging still works."""
        return [
            F.cross_entropy(energies[i].float(), targets[..., i])
            for i in range(len(energies))
        ]

    @torch.no_grad()
    def _crm_metrics(self, energies, targets):
        """Joint accuracy under the CRM decision rule, plus the baseline rule."""
        out = {}
        for name, extrapolate in (("crm_acc", True), ("crm_naive_acc", False)):
            pred = self.predict_groups(energies, extrapolate=extrapolate)
            out[name] = (pred == targets).all(dim=1).float().mean().item() * 100.0
        if self.report_baseline_metrics:
            corr = torch.ones(targets.shape[0], dtype=torch.bool, device=targets.device)
            for i, e in enumerate(energies):
                corr = corr & (e.argmax(dim=-1) == targets[..., i])
            out["baseline_acc"] = corr.float().mean().item() * 100.0
        return out

    def _log_dict(self, loss, energies, targets, crm_metrics=True):
        attr_losses = self._attribute_losses(energies, targets)
        metrics, attr_metrics = self._compute_metrics(energies, targets)
        log_dict = self._compose_logging_dict(loss, attr_losses, metrics, attr_metrics)
        if crm_metrics:
            log_dict |= self._crm_metrics(energies, targets)
        return log_dict

    def train_step(self, x, y, optimizer, amp_scaler=None, **kwargs):
        step_optimizer = kwargs.get("step_optimizer", True)
        grad_accum_steps = kwargs.get("grad_accum_steps", 1)
        divisor = max(1, grad_accum_steps)
        if amp_scaler:
            with torch.amp.autocast("cuda"):
                energies, aux = self.energies_and_aux(x, y)
            # the group softmax itself always runs in float32: a logsumexp over
            # up to 5e5 terms is not safe in half precision
            loss, targets = self._compute_loss(energies, y)
            total = loss + self.aux_loss_weight * aux.float()
            amp_scaler.scale(total / divisor).backward()
            if step_optimizer:
                # `unscale_` before clipping, or the norm is measured on the
                # scaled gradients (scale * true norm) and the clip threshold is
                # meaningless. Then always call step()/update(): the scaler
                # skips the step itself when it finds inf/nan, and update() is
                # what backs the scale off afterwards. Gating update() behind a
                # finiteness check deadlocks training -- the scale never
                # decreases, so every later step overflows too and the model
                # stops updating for the rest of the run.
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e3)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            energies, aux = self.energies_and_aux(x, y)
            loss, targets = self._compute_loss(energies, y)
            total = loss + self.aux_loss_weight * aux.float()
            (total / divisor).backward()
            if step_optimizer:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=1e3
                )
                if grad_norm.isfinite():
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            log_dict = self._log_dict(
                loss.detach(),
                [e.detach().float() for e in energies],
                targets,
                crm_metrics=self.crm_metrics_on_train,
            )
            return log_dict | self._aux_log(aux, total)

    @torch.no_grad()
    def validation_step(self, x, y=None, **kwargs):
        energies, aux = self.energies_and_aux(x, y)
        energies = [e.float() for e in energies]
        loss, targets = self._compute_loss(energies, y, grad=False)
        log_dict = self._log_dict(loss, energies, targets)
        return log_dict | self._aux_log(aux, loss + self.aux_loss_weight * aux.float())

    def _aux_log(self, aux, total):
        if self.aux_loss_weight == 0.0:
            return {}
        return {"mixer_loss": aux.item(), "total_loss": total.item()}

    # -- step 2: extrapolated bias ----------------------------------------

    @torch.no_grad()
    def compute_extrapolated_bias(
        self,
        loader,
        device,
        max_samples=None,
        group_chunk=None,
        verbose=False,
    ):
        """Closed-form ``B_star(z) = log Z(z)`` for every evaluation group.

        Implements Eq. (11) of the paper in the numerically stable double
        log-sum-exp form::

            inner(x)  = LSE_{z' in Z_train} [ s(x,z') + log p(z') - B_hat(z') ]
            B_star(z) = LSE_{x in D}        [ s(x,z) - inner(x) ] - log N

        The outer LSE is accumulated across batches with a running maximum, so
        peak memory is ``O(batch * group_chunk)`` regardless of dataset size.
        Also records the effective sample size of the importance-sampling
        estimator per group -- the key diagnostic once the group count is large,
        since ``B_star`` is self-normalised importance sampling with the training
        marginal ``p(x)`` as proposal.
        """
        group_chunk = group_chunk or self.group_chunk
        was_training = self.training
        self.eval()
        num_eval = self.support.num_eval_groups
        running_max = torch.full((num_eval,), float("-inf"), device=device)
        running_sum = torch.zeros(num_eval, device=device)
        running_sum_sq = torch.zeros(num_eval, device=device)
        seen = 0

        for x, _ in loader:
            if max_samples is not None and seen >= max_samples:
                break
            energies = [e.float() for e in self.energies(x.to(device))]
            n = energies[0].shape[0]
            if max_samples is not None and seen + n > max_samples:
                n = max_samples - seen
                energies = [e[:n] for e in energies]
            inner = chunked_group_logsumexp(
                energies,
                self.support.train_groups,
                bias=self.b_hat,
                log_prior=self.log_p_hat,
                chunk=group_chunk,
            )
            for start, stop in _group_chunks(num_eval, group_chunk):
                groups = self.support.eval_groups[start:stop]
                w = crm_group_energy(energies, groups) - inner.unsqueeze(1)
                new_max = torch.maximum(running_max[start:stop], w.max(dim=0).values)
                rescale = torch.where(
                    torch.isfinite(new_max),
                    torch.exp(running_max[start:stop] - new_max),
                    torch.zeros_like(new_max),
                )
                contrib = torch.exp(w - new_max.unsqueeze(0))
                running_sum[start:stop] = running_sum[
                    start:stop
                ] * rescale + contrib.sum(dim=0)
                running_sum_sq[start:stop] = running_sum_sq[start:stop] * rescale.pow(
                    2
                ) + contrib.pow(2).sum(dim=0)
                running_max[start:stop] = new_max
            seen += n
            if verbose:
                print(f"[crm] B* pass: {seen} samples", flush=True)

        if seen == 0:
            raise RuntimeError("CRM step 2 saw no samples")
        b_star = torch.log(running_sum.clamp_min(1e-38)) + running_max
        b_star = b_star - float(np.log(seen))
        ess = running_sum.pow(2) / running_sum_sq.clamp_min(1e-38)
        self.b_star.copy_(b_star)
        self.b_star_ess.copy_(ess)
        self.has_b_star.fill_(True)
        if was_training:
            self.train()
        return {
            "crm/b_star_samples": seen,
            "crm/b_star_ess_min": float(ess.min().item()),
            "crm/b_star_ess_p05": float(torch.quantile(ess, 0.05).item()),
            "crm/b_star_ess_median": float(ess.median().item()),
        }
