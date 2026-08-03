"""Trainer for Compositional Risk Minimization.

Two stages, as in Ahuja et al. (arXiv:2410.06303):

1. the usual training loop, but with the CRM group head (``CRMWrapper``) and a
   single softmax over the observed group set;
2. a closed-form post-hoc pass that produces the extrapolated bias
   ``B_star(z) = log Z(z)`` for every group, including unseen combinations.

The paper's ablation shows stage 2 is the whole method (worst-group accuracy on
Waterbirds drops 78.7 -> 55.7 without it), so ``crm_acc`` (with ``B_star``) and
``crm_naive_acc`` (with the learned ``B_hat``) are both reported.
"""

import os

import torch
from torch.utils.data import DataLoader

from visgen.datasets.non_iid import NonIIDWrapper
from visgen.models.crm import CRMWrapper, GroupSupport
from visgen.utils.general import load_checkpoint

from .optimizers import get_optimizer
from .trainer import BaseTrainer


def unwrapped_train_loader(loader):
    """A loader over the same samples, without the ``NonIIDWrapper`` resampling.

    ``B_star`` is an expectation over the training marginal ``p(x)``, so it must
    be estimated with each training sample contributing once. The 4-view wrapper
    resamples quadruples, which would silently reweight the proposal
    distribution of the importance-sampling estimator.
    """
    dataset = loader.dataset
    if not isinstance(dataset, NonIIDWrapper):
        return loader
    return DataLoader(
        dataset.dataset,
        batch_size=loader.batch_size * 4,
        num_workers=loader.num_workers,
        pin_memory=True,
    )


class CRMTrainer(BaseTrainer):
    """``training.trainer: crm`` -- wraps any model of the repo with CRM."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        crm_cfg = cfg.get("crm", {}) or {}
        self.crm_cfg = crm_cfg
        self.b_weight_decay = float(crm_cfg.get("b_weight_decay", 0.0))
        self.eval_group_set = crm_cfg.get("eval_group_set", "full_product")
        self.test_prior = crm_cfg.get("test_prior", "uniform")
        self.group_chunk = int(crm_cfg.get("group_chunk", 65536))
        self.extrapolate_every = int(crm_cfg.get("extrapolate_every_n_epochs", 0))
        self.extrapolate_samples = crm_cfg.get("extrapolate_num_samples", 20000)
        self.report_baseline_metrics = bool(
            crm_cfg.get("report_baseline_metrics", True)
        )
        self.crm_metrics_on_train = bool(crm_cfg.get("crm_metrics_on_train", False))

    # -- stage 1 ----------------------------------------------------------

    def build_model(self, model, d_dataloaders, writer=None):
        train_loader = d_dataloaders["training"]
        attribute_sizes = self._attribute_sizes(model)
        support = GroupSupport.from_loader(
            train_loader, attribute_sizes, eval_group_set=self.eval_group_set
        )
        summary = support.summary()
        print(
            "[crm] " + ", ".join(f"{k.split('/')[-1]}={v}" for k, v in summary.items())
        )
        if writer is not None:
            writer.write(summary)
        aux_weight = self._aux_loss_weight(model)
        if aux_weight:
            print(
                f"[crm] auxiliary term active: "
                f"{getattr(model, 'mixer_mode', '?')} x {aux_weight}"
            )
        wrapped = CRMWrapper(
            model,
            support,
            test_prior=self.test_prior,
            report_baseline_metrics=self.report_baseline_metrics,
            group_chunk=self.group_chunk,
            crm_metrics_on_train=self.crm_metrics_on_train,
            aux_loss_weight=aux_weight,
        ).to(self.device)
        self._support_summary = summary
        return wrapped

    def _aux_loss_weight(self, model):
        """Weight of the model's compositional term (mixer / algebraic) under CRM.

        Defaults to the model config's own ``mixer.loss_weight`` -- the same knob
        that governs it outside CRM -- and is 0 for models that expose no such
        term. ``training.crm.aux_loss_weight`` overrides it.
        """
        override = self.crm_cfg.get("aux_loss_weight")
        if override is not None:
            return float(override)
        if not hasattr(model, "crm_outputs"):
            return 0.0
        return float(getattr(model, "mixer_loss_weight", 0.0))

    @staticmethod
    def _attribute_sizes(model):
        sizes = getattr(model, "attribute_sizes", None)
        if sizes:
            return [int(s) for s in sizes]
        raise ValueError(
            f"{type(model).__name__} does not expose attribute_sizes; CRM needs "
            "the per-attribute cardinalities to build the group space"
        )

    def build_optimizer(self, model):
        """Give ``B_hat`` its own weight decay.

        With one scalar per group and as little as one sample per group (cars3d),
        ``B_hat`` is a high-variance estimate; shrinking it towards 0 is the
        cheapest available regulariser.
        """
        opt_cfg = {k: v for k, v in self.cfg["optimizer"].items()}
        b_params = [p for n, p in model.named_parameters() if n == "b_hat"]
        rest = [p for n, p in model.named_parameters() if n != "b_hat"]
        if not b_params:
            return get_optimizer(self.cfg["optimizer"], model.parameters())
        groups = [
            {"params": rest},
            {"params": b_params, "weight_decay": self.b_weight_decay},
        ]
        return get_optimizer(opt_cfg, groups)

    # -- stage 2 ----------------------------------------------------------

    def on_epoch_end(self, model, d_dataloaders, i_epoch, logams):
        if self.extrapolate_every <= 0 or i_epoch % self.extrapolate_every != 0:
            return {}
        stats = model.compute_extrapolated_bias(
            unwrapped_train_loader(d_dataloaders["training"]),
            device=self.device,
            max_samples=self.extrapolate_samples,
            group_chunk=self.group_chunk,
        )
        return stats

    def finalize(self, model, d_dataloaders, savepath, best_ams, prefix=""):
        """Full ``B_star`` pass on the selected checkpoint, then a final eval."""
        model_best = os.path.join(savepath, "model_best.pth.tar")
        epoch = None
        if os.path.exists(model_best):
            model, _, epoch, _ = load_checkpoint(model_best, model, None, self.device)
            print(f"[crm] step 2 on best checkpoint (epoch {epoch})")
        else:
            print("[crm] step 2 on the last-epoch model (no checkpoint on disk)")

        stats = model.compute_extrapolated_bias(
            unwrapped_train_loader(d_dataloaders["training"]),
            device=self.device,
            max_samples=self.crm_cfg.get("final_extrapolate_num_samples"),
            group_chunk=self.group_chunk,
            verbose=bool(self.cfg.get("verbose", False)),
        )
        out = dict(stats)
        out |= getattr(self, "_support_summary", {})

        # `model_best` is written during the training loop, when b_star is still
        # zeros -- step 2 only runs here. Without this the extrapolated bias is
        # used for the final_* metrics and then lost, so re-evaluating with the
        # CRM decision rule later would mean redoing the whole pass over the
        # training set.
        crm_path = os.path.join(savepath, "model_crm.pth.tar")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": None,
                "best_ams": best_ams,
            },
            crm_path,
        )
        print(f"[crm] B* guardado en {crm_path}")

        model.eval()
        for name in ("validation", "testing"):
            loader = d_dataloaders.get(name)
            if loader is None:
                continue
            short = {"validation": "val", "testing": "test"}[name]
            out |= {
                f"{prefix}final_{short}_{k}": v
                for k, v in self._evaluate(model, loader).items()
            }
        ood_keys = sorted(
            (k for k in d_dataloaders if k.startswith("ood_validation_")),
            key=lambda k: int(k.split("ood_validation_")[1]),
        )
        ood_accs = []
        for i, key in enumerate(ood_keys):
            res = self._evaluate(model, d_dataloaders[key])
            ood_accs.append(res["crm_acc"])
            out |= {f"{prefix}final_ood_val_{i}_{k}": v for k, v in res.items()}
        if ood_accs and f"{prefix}final_val_crm_acc" in out:
            out[f"{prefix}final_wio_crm_acc"] = (
                out[f"{prefix}final_val_crm_acc"] + (min(ood_accs) - 100) / 10
            )
        return out

    @torch.no_grad()
    def _evaluate(self, model, loader):
        totals = {}
        count = 0
        for x, y in loader:
            res = model.validation_step(x=x.to(self.device), y=y.to(self.device))
            n = y.shape[0]
            for k, v in res.items():
                totals[k] = totals.get(k, 0.0) + float(v) * n
            count += n
        if count == 0:
            return {}
        return {k: v / count for k, v in totals.items()}
