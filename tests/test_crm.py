"""Tests for the CRM (Compositional Risk Minimization) implementation.

The critical one is ``test_reduces_to_baseline_*``: with ``B_hat == 0``, a
uniform prior and the full cartesian product as group set, the joint group
cross-entropy must factorise exactly into the sum of the per-attribute
cross-entropies. If that holds, the gather/indexing machinery is correct.
"""

import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from visgen.models.crm import (
    CRMWrapper,
    GroupSupport,
    chunked_group_argmax,
    chunked_group_logsumexp,
    crm_group_energy,
    crm_group_logits,
)
from visgen.models.losses import AttributeCrossEntropyLoss
from visgen.models.metrics import MultiAccuracy

ATTRIBUTE_SIZES = [3, 4, 2]


def full_product_targets(attribute_sizes):
    return np.array(
        list(itertools.product(*[range(s) for s in attribute_sizes])), dtype=np.int64
    )


def random_energies(batch, attribute_sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, s, generator=g) for s in attribute_sizes]


# -- GroupSupport ---------------------------------------------------------


def test_support_indexing_roundtrip():
    targets = full_product_targets(ATTRIBUTE_SIZES)
    support = GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)
    assert support.num_train_groups == int(np.prod(ATTRIBUTE_SIZES))
    assert support.num_eval_groups == int(np.prod(ATTRIBUTE_SIZES))
    idx = support.train_index(torch.as_tensor(targets))
    assert torch.equal(support.train_groups[idx], torch.as_tensor(targets))


def test_support_marks_absent_groups():
    targets = full_product_targets(ATTRIBUTE_SIZES)
    keep = ~((targets[:, 0] == 2) & (targets[:, 1] == 3))
    support = GroupSupport.from_targets(targets[keep], ATTRIBUTE_SIZES)
    assert support.num_train_groups == int(np.prod(ATTRIBUTE_SIZES)) - 2
    absent = torch.as_tensor(np.array([[2, 3, 0]], dtype=np.int64))
    assert int(support.train_index(absent).item()) == -1
    # every training group must still exist in the evaluation set
    assert bool((support.train_to_eval >= 0).all())


def test_support_rejects_out_of_range_values():
    targets = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.int64)
    with pytest.raises(ValueError):
        GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)


def test_log_p_hat_matches_empirical_frequencies():
    targets = np.array([[0, 0, 0]] * 3 + [[1, 2, 1]] * 1, dtype=np.int64)
    support = GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)
    probs = support.log_p_hat.exp()
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
    order = support.train_index(torch.as_tensor(np.array([[0, 0, 0], [1, 2, 1]])))
    assert torch.allclose(probs[order], torch.tensor([0.75, 0.25]), atol=1e-6)


# -- reduction to the baseline -------------------------------------------


def test_reduces_to_baseline_loss():
    """Joint group CE == sum of per-attribute CEs when B_hat=0, prior uniform."""
    targets = full_product_targets(ATTRIBUTE_SIZES)
    support = GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)
    energies = random_energies(16, ATTRIBUTE_SIZES, seed=1)
    y = torch.as_tensor(targets[:16])

    logits = crm_group_logits(energies, support.train_groups)
    joint = F.cross_entropy(logits, support.train_index(y))
    factorised = sum(
        F.cross_entropy(energies[i], y[:, i]) for i in range(len(ATTRIBUTE_SIZES))
    )
    assert torch.allclose(joint, factorised, atol=1e-5)


def test_reduces_to_baseline_argmax():
    targets = full_product_targets(ATTRIBUTE_SIZES)
    support = GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)
    energies = random_energies(32, ATTRIBUTE_SIZES, seed=2)

    idx = chunked_group_argmax(energies, support.eval_groups, chunk=5)
    joint_pred = support.eval_groups[idx]
    per_attribute = torch.stack([e.argmax(dim=-1) for e in energies], dim=1)
    assert torch.equal(joint_pred, per_attribute)


# -- chunking / numerics --------------------------------------------------


def test_chunked_logsumexp_matches_dense():
    targets = full_product_targets(ATTRIBUTE_SIZES)
    support = GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)
    energies = random_energies(8, ATTRIBUTE_SIZES, seed=3)
    bias = torch.randn(support.num_train_groups)
    dense = torch.logsumexp(
        crm_group_logits(
            energies, support.train_groups, bias=bias, log_prior=support.log_p_hat
        ),
        dim=1,
    )
    for chunk in (1, 5, 7, 1000):
        got = chunked_group_logsumexp(
            energies,
            support.train_groups,
            bias=bias,
            log_prior=support.log_p_hat,
            chunk=chunk,
        )
        assert torch.allclose(got, dense, atol=1e-5), chunk


def test_masked_groups_get_zero_probability():
    targets = full_product_targets(ATTRIBUTE_SIZES)
    support = GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)
    energies = random_energies(4, ATTRIBUTE_SIZES, seed=4)
    log_prior = support.log_p_hat.clone()
    log_prior[0] = float("-inf")
    logits = crm_group_logits(energies, support.train_groups, log_prior=log_prior)
    probs = torch.softmax(logits, dim=1)
    assert torch.all(probs[:, 0] == 0.0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)


def test_chunked_logsumexp_survives_a_fully_masked_chunk():
    """A chunk with only absent groups must contribute 0, not NaN."""
    targets = full_product_targets(ATTRIBUTE_SIZES)
    support = GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)
    energies = random_energies(4, ATTRIBUTE_SIZES, seed=6)
    log_prior = support.log_p_hat.clone()
    log_prior[:5] = float("-inf")  # chunk size 5 below => first chunk all masked

    got = chunked_group_logsumexp(
        energies, support.train_groups, log_prior=log_prior, chunk=5
    )
    dense = torch.logsumexp(
        crm_group_logits(energies, support.train_groups, log_prior=log_prior), dim=1
    )
    assert torch.isfinite(got).all()
    assert torch.allclose(got, dense, atol=1e-5)


def test_gauge_invariance_of_predictions():
    """Adding a per-sample constant to any E_i must not change predictions."""
    targets = full_product_targets(ATTRIBUTE_SIZES)
    support = GroupSupport.from_targets(targets, ATTRIBUTE_SIZES)
    energies = random_energies(16, ATTRIBUTE_SIZES, seed=5)
    shifted = [e.clone() for e in energies]
    shifted[1] = shifted[1] + 3.7

    base = chunked_group_argmax(energies, support.eval_groups)
    got = chunked_group_argmax(shifted, support.eval_groups)
    assert torch.equal(base, got)


# -- step 2: B* recovers log Z -------------------------------------------


class _LinearAED(torch.nn.Module):
    """Toy model whose per-attribute logits are a fixed linear map of x."""

    def __init__(self, weights, attributes, attribute_sizes):
        super().__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(w, requires_grad=False) for w in weights]
        )
        self.attributes = attributes
        self.attribute_sizes = attribute_sizes
        self.objective = "classification"
        self.loss_fn = None
        self.metric_fns = []

    def forward(self, x):
        return [x @ w for w in self.weights]


def test_b_star_recovers_log_partition_function():
    """B*(z) must equal log Z(z), including for groups never seen in training.

    Construct an exact AED over a finite input space: x is one of ``n_x``
    one-hot vectors, ``-E_i(x, v) = W_i[x, v]``, so
    ``Z(z) = sum_x exp(sum_i W_i[x, z_i])``. Training samples are drawn from
    ``p(x) = sum_z p(z) p(x|z)``, and B* is estimated from those samples.
    """
    torch.manual_seed(0)
    sizes = [3, 3]
    n_x = 40
    weights = [torch.randn(n_x, s) * 0.5 for s in sizes]
    all_groups = torch.as_tensor(full_product_targets(sizes))

    # exact log Z(z) for every group
    scores = crm_group_energy([w for w in weights], all_groups)  # (n_x, G)
    log_Z = torch.logsumexp(scores, dim=0)

    # p(x | z) and the training marginal p(x) under a uniform group prior over
    # a strict subset of the groups (so some groups are never observed)
    seen = torch.as_tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
    seen_idx = torch.as_tensor(
        [int(((all_groups == g).all(dim=1)).nonzero()[0, 0]) for g in seen]
    )
    log_px_given_z = scores[:, seen_idx] - log_Z[seen_idx].unsqueeze(0)
    px = (log_px_given_z.exp() / len(seen_idx)).sum(dim=1)
    px = px / px.sum()

    # a large sample from p(x); B* is an expectation, so the estimator is
    # consistent as the sample grows
    n = 400_000
    xs = torch.multinomial(px, n, replacement=True)
    x_onehot = F.one_hot(xs, num_classes=n_x).float()

    model = _LinearAED(weights, ["a", "b"], sizes)
    targets = seen.repeat(2, 1).numpy()  # every seen group observed, uniform
    support = GroupSupport.from_targets(targets, sizes)
    wrapper = CRMWrapper(model, support, report_baseline_metrics=False)

    loader = [(x_onehot, torch.zeros(n, len(sizes), dtype=torch.long))]
    wrapper.compute_extrapolated_bias(loader, device="cpu")

    # B* is identified up to a global additive constant; compare after centring
    got = wrapper.b_star - wrapper.b_star.mean()
    want = log_Z - log_Z.mean()
    assert torch.allclose(got, want, atol=0.05), (got, want)


def test_b_star_chunking_is_consistent():
    torch.manual_seed(1)
    sizes = [3, 3]
    n_x = 16
    weights = [torch.randn(n_x, s) for s in sizes]
    model = _LinearAED(weights, ["a", "b"], sizes)
    support = GroupSupport.from_targets(full_product_targets(sizes), sizes)

    x = torch.eye(n_x)
    loader = [(x, torch.zeros(n_x, 2, dtype=torch.long))]

    results = []
    for chunk in (1, 4, 1000):
        wrapper = CRMWrapper(
            model, support, report_baseline_metrics=False, group_chunk=chunk
        )
        wrapper.compute_extrapolated_bias(loader, device="cpu", group_chunk=chunk)
        results.append(wrapper.b_star.clone())
    for other in results[1:]:
        assert torch.allclose(results[0], other, atol=1e-5)


def test_b_star_changes_predictions():
    """The post-hoc bias must actually alter the decision rule."""
    torch.manual_seed(2)
    sizes = [3, 3]
    n_x = 16
    weights = [torch.randn(n_x, s) for s in sizes]
    model = _LinearAED(weights, ["a", "b"], sizes)
    seen = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 3, dtype=np.int64)
    support = GroupSupport.from_targets(seen, sizes)
    wrapper = CRMWrapper(model, support, report_baseline_metrics=False)
    with torch.no_grad():
        wrapper.b_hat.normal_(std=1.0)

    x = torch.eye(n_x)
    energies = wrapper.energies(x)
    naive = wrapper.predict_groups(energies, extrapolate=False)
    # before step 2 the "extrapolate" path falls back to the naive one
    assert torch.equal(wrapper.predict_groups(energies, extrapolate=True), naive)

    wrapper.compute_extrapolated_bias(
        [(x, torch.zeros(n_x, 2, dtype=torch.long))], device="cpu"
    )
    extrapolated = wrapper.predict_groups(energies, extrapolate=True)
    # the naive rule can never predict an unseen group (they are hard-masked)
    naive_codes = {tuple(g.tolist()) for g in naive}
    assert naive_codes <= {tuple(g) for g in seen.tolist()}
    assert not torch.equal(naive, extrapolated)


# -- CRM combined with the analogical (algebraic) mixer term --------------


class _FakeSplitMixer(torch.nn.Module):
    """Minimal stand-in for SplitResNet18Mixer's CRM-facing surface.

    Mirrors `crm_outputs`: one encoder pass yields both the per-attribute logits
    over the flattened `batch * views` rows and the parallelogram residual over
    the per-view representations.
    """

    def __init__(self, attribute_sizes, rep_dim=6):
        super().__init__()
        self.attributes = [f"a{i}" for i in range(len(attribute_sizes))]
        self.attribute_sizes = attribute_sizes
        self.objective = "classification"
        self.loss_fn = AttributeCrossEntropyLoss()
        self.metric_fns = [MultiAccuracy()]
        self.mixer_mode = "algebraic"
        self.algebraic_use_all_terms = True
        self.mixer_loss_weight = 1.0
        self.encoder = torch.nn.Linear(rep_dim, rep_dim, bias=False)
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(rep_dim, s, bias=False) for s in attribute_sizes]
        )
        self.calls = 0

    def _reps(self, x_flat):
        self.calls += 1
        return self.encoder(x_flat)

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        reps = self._reps(x)
        return [h(reps) for h in self.heads]

    def mixer_loss_from_reps(self, reps, y=None):
        residual = reps[:, 0] - reps[:, 1] - reps[:, 2] + reps[:, 3]
        loss = residual.pow(2).mean()
        return 4.0 * loss if self.algebraic_use_all_terms else loss

    def crm_outputs(self, x, y=None):
        if x.dim() == 3:
            b, v = x.shape[:2]
            reps_flat = self._reps(x.reshape(b * v, -1))
            logits = [h(reps_flat) for h in self.heads]
            return logits, self.mixer_loss_from_reps(reps_flat.view(b, v, -1), y)
        reps = self._reps(x)
        return [h(reps) for h in self.heads], torch.zeros((), device=x.device)


def _four_view_batch(support, batch=5, rep_dim=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 4, rep_dim, generator=g)
    groups = support.train_groups
    idx = torch.randint(0, len(groups), (batch, 4), generator=g)
    y = groups[idx].unsqueeze(2)  # (batch, 4, 1, m) -- the object axis
    return x, y


def test_aux_weight_requires_model_support():
    support = GroupSupport.from_targets(
        full_product_targets(ATTRIBUTE_SIZES), ATTRIBUTE_SIZES
    )
    plain = _LinearAED(
        [torch.randn(4, s) for s in ATTRIBUTE_SIZES],
        [f"a{i}" for i in range(3)],
        ATTRIBUTE_SIZES,
    )
    with pytest.raises(ValueError, match="crm_outputs"):
        CRMWrapper(plain, support, aux_loss_weight=1.0)


def test_crm_outputs_matches_forward_and_runs_encoder_once():
    sizes = [3, 2]
    support = GroupSupport.from_targets(full_product_targets(sizes), sizes)
    model = _FakeSplitMixer(sizes)
    x, _ = _four_view_batch(support)

    model.calls = 0
    via_forward = model(x)
    assert model.calls == 1

    model.calls = 0
    via_crm, aux = model.crm_outputs(x)
    assert model.calls == 1  # no second encoder pass for the auxiliary term

    for a, b in zip(via_forward, via_crm):
        assert torch.allclose(a, b, atol=1e-6)
    assert aux.item() > 0.0


def test_aux_term_enters_total_loss_with_its_weight():
    sizes = [3, 2]
    support = GroupSupport.from_targets(full_product_targets(sizes), sizes)
    model = _FakeSplitMixer(sizes)
    x, y = _four_view_batch(support, seed=1)

    off = CRMWrapper(model, support, report_baseline_metrics=False)
    on = CRMWrapper(model, support, report_baseline_metrics=False, aux_loss_weight=2.5)

    energies, aux = on.energies_and_aux(x, y)
    crm_loss, _ = on._compute_loss(energies, y)

    log_off = off.validation_step(x=x, y=y)
    log_on = on.validation_step(x=x, y=y)

    # the group cross-entropy itself is untouched by the auxiliary term
    assert abs(log_off["loss"] - log_on["loss"]) < 1e-5
    # ... and total_loss is exactly crm_loss + w * aux
    assert "mixer_loss" not in log_off
    assert abs(log_on["mixer_loss"] - aux.item()) < 1e-5
    assert abs(log_on["total_loss"] - (crm_loss.item() + 2.5 * aux.item())) < 1e-4


def test_aux_term_reaches_the_encoder_gradient():
    sizes = [3, 2]
    support = GroupSupport.from_targets(full_product_targets(sizes), sizes)
    x, y = _four_view_batch(support, seed=2)

    grads = {}
    for weight in (0.0, 5.0):
        model = _FakeSplitMixer(sizes)
        torch.manual_seed(0)
        with torch.no_grad():
            model.encoder.weight.copy_(torch.eye(6) + 0.1)
        wrapper = CRMWrapper(
            model, support, report_baseline_metrics=False, aux_loss_weight=weight
        )
        energies, aux = wrapper.energies_and_aux(x, y)
        loss, _ = wrapper._compute_loss(energies, y)
        (loss + weight * aux).backward()
        grads[weight] = model.encoder.weight.grad.clone()

    assert not torch.allclose(grads[0.0], grads[5.0], atol=1e-6)


def test_four_view_targets_align_with_flattened_logits():
    """y is (B, 4, 1, m); it must flatten to the same B*4 rows as the logits."""
    sizes = [3, 2]
    support = GroupSupport.from_targets(full_product_targets(sizes), sizes)
    model = _FakeSplitMixer(sizes)
    wrapper = CRMWrapper(
        model, support, report_baseline_metrics=False, aux_loss_weight=1.0
    )
    x, y = _four_view_batch(support, batch=7, seed=3)

    energies, _ = wrapper.energies_and_aux(x, y)
    targets = wrapper._flatten_targets(y)
    assert energies[0].shape[0] == 7 * 4
    assert targets.shape == (7 * 4, 2)
    assert bool((support.train_index(targets) >= 0).all())


def _real_split_mixer(attribute_sizes, mixer_mode="algebraic"):
    """A real SplitResNet18Mixer, small enough to run on CPU."""
    from torch import nn

    from visgen.models.ain import SplitResNet18Mixer

    return SplitResNet18Mixer(
        in_channels=1,
        out_dim=512,
        maxpool=1,
        split_layers=0,
        preprocessing=None,
        head=None,
        attributes=[f"a{i}" for i in range(len(attribute_sizes))],
        attribute_sizes=attribute_sizes,
        objective="classification",
        loss_fn=AttributeCrossEntropyLoss(),
        metric_fns=[MultiAccuracy()],
        activation=nn.ReLU(inplace=True),
        mixer_mode=mixer_mode,
        mixer_loss_weight=1.0,
        head_bias=False,
    )


def test_real_split_mixer_crm_outputs_match_forward():
    torch.manual_seed(0)
    sizes = [3, 2]
    model = _real_split_mixer(sizes).eval()
    x = torch.randn(2, 4, 1, 32, 32)

    with torch.no_grad():
        logits_crm, aux = model.crm_outputs(x)
        # forward() keeps only the last view, so compare against that slice
        logits_fwd = model(x)
    assert logits_crm[0].shape == (8, 3)
    for i in range(len(sizes)):
        assert torch.allclose(logits_crm[i][3::4], logits_fwd[i], atol=1e-5)
    assert torch.isfinite(aux) and aux.item() > 0.0


def test_real_split_mixer_algebraic_term_is_the_parallelogram_residual():
    torch.manual_seed(1)
    sizes = [3, 2]
    model = _real_split_mixer(sizes).eval()
    x = torch.randn(2, 4, 1, 32, 32)

    with torch.no_grad():
        _, reps_flat, _ = model._encode_split(x.reshape(8, 1, 32, 32))
        reps = reps_flat.view(2, 4, -1)
        residual = reps[:, 0] - reps[:, 1] - reps[:, 2] + reps[:, 3]
        expected = 4.0 * residual.pow(2).mean()  # algebraic_use_all_terms=True
        _, aux = model.crm_outputs(x)
    assert torch.allclose(aux, expected, atol=1e-6)


def test_real_split_mixer_refactor_preserves_compute_losses():
    """_compute_losses must still return the same mixer term after the refactor."""
    torch.manual_seed(2)
    sizes = [3, 2]
    model = _real_split_mixer(sizes).eval()
    x = torch.randn(2, 4, 1, 32, 32)
    y = torch.zeros(2, 4, 1, 2, dtype=torch.long)

    with torch.no_grad():
        _, mixer_loss, _ = model._compute_losses(x, y)
        _, aux = model.crm_outputs(x, y)
    assert torch.allclose(mixer_loss, aux, atol=1e-6)


def test_real_split_mixer_without_four_views_has_no_aux_term():
    torch.manual_seed(3)
    model = _real_split_mixer([3, 2]).eval()
    with torch.no_grad():
        _, aux = model.crm_outputs(torch.randn(4, 1, 32, 32))
    assert aux.item() == 0.0


# -- que aporta exactamente el paso 2 -------------------------------------
#
# Con Ze = producto cartesiano completo, prior de test uniforme y energias
# aditivas (s(x,z) = sum_i E_i(x, z_i)), el argmax conjunto de CRM factoriza
# exactamente en el argmax por atributo. Es decir: sin B*, `crm_acc` no puede
# diferir de `baseline_acc`, y toda la diferencia entre ambos mide el aporte
# del paso 2. Esto sostiene la ablacion "solo perdida de CRM" vs "CRM full".


def _wrapper_on_full_product(sizes=ATTRIBUTE_SIZES, seed=0):
    targets = full_product_targets(sizes)
    support = GroupSupport.from_targets(targets, sizes)
    torch.manual_seed(seed)
    model = _FakeSplitMixer(sizes)
    return CRMWrapper(model, support, report_baseline_metrics=True)


def test_joint_decision_equals_per_attribute_argmax_without_b_star():
    wrapper = _wrapper_on_full_product()
    assert wrapper.support.eval_groups.shape[0] == int(
        torch.tensor(ATTRIBUTE_SIZES).prod()
    )
    energies = random_energies(64, ATTRIBUTE_SIZES, seed=7)

    # b_star arranca en ceros y el prior de test es uniforme
    assert torch.count_nonzero(wrapper.b_star) == 0
    assert wrapper.test_prior == "uniform"

    joint = wrapper.predict_groups(energies, extrapolate=True)
    per_attribute = torch.stack([e.argmax(dim=-1) for e in energies], dim=1)
    assert torch.equal(joint, per_attribute)


def test_crm_and_baseline_accuracy_coincide_without_b_star():
    wrapper = _wrapper_on_full_product()
    energies = random_energies(64, ATTRIBUTE_SIZES, seed=11)
    targets = torch.stack(
        [torch.randint(0, s, (64,)) for s in ATTRIBUTE_SIZES], dim=1
    )

    metrics = wrapper._crm_metrics(energies, targets)
    assert metrics["crm_acc"] == pytest.approx(metrics["baseline_acc"])


def test_a_non_zero_b_star_is_what_makes_them_differ():
    # El contrapunto: en cuanto B* deja de ser constante, la decision conjunta
    # se separa del argmax por atributo. Ese delta es lo que reporta la
    # ablacion como aporte (o dano) del paso 2.
    wrapper = _wrapper_on_full_product()
    energies = random_energies(64, ATTRIBUTE_SIZES, seed=13)
    before = wrapper.predict_groups(energies, extrapolate=True)

    torch.manual_seed(3)
    wrapper.b_star.copy_(torch.randn_like(wrapper.b_star) * 5.0)
    wrapper.has_b_star.fill_(True)
    after = wrapper.predict_groups(energies, extrapolate=True)

    assert not torch.equal(before, after)


def test_a_constant_b_star_leaves_the_decision_unchanged():
    # B* solo esta identificado salvo constante aditiva: sumarle un escalar a
    # todos los grupos no puede cambiar el argmax.
    wrapper = _wrapper_on_full_product()
    energies = random_energies(64, ATTRIBUTE_SIZES, seed=17)
    before = wrapper.predict_groups(energies, extrapolate=True)

    wrapper.b_star.fill_(3.7)
    wrapper.has_b_star.fill_(True)
    assert torch.equal(wrapper.predict_groups(energies, extrapolate=True), before)


def test_naive_rule_cannot_hit_a_group_absent_from_training():
    # Por que crm_naive_acc sale 0 en nuestros splits: la regla naive enmascara
    # a -inf todo grupo no visto en train, y el test composicional esta hecho
    # exactamente de grupos no vistos.
    sizes = ATTRIBUTE_SIZES
    full = full_product_targets(sizes)
    seen = full[: len(full) // 2]
    # eval sigue siendo el producto completo; train ve solo la mitad
    support = GroupSupport.from_targets(seen, sizes, eval_group_set="full_product")
    torch.manual_seed(0)
    wrapper = CRMWrapper(_FakeSplitMixer(sizes), support)

    energies = random_energies(64, sizes, seed=19)
    pred = wrapper.predict_groups(energies, extrapolate=False)
    seen_codes = {tuple(int(v) for v in g) for g in seen}
    assert all(tuple(int(v) for v in p) in seen_codes for p in pred)
