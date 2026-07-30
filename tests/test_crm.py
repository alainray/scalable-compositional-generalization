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
