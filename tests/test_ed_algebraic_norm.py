"""La perdida algebraica de ED y su interaccion con la normalizacion.

El readout de ED clasifica con ``F.normalize`` seguido de ``cosine_similarity``
(``visgen/models/modules/readouts.py``), asi que la perdida de clasificacion
solo depende de la DIRECCION de cada pieza de la representacion. El residuo
algebraico crudo, en cambio, escala con ``||rep||^2``. Encoger la norma lo lleva
a cero sin costo para la clasificacion, que es un minimo degenerado. AIN no lo
sufre porque su clasificador es lineal y si depende de la magnitud.
"""

import pytest
import torch

from visgen.models.ed import ExpDisentanglementMixer

Z_DIM = 4
NUM_PIECES = 3


def _loss_fn(algebraic_norm):
    model = object.__new__(ExpDisentanglementMixer)
    model.z_dim = Z_DIM
    model.algebraic_norm = algebraic_norm
    return model._algebraic_loss


def _reps(seed=0, batch=64):
    torch.manual_seed(seed)
    return torch.randn(batch, 4, Z_DIM * NUM_PIECES)


def _compositional_reps(seed=0, batch=64):
    """Cuadrilatero de un espacio aditivo: rep = f(fila) + g(columna)."""
    torch.manual_seed(seed)
    dim = Z_DIM * NUM_PIECES
    a, b, c, d = (torch.randn(batch, dim) for _ in range(4))
    return torch.stack([a + c, a + d, b + c, b + d], dim=1)


def test_raw_residual_rewards_shrinking_the_representation():
    # Es el problema: sin normalizar, encoger la norma es una salida gratis.
    loss = _loss_fn("none")
    reps = _reps()
    assert loss(reps * 0.01).item() < loss(reps).item() * 1e-3


@pytest.mark.parametrize("algebraic_norm", ["unit", "relative"])
def test_normalized_residual_is_invariant_to_the_representation_scale(
    algebraic_norm
):
    loss = _loss_fn(algebraic_norm)
    reps = _reps()
    reference = loss(reps).item()
    for scale in (100.0, 0.1, 0.01):
        assert loss(reps * scale).item() == pytest.approx(reference, rel=1e-4)


@pytest.mark.parametrize("algebraic_norm", ["none", "unit", "relative"])
def test_residual_still_separates_compositional_from_random(algebraic_norm):
    loss = _loss_fn(algebraic_norm)
    assert loss(_compositional_reps()).item() < loss(_reps()).item()


def test_relative_residual_vanishes_on_a_perfectly_additive_representation():
    # 'unit' rompe la identidad f(a)+g(c) al normalizar cada pieza, asi que
    # penaliza la solucion ideal; 'relative' conserva la semantica aditiva.
    assert _loss_fn("relative")(_compositional_reps()).item() < 1e-10
    assert _loss_fn("unit")(_compositional_reps()).item() > 1e-3


def test_unknown_algebraic_norm_is_rejected():
    with pytest.raises(ValueError, match="algebraic_norm"):
        ExpDisentanglementMixer(
            preprocessing=None, feature_extractors=[], readouts=[],
            z_dim=Z_DIM, algebraic_norm="l2",
        )
