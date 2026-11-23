"""The two penalties of the training loss, and that neither reaches the validation loss.

``correction_penalty`` adds ``lambda * mean(delta**2)`` over the raw network outputs along
each trajectory: the log-changes of the rates and the logit-changes of the fractions. A
calibrated hybrid at zero output is the mechanistic model, so this is a prior centred on
the calibration. ``regularization_lambda`` adds ``lambda * ||W||^2`` over the network's
weights, which is no such prior: the output-layer bias, which sets the level of every
correction, is not among the weights it covers.

Both belong in the training gradient, not in the validation loss: that loss chooses the
checkpoint and ranks the trials of a search, and with a penalty in it, it would fall with
lambda while the fit stays as it was.
"""

from __future__ import annotations

from types import SimpleNamespace

import diffrax as dx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import correction_mlp

from adm1_package.experiments.registry import MODEL_KEYS, get_spec
from adm1_package.utils.corrections import CorrectionOptions
from adm1_package.utils.ude_trainer import loss_targets

NARROW = CorrectionOptions("calibrated", corrected=("f_ch", "k_hyd_ch"), bound=float(np.log(2.0)))
#: What the network outputs everywhere in these tests: f_ch and k_hyd_ch changes.
OUTPUT = (0.2, -0.4)
CORRECTION, WEIGHTS = 0.5, 0.3


def _constant(mlp):
    """Zeroed output weights and a set bias: the same output at every input."""
    return eqx.tree_at(lambda m: m.layers[-1].bias, mlp, jnp.asarray(OUTPUT))


@pytest.mark.requires_data
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_the_solution_carries_the_raw_network_output(model_key, installed_dataset):
    spec = get_spec(model_key)
    dataset = installed_dataset(model_key, ["LaRA-temp"])
    x, _ = dataset[0]
    mlp = _constant(correction_mlp(spec, spec.nn_in_size(dataset), NARROW, width=8,
                                   zero_output=True))
    interp = dx.CubicInterpolation(x["input"]["time"], x["input"]["coeffs"])
    scaling = (x["input"]["means"], x["input"]["stds"])
    solution = SimpleNamespace(ts=x["input"]["time"][:5], ys=jnp.tile(x["initial_values"], (5, 1)))
    args = (x["parameters"], interp, mlp, jr.PRNGKey(1), scaling)
    output = spec.ude(NARROW)._calculate_output(solution, args)
    np.testing.assert_allclose(np.asarray(output["network_output"]), np.tile(OUTPUT, (5, 1)))


@pytest.fixture
def penalised(r4_trainer, short_r4_batch):
    """A trainer with both penalties, its network constant at OUTPUT, and a batch:
    ``(trainer, model, data, key)``."""
    _, inputs, targets = short_r4_batch
    trainer = r4_trainer(NARROW, width_size=8, correction_penalty=CORRECTION,
                         regularization_lambda=WEIGHTS)
    return trainer, _constant(trainer.state.model), (inputs, loss_targets(targets)), jr.PRNGKey(0)


def _loss(trainer, model, data, key, *penalties) -> float:
    """The training loss; ``penalties`` (weights, correction) override the trainer's."""
    return float(trainer.loss_fn(model, trainer.ude, data, key, *penalties)[0])


@pytest.mark.requires_data
def test_each_penalty_adds_its_term_to_the_training_loss(penalised):
    """Each lambda is an argument of the jitted loss, not a read of the trainer that the
    first trace would freeze, so either can be switched off."""
    trainer, model, data, key = penalised
    weights = WEIGHTS * float(model.regularization_loss())
    assert weights > 0
    plain = _loss(trainer, model, data, key, 0.0, 0.0)
    assert _loss(trainer, model, data, key, 0.0, None) - plain == pytest.approx(
        CORRECTION * np.mean(np.square(OUTPUT)), rel=1e-9)
    assert _loss(trainer, model, data, key, None, 0.0) - plain == pytest.approx(weights, rel=1e-9)
    assert _loss(trainer, model, data, key) - plain == pytest.approx(
        weights + CORRECTION * np.mean(np.square(OUTPUT)), rel=1e-9)


@pytest.mark.requires_data
def test_there_is_no_penalty_unless_asked_for(r4_trainer, short_r4_batch):
    _, inputs, targets = short_r4_batch
    trainer = r4_trainer(NARROW, width_size=8)
    model, data, key = _constant(trainer.state.model), (inputs, loss_targets(targets)), jr.PRNGKey(0)
    assert _loss(trainer, model, data, key) == pytest.approx(
        _loss(trainer, model, data, key, 0.0, 0.0), rel=1e-12)


@pytest.mark.requires_data
def test_validation_reports_the_fit_without_the_penalties(penalised, short_r4_batch):
    """And reports each penalty beside it."""
    trainer, model, data, key = penalised
    _, inputs, targets = short_r4_batch
    state = trainer.state._replace(model=model)
    _, metrics = trainer.eval_step(state, (inputs, targets))
    plain = trainer.loss_fn(model, trainer.ude, data, state.key, 0.0, 0.0)[0]
    assert float(metrics["loss"]) == pytest.approx(float(plain), rel=1e-9)
    assert float(metrics["correction"]) == pytest.approx(np.mean(np.square(OUTPUT)), rel=1e-9)
    assert float(metrics["penalty"]) == pytest.approx(
        WEIGHTS * float(model.regularization_loss()), rel=1e-9)
