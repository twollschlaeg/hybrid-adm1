"""The training loss of the hybrid models (utils/ude_trainer.py)."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from adm1_package.experiments.registry import get_spec
from adm1_package.utils.losses import robust_target_range
from adm1_package.utils.ude_trainer import loss_targets, objective_mse, scored_points


def test_the_first_point_and_the_spin_up_are_not_scored():
    """A record that starts on day 5, with a 3-day spin-up: days 8-14 are scored."""
    indices = jnp.ones((2, 10, 1))
    time = jnp.tile(5.0 + jnp.arange(10.0), (2, 1))
    scored = np.asarray(scored_points(indices, time, 3.0))[..., 0]
    np.testing.assert_array_equal(scored, np.tile([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], (2, 1)))
    without = np.asarray(scored_points(indices, time, 0.0))[..., 0]
    np.testing.assert_array_equal(without, np.tile([0] + [1] * 9, (2, 1)))


def test_unmeasured_points_stay_unscored():
    indices = jnp.asarray([[1.0], [1.0], [0.0], [1.0]])
    scored = np.asarray(scored_points(indices, jnp.arange(4.0), 0.0))[:, 0]
    np.testing.assert_array_equal(scored, [0, 1, 0, 1])


# ---------------------------------------------------------------------------
# The residual scale
# ---------------------------------------------------------------------------
def test_the_loss_scale_is_the_interpercentile_range_with_no_unit_floor():
    # Column 0 spans ~6 (biogas-like), column 1 spans ~0.4 (acid-like). With linear
    # interpolation Q90 - Q10 is 15.4 - 10.6 and 0.44 - 0.13; both exceed the std.
    targets = jnp.asarray(
        [[10.0, 0.10], [12.0, 0.20], [14.0, 0.30], [16.0, 0.50]], dtype=float
    )
    scale = np.asarray(robust_target_range(targets, axis=0))
    np.testing.assert_allclose(scale, [[4.8, 0.31]], rtol=1e-12)


def test_the_loss_scale_comes_from_the_scored_points_only():
    targets = jnp.asarray([[100.0], [10.0], [12.0], [14.0], [16.0], [16.0]])
    scored = jnp.asarray([[False], [True], [True], [True], [True], [False]])  # spin-up, padding
    scale = robust_target_range(targets, axis=0, where=scored)
    assert float(scale[0, 0]) == pytest.approx(4.8, rel=1e-12)


def test_the_loss_scale_never_divides_by_zero():
    assert bool((robust_target_range(jnp.zeros((5, 2), dtype=float), axis=0) > 0).all())


@pytest.mark.requires_data
def test_the_loss_ignores_the_spin_up_window(r4_trainer, short_r4_batch):
    _, batch, targets = short_r4_batch
    trainer = r4_trainer(zero_output=False)
    model, key = trainer.state.model, jr.PRNGKey(0)

    def loss(values):
        data = (batch, {"values": values, "indices": targets["indices"],
                        "object_indices": targets["object_indices"]})
        return float(trainer.loss_fn(model, trainer.ude, data, key)[0])

    reference = loss(targets["values"])
    assert loss(targets["values"].at[:, 1:30].multiply(3.0)) == pytest.approx(reference, rel=1e-12)
    assert loss(targets["values"].at[:, 35].multiply(3.0)) != pytest.approx(reference, rel=1e-6)


@pytest.mark.requires_data
def test_each_reactor_is_scaled_by_its_own_range(r4_trainer, short_r4_batch):
    """Each reactor's residuals are divided by that reactor's own range of the output.
    It must not change: every run of the paper trained this way."""
    _, inputs, targets = short_r4_batch
    spec = get_spec("r4")
    trainer = r4_trainer()
    data, key = (inputs, loss_targets(targets)), jr.PRNGKey(0)
    solution, mse = trainer.loss_fn(trainer.state.model, trainer.ude, data, key)[1]

    columns = [targets["object_indices"][name] for name in spec.objective_list]
    values = targets["values"][..., columns]
    scored = scored_points(targets["indices"][..., columns], inputs["input"]["time"], 30.0)
    predicted = jnp.stack([solution[name] for name in spec.objective_list], axis=-1)
    expected = objective_mse((predicted - values) / robust_target_range(values, 1, where=scored > 0),
                             scored)
    np.testing.assert_allclose(np.asarray(mse), np.asarray(expected), rtol=1e-9)
