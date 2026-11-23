"""What the correction network reads.

The current forcing -- temperature, loading rate and influent -- z-scored with
the training statistics. Absolute experiment time is not an input: day 100 of
one experiment says nothing about day 100 of another.
"""

from __future__ import annotations

import diffrax as dx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from helpers import correction_mlp, first_days

from adm1_package.experiments.registry import MODEL_KEYS, get_spec
from adm1_package.utils.corrections import DEFAULT
from adm1_package.utils.nn_architectures import network_input


def test_the_input_is_the_forcing_z_scored():
    means = jnp.asarray([310.0, 0.02, 1.0, 900.0])
    stds = jnp.asarray([2.0, 0.01, 0.5, 10.0])
    got = network_input(312.0, 0.03, jnp.asarray([2.0, 880.0]), (means, stds))
    np.testing.assert_allclose(np.asarray(got), [1.0, 1.0, 2.0, -2.0], rtol=1e-12)


@pytest.mark.requires_data
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_the_ude_feeds_the_network_exactly_the_inputs_the_registry_sizes(model_key,
                                                                        installed_dataset):
    """Temperature, loading rate and the influent species."""
    spec = get_spec(model_key)
    dataset = installed_dataset(model_key)
    x, _ = dataset[0]
    ude = spec.ude(DEFAULT)
    width = spec.nn_in_size(dataset)
    assert width == len(spec.influent_species) + 2
    mlp = correction_mlp(spec, width, DEFAULT)
    interp = dx.CubicInterpolation(x["input"]["time"], x["input"]["coeffs"])
    scaling = (x["input"]["means"], x["input"]["stds"])
    dy = ude._rhs(5.0, x["initial_values"], (x["parameters"], interp, mlp, jr.PRNGKey(1), scaling))
    assert np.all(np.isfinite(np.asarray(dy)))


@pytest.mark.requires_data
def test_a_shifted_clock_changes_nothing(installed_dataset):
    """The same reactor, its time stamps moved 100 days later."""
    spec = get_spec("r4")
    dataset = installed_dataset("r4")
    x, _ = dataset[dataset.labels.index("LaRA-temp_F4")]
    inputs = first_days(x["input"], 20.0)
    later = {**inputs, "time": inputs["time"] + 100.0}
    ude = spec.ude(DEFAULT)
    mlp = correction_mlp(spec, spec.nn_in_size(dataset), DEFAULT)

    def biogas(forcing):
        out, _ = ude(x["parameters"], x["initial_values"], forcing, mlp, jr.PRNGKey(1))
        return np.asarray(out["q_gasout"])

    # The adaptive steps turn roundoff into ~2e-8: scaling the forcing by 1 +- 2e-15,
    # without any shift, moves the biogas as much. A clock read by the model would move it
    # by orders of magnitude more.
    relative = np.max(np.abs(biogas(later) - biogas(inputs)) / biogas(inputs))
    assert relative < 1e-6
