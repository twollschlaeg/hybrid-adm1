"""The split correction network.

The degradability factors describe the substrate, so they come from a network
that reads only the current forcing, the first ``factor_inputs`` inputs. The rate
corrections come from a second network that reads every input. In the paper's runs both
read the forcing alone (the memory that the rate network also read was retired), but the
network keeps the distinction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from adm1_package.utils.nn_architectures import CorrectionMLP

#: 7 forcing columns, then 5 further columns; 2 factors, then 3 rates.
FORCING, MEMORY, FACTORS, RATES = 7, 5, 2, 3


def _network(**overrides):
    settings = dict(in_size=FORCING + MEMORY, out_size=FACTORS + RATES, width_size=8, depth=3,
                    activation=jax.nn.tanh, final_activation=lambda z: z, use_final_bias=True,
                    factor_inputs=FORCING, factor_outputs=FACTORS)
    settings.update(overrides)
    return CorrectionMLP(**settings, key=jr.PRNGKey(0))


def _inputs(seed):
    return jnp.asarray(np.random.default_rng(seed).normal(size=FORCING + MEMORY))


def test_the_factors_do_not_see_inputs_beyond_the_forcing():
    network = _network()
    x = _inputs(0)
    changed = x.at[FORCING:].add(3.0)
    a, b = network(x, jr.PRNGKey(1)), network(changed, jr.PRNGKey(1))
    np.testing.assert_array_equal(np.asarray(a[:FACTORS]), np.asarray(b[:FACTORS]))
    assert not np.allclose(np.asarray(a[FACTORS:]), np.asarray(b[FACTORS:]))


def test_the_factors_follow_the_forcing():
    network = _network()
    x = _inputs(0)
    changed = x.at[0].add(3.0)
    a, b = network(x, jr.PRNGKey(1)), network(changed, jr.PRNGKey(1))
    assert not np.allclose(np.asarray(a[:FACTORS]), np.asarray(b[:FACTORS]))


def test_both_networks_start_at_zero_output():
    """The UDE starts at its base model, whichever network an output comes from."""
    network = _network(zero_output=True)
    np.testing.assert_array_equal(np.asarray(network(_inputs(0), jr.PRNGKey(1))), np.zeros(FACTORS + RATES))


def test_the_penalty_covers_both_networks():
    network = _network()
    single = CorrectionMLP(FORCING + MEMORY, FACTORS + RATES, 8, 3, jax.nn.tanh, lambda z: z, True,
                           key=jr.PRNGKey(0))
    weights = sum(float(jnp.sum(layer.weight ** 2)) for layer in network.layers + network.factor_layers)
    assert float(network.regularization_loss()) == pytest.approx(weights)
    assert len(network.factor_layers) == 3 and len(single.factor_layers) == 0


def test_with_depth_one_both_networks_are_a_linear_map_of_their_inputs():
    """The linear baseline (fold_fast_narrow_tan_linear): no hidden layer, so every output is
    an affine function of the inputs before the links bound it."""
    network = _network(depth=1)
    assert len(network.layers) == 1 and len(network.factor_layers) == 1
    x, z = _inputs(0), _inputs(1)
    mixed = np.asarray(network(0.3 * x + 0.7 * z, jr.PRNGKey(1)))
    np.testing.assert_allclose(mixed, 0.3 * np.asarray(network(x, jr.PRNGKey(1)))
                               + 0.7 * np.asarray(network(z, jr.PRNGKey(1))), rtol=1e-5, atol=1e-6)


def test_without_a_split_every_output_sees_every_input():
    network = _network(factor_inputs=0, factor_outputs=0)
    x = _inputs(0)
    a, b = network(x, jr.PRNGKey(1)), network(x.at[FORCING:].add(3.0), jr.PRNGKey(1))
    assert not np.allclose(np.asarray(a[:FACTORS]), np.asarray(b[:FACTORS]))
