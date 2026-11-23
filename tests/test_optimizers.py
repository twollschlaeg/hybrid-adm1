"""Optimizer and learning-rate schedule construction (the curriculum's schedule is in
test_length_schedule.py).

Every hybrid trains with Adam on a warmup-cosine-decay learning rate after clipping
the gradient's global norm; the other optimizers and schedules were retired.
"""

from __future__ import annotations

import logging

import optax
import pytest

from adm1_package.utils.optimizers import adam_optimizer, build_learning_rate_schedule


def test_warmup_cosine_decay_rises_then_falls():
    schedule = build_learning_rate_schedule(1e-3, warmup=10, total_steps=100)
    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(10)) == pytest.approx(1e-3)
    assert float(schedule(99)) < 1e-3


def test_warmup_is_clamped_to_the_available_steps(caplog):
    # A short run, such as a one-epoch smoke run, must not fail inside optax.
    with caplog.at_level(logging.WARNING):
        schedule = build_learning_rate_schedule(1e-3, warmup=500, total_steps=4)
    assert "clamped" in caplog.text
    assert float(schedule(0)) >= 0.0


def test_chained_optimizer_updates_parameters():
    import jax.numpy as jnp

    optimizer = adam_optimizer(optax.constant_schedule(0.1), gradient_clip=1.0)
    params = {"w": jnp.ones(3)}
    state = optimizer.init(params)
    updates, _ = optimizer.update({"w": jnp.ones(3)}, state, params)
    assert float(jnp.abs(updates["w"]).sum()) > 0.0


@pytest.mark.parametrize("end_value, expected", [(None, 1e-5), (2e-5, 2e-5)],
                         ids=["a hundredth of the rate by default", "as given"])
def test_the_rate_decays_to_the_end_value(end_value, expected):
    options = {} if end_value is None else {"end_value": end_value}
    schedule = build_learning_rate_schedule(1e-3, warmup=5, total_steps=50, **options)
    assert float(schedule(50)) == pytest.approx(expected)


def test_the_optimizer_is_the_clip_then_adam_chain_the_runs_trained_with():
    """The paper's numbers depend on it: clipping by global norm, then Adam."""
    import jax.numpy as jnp

    schedule = build_learning_rate_schedule(1e-3, warmup=5, total_steps=50)
    params = {"w": jnp.linspace(-1.0, 1.0, 3)}
    gradient = {"w": jnp.asarray([30.0, -4.0, 0.5])}
    ours = adam_optimizer(schedule, gradient_clip=10.0)
    reference = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    state, expected_state = ours.init(params), reference.init(params)
    for _ in range(3):
        update, state = ours.update(gradient, state, params)
        expected, expected_state = reference.update(gradient, expected_state, params)
        assert jnp.array_equal(update["w"], expected["w"])
