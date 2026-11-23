"""Optimizer, learning-rate schedule and curriculum schedule construction.

Every hybrid is trained with Adam on a warmup-cosine-decay learning rate, after
clipping the gradient's global norm. The other optimizers (SGD, AdamW, L-BFGS,
SOAP), the other schedules and staged training with several optimizers were
retired before this release; see ``experiments/retired.py``.

The curriculum schedule is the ADM1-specific piece: training starts on a short
prefix of each time series and lengthens it in stages, which keeps the early
optimization out of local minima (Kidger, 2022).
"""

from __future__ import annotations

import logging
from typing import Sequence

import optax

__all__ = [
    "OPTIMIZER",
    "LR_SCHEDULER",
    "build_length_schedule",
    "build_learning_rate_schedule",
    "adam_optimizer",
]

#: The optimizer and the learning-rate schedule a trainer's ``optimizer_hparams`` name.
#: The optimizer's name is also the step prefix of the checkpoints (``adam_200``).
OPTIMIZER = "adam"
LR_SCHEDULER = "warmup_cosine_decay"


def build_length_schedule(
    schedule: Sequence[Sequence[float]] | None,
    num_epochs: int,
    steps_per_epoch: int = 1,
):
    """Curriculum over the simulated fraction of each time series.

    ``schedule`` is a list of ``(epoch_fraction, length_fraction)`` pairs; the
    first pair gives the starting length and each later pair raises it from that
    point in training onwards. ``None`` simulates the full series throughout.

    ``steps_per_epoch`` converts the epoch fractions into the optimizer-step
    units the returned schedule is evaluated in. It is not optional in practice:
    the trainer calls ``self.length_schedule(state.step)``, and ``state.step``
    counts optimizer steps, not epochs. Leaving it at 1 places every boundary
    ``steps_per_epoch`` times too early -- with 12 reactors at batch size 4 that
    is a factor of three, so a 50-epoch run would reach full length at epoch 9
    instead of epoch 25, and training would be both slower and less curricular
    than intended.
    """
    if schedule is None:
        return optax.constant_schedule(1.0)

    if not (
        isinstance(schedule, list)
        and all(
            isinstance(item, (tuple, list)) and all(isinstance(x, (float, int)) for x in item)
            for item in schedule
        )
    ):
        raise TypeError("length_schedule must be a list of (epoch fraction, length fraction) pairs")

    initial_length = schedule[0][1]
    boundaries_and_scales: dict[int, float] = {}
    previous_length = initial_length
    for epoch_fraction, length_fraction in schedule[1:]:
        if not 0.0 < epoch_fraction < 1.0:
            raise ValueError("length schedule boundaries must lie strictly between 0 and 1")
        if length_fraction <= 0.0:
            raise ValueError("length schedule scales must be positive")
        boundary = int(num_epochs * epoch_fraction * steps_per_epoch)
        boundaries_and_scales[boundary] = length_fraction / previous_length
        previous_length = length_fraction

    return optax.piecewise_constant_schedule(
        init_value=initial_length, boundaries_and_scales=boundaries_and_scales
    )


def build_learning_rate_schedule(
    learning_rate: float,
    warmup: int,
    total_steps: int,
    end_value: float | None = None,
):
    """The warmup-cosine-decay learning rate, with the warmup clamped to the available steps.

    optax needs at least one decay step after the warmup, so a short run (a smoke
    test, or a small tuning budget) would otherwise fail deep inside optax.
    ``end_value`` defaults to 1% of ``learning_rate``.
    """
    if warmup >= total_steps:
        warmup = max(0, total_steps - 1)
        logging.warning(
            "warmup exceeds the %d available optimizer steps; clamped to %d", total_steps, warmup
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup,
        decay_steps=max(total_steps, warmup + 1),
        end_value=0.01 * learning_rate if end_value is None else end_value,
    )


def adam_optimizer(learning_rate_schedule, gradient_clip: float = 1.0):
    """Gradient clipping by global norm, then Adam on ``learning_rate_schedule``."""
    return optax.chain(optax.clip_by_global_norm(gradient_clip),
                       optax.adam(learning_rate_schedule))
