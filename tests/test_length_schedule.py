"""The curriculum over the simulated fraction of each time series.

``UDETrainer.train_step`` evaluates the schedule at ``state.step``, which
counts optimizer steps. The schedule is specified in epoch fractions, so the
conversion has to happen when it is built -- and it did not, which put every
boundary ``steps_per_epoch`` times too early.
"""

from __future__ import annotations

import pytest

from adm1_package.utils.optimizers import build_length_schedule

#: The schedule the study trains with: start on a quarter of each series, half
#: from 25% of training, all of it from 50%.
SPEC = [[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]]


def _per_epoch(schedule, num_epochs: int, steps_per_epoch: int) -> list[float]:
    """The simulated fraction at the first optimizer step of each epoch."""
    return [float(schedule(epoch * steps_per_epoch)) for epoch in range(num_epochs)]


def test_boundaries_land_on_the_intended_epochs():
    """A 50-epoch run at 3 steps/epoch is 150 steps.

    The boundaries fall at step 37 (25%) and step 75 (50%). Step 37 is inside
    epoch 13, whose steps are 36-38, so sampling each epoch at its *first* step
    shows the change one epoch later. That is the schedule being step-accurate
    rather than epoch-quantised, which is the point.
    """
    schedule = build_length_schedule(SPEC, num_epochs=50, steps_per_epoch=3)
    lengths = _per_epoch(schedule, 50, 3)

    assert lengths[0] == pytest.approx(0.25)
    assert lengths[12] == pytest.approx(0.25)   # epoch 13, boundary mid-epoch
    assert lengths[13] == pytest.approx(0.5)    # epoch 14
    assert lengths[24] == pytest.approx(0.5)    # epoch 25
    assert lengths[25] == pytest.approx(1.0)    # epoch 26, exactly halfway

    # The boundary really is a step, not an epoch.
    assert float(schedule(36)) == pytest.approx(0.25)
    assert float(schedule(37)) == pytest.approx(0.5)


def test_the_schedule_is_evaluated_in_optimizer_steps():
    """Boundaries must scale with steps_per_epoch, not sit at epoch numbers.

    With 12 reactors at batch size 4 there are 3 optimizer steps per epoch, so
    ignoring that reached full length at epoch 10 of 50 instead of epoch 26 and
    spent 41 epochs on the full series rather than 25 -- slower, and barely a
    curriculum.
    """
    steps_per_epoch = 3
    correct = build_length_schedule(SPEC, num_epochs=50, steps_per_epoch=steps_per_epoch)
    uncorrected = build_length_schedule(SPEC, num_epochs=50, steps_per_epoch=1)

    first_full = lambda values: next(  # noqa: E731
        index for index, value in enumerate(values) if value >= 1.0
    )
    assert first_full(_per_epoch(uncorrected, 50, steps_per_epoch)) == 9
    assert first_full(_per_epoch(correct, 50, steps_per_epoch)) == 25


def test_a_longer_epoch_pushes_the_boundaries_further_out():
    """More optimizer steps per epoch must not shorten the curriculum in epochs."""
    boundaries = []
    for steps_per_epoch in (1, 2, 4):
        schedule = build_length_schedule(SPEC, num_epochs=40, steps_per_epoch=steps_per_epoch)
        lengths = _per_epoch(schedule, 40, steps_per_epoch)
        boundaries.append(next(i for i, v in enumerate(lengths) if v >= 1.0))
    # The epoch at which full length is reached is a property of the schedule
    # spec, not of the batch size.
    assert len(set(boundaries)) == 1
    assert boundaries[0] == 20  # 50% of 40 epochs


def test_no_schedule_means_the_full_series_throughout():
    schedule = build_length_schedule(None, num_epochs=50, steps_per_epoch=3)
    assert float(schedule(0)) == pytest.approx(1.0)
    assert float(schedule(10_000)) == pytest.approx(1.0)


def test_boundaries_outside_the_unit_interval_are_rejected():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        build_length_schedule([[0.0, 0.25], [1.5, 1.0]], num_epochs=50, steps_per_epoch=3)


def test_non_positive_fractions_are_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        build_length_schedule([[0.0, 0.25], [0.5, 0.0]], num_epochs=50, steps_per_epoch=3)


def test_a_malformed_schedule_is_rejected():
    with pytest.raises(TypeError, match="list of"):
        build_length_schedule("quarter-then-full", num_epochs=50, steps_per_epoch=3)
