"""Sample ordering decides the fold assignment, so it is pinned down here."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from adm1_package.utils.datasets import InputScalingMixin, list_input_files


def _touch(directory, names):
    for name in names:
        (directory / name).write_text("", encoding="utf-8")


def test_files_are_ordered_naturally_not_lexicographically(tmp_path):
    # Lexicographic order would put input_R100 between input_R10 and input_R20,
    # which would mis-group the rice-bran replicate blocks.
    _touch(tmp_path, ["input_R100.csv", "input_R20.csv", "input_R10.csv", "input_R3.csv"])
    assert list_input_files(tmp_path.as_posix()) == [
        "input_R3.csv",
        "input_R10.csv",
        "input_R20.csv",
        "input_R100.csv",
    ]


def test_only_input_csv_files_are_listed(tmp_path):
    _touch(tmp_path, ["input_F1.csv", "output_F1.csv", "R4-IVs_F1.csv", "input_F2.txt"])
    assert list_input_files(tmp_path.as_posix()) == ["input_F1.csv"]


def test_ordering_is_independent_of_creation_order(tmp_path_factory):
    forward = tmp_path_factory.mktemp("forward")
    backward = tmp_path_factory.mktemp("backward")
    names = ["input_F1.csv", "input_F2.csv", "input_F10.csv"]
    _touch(forward, names)
    _touch(backward, list(reversed(names)))
    assert list_input_files(forward.as_posix()) == list_input_files(backward.as_posix())


class _FakeDataset(InputScalingMixin):
    """Minimal stand-in exercising the scaling mixin without loading real data."""

    def __init__(self, samples):
        self._init_scaling()
        self.data = [{"input": {}} for _ in samples]
        self.targets = [{"label": f"sample_{i}"} for i in range(len(samples))]
        for temperature, hrt, y_in in samples:
            self._record_scaling_signals(jnp.asarray(temperature), jnp.asarray(hrt), jnp.asarray(y_in))
        self._apply_scaling()


def _dataset():
    return _FakeDataset(
        [
            ([310.0, 311.0], [0.1, 0.1], [[1.0, 2.0], [1.0, 2.0]]),
            ([320.0, 321.0], [0.2, 0.2], [[3.0, 4.0], [3.0, 4.0]]),
            ([330.0, 331.0], [0.3, 0.3], [[5.0, 6.0], [5.0, 6.0]]),
        ]
    )


def test_scaling_vector_has_one_entry_per_network_input():
    dataset = _dataset()
    # temperature + inflow/V_liq + two influent species
    assert dataset.data[0]["input"]["means"].shape == (4,)


def test_refit_excludes_held_out_samples():
    dataset = _dataset()
    everything = float(dataset.data[0]["input"]["means"][0])

    dataset.refit_scaling([0, 1])
    training_only = float(dataset.data[0]["input"]["means"][0])

    assert everything == pytest.approx(320.5)
    assert training_only == pytest.approx(315.5)
    assert dataset.scaling_indices == (0, 1)


def test_refit_writes_the_same_statistics_to_every_sample():
    dataset = _dataset()
    dataset.refit_scaling([0, 2])
    reference = dataset.data[0]["input"]["means"]
    for entry in dataset.data:
        assert jnp.allclose(entry["input"]["means"], reference)


def test_constant_signals_do_not_produce_a_zero_divisor():
    dataset = _FakeDataset([([300.0, 300.0], [0.1, 0.1], [[1.0]])])
    assert bool(jnp.all(dataset.data[0]["input"]["stds"] > 0))


def test_refit_rejects_unknown_and_empty_index_sets():
    dataset = _dataset()
    with pytest.raises(IndexError):
        dataset.refit_scaling([0, 99])
    with pytest.raises(ValueError):
        dataset.refit_scaling([])
