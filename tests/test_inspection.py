"""Looking at single reactors (adm1_package/experiments/inspection.py).

What it shows has to be computed the way evaluation computes it: the same scoring
mask, the same fold preparation, the same mechanistic start.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import diffrax as dx
import jax.numpy as jnp
import numpy as np
import pytest

from adm1_package.experiments import inspection
from adm1_package.experiments.calibrations import installed_forcing
from adm1_package.experiments.registry import MODEL_KEYS, get_spec
from adm1_package.experiments.splits import DataSplit
from adm1_package.utils import plot_style
from adm1_package.utils.corrections import DEFAULT
from adm1_package.utils.plotting import forcing_series

SPLIT = DataSplit(
    test=["rice-bran_R90"],
    folds=[["LaRA-temp_F1", "rice-bran_R30"], ["LaRA-temp_F2", "rice-bran_R60"]],
    seed=0,
)
REACTOR = "LaRA-temp_F4"


# ---------------------------------------------------------------------------
# Finding a reactor, a fold and a run
# ---------------------------------------------------------------------------
class _Samples:
    labels = ["LaRA-temp_F1", "LaRA-temp_F2"]

    def __getitem__(self, index):
        return {"index": index}, {"label": self.labels[index]}


def test_a_sample_is_found_by_its_label():
    x, y = inspection.sample(_Samples(), "LaRA-temp_F2")
    assert (x["index"], y["label"]) == (1, "LaRA-temp_F2")


def test_an_unknown_label_names_the_known_ones():
    with pytest.raises(KeyError, match="LaRA-temp_F1, LaRA-temp_F2"):
        inspection.sample(_Samples(), "LaRA-temp_F9")


def test_the_default_fold_is_the_one_that_held_the_reactor_out():
    assert inspection.held_out_fold(SPLIT, "rice-bran_R60") == 2
    # every fold holds a test reactor out, so none is the one
    assert inspection.held_out_fold(SPLIT, "rice-bran_R90") is None
    with pytest.raises(KeyError, match="LaRA-temp_F9"):
        inspection.held_out_fold(SPLIT, "LaRA-temp_F9")


def test_a_reactor_has_one_role_per_fold():
    assert inspection.role_in_fold(SPLIT, 1, "LaRA-temp_F1") == "validation"
    assert inspection.role_in_fold(SPLIT, 2, "LaRA-temp_F1") == "train"
    assert inspection.role_in_fold(SPLIT, 2, "rice-bran_R90") == "test"
    with pytest.raises(KeyError, match="LaRA-temp_F9"):
        inspection.role_in_fold(SPLIT, 1, "LaRA-temp_F9")


def _run(root, name, timestamp, recorded=True, split=True):
    run = root / "r4" / name
    for fold in ("fold_1", "fold_2"):
        (run / fold).mkdir(parents=True)
        record = {"correction": DEFAULT.to_dict(), "initial_state": "calibrated",
                  "spin_up_days": 30.0, "forcing": "smoothstep", "calibration": "per-fold"}
        if not recorded and fold == "fold_2":
            record = {"correction": DEFAULT.to_dict()}
        (run / fold / "hparams.json").write_text(json.dumps(record), encoding="utf-8")
    if split:
        (run / "split.json").write_text(json.dumps(SPLIT.to_dict()), encoding="utf-8")
    (run / "metadata.json").write_text(json.dumps({"timestamp_utc": timestamp}), encoding="utf-8")
    return run


def test_only_runs_the_current_code_can_restore_are_offered_newest_first(tmp_path):
    first = _run(tmp_path, "z_single", "2026-10-01T15:00:00+00:00")
    second = _run(tmp_path, "a_split", "2026-10-03T09:00:00+00:00")
    _run(tmp_path, "earlier_version", "2026-10-04T09:00:00+00:00", recorded=False)
    _run(tmp_path, "no_split", "2026-10-05T09:00:00+00:00", split=False)
    assert inspection.restorable_runs("r4", root=tmp_path) == [second, first]
    assert inspection.latest_run("r4", root=tmp_path) == second


def _ensemble(root, name, members):
    """An ensemble directory as evaluate.py --ensemble leaves it beside its runs."""
    directory = root / "r4" / name
    directory.mkdir(parents=True)
    (directory / "split.json").write_text(json.dumps(SPLIT.to_dict()), encoding="utf-8")
    (directory / "ensemble.json").write_text(json.dumps({
        "model_key": "r4", "members": [member.name for member in members], "seeds": [42, 43]}),
        encoding="utf-8")
    return directory


def test_ensembles_are_offered_beside_the_runs_but_the_latest_run_is_trained(tmp_path):
    """Listed by when they were made; one whose members are gone is not offered."""
    first = _run(tmp_path, "a_seed42", "2026-10-01T15:00:00+00:00")
    second = _run(tmp_path, "a_seed43", "2026-10-03T09:00:00+00:00")
    ensemble = _ensemble(tmp_path, "a_ensemble_s42-43", [first, second])
    _ensemble(tmp_path, "b_ensemble_s42-44", [first, tmp_path / "r4" / "a_seed44"])
    assert inspection.restorable_runs("r4", root=tmp_path) == [ensemble, second, first]
    assert inspection.latest_run("r4", root=tmp_path) == second


def test_without_a_restorable_run_the_error_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="r4"):
        inspection.latest_run("r4", root=tmp_path)


# ---------------------------------------------------------------------------
# Scores and figures
# ---------------------------------------------------------------------------
def _forcing(days=60, start=5.0):
    """A forcing signal laid out as the datasets lay it out, padded by 3 points."""
    times = start + jnp.arange(float(days))
    values = jnp.stack([jnp.full(days, 310.15), jnp.full(days, 0.5),
                        jnp.full(days, 1.0), jnp.full(days, 950.0)], axis=1)
    coeffs = dx.backward_hermite_coefficients(times, values)
    padded_times = jnp.pad(times, (0, 3), mode="edge")
    padded_coeffs = tuple(jnp.pad(c, ((0, 3), (0, 0))) for c in coeffs)
    return {"input": {"time": padded_times, "coeffs": padded_coeffs}}


def _targets(outputs, days=60, label=REACTOR):
    values = np.zeros((days + 3, len(outputs)))
    indices = np.ones((days + 3, len(outputs)))
    indices[days:] = 0.0  # the padded tail
    return {"values": values, "indices": indices, "label": label,
            "object_indices": {name: column for column, name in enumerate(outputs)}}


def _simulation(key, solution, outputs=("q_gasout", "S_TAN")):
    return inspection.Simulation(
        name=get_spec(key).display_name, inputs=_forcing(), targets=_targets(outputs),
        solution={name: np.asarray(value, dtype=float) for name, value in solution.items()},
        outputs=tuple(outputs), style=dict(inspection.VARIANT_LINES[key]),
    )


def test_scores_skip_the_first_point_and_the_spin_up_as_evaluation_does():
    time = np.asarray(_forcing()["input"]["time"])
    gas = np.where(time < 35.0, 100.0, 1.0)   # wrong by 99 in the spin-up, by 1 after it
    table = inspection.scores([_simulation("r4", {"q_gasout": gas, "S_TAN": np.full(63, 2.0)})],
                              spin_up_days=30.0)
    assert table.loc["q_gasout", "ADM1-R4"] == pytest.approx(1.0)
    assert table.loc["S_TAN", "ADM1-R4"] == pytest.approx(2.0)


def test_an_output_a_model_does_not_produce_is_left_empty():
    ones = np.ones(63)
    table = inspection.scores([
        _simulation("r4", {"q_gasout": ones}, outputs=("q_gasout",)),
        _simulation("r3", {"q_gasout": ones, "S_ac": ones}, outputs=("q_gasout", "S_ac")),
    ])
    assert list(table.index) == ["q_gasout", "S_ac"]
    assert list(table.columns) == ["ADM1-R4", "ADM1-R3"]
    assert np.isnan(table.loc["S_ac", "ADM1-R4"])


def test_the_overlay_has_a_panel_per_output_and_one_each_for_temperature_and_inflow():
    ones = np.ones(63)
    figure = inspection.overlay([
        _simulation("r4", {"q_gasout": ones, "S_TAN": ones}),
        _simulation("r3", {"q_gasout": ones, "S_ac": ones}, outputs=("q_gasout", "S_ac")),
    ])
    titles = [ax.get_title(loc="left") for ax in figure.axes if ax.axison]
    assert titles == ["Biogas production rate", "Total ammoniacal nitrogen", "Acetic acid",
                      "Temperature", "Inflow"]
    # no twin axes: a second y-scale on a panel would add an axes object
    assert len(figure.axes) == 6   # a 2 x 3 grid, the last panel switched off


def test_a_variant_keeps_its_colour_whichever_variants_are_shown():
    ones = np.ones(63)
    alone = inspection.overlay([_simulation("r4", {"q_gasout": ones, "S_TAN": ones})])
    together = inspection.overlay([
        _simulation("r2", {"q_gasout": ones, "S_TAN": ones}),
        _simulation("r4", {"q_gasout": ones, "S_TAN": ones}),
    ])

    def colour(figure, name):
        return next(line.get_color() for line in figure.axes[0].get_lines()
                    if line.get_label() == name)

    assert colour(alone, "ADM1-R4") == colour(together, "ADM1-R4")
    assert colour(together, "ADM1-R2") != colour(together, "ADM1-R4")


def test_every_variant_an_inspection_shows_has_its_own_colour_and_dash():
    # the study's variants and the full ADM1, told apart in colour and in greyscale
    assert set(inspection.VARIANT_LINES) == set(MODEL_KEYS) | {"base"}
    for feature in ("color", "linestyle"):
        values = [str(style[feature]) for style in inspection.VARIANT_LINES.values()]
        assert len(set(values)) == len(values), feature


def test_the_overlay_draws_the_measurements_once_and_not_the_padded_tail():
    figure = inspection.overlay([_simulation("r4", {"q_gasout": np.ones(63), "S_TAN": np.ones(63)})])
    measured = [line for line in figure.axes[0].get_lines() if line.get_label() == "Measured"]
    assert len(measured) == 1
    assert len(measured[0].get_xdata()) == 60


def _spans(ax):
    return [patch for patch in ax.patches if patch.get_label() == plot_style.SPIN_UP_LABEL]


def _extent(span):
    return span.get_x(), span.get_x() + span.get_width()


def test_the_overlay_shades_the_spin_up_on_every_output_but_not_on_the_forcing():
    ones = np.ones(63)
    simulation = replace(_simulation("r4", {"q_gasout": ones, "S_TAN": ones}), spin_up_days=20.0)
    figure = inspection.overlay([simulation])
    outputs, forcing = figure.axes[:2], figure.axes[2:4]
    # from the sample's first day, which is day 5 here
    assert [_extent(span) for ax in outputs for span in _spans(ax)] == [(5.0, 25.0)] * 2
    assert not any(_spans(ax) for ax in forcing)
    legend = [text.get_text() for text in figure.legends[0].get_texts()]
    assert legend.count(plot_style.SPIN_UP_LABEL) == 1


def test_a_simulation_outside_a_run_is_shaded_over_the_spin_up_the_scores_leave_out():
    figure = inspection.overlay([_simulation("r4", {"q_gasout": np.ones(63), "S_TAN": np.ones(63)})])
    (span,) = _spans(figure.axes[0])
    assert _extent(span) == (5.0, 5.0 + inspection.SPIN_UP_DAYS)


def _fold(fold, role, mechanistic, hybrid, **fields):
    """One fold's inspection of the reactor, with constant solutions; ``fields`` go to
    both of its simulations."""
    def simulation(name, value, style):
        constant = np.full(63, float(value))
        return inspection.Simulation(
            name=name, inputs=_forcing(), targets=_targets(("q_gasout",)),
            solution={"q_gasout": constant}, outputs=("q_gasout",), style=dict(style), **fields)

    return inspection.RunInspection(
        model_key="r4", run=Path("runs/r4/split"), fold=fold, role=role, settings={},
        hybrid=simulation("ADM1-R4 hybrid", hybrid, inspection.LINES["ude"]),
        mechanistic=simulation("ADM1-R4 mechanistic", mechanistic, inspection.LINES["mech"]),
        record=None, checkpoint=("last", 1))


def _labels(figure):
    return [line.get_label() for line in figure.axes[0].get_lines()]


def test_the_fold_overlay_draws_the_held_out_fold_bold_over_the_other_folds():
    figure = inspection.fold_overlay([
        _fold(1, "train", 1.0, 4.0), _fold(2, "validation", 2.0, 5.0), _fold(3, "train", 3.0, 6.0),
    ])
    lines = {label: [line for line in figure.axes[0].get_lines() if line.get_label() == label]
             for label in _labels(figure)}
    held_out = lines["ADM1-R4 hybrid, fold 2 (validation)"]
    assert len(held_out) == 1 and held_out[0].get_ydata()[0] == 5.0
    assert sorted(line.get_ydata()[0] for line in lines["ADM1-R4 hybrid, other folds"]) == [4.0, 6.0]
    assert len(lines["ADM1-R4 mechanistic, other folds"]) == 2
    assert held_out[0].get_linewidth() > lines["ADM1-R4 hybrid, other folds"][0].get_linewidth()
    # a band over the folds' range per model
    assert len(figure.axes[0].collections) == 2
    assert [ax.get_title(loc="left") for ax in figure.axes if ax.axison] == [
        "Biogas production rate", "Temperature", "Inflow"]


def test_the_fold_overlay_draws_a_model_the_folds_simulate_alike_once():
    figure = inspection.fold_overlay([_fold(1, "validation", 2.0, 4.0), _fold(2, "train", 2.0, 5.0)])
    assert _labels(figure).count("ADM1-R4 mechanistic") == 1
    assert not any(label.startswith("ADM1-R4 mechanistic,") for label in _labels(figure))
    assert len(figure.axes[0].collections) == 1


def test_the_fold_overlay_of_a_test_reactor_draws_the_folds_mean():
    figure = inspection.fold_overlay([_fold(1, "test", 1.0, 4.0), _fold(2, "test", 3.0, 6.0)])
    mean = next(line for line in figure.axes[0].get_lines()
                if line.get_label() == "ADM1-R4 hybrid, mean of 2 folds")
    assert mean.get_ydata()[0] == pytest.approx(5.0)
    assert _labels(figure).count("ADM1-R4 hybrid, individual folds") == 2


def test_the_fold_overlay_refuses_folds_of_different_reactors():
    other = _fold(2, "train", 1.0, 1.0)
    other.hybrid.targets = _targets(("q_gasout",), label="LaRA-temp_F1")
    with pytest.raises(ValueError, match="different reactors"):
        inspection.fold_overlay([_fold(1, "validation", 1.0, 1.0), other])


def test_the_fold_overlay_shades_the_spin_up_the_folds_trained_with():
    figure = inspection.fold_overlay([_fold(1, "validation", 2.0, 4.0, spin_up_days=12.0),
                                      _fold(2, "train", 2.0, 5.0, spin_up_days=12.0)])
    (span,) = _spans(figure.axes[0])
    assert _extent(span) == (5.0, 17.0)


# ---------------------------------------------------------------------------
# Tables of a dataset's forcing and measurements
# ---------------------------------------------------------------------------
class _Reactors:
    labels = [REACTOR]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return _forcing(), _targets(("q_gasout", "S_TAN"))


def test_the_forcing_table_names_the_columns_and_drops_the_padding():
    table = inspection.forcing_frame(_Reactors(), ("S_TAN", "S_h2o"))
    assert list(table["variable"].unique()) == ["T_op", "q_in", "S_TAN", "S_h2o"]
    assert (table.groupby("variable").size() == 60).all()
    temperature = table[table["variable"] == "T_op"]["value"]
    assert temperature.to_numpy() == pytest.approx(37.0)   # Kelvin on load, Celsius here
    assert set(table["campaign"]) == {"LaRA-temp"}


def test_the_measurement_table_drops_the_padded_tail():
    table = inspection.measurement_frame(_Reactors())
    assert sorted(table["output"].unique()) == ["S_TAN", "q_gasout"]
    assert (table.groupby("output").size() == 60).all()
    assert table["time"].min() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# On the data
# ---------------------------------------------------------------------------
@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2", "base"])
def test_a_mechanistic_simulation_is_the_variants_own(key, data_dir):
    spec = get_spec(key)
    simulation = inspection.simulate_mechanistic(key, REACTOR, data_dir=data_dir)
    # with the solids, which an inspection draws and which start at their measurement, and
    # the forcing the installed calibration was fitted to
    dataset = spec.dataset(data_dir, data_to_use=["LaRA-temp"], diagnostics=True,
                           forcing=installed_forcing(data_dir))
    x, y = inspection.sample(dataset, REACTOR)
    direct, _ = spec.mechanistic()(x["parameters"], x["initial_values"], x["input"])
    assert simulation.name == spec.display_name and simulation.outputs == spec.drawn_outputs
    for name in spec.drawn_outputs:
        np.testing.assert_allclose(simulation.solution[name], np.asarray(direct[name]), rtol=1e-10)


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r2"])
def test_the_forcing_table_follows_the_variants_forcing_layout(key, installed_dataset):
    spec = get_spec(key)
    dataset = installed_dataset(key, ["LaRA-temp"])
    table = inspection.forcing_frame(dataset, spec.influent_species)
    assert list(table["variable"].unique()) == ["T_op", "q_in", *spec.influent_species]
    x, _ = inspection.sample(dataset, REACTOR)
    times, celsius, _ = forcing_series(x)
    valid = int(np.argmax(times)) + 1
    mine = table[(table["label"] == REACTOR) & (table["variable"] == "T_op")]["value"]
    np.testing.assert_allclose(mine.to_numpy(), celsius[:valid])


@pytest.mark.requires_data
def test_the_measurement_table_holds_every_output(installed_dataset):
    spec = get_spec("r3")
    table = inspection.measurement_frame(installed_dataset("r3", ["LaRA-temp"]))
    assert set(table["output"]) == set(spec.outputs)
