"""Figure construction, on synthetic tables and samples.

Only the parts that need neither trained checkpoints nor the reactor data are
covered here: the fold figures drawn from synthetic predictions, and the
component metadata, which is pure bookkeeping.

Every test closes its figure, because a test session that leaks figures
eventually trips matplotlib's open-figure warning and hides real problems.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adm1_package.experiments.figures import _symbol, fold_prediction_grid  # noqa: E402
from adm1_package.experiments.predictions import FoldPrediction, predict_reactor  # noqa: E402
from adm1_package.experiments.registry import MODEL_KEYS, get_spec  # noqa: E402
from adm1_package.utils import plot_style  # noqa: E402
from adm1_package.utils.plotting import COMPONENTS, axis_label, describe  # noqa: E402


# ---------------------------------------------------------------------------
# Component metadata
# ---------------------------------------------------------------------------
def test_every_model_output_has_presentation_metadata():
    """A component a model can produce must be drawable without a fallback."""
    for key in MODEL_KEYS:
        for output in get_spec(key).outputs:
            assert output in COMPONENTS, f"{key}: no metadata for {output!r}"


def test_describe_falls_back_for_unknown_components():
    component = describe("X_bac")
    assert component.key == "X_bac"
    assert component.title == "X_bac"
    assert component.unit == ""


def test_axis_label_uses_the_unit():
    # norm litres: dry gas at 0 degC and 1.01325 bar, as the campaigns report it
    assert axis_label("q_gasout") == "$q_{gas}$ [NL/d]"
    assert axis_label("ch4_out") == "$q_{CH_4}$ [NL/d]"
    assert axis_label("S_ac", compact=False) == "Concentration [g/L]"


# ---------------------------------------------------------------------------
# Rate symbols
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("k_hyd_ch", "$k_{hyd,ch}$"),
        ("k_dec_Xbac", "$k_{dec,Xbac}$"),
        ("f_ch", "$f_{ch}$"),
        ("pH", "$pH$"),
    ],
)
def test_symbol_renders_subscripts(name, expected):
    assert _symbol(name) == expected


# ---------------------------------------------------------------------------
# Every fold on every reactor
# ---------------------------------------------------------------------------
def _prediction(label, fold, split, hybrid, mechanistic, days=10):
    """A reactor as one fold predicts it, with constant curves and a time grid
    padded by 2 points, as the datasets pad it."""
    times = np.pad(np.arange(float(days)), (0, 2), mode="edge")
    valid = np.arange(days + 2) < days
    return FoldPrediction(
        label=label, fold=fold, split=split, times=times,
        hybrid={"q_gasout": np.full(days + 2, float(hybrid))},
        mechanistic={"q_gasout": np.full(days + 2, float(mechanistic))},
        measured={"q_gasout": (np.full(days + 2, 2.0), valid)},
    )


def _lines(ax, label):
    return [line for line in ax.get_lines() if line.get_label() == label]


def test_fold_prediction_grid_draws_every_fold_with_the_held_out_one_bold():
    records = [
        _prediction("LaRA-temp_F1", 1, "train", 4.0, 1.0),
        _prediction("LaRA-temp_F1", 2, "validation", 5.0, 2.0),
        _prediction("LaRA-temp_F1", 3, "train", 6.0, 3.0),
        _prediction("LaRA-temp_F2", 1, "validation", 4.0, 1.0),
        _prediction("LaRA-temp_F2", 2, "train", 5.0, 1.0),
    ]
    figure = fold_prediction_grid(records)
    first, second = figure.axes[:2]
    assert [ax.get_title(loc="left") for ax in (first, second)] == ["(a) LaRA-temp_F1 (fold 2)",
                                                          "(b) LaRA-temp_F2 (fold 1)"]
    (bold,) = _lines(first, "Hybrid (UDE), held-out fold")
    assert bold.get_ydata()[0] == 5.0
    assert len(bold.get_xdata()) == 10   # not the padded tail
    assert sorted(line.get_ydata()[0] for line in _lines(first, "Hybrid (UDE), other folds"))         == [4.0, 6.0]
    assert len(_lines(first, "Mechanistic, other folds")) == 2
    # F2's folds share their mechanistic curve: one line, no band for it
    assert len(_lines(second, "Mechanistic")) == 1
    assert len(second.collections) == 1
    assert len(_lines(first, "Measured")) == 1
    assert not any(ax.axison for ax in figure.axes[2:])
    plt.close(figure)


def test_fold_prediction_grid_draws_the_folds_mean_on_a_test_reactor():
    figure = fold_prediction_grid([_prediction("rice-bran_R90", 1, "test", 4.0, 1.0),
                                   _prediction("rice-bran_R90", 2, "test", 6.0, 1.0)])
    ax = figure.axes[0]
    assert ax.get_title(loc="left") == "(a) rice-bran_R90"
    (mean,) = _lines(ax, "Hybrid (UDE), mean of 2 folds")
    assert mean.get_ydata()[0] == pytest.approx(5.0)
    assert len(_lines(ax, "Hybrid (UDE), individual folds")) == 2
    plt.close(figure)


def test_fold_prediction_grid_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="no reactors"):
        fold_prediction_grid([_prediction("LaRA-temp_F1", 1, "validation", 1.0, 1.0)],
                             labels=["LaRA-temp_F9"])


# ---------------------------------------------------------------------------
# The spin-up, which the scores leave out
# ---------------------------------------------------------------------------
def _spans(ax):
    return [patch for patch in ax.patches if patch.get_label() == plot_style.SPIN_UP_LABEL]


def _extent(span):
    return span.get_x(), span.get_x() + span.get_width()


def _legend_labels(figure):
    return [text.get_text() for legend in figure.legends for text in legend.get_texts()]


def test_the_fold_figure_shades_the_spin_up_the_scores_leave_out():
    records = []
    for label in ("LaRA-temp_F1", "LaRA-temp_F2"):
        record = _prediction(label, 1, "validation", 4.0, 1.0)
        records.append(replace(record, times=record.times + 5.0, spin_up_days=3.0))
    figure = fold_prediction_grid(records)
    try:
        for ax in figure.axes[:2]:
            (span,) = _spans(ax)
            assert _extent(span) == (5.0, 8.0)   # from the first day, not from day 0
            assert span.get_zorder() < min(line.get_zorder() for line in ax.get_lines())
        assert not _spans(figure.axes[2])         # the unused panel
        assert _legend_labels(figure).count(plot_style.SPIN_UP_LABEL) == 1
    finally:
        plt.close(figure)


def test_a_record_without_a_spin_up_is_not_shaded():
    figure = fold_prediction_grid([_prediction("LaRA-temp_F1", 1, "validation", 4.0, 1.0)])
    try:
        assert not _spans(figure.axes[0])
        assert plot_style.SPIN_UP_LABEL not in _legend_labels(figure)
    finally:
        plt.close(figure)


def test_the_spin_up_is_shaded_no_further_than_the_series_goes():
    figure, ax = plt.subplots()
    try:
        assert _extent(plot_style.shade_spin_up(ax, np.arange(5.0, 25.0), 30.0)) == (5.0, 24.0)
        assert plot_style.shade_spin_up(ax, np.arange(5.0, 25.0), 0.0) is None
        assert len(ax.patches) == 1
    finally:
        plt.close(figure)


def test_a_folds_prediction_carries_the_spin_up_the_fold_trained_with():
    times = np.arange(10.0)
    sample = ({"parameters": {"k_hyd_ch": 0.5}, "initial_values": None, "input": {"time": times}},
              {"label": "LaRA-temp_F1", "values": np.zeros((10, 1)), "indices": np.ones((10, 1)),
               "object_indices": {"q_gasout": 0}})
    opened = SimpleNamespace(
        dataset=[sample], fold=2, model=None, key=None, settings={"spin_up_days": 12.0},
        hybrid=lambda parameters, initial, forcing, model, key: ({"q_gasout": times}, None),
        mechanistic=lambda parameters, initial, forcing: ({"q_gasout": times}, None),
    )
    record = predict_reactor(opened, 0, "validation")
    assert (record.label, record.fold, record.spin_up_days) == ("LaRA-temp_F1", 2, 12.0)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def test_figure_style_restores_the_previous_rcparams():
    """The point of the context manager: importing a plot must not restyle a notebook."""
    before = matplotlib.rcParams["font.size"]
    with plot_style.figure_style():
        assert matplotlib.rcParams["font.size"] == plot_style.RC_PARAMS["font.size"]
    assert matplotlib.rcParams["font.size"] == before


def test_figure_style_accepts_underscore_overrides():
    with plot_style.figure_style(font_size=13):
        assert matplotlib.rcParams["font.size"] == 13


def test_model_roles_stay_distinguishable_without_colour():
    """Greyscale printing must not collapse two models onto one line."""
    patterns = [plot_style.LINES[role]["linestyle"] for role in ("mech", "ude", "base", "reference")]
    assert len(set(map(str, patterns))) == len(patterns)


def test_model_roles_have_distinct_colours():
    colours = [plot_style.COLORS[role] for role in ("data", "mech", "ude", "base")]
    assert len(set(colours)) == len(colours)


def test_the_double_column_width_is_elseviers():
    assert plot_style.WIDTH_DOUBLE == pytest.approx(190 / 25.4)


def test_save_figure_writes_each_requested_format_into_a_new_directory(tmp_path):
    figure = plt.figure()
    written = plot_style.save_figure(figure, tmp_path / "nested" / "figure", formats=("png", "pdf"))
    assert written == [tmp_path / "nested" / "figure.png", tmp_path / "nested" / "figure.pdf"]
    assert all(path.exists() for path in written)
    plt.close(figure)


def test_label_panels_skips_axes_that_were_switched_off():
    figure, axes = plt.subplots(1, 3)
    try:
        axes[2].axis("off")
        plot_style.label_panels(axes)
        drawn = [text.get_text() for ax in axes for text in ax.texts]
        assert drawn == ["(a)", "(b)"]
    finally:
        plt.close(figure)
