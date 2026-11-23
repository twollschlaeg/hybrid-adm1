"""The figures drawn from a cross-validated run's predictions (:mod:`.predictions`).

Under cross-validation every development reactor is held out in exactly one fold and
trained on in the others, so the figures draw every fold:

:func:`fold_prediction_grid`
    every reactor drawn as predicted by every fold, hybrid and mechanistic, the
    one that held it out in bold: how much a prediction depends on the fold.
:func:`kinetic_trajectories`
    the learned rate corrections of *all* folds at once. A single model cannot
    distinguish a learned process signal from one fit's idiosyncrasy; agreement
    across folds can.

:func:`draw_folds`, the band-and-lines of one model over the folds, is shared with
:func:`.inspection.fold_overlay`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.plot_style import (
    CAMPAIGN_COLORS,
    COLORS,
    LINES,
    WIDTH_DOUBLE,
    figure_style,
    label_panels,
    shade_spin_up,
)
from ..utils.plotting import axis_label, describe
from .predictions import FoldPrediction
from .registry import ModelSpec
from .splits import campaign_of

__all__ = [
    "fold_prediction_grid",
    "draw_folds",
    "kinetic_trajectories",
]

#: Units of the two blocks the network emits.
_NN_UNITS = {"dynamic_rates": "d$^{-1}$", "nutrient_factors": "-"}


def _symbol(name: str) -> str:
    """``k_hyd_ch`` -> ``$k_{hyd,ch}$``, matching the paper's notation."""
    head, _, tail = name.partition("_")
    if not tail:
        return f"${head}$"
    return f"${head}_{{{tail.replace('_', ',')}}}$"


# ---------------------------------------------------------------------------
# Figure: predictions over time
# ---------------------------------------------------------------------------
def _group_by_label(records: Sequence[FoldPrediction]) -> dict[str, list[FoldPrediction]]:
    """The records by reactor, in the order they come."""
    grouped: dict[str, list[FoldPrediction]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)
    return grouped


def draw_folds(
    ax,
    times: np.ndarray,
    curves: np.ndarray,
    style: Mapping[str, Any],
    name: str,
    held_out: int | None = None,
    held_out_label: str | None = None,
) -> None:
    """One model's prediction of a reactor by every fold, on one axis.

    ``curves`` holds one row per fold. They are drawn as a band over their range
    and a thin line each, in the colour of ``style``; the row ``held_out``, the
    fold that never trained on the reactor, goes on top in the full ``style`` and
    is labelled ``held_out_label``. Without it (a test reactor, which every fold
    held out) the folds' mean is drawn bold instead. Folds that predict alike,
    such as a mechanistic model on a calibration they share, are one line,
    labelled ``name``.
    """
    curves = np.asarray(curves, dtype=float)
    if np.allclose(curves, curves[0], rtol=1e-10, atol=0.0):
        ax.plot(times, curves[0], label=name, **style)
        return
    color = style["color"]
    ax.fill_between(times, curves.min(axis=0), curves.max(axis=0), color=color,
                    alpha=0.15, linewidth=0, zorder=1.0)
    # the bold line first, so a legend lists it above the model's other folds
    if held_out is not None:
        ax.plot(times, curves[held_out], label=held_out_label or f"{name}, held-out fold",
                **style)
        others = "other folds"
    else:
        ax.plot(times, curves.mean(axis=0), label=f"{name}, mean of {len(curves)} folds",
                **style)
        others = "individual folds"
    for row, curve in enumerate(curves):
        if row != held_out:
            ax.plot(times, curve, color=color, linewidth=0.6, alpha=0.5, zorder=1.2,
                    label=f"{name}, {others}")


def fold_prediction_grid(
    records: Sequence[FoldPrediction],
    component: str = "q_gasout",
    labels: Sequence[str] | None = None,
    ncols: int = 3,
    width: float = WIDTH_DOUBLE,
    panel_height: float = 1.45,
):
    """One panel per reactor, each predicted by every fold.

    The fold that held the reactor out is drawn in bold, the folds that trained on
    it beside it (:func:`draw_folds`): the hybrid's and the mechanistic model's
    (MECH-CV's) spread says how much a prediction depends on the reactors a fold
    saw, and how far the out-of-sample fold sits from the fitted ones. The panel's
    title names the fold that held the reactor out. The spin-up the scores leave
    out is shaded.

    Args:
      records: :func:`.predictions.collect_predictions` with ``splits=("validation", "train")``
        (and ``"test"`` if the split has test reactors), so that every fold
        predicts every reactor.
      component: which model output to draw.
      labels: restrict and order the panels; default is every label, campaign by
        campaign.
    """
    import matplotlib.pyplot as plt

    grouped = _group_by_label(records)
    if labels is None:
        labels = sorted(grouped, key=lambda name: (campaign_of(name), name))
    labels = [name for name in labels if name in grouped]
    if not labels:
        raise ValueError("no reactors to draw")

    nrows = int(np.ceil(len(labels) / ncols))

    with figure_style():
        figure, axes = plt.subplots(nrows, ncols, figsize=(width, panel_height * nrows),
                                    layout="constrained", squeeze=False)
        flat = axes.ravel()

        for ax, label in zip(flat, labels):
            entries = sorted(grouped[label], key=lambda entry: entry.fold)
            first = entries[0]
            length = int(np.argmax(first.times)) + 1   # before the padded tail
            times = first.times[:length]
            held_out = [row for row, entry in enumerate(entries) if entry.split == "validation"]
            held_out = held_out[0] if len(held_out) == 1 else None

            for model, name in (("mechanistic", "Mechanistic"), ("hybrid", "Hybrid (UDE)")):
                drawn = [getattr(entry, model)[component][:length] for entry in entries
                         if component in getattr(entry, model)]
                if len(drawn) == len(entries):
                    draw_folds(ax, times, np.stack(drawn),
                               LINES["mech" if model == "mechanistic" else "ude"], name,
                               held_out=held_out)

            if component in first.measured:
                values, valid = first.measured[component]
                valid = valid[:length]
                ax.plot(times[valid], values[:length][valid], label="Measured",
                        **LINES["data_line"])
            shade_spin_up(ax, times, first.spin_up_days)

            title = label if held_out is None else f"{label} (fold {entries[held_out].fold})"
            ax.set_title(title, color=CAMPAIGN_COLORS.get(campaign_of(label), "black"))
            ax.set_xlabel("Time [d]")
            ax.set_ylabel(axis_label(component))

        for ax in flat[len(labels):]:
            ax.axis("off")

        label_panels(flat)

        handles: dict[str, Any] = {}
        for ax in flat[: len(labels)]:
            for handle, name in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(name, handle)
        # three columns keep each model's bold line above its other folds
        figure.legend(list(handles.values()), list(handles), loc="outside lower center",
                      ncols=min(len(handles), 3))
        figure.suptitle(f"{describe(component).title}, every fold")

    return figure


# ---------------------------------------------------------------------------
# Figure: learned kinetic trajectories
# ---------------------------------------------------------------------------
def kinetic_trajectories(
    records: Sequence[FoldPrediction],
    spec: ModelSpec,
    label: str,
    rates: Sequence[str] | None = None,
    include_factors: bool = True,
    ncols: int = 3,
    width: float = WIDTH_DOUBLE,
    panel_height: float = 1.5,
):
    """Learned kinetic rates and influent degradability factors over time.

    A single rate trajectory drawn against the constant mechanistic value cannot
    separate a learned process signal from one fit's idiosyncrasy, which matters
    when neural-network outputs are read mechanistically.

    The figure therefore draws every fold. Where the folds agree, the correction is
    a property of the data. Where they scatter, it is not, and the figure says so
    rather than hiding it behind a single line.

    The dashed reference is the reactor's own **calibrated** mechanistic value,
    read from the sample's parameter set -- the quantity the caption claims it is.

    Args:
      records: output of :func:`.predictions.collect_predictions`, which must have been called
        with ``extra_keys`` including ``"dynamic_rates"`` and ``"nutrient_factors"``.
      spec: the variant's :class:`~adm1_package.experiments.registry.ModelSpec`,
        which carries the name of each column of ``dynamic_rates``.
      label: which reactor to show.
      rates: restrict to a subset of ``spec.rate_names``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    entries = [record for record in records if record.label == label]
    if not entries:
        raise ValueError(f"no predictions for reactor {label!r}")
    if not spec.rate_names:
        raise ValueError(f"{spec.display_name} has no rate_names in its ModelSpec")

    available = [entry for entry in entries if "dynamic_rates" in entry.hybrid]
    if not available:
        raise ValueError(
            "records carry no 'dynamic_rates'; call collect_predictions with "
            "extra_keys=('dynamic_rates', 'nutrient_factors')"
        )

    rate_names = list(rates) if rates is not None else list(spec.rate_names)
    unknown = set(rate_names) - set(spec.rate_names)
    if unknown:
        raise KeyError(f"{spec.display_name} does not learn {sorted(unknown)}")

    panels: list[tuple[str, str, np.ndarray]] = []
    for name in rate_names:
        column = spec.rate_names.index(name)
        stacked = np.stack([entry.hybrid["dynamic_rates"][:, column] for entry in available])
        panels.append((name, "dynamic_rates", stacked))

    if include_factors and "nutrient_factors" in available[0].hybrid:
        for column, name in enumerate(spec.factor_names):
            stacked = np.stack(
                [entry.hybrid["nutrient_factors"][:, column] for entry in available]
            )
            panels.append((name, "nutrient_factors", stacked))

    times = available[0].times
    parameters = available[0].parameters
    nrows = int(np.ceil(len(panels) / ncols))

    with figure_style():
        figure, axes = plt.subplots(
            nrows, ncols,
            figsize=(width, panel_height * nrows),
            layout="constrained",
            squeeze=False,
        )
        flat = axes.ravel()

        for position, (name, kind, curves) in enumerate(panels):
            ax = flat[position]

            if len(curves) > 1:
                ax.fill_between(
                    times, curves.min(axis=0), curves.max(axis=0),
                    color=COLORS["ude"], alpha=0.20, linewidth=0,
                )
                for curve in curves:
                    ax.plot(times, curve, color=COLORS["ude"], linewidth=0.5, alpha=0.55)
            ax.plot(times, curves.mean(axis=0), color=COLORS["ude"], linewidth=1.4, zorder=3)

            # The calibrated mechanistic constant, where the model has one.
            if name in parameters:
                ax.axhline(parameters[name], **LINES["reference"])

            ax.set_title(_symbol(name))
            ax.set_xlabel("Time [d]")
            ax.set_ylabel(f"[{_NN_UNITS[kind]}]")

        for ax in flat[len(panels):]:
            ax.axis("off")

        label_panels(flat)
        figure.suptitle(f"{spec.display_name}: learned corrections, {label}")
        figure.legend(
            handles=[
                Line2D([], [], color=COLORS["ude"], linewidth=1.4,
                       label=f"Hybrid, mean of {len(available)} folds"),
                Line2D([], [], color=COLORS["ude"], linewidth=0.5, alpha=0.55,
                       label="Individual folds"),
                Line2D([], [], label="Calibrated mechanistic value", **LINES["reference"]),
            ],
            loc="outside lower center",
            ncols=3,
        )

    return figure
