"""Shared figure style: colours, sizes and rcParams.

Every plotting function takes its figure size and colours from here, and none
calls bare ``plt.rc(...)``, which would change the appearance of every figure
drawn later in the same session.

Three things are fixed here.

**Size.** Figures are built at their final printed width, so the journal does
not rescale them and the font sizes land where they were set: Elsevier's 190 mm
double column.

**Colour.** Green and red are the classic confusion pair for red-green colour
vision deficiency, which affects roughly 8% of men, and they convert to nearly
the same grey. The palette here is
Okabe-Ito, which is designed to stay distinguishable under all common forms of
colour blindness.

**Grayscale.** Hue is never the only carrier of meaning: every model role also
has its own dash pattern, so the figures survive being printed in black and
white regardless of the palette.

Nothing here mutates global state -- :func:`figure_style` is a context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "COLORS",
    "CAMPAIGN_COLORS",
    "LINES",
    "SPIN_UP_LABEL",
    "WIDTH_DOUBLE",
    "figure_style",
    "label_panels",
    "save_figure",
    "shade_spin_up",
]


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
#: Okabe-Ito, the standard colour-vision-deficiency-safe qualitative palette.
#: Okabe & Ito (2008), "Color Universal Design".
_OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

#: Role -> colour. Blue/vermillion is the most robust pair in the palette: it
#: separates under every common colour vision deficiency *and* in luminance, so
#: it also survives greyscale conversion.
COLORS: dict[str, str] = {
    "data": "#4D4D4D",               # measurements: neutral grey, never a hue
    "mech": _OKABE_ITO["blue"],      # mechanistic ADM1 variant
    "anchor": _OKABE_ITO["sky_blue"],  # the mechanistic model a hybrid is built on
    "ude": _OKABE_ITO["vermillion"], # hybrid (UDE) model
    "base": _OKABE_ITO["green"],     # full ADM1 reference
    "reference": "#9A9A9A",          # calibrated constant / baseline value
    "highlight": _OKABE_ITO["purple"],
}

#: Experimental campaign -> colour, for figures that group by campaign.
CAMPAIGN_COLORS: dict[str, str] = {
    "LaRA-temp": _OKABE_ITO["blue"],
    "LaRA-substrate": _OKABE_ITO["vermillion"],
    "rice-bran": _OKABE_ITO["green"],
}

#: Role -> complete line style. The dash pattern is what carries the distinction
#: in greyscale, so it must stay different even if the colours are overridden.
LINES: dict[str, dict[str, Any]] = {
    "data": dict(
        color=COLORS["data"], linestyle="none", marker="o", markersize=2.6,
        markerfacecolor="none", markeredgewidth=0.7, zorder=2.5,
    ),
    "data_line": dict(
        color=COLORS["data"], linestyle=(0, (1, 1.4)), linewidth=1.1, zorder=2.5,
    ),
    "mech": dict(color=COLORS["mech"], linestyle=(0, (5, 1.8)), linewidth=1.3, zorder=2.0),
    "anchor": dict(color=COLORS["anchor"], linestyle=(0, (6, 1.5, 1.5, 1.5)), linewidth=1.3,
                   zorder=2.1),
    "ude": dict(color=COLORS["ude"], linestyle="-", linewidth=1.4, zorder=2.2),
    "base": dict(color=COLORS["base"], linestyle=(0, (3, 1.2, 1, 1.2)), linewidth=1.2, zorder=1.8),
    "reference": dict(color=COLORS["reference"], linestyle=(0, (4, 2)), linewidth=1.0, zorder=1.5),
}

#: The legend entry of the days the scores leave out (:func:`shade_spin_up`).
SPIN_UP_LABEL = "Spin-up (not scored)"


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------
_MM = 1.0 / 25.4

#: Elsevier's double-column width, in inches (7.48").
WIDTH_DOUBLE = 190 * _MM


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
# Sized for a figure drawn at its final printed width: Elsevier asks for at
# least 7 pt in the printed figure, so nothing here goes below that.
RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.transparent": False,

    # Arial/Helvetica if the system has them, DejaVu Sans (shipped with
    # matplotlib) otherwise. Matplotlib falls back silently, so this is safe
    # everywhere; the mathtext set is fixed to DejaVu so symbols do not change
    # shape depending on which machine drew the figure.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",

    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,

    # Open frames read better at small sizes than a full box.
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.titlelocation": "left",
    "axes.titlepad": 3.0,
    "axes.labelpad": 2.0,
    "axes.axisbelow": True,

    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,

    # A grid should recede behind the data, not compete with it.
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "axes.grid.axis": "both",

    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.2,
    "legend.handletextpad": 0.5,

    "lines.linewidth": 1.3,
    "lines.markersize": 2.6,

    # Keep text as text in vector output, so the publisher can reflow it and
    # readers can search it.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


@contextmanager
def figure_style(**overrides: Any) -> Iterator[None]:
    """Apply the shared style for the duration of the block.

    Use this rather than ``plt.rc`` / ``plt.style.use``: it restores whatever was
    set before, so importing a plotting function cannot change the appearance of
    a session's other figures.

    >>> with figure_style(font_size=9):
    ...     fig, ax = plt.subplots()

    Keyword names may be given with underscores (``font_size``) or as rcParam
    keys (``{"font.size": 9}`` passed through ``**``).
    """
    import matplotlib as mpl

    resolved = {key.replace("_", "."): value for key, value in overrides.items()}
    with mpl.rc_context({**RC_PARAMS, **resolved}):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def label_panels(
    axes: Iterable[Any],
    labels: Sequence[str] | None = None,
    x: float = -0.02,
    y: float = 1.06,
    weight: str = "bold",
) -> None:
    """Put ``(a)``, ``(b)``, ... on each axes, as the captions refer to them.

    A panel that already has a title gets the label prepended to it, because a
    separate text artist would land on top of the title -- both sit at the upper
    left once ``axes.titlelocation`` is ``"left"``. Panels without a title get
    the label as its own artist above the axes.

    Axes that were switched off with ``ax.axis("off")`` are skipped, so a grid
    with an unused trailing cell still labels its panels consecutively.
    """
    import matplotlib as mpl

    axes = [ax for ax in axes if ax is not None and ax.axison]
    if labels is None:
        labels = [f"({chr(ord('a') + index)})" for index in range(len(axes))]

    location = mpl.rcParams.get("axes.titlelocation", "center")

    for ax, label in zip(axes, labels):
        title = ax.get_title(loc=location)
        if title:
            ax.set_title(f"{label} {title}", loc=location)
        else:
            ax.text(
                x, y, label,
                transform=ax.transAxes,
                fontweight=weight,
                va="bottom",
                ha="left",
            )


def shade_spin_up(ax, times: Sequence[float], days: float):
    """Shade the first ``days`` of ``times``: the spin-up, which follows from the
    initial state rather than from the model, so the scores leave it out.

    The span starts at the first time point, from which the scores count the
    spin-up, and stops at the last, so a short series does not widen the axis. It
    sits behind the grid and every line, labelled :data:`SPIN_UP_LABEL` for the
    legend. Returns the span, or ``None`` when there is no spin-up to shade.
    """
    if not days > 0:
        return None
    start = float(times[0])
    return ax.axvspan(start, min(start + float(days), float(max(times))), color="#000000",
                      alpha=0.07, linewidth=0, zorder=0, label=SPIN_UP_LABEL)


def save_figure(
    figure,
    path: str | Path,
    formats: Sequence[str] = ("pdf",),
    close: bool = True,
) -> list[Path]:
    """Write ``figure`` once per requested format and return the paths written.

    The parent directory is created; a missing output directory was a recurring
    way for a long plotting run to fail at the very end.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written = []
    # Inside the style context: matplotlib resolves savefig.dpi, savefig.bbox and
    # the font types when the file is written, not when the figure was built, so
    # saving outside would silently fall back to the defaults.
    with figure_style():
        for suffix in formats:
            target = path.with_suffix(f".{suffix.lstrip('.')}")
            figure.savefig(target)
            written.append(target)

    if close:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return written
