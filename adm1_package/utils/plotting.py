"""What every ADM1 output is called and measured in, for the figures of every variant.

The panel titles, the axis labels and the detection limits are described once here, in
:data:`COMPONENTS`, with the helpers that read a sample's measurements and forcing for
a figure. Parallel per-variant lists indexed by position would have to be edited
together to add a component, and one of them out of order would relabel a panel
silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


__all__ = [
    "Component",
    "COMPONENTS",
    "describe",
    "axis_label",
    "target_series",
    "forcing_series",
]


# ---------------------------------------------------------------------------
# Component metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Component:
    """How one model output is presented in a figure."""

    key: str
    title: str
    #: Compact mathtext symbol, for axis labels where the title is the panel title.
    symbol: str
    unit: str
    #: Measurements below this value are at or under the analytical detection
    #: limit; the region is shaded so a reader does not read structure into it.
    detection_limit: float | None = None


_COMPONENT_LIST = (
    Component("q_gasout", "Biogas production rate", r"$q_{gas}$", "NL/d"),
    Component("ch4_out", "Methane production rate", r"$q_{CH_4}$", "NL/d"),
    Component("S_ac", "Acetic acid", r"$S_{ac}$", "g/L", detection_limit=0.02),
    Component("S_va", "Valeric acid", r"$S_{va}$", "g/L", detection_limit=0.02),
    Component("S_bu", "Butyric acid", r"$S_{bu}$", "g/L", detection_limit=0.02),
    Component("S_pro", "Propionic acid", r"$S_{pro}$", "g/L", detection_limit=0.02),
    Component("S_TAN", "Total ammoniacal nitrogen", r"$S_{TAN}$", "g/L"),
    Component("S_h2o", "Water content", r"$S_{H_2O}$", "g/L"),
    Component("TS", "Total solids", r"$TS$", "g/L"),
    Component("VS", "Volatile solids", r"$VS$", "g/L"),
    Component("X_inert", "Non-degradable organics", r"$X_{inert}$", "g/L"),
    Component("X_ash", "Ash", r"$X_{ash}$", "g/L"),
    Component("pH", "pH", "pH", "-"),
    Component("T_op", "Temperature and inflow", r"$T_{op}$", "°C"),
)

#: Component key -> presentation metadata.
COMPONENTS: dict[str, Component] = {item.key: item for item in _COMPONENT_LIST}


def describe(key: str) -> Component:
    """Metadata for one component, falling back to the bare key if unknown.

    Unknown keys are tolerated rather than raising: a diagnostic plot of an
    intermediate state is a legitimate thing to want, and it should not need a
    table entry first.
    """
    if key in COMPONENTS:
        return COMPONENTS[key]
    return Component(key=key, title=key, symbol=key, unit="")


def axis_label(key: str, compact: bool = True) -> str:
    """``"$S_{ac}$ [g/L]"`` (compact) or ``"Concentration [g/L]"``."""
    component = describe(key)
    if not component.unit:
        return component.title
    if compact:
        return f"{component.symbol} [{component.unit}]"
    return f"{'Concentration' if component.unit == 'g/L' else 'Flow'} [{component.unit}]"


# ---------------------------------------------------------------------------
# Pulling series out of a dataset sample
# ---------------------------------------------------------------------------
def target_series(targets: Mapping[str, Any], key: str) -> tuple[np.ndarray, np.ndarray]:
    """``(values, valid)`` for one measured component.

    Columns are resolved through ``targets["object_indices"]`` rather than by
    position. The previous comparison plots indexed ``y["values"]`` with literal
    column numbers, which happened to be correct only as long as nobody
    reordered ``ModelSpec.outputs``.

    ``valid`` marks the records that hold a measurement (``targets["measured"]``,
    ``datasets.measured_points``): the lab days of the weekly analyses and the days
    the gas was read, as the scores count them, not the interpolation between them.
    It is False on the padded tail too: targets are edge-padded to a common length
    for batching, and drawing that tail invents a measurement that was never taken.
    """
    column = targets["object_indices"][key]
    values = np.asarray(targets["values"][..., column], dtype=float)
    if "measured" in targets:
        valid = np.asarray(targets["measured"][..., column], dtype=float) > 0
    elif "indices" in targets:
        valid = np.asarray(targets["indices"][..., column], dtype=float) > 0
    else:
        valid = np.ones(values.shape, dtype=bool)
    return values, valid


def forcing_series(inputs: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(time, temperature_celsius, inflow)`` read off the forcing signal."""
    import diffrax as dx
    import jax

    # Accepts either the whole sample (``x``) or just its forcing signal.
    forcing = inputs["input"] if "input" in inputs else inputs
    times = np.asarray(forcing["time"], dtype=float)
    interpolation = dx.CubicInterpolation(forcing["time"], forcing["coeffs"])
    evaluated = np.asarray(jax.vmap(interpolation.evaluate)(forcing["time"]), dtype=float)
    return times, evaluated[:, 0] - 273.15, evaluated[:, 1]
