"""Where a fold's held-out reactors start: values fitted on their unscored first days.

Every reactor starts from its measured chemistry and from fitted biology, its biomass and
particulates (:mod:`adm1_package.utils.initial_state`). A per-fold calibration
(``scripts/calibrate/calibrate_folds.py``) fits the biology of the fold's training reactors
on their whole record, and copies a held-out reactor's as installed: fitted by the
calibration on all reactors, on the whole record of the reactor that is then scored and
with parameters calibrated on it. On LaRA-temp those parameters sit in the corner of slow
hydrolysis, so the pool they leave a reactor is several times what a faster anchor holds,
and the hybrid built on the fast calibration burns it off in a start-up burst.

``scripts/calibrate/fit_held_out_starts.py`` fits the held-out reactors' biology again for
each per-fold calibration, with that calibration's own parameters and model, on the first
``days`` of the record only, which no score counts, and writes

    <benchmarks>/fold_<k>/held-out-starts/<campaign>.json

Evaluation, the figures and the inspection helpers start the fold's held-out reactors
there (``"fitted"``), the hybrid where its anchor starts; ``"installed"`` starts them as
the installed calibration does, for comparison. The training reactors keep the start values
their calibration fitted, and the fingerprint a run recorded of its calibration covers the
files as copied, so runs trained before the starts were fitted are scored on them as well.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from .splits import DataSplit, campaign_of

__all__ = [
    "HELD_OUT_STARTS",
    "DIRECTORY",
    "resolve_held_out_starts",
    "starts_file",
    "state_names",
    "read_starts",
    "with_starts",
    "held_out_view",
]

#: How a fold's held-out reactors start: from their first days (``fitted``), or from the
#: installed calibration, fitted on their whole record (``installed``).
HELD_OUT_STARTS = ("fitted", "installed")

#: Where a per-fold calibration keeps them: ``<benchmarks>/fold_<k>/<DIRECTORY>/``.
DIRECTORY = "held-out-starts"

_SCRIPT = "scripts/calibrate/fit_held_out_starts.py"


def resolve_held_out_starts(value: str) -> str:
    """Validate a way to start the held-out reactors (:data:`HELD_OUT_STARTS`)."""
    if value not in HELD_OUT_STARTS:
        raise ValueError(f"unknown held-out start {value!r}; expected one of "
                         f"{', '.join(HELD_OUT_STARTS)}")
    return value


def starts_file(benchmarks: Path, fold: int, campaign: str) -> Path:
    """The file holding fold ``fold``'s fitted starts of ``campaign``'s held-out reactors."""
    return Path(benchmarks) / f"fold_{fold}" / DIRECTORY / f"{campaign}.json"


def state_names(key: str) -> tuple[str, ...]:
    """Variant ``key``'s state vector, in the order its dataset keeps it."""
    from adm1_package.models.r2.model import STATES as R2
    from adm1_package.models.r3.model import STATES as R3
    from adm1_package.models.r4.model import STATES as R4

    return {"r4": R4, "r3": R3, "r2": R2}[key].names


def _missing(benchmarks: Path, what: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{what}: fit the held-out reactors' start values first, with "
        f"python {_SCRIPT} --benchmarks {benchmarks}, or start them as installed "
        "(held-out starts 'installed')")


def read_starts(benchmarks: Path, fold: int, key: str,
                labels: Sequence[str]) -> tuple[dict[str, dict[str, float]], float]:
    """Variant ``key``'s fitted starts of ``labels``, held out by fold ``fold``.

    Returns ``({label: {state: value}}, days)``, ``days`` being the longest first stretch
    of the records any of them was fitted on.
    """
    starts: dict[str, dict[str, float]] = {}
    days = 0.0
    records: dict[str, dict[str, Any]] = {}
    for label in labels:
        campaign = campaign_of(label)
        if campaign not in records:
            path = starts_file(benchmarks, fold, campaign)
            if not path.exists():
                raise _missing(benchmarks, f"{path} does not exist")
            records[campaign] = json.loads(path.read_text(encoding="utf-8"))
        record = records[campaign]
        fitted = record.get("variants", {}).get(key, {}).get(label)
        if fitted is None:
            raise _missing(benchmarks, f"fold {fold} has no fitted {key} start of {label}")
        starts[label] = {name: float(value) for name, value in fitted["start"].items()}
        days = max(days, float(record["days"]))
    return starts, days


def with_starts(dataset, starts: Mapping[str, Mapping[str, float]], names: Sequence[str]):
    """A copy of ``dataset`` whose reactors in ``starts`` start from those values.

    ``starts`` maps a reactor's label to the states to set and their values; ``names`` is
    the dataset's state vector (:func:`state_names`). Only those entries change: the
    datasets set the chemistry from the first measurements, which the biology does not
    enter. The copy shares everything else with ``dataset``, which starts as before.
    """
    labels = list(dataset.labels)
    unknown = sorted(set(starts) - set(labels))
    if unknown:
        raise KeyError(f"the dataset holds no reactor {', '.join(unknown)}")
    view = copy.copy(dataset)
    view.data = list(dataset.data)
    for label, values in starts.items():
        index = labels.index(label)
        sample = dict(view.data[index])
        # JAX drops an update outside the vector without a word
        if jnp.shape(sample["initial_values"])[-1] != len(names):
            raise ValueError(f"{label} has {jnp.shape(sample['initial_values'])[-1]} states, "
                             f"not the {len(names)} of {', '.join(names)}")
        positions = jnp.array([list(names).index(name) for name in values])
        sample["initial_values"] = jnp.asarray(sample["initial_values"]).at[positions].set(
            jnp.array([float(value) for value in values.values()]))
        view.data[index] = sample
    return view


def held_out_view(dataset, key: str, benchmarks: Path, split: DataSplit, fold: int,
                  held_out_starts: str, spin_up_days: float):
    """``dataset``, a fold's per-fold calibration, with the fold's held-out reactors (and
    the split's test reactors) started as ``held_out_starts`` says.

    Returns ``(dataset, days)``: for ``"fitted"`` a copy and the stretch of the records the
    starts were fitted on, which must end where the scores begin, at ``spin_up_days``; for
    ``"installed"`` the dataset itself and ``None``.
    """
    if resolve_held_out_starts(held_out_starts) == "installed":
        return dataset, None
    labels = [label for label in [*split.validation_labels(fold), *split.test]
              if label in dataset.labels]
    starts, days = read_starts(benchmarks, fold, key, labels)
    if days > spin_up_days:
        raise ValueError(f"the held-out starts of fold {fold} were fitted on the first {days:g} "
                         f"days, which the scores count from day {spin_up_days:g}")
    return with_starts(dataset, starts, state_names(key)), days
