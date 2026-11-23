"""What a calibration records about itself, and the checks that keep a hybrid on its own.

Two kinds of calibration live in data directories:

* the installed one in ``data/`` (``scripts/calibrate/calibrate_benchmarks.py``), which
  writes ``<campaign>/calibration.json`` beside a campaign's parameters: the forcing they
  were fitted to (``utils/forcing.FORCINGS``). A calibration made before the record was
  fitted to the spline, which was retired: :func:`campaign_forcing` says so, and a dataset
  refuses to be built on it;
* the per-fold ones (``scripts/calibrate/calibrate_folds.py``), one per fold on the fold's
  training reactors, under ``<benchmarks>/fold_<k>/data``. Every hybrid is built on its
  fold's (:func:`fold_benchmark_data`); ``<benchmarks>/model.json`` says which model they
  calibrated and to which forcing (:func:`benchmark_model`), and a fold records a
  fingerprint of the calibration it trained on (:func:`calibration_fingerprint`), so that
  it is never scored on another (:func:`check_forcing`, :func:`check_calibration`).

Nothing here imports JAX.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .registry import DATASETS
from .splits import DataSplit

__all__ = [
    "CALIBRATION_FILE",
    "campaign_forcing",
    "installed_forcing",
    "fold_benchmark_data",
    "benchmark_model",
    "calibration_fingerprint",
    "check_forcing",
    "check_calibration",
]

#: Where a calibration records the forcing it was fitted to: ``<campaign>/CALIBRATION_FILE``.
CALIBRATION_FILE = "calibration.json"


# ---------------------------------------------------------------------------
# The installed calibration
# ---------------------------------------------------------------------------
def campaign_forcing(data_dir: Path, campaign: str) -> str:
    """The forcing ``campaign``'s calibration in ``data_dir`` was fitted to."""
    path = Path(data_dir) / campaign / CALIBRATION_FILE
    recorded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return recorded.get("forcing", "spline")


def installed_forcing(data_dir: Path) -> str | None:
    """The forcing the calibration in ``data_dir`` was fitted to, the same for every
    campaign; ``None`` where ``data_dir`` holds no campaign."""
    forcings = {campaign: campaign_forcing(data_dir, campaign) for campaign in DATASETS
                if (Path(data_dir) / campaign).is_dir()}
    if not forcings:
        return None
    if len(set(forcings.values())) > 1:
        raise ValueError(f"the campaigns in {data_dir} were calibrated to different forcings: "
                         + ", ".join(f"{name} {forcing}" for name, forcing in forcings.items()))
    return next(iter(forcings.values()))


# ---------------------------------------------------------------------------
# The per-fold calibrations
# ---------------------------------------------------------------------------
def fold_benchmark_data(benchmarks: Path, split: DataSplit, fold: int) -> Path:
    """The data directory of fold ``fold``'s per-fold calibration.

    Refused unless it was calibrated for the same split and every campaign of the
    fold is finished: a campaign still being calibrated holds the installed values.
    """
    benchmarks = Path(benchmarks)
    path = benchmarks / "split.json"
    if not path.exists():
        raise FileNotFoundError(f"{benchmarks} holds no per-fold benchmarks (no split.json); "
                                "run scripts/calibrate/calibrate_folds.py, or copy its output")
    stored = DataSplit.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if not stored.same_assignment(split):
        raise ValueError(f"{benchmarks} was calibrated for another split than this run's")
    unfinished = []
    for campaign in DATASETS:
        record = benchmarks / f"fold_{fold}" / f"{campaign}.json"
        status = (json.loads(record.read_text(encoding="utf-8")).get("status")
                  if record.exists() else None)
        if status not in ("calibrated", "installed"):
            unfinished.append(campaign)
    if unfinished:
        raise FileNotFoundError(
            f"the per-fold benchmark of fold {fold} is not finished for {', '.join(unfinished)}; "
            "run scripts/calibrate/calibrate_folds.py")
    return benchmarks / f"fold_{fold}" / "data"


def benchmark_model(benchmarks: Path) -> dict[str, Any]:
    """The model per-fold benchmarks were calibrated with, the bounds they were searched
    within beyond the stored ones and the forcing they were fitted to:
    ``{"arrhenius": bool, "bounds": {name: [low, high]}, "forcing": str}``.

    ``calibrate_folds.py`` records them in ``model.json``; benchmarks made before the
    options were calibrated without Arrhenius, within the stored bounds, to the spline
    forcing, which was retired (``forcing.resolve_forcing`` refuses them). MECH-CV
    runs that model, and a hybrid built on the benchmarks needs the same physics,
    bounds and forcing.
    """
    from adm1_package.utils.forcing import resolve_forcing

    path = Path(benchmarks) / "model.json"
    recorded = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {"arrhenius": bool(recorded.get("arrhenius", False)),
            "bounds": {name: [float(low), float(high)]
                       for name, (low, high) in sorted(recorded.get("bounds", {}).items())},
            "forcing": resolve_forcing(recorded.get("forcing"))}


def calibration_fingerprint(dataset) -> str:
    """A hash of every reactor's parameters and start values in ``dataset``.

    A fold built on its per-fold calibration records it, so that it is never
    scored on another one, e.g. after a recalibration or from a stale copy on
    another machine. Twelve significant digits: two machines' CSV parsers may
    differ in the last bit.
    """
    def digits(values) -> list[str]:
        return [f"{value:.12g}" for value in np.asarray(values, dtype=float).ravel()]

    record = {}
    for index, label in enumerate(dataset.labels):
        x, _ = dataset[index]
        record[label] = {
            "parameters": {name: digits(value) for name, value in sorted(x["parameters"].items())},
            "initial_values": digits(x["initial_values"]),
        }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()


def check_forcing(settings: dict[str, Any], benchmarks: Path, where: str) -> None:
    """Refuse to anchor a fold on per-fold calibrations fitted to another forcing than the
    one it trained with (``settings["forcing"]``)."""
    fitted_to = benchmark_model(benchmarks)["forcing"]
    if fitted_to != settings["forcing"]:
        raise ValueError(f"{where} was trained with the {settings['forcing']} forcing, but the "
                         f"calibrations in {benchmarks} were fitted to the {fitted_to}; score it "
                         "with calibrations fitted to its own")


def check_calibration(dataset, settings: dict[str, Any], where: str) -> None:
    """Refuse to score a fold on a calibration other than the one it trained on, which its
    ``settings["calibration_fingerprint"]`` records."""
    recorded = settings.get("calibration_fingerprint")
    if not recorded:
        return
    if calibration_fingerprint(dataset) != recorded:
        raise ValueError(
            f"{where}: the per-fold calibration has changed since this fold was trained on it; "
            "score it with the benchmarks it was trained with")
