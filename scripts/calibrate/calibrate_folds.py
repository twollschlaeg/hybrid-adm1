"""Calibrate the mechanistic benchmarks once per fold, without the fold's held-out reactors.

The installed benchmarks are calibrated per campaign on every reactor
(``calibrate_benchmarks.py``), so on a fold's held-out reactors they are in-sample,
while the hybrid has never seen those reactors. This script calibrates them again for
every fold and campaign as the installed calibration was run (both stages, R4 then R3
then R2, ``--reset-particulates``), but with the fold's held-out reactors left out, on a
copy of the data:

    <output>/split.json                  the split the folds come from
    <output>/model.json                  the model calibrated: {"arrhenius": true|false,
                                         "bounds": {name: [low, high]} given by --bound,
                                         "forcing": "smoothstep", the forcing it was fitted to}
    <output>/fold_<k>/data/<campaign>/   the copy, with the fold's parameter file
    <output>/fold_<k>/<campaign>.json    what was done, written once it is done
    <output>/fold_<k>/<campaign>.log     the calibration's output
    <output>/fold_<k>/held-out-starts/<campaign>.json
                                         the held-out reactors' start values, fitted on
                                         their first days (fit_held_out_starts.py)

``data/`` is never written. The copy keeps the held-out reactors' installed start values,
which the calibration on all reactors fitted on their whole record. Once a campaign is
calibrated, ``fit_held_out_starts.py`` fits them again with the fold's parameters on the
first days of each held-out reactor, which no score counts; MECH-CV and the hybrid
built on the fold start there, so the two differ only in what they learned from the
training reactors. A campaign of which a fold holds no reactor out is copied as
installed: calibrated on all its reactors, it already is the fold's.

``evaluate.py`` scores a run against these benchmarks as well when they exist for the
run's split (by default in ``<runs>/benchmarks``).

``--arrhenius`` calibrates the temperature-dependent model (calibrate_benchmarks.py
``--arrhenius``), into an output of its own: an output holds one model's calibrations, and
MECH-CV and the hybrids built on them read which from ``model.json``. ``--bound`` searches
parameters within other bounds (calibrate_benchmarks.py ``--bound``), likewise into an
output of its own. Both calibrate the campaigns a fold holds nothing out of as well, since
the installed calibration is that of the model without Arrhenius, within the stored bounds.

    python scripts/calibrate/calibrate_folds.py --workers 6
    python scripts/calibrate/calibrate_folds.py --folds 1 2 --campaigns LaRA-temp
    python scripts/calibrate/calibrate_folds.py --arrhenius --output runs/benchmarks-smoothstep-arrhenius
    python scripts/calibrate/calibrate_folds.py --arrhenius --bound k_hyd_ch 0.1 2.88 --output runs/benchmarks-smoothstep-fast

A rerun skips what is done and redoes what failed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from adm1_package.experiments.calibrations import campaign_forcing  # noqa: E402
from adm1_package.experiments.config import data_dir as default_data_dir  # noqa: E402
from adm1_package.experiments.config import run_dir  # noqa: E402
from adm1_package.experiments.registry import DATASETS  # noqa: E402
from adm1_package.experiments.splits import DataSplit, campaign_of, load_split  # noqa: E402

CALIBRATION = Path(__file__).resolve().parent / "calibrate_benchmarks.py"
HELD_OUT_STARTS = Path(__file__).resolve().parent / "fit_held_out_starts.py"

#: What the datasets read from a campaign directory (models/*/dataset.py), and the
#: forcing its calibration was fitted to (calibrate_benchmarks.py).
CAMPAIGN_FILES = ("{campaign}-parameters.json", "input_*.csv", "output_*.csv",
                  "R4-IVs_*.csv", "R3-IVs_*.csv", "R2-IVs_*.csv", "calibration.json")

#: The forcing the calibrations are fitted to (utils/forcing.forcing_coefficients).
FORCING = "smoothstep"

#: How the installed calibration ran (data/<campaign>/calibration-backup/*/report.json).
INSTALLED_FLAGS = ("--reset-particulates",)

Runner = Callable[[Sequence[str], Path, dict], int]


@dataclass(frozen=True)
class Job:
    """One campaign's calibration for one fold."""

    fold: int
    campaign: str
    #: The campaign's reactors the fold holds out (and the split's test reactors),
    #: left out of the fit.
    held_out: tuple[str, ...]

    @property
    def calibrates(self) -> bool:
        """False where the fold holds out none of the campaign's reactors."""
        return bool(self.held_out)


def plan(split: DataSplit, folds: Iterable[int], campaigns: Iterable[str]) -> list[Job]:
    campaigns = list(campaigns)
    return [Job(fold, campaign, tuple(sorted(
                label for label in [*split.validation_labels(fold), *split.test]
                if campaign_of(label) == campaign)))
            for fold in folds for campaign in campaigns]


def fold_data(output: Path, fold: int) -> Path:
    """The data directory a fold's benchmark is read from."""
    return Path(output) / f"fold_{fold}" / "data"


def copy_campaign(data_dir: Path, target_root: Path, campaign: str) -> Path:
    """Copy what the datasets read of one campaign; anything there is overwritten."""
    target = Path(target_root) / campaign
    target.mkdir(parents=True, exist_ok=True)
    for pattern in CAMPAIGN_FILES:
        for path in sorted((Path(data_dir) / campaign).glob(pattern.format(campaign=campaign))):
            shutil.copy2(path, target / path.name)
    return target


def check_bounds(given) -> dict[str, tuple[float, float]]:
    """``--bound`` as calibrate_benchmarks.py reads it, refused here as it would be there.
    That script imports JAX, so it is loaded only when there are bounds to check."""
    if not given:
        return {}
    spec = importlib.util.spec_from_file_location("calibrate_benchmarks", CALIBRATION)
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    return calibration.check_bounds(given, calibration.VARIANTS)


def prepare_output(output: Path, split: DataSplit, arrhenius: bool = False,
                   bounds: dict[str, tuple[float, float]] | None = None) -> None:
    """Record the split and the model, or refuse an output that holds the benchmarks of
    another split, of the model with(out) Arrhenius, fitted within other bounds, or to
    another forcing (those made before the record were fitted to the spline)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "split.json"
    existed = path.exists()
    if existed:
        stored = DataSplit.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if not stored.same_assignment(split):
            raise ValueError(f"{output} holds the benchmarks of another split; "
                             "choose another --output")
    else:
        path.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    model = output / "model.json"
    requested = {"arrhenius": bool(arrhenius),
                 "bounds": {name: [float(low), float(high)]
                            for name, (low, high) in sorted((bounds or {}).items())},
                 "forcing": FORCING}
    # Benchmarks made before the options were calibrated without Arrhenius, within the
    # stored bounds.
    recorded = (json.loads(model.read_text(encoding="utf-8")) if model.exists()
                else {} if existed else None)
    if recorded is not None:
        if bool(recorded.get("arrhenius", False)) != requested["arrhenius"]:
            calibrated = "with" if recorded.get("arrhenius") else "without"
            raise ValueError(f"{output} holds benchmarks calibrated {calibrated} Arrhenius; "
                             "choose another --output")
        if recorded.get("bounds", {}) != requested["bounds"]:
            raise ValueError(f"{output} holds benchmarks fitted within other bounds "
                             f"({recorded.get('bounds') or 'the stored ones'}); "
                             "choose another --output")
        if recorded.get("forcing", "spline") != requested["forcing"]:
            raise ValueError(f"{output} holds benchmarks fitted to the "
                             f"{recorded.get('forcing', 'spline')} forcing; choose another --output")
    model.write_text(json.dumps(requested, indent=2), encoding="utf-8")


def calibration_command(job: Job, output: Path, passthrough: Sequence[str]) -> list[str]:
    command = [sys.executable, str(CALIBRATION), "--campaign", job.campaign,
               "--data-dir", str(fold_data(output, job.fold)), *INSTALLED_FLAGS, *passthrough]
    return command + (["--exclude", *job.held_out] if job.held_out else [])


def held_out_starts_command(job: Job, output: Path) -> list[str]:
    """Fit the start values of the job's held-out reactors with the fold's calibration, on
    their first days (fit_held_out_starts.py); afresh, as the calibration is."""
    return [sys.executable, str(HELD_OUT_STARTS), "--benchmarks", str(output),
            "--folds", str(job.fold), "--campaigns", job.campaign, "--force"]


def _environment() -> dict:
    """One core per calibration, as run_experiment.sh trains one fold per core."""
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1",
                       XLA_FLAGS="--xla_force_host_platform_device_count=1")
    environment.setdefault("JAX_PLATFORMS", "cpu")
    return environment


def _run(command: Sequence[str], log: Path, environment: dict) -> int:
    with open(log, "w", encoding="utf-8") as handle:
        return subprocess.run(list(command), stdout=handle, stderr=subprocess.STDOUT,
                              env=environment, cwd=REPO).returncode


def run_job(job: Job, data_dir: Path, output: Path, passthrough: Sequence[str],
            runner: Runner = _run, every: bool = False) -> dict:
    """``every`` calibrates a campaign the fold holds nothing out of as well: needed when
    the installed calibration is of another model (``--arrhenius``) or was fitted within
    other bounds (``--bound``)."""
    record_path = Path(output) / f"fold_{job.fold}" / f"{job.campaign}.json"
    if record_path.exists():
        return {**json.loads(record_path.read_text(encoding="utf-8")), "skipped": True}
    # A fresh copy each time, so a calibration that failed halfway starts over.
    copy_campaign(data_dir, fold_data(output, job.fold), job.campaign)
    record = {"fold": job.fold, "campaign": job.campaign, "held_out": list(job.held_out)}
    if not job.calibrates and not every:
        record["status"] = "installed"
    else:
        command = calibration_command(job, output, passthrough)
        log = record_path.with_suffix(".log")
        began = time.perf_counter()
        code = runner(command, log, _environment())
        if code != 0:
            return {**record, "status": "failed", "returncode": code, "log": str(log)}
        record.update(status="calibrated", seconds=round(time.perf_counter() - began),
                      command=list(command[1:]))
        if job.held_out:
            starts = record_path.with_name(f"{job.campaign}.held-out-starts.log")
            code = runner(held_out_starts_command(job, output), starts, _environment())
            if code != 0:
                return {**record, "status": "failed", "returncode": code, "log": str(starts)}
            record["held_out_starts"] = "fitted"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main(argv=None, runner: Runner = _run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, default=None,
                        help="the split the runs use (default: configs/split.json)")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="folds to calibrate (default: all)")
    parser.add_argument("--campaigns", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="the installed data, read only (default: $ADM1_DATA_DIR or ./data)")
    parser.add_argument("--output", type=Path, default=run_dir() / "benchmarks",
                        help="where the per-fold benchmarks go (default: <runs>/benchmarks)")
    parser.add_argument("--workers", type=int, default=4,
                        help="calibrations at a time, one core each (default 4)")
    for name in ("--de-maxiter", "--de-popsize", "--local-maxiter", "--seed"):
        parser.add_argument(name, type=int, default=None,
                            help="passed on to calibrate_benchmarks.py (default: its own)")
    parser.add_argument("--arrhenius", action="store_true",
                        help="calibrate the temperature-dependent model (rates at 35 C), into "
                             "an output of its own")
    parser.add_argument("--bound", nargs=3, action="append", metavar=("NAME", "LOW", "HIGH"),
                        help="passed on to calibrate_benchmarks.py: search NAME within LOW and "
                             "HIGH; into an output of its own")
    args = parser.parse_args(argv)
    try:
        bounds = check_bounds(args.bound)
    except ValueError as error:
        parser.error(f"--bound: {error}")

    split = load_split(args.split)
    folds = args.folds or list(range(1, split.k_folds + 1))
    # The installed calibration is a fold's own only for the model it was made for, which
    # is the model without Arrhenius, within the stored bounds; and only if it was fitted to
    # the forcing these benchmarks are, for a campaign a fold holds nothing out of.
    every = args.arrhenius or bool(bounds)
    copied = sorted({job.campaign for job in plan(split, folds, args.campaigns)
                     if not job.calibrates and not every
                     and campaign_forcing(args.data_dir, job.campaign) != FORCING})
    if copied:
        print(f"the installed calibration of {', '.join(copied)} was fitted to another forcing "
              f"than the {FORCING}; calibrate it again first (calibrate_benchmarks.py)",
              file=sys.stderr)
        return 1
    prepare_output(args.output, split, arrhenius=args.arrhenius, bounds=bounds)
    passthrough = [token for name in ("de_maxiter", "de_popsize", "local_maxiter", "seed")
                   if getattr(args, name) is not None
                   for token in (f"--{name.replace('_', '-')}", str(getattr(args, name)))]
    if args.arrhenius:
        passthrough.append("--arrhenius")
    for name, (low, high) in bounds.items():
        passthrough += ["--bound", name, repr(low), repr(high)]
    jobs = plan(split, folds, args.campaigns)
    print(f"{len(jobs)} fold/campaign pairs into {args.output}, {args.workers} at a time"
          + (", with Arrhenius" if args.arrhenius else "")
          + "".join(f", {name} in [{low:g}, {high:g}]" for name, (low, high) in bounds.items())
          + ":", flush=True)
    for job in jobs:
        what = (f"without {', '.join(job.held_out)}" if job.calibrates
                else "on all its reactors (none held out)" if every
                else "installed (none held out)")
        print(f"  fold {job.fold} {job.campaign}: {what}", flush=True)

    def work(job: Job) -> dict:
        result = run_job(job, args.data_dir, args.output, passthrough, runner, every=every)
        state = "done before" if result.get("skipped") else result["status"]
        print(f"fold {job.fold} {job.campaign}: {state}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(work, jobs))
    failed = [result for result in results if result["status"] == "failed"]
    for result in failed:
        print(f"FAILED fold {result['fold']} {result['campaign']}: see {result['log']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
