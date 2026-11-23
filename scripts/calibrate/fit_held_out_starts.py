"""Fit the start values of each fold's held-out reactors on their first days, with the fold's
own calibration.

A per-fold calibration (``calibrate_folds.py``) fits the biology a reactor starts with, its
biomass and particulates, for the fold's training reactors only; a held-out reactor keeps the
installed values, which the calibration on all reactors fitted on that reactor's whole record,
scored part included, with parameters calibrated on it. This script fits them again for every
held-out reactor and variant (R4, R3, R2):

* on the first ``--days`` of the record (default ``SPIN_UP_DAYS``, 30), which no score counts,
  from the second point on (the datasets set the first from the measurements);
* with the fold's calibrated parameters and model (``model.json``: with or without Arrhenius),
  so the start is the one the anchor, and the hybrid with it, would reach from that start-up;
* from two places, keeping the better fit: the model's own warm-up at the reactor's
  first-week forcing (``utils/initial_state.py``), which reads only the first
  measurements, and the median start the fold's calibration gave the campaign's training
  reactors (a warm-up can wash biomass out: R2's acid degraders at fast hydrolysis on
  LaRA-temp, from which its acids pile up);
* with the particulates capped at the warm-up's first-week steady state, as in the
  calibration, and the biomass below twice the largest start the fold's calibration gave
  the campaign's training reactors (at least 10 g/L), where the calibration bounds a start
  at twice its stored value;
* on the loss the calibration fits, its residuals scaled by the median range of the campaign's
  training reactors, so that nothing of the held-out reactor's later record enters.

Nothing else of the held-out reactor is read. The values go to

    <benchmarks>/fold_<k>/held-out-starts/<campaign>.json

where evaluation and the figures take them
(``adm1_package/experiments/held_out_starts.py``). A fold and campaign already fitted is
skipped unless ``--force``.

    python scripts/calibrate/fit_held_out_starts.py --benchmarks runs/benchmarks-smoothstep-fast
    python scripts/calibrate/fit_held_out_starts.py --benchmarks runs/benchmarks-smoothstep --folds 3 --campaigns LaRA-temp
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _bootstrap import configure_jax  # noqa: E402

configure_jax()
if __name__ == "__main__":
    os.environ.setdefault("ADM1_SOLVER_MAX_STEPS", "20000")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import calibrate_benchmarks as calibration  # noqa: E402
from adm1_package.experiments.config import run_dir  # noqa: E402
from adm1_package.experiments.calibrations import benchmark_model  # noqa: E402
from adm1_package.experiments.held_out_starts import starts_file, state_names  # noqa: E402
from adm1_package.experiments.registry import DATASETS, get_spec  # noqa: E402
from adm1_package.experiments.splits import DataSplit, campaign_of  # noqa: E402
from adm1_package.utils.initial_state import warm_start  # noqa: E402
from adm1_package.utils.losses import robust_target_range  # noqa: E402

VARIANTS = calibration.VARIANTS
#: The first days of a record the fit reads: the spin-up the scores leave out.
DAYS = calibration.SPIN_UP_DAYS
#: L-BFGS-B iterations per reactor; four to eight start values converge within a few dozen.
MAXITER = 300


def window_mask(measured, time, days=DAYS):
    """The measured points the fit reads: after the first, before ``days`` from the start."""
    time = np.asarray(time, dtype=float)
    inside = (time > time[0]) & (time < time[0] + days)
    return np.asarray(measured, dtype=bool) & inside[:, None]


def first_days(x, y, days=DAYS):
    """``(x, y)`` cut to the first point at or past ``days`` from the start: all a fit on
    the first ``days`` needs to simulate."""
    time = np.asarray(x["input"]["time"], dtype=float)
    knots = int(np.searchsorted(time, time[0] + days)) + 1
    inputs = {**x["input"], "time": x["input"]["time"][:knots],
              "coeffs": tuple(c[: knots - 1] for c in x["input"]["coeffs"])}
    return ({**x, "input": inputs},
            {**y, "values": y["values"][:knots], "indices": y["indices"][:knots]})


def campaign_ranges(spec, samples):
    """The loss scale of each objective: the median over ``samples``, the campaign's training
    reactors, of the range the calibration scales each reactor's residuals by."""
    objectives = spec.objective_list
    ranges = []
    for x, y in samples:
        columns = [y["object_indices"][name] for name in objectives]
        measured = y["indices"][..., columns] > 0
        scored, _ = calibration.stage_masks(measured, x["input"]["time"])
        ranges.append(robust_target_range(y["values"][..., columns], axis=0, where=scored)[0])
    if not ranges:
        raise ValueError("no training reactor of the campaign to scale the loss by")
    return jnp.median(jnp.stack(ranges), axis=0)


def training_ceilings(key, samples):
    """The largest biomass start the calibration gave ``samples``, the campaign's training
    reactors, per biomass state of variant ``key``."""
    names = state_names(key)
    return {name: max(float(x["initial_values"][names.index(name)]) for x, _ in samples)
            for name in calibration.INITIAL_VALUES[key] if name in calibration.BIOMASS}


def training_medians(key, samples):
    """The median start the calibration gave ``samples``, the campaign's training reactors,
    of each of variant ``key``'s fitted states: a second place to start the fit from."""
    names = state_names(key)
    return {name: float(np.median([float(x["initial_values"][names.index(name)])
                                   for x, _ in samples]))
            for name in calibration.INITIAL_VALUES[key]}


def start_bounds(key, guess, caps, ceilings):
    """The bounds of variant ``key``'s fitted starts, in ``INITIAL_VALUES`` order.

    A substrate stays at most at its first-week steady state (``caps``), as in the
    calibration. The calibration lets a biomass start reach twice its stored value, at
    least 10 g/L; a held-out reactor has no stored value it may read, so the largest start
    the fold's calibration gave the campaign's training reactors (``ceilings``) stands in,
    next to the warm-up's ``guess``.
    """
    bounds = []
    for name in calibration.INITIAL_VALUES[key]:
        if name in caps:
            bounds.append(calibration.initial_value_bounds(name, guess[name], cap=caps[name]))
        else:
            reference = max(guess[name], ceilings.get(name, 0.0))
            bounds.append(calibration.initial_value_bounds(name, reference))
    return bounds


def fit_reactor(key, x, y, ranges, arrhenius, days=DAYS, maxiter=MAXITER, ceilings=None,
                alternatives=None):
    """Variant ``key``'s start values of one reactor, fitted on its first ``days``.

    ``x`` carries the parameters of the calibration the reactor is scored on; its stored
    start values of the fitted states are not read. ``ceilings``: see :func:`start_bounds`.

    The fit starts from the model's warm-up and from each of ``alternatives``
    (``{origin: {state: value}}``, such as :func:`training_medians`), and keeps the best
    fit of the first days. A warm-up can wash biomass out, R2's acid degraders at fast
    hydrolysis on LaRA-temp, and a fit from there stays in that basin, its acids piling
    up for the rest of the record.
    """
    spec = get_spec(key)
    names = state_names(key)
    unknown = calibration.INITIAL_VALUES[key]
    positions = jnp.array([names.index(name) for name in unknown])
    model = spec.mechanistic(initial_state="calibrated", arrhenius=arrhenius)
    objectives = spec.objective_list
    weights = jnp.array([spec.objective_weights[name] for name in objectives])
    x, y = first_days(x, y, days)
    columns = [y["object_indices"][name] for name in objectives]
    targets = y["values"][..., columns][None]
    mask = jnp.asarray(window_mask(y["indices"][..., columns] > 0, x["input"]["time"], days))[None]
    scale = jnp.asarray(ranges)[None, None, :]

    warmed = eqx.filter_jit(lambda params, y0, inputs: warm_start(
        calibration.LAYOUTS[key], model._rhs, lambda forcing: (params, forcing), params, y0,
        inputs))(x["parameters"], x["initial_values"], x["input"])
    guess = np.asarray(warmed)[np.asarray(positions)]
    # The warm-up is the first-week steady state, which caps the particulates.
    caps = {name: float(guess[j]) for j, name in enumerate(unknown)
            if name in calibration.SUBSTRATES[key]}

    def predict(theta):
        y0 = x["initial_values"].at[positions].set(theta)
        out = model(x["parameters"], y0, x["input"])[0]
        return jnp.stack([out[name] for name in objectives], axis=-1)[None]

    def loss(theta):
        return calibration.weighted_loss(predict(theta), targets, mask, scale, weights)

    value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    evaluate = eqx.filter_jit(loss)
    guessed = {name: float(guess[j]) for j, name in enumerate(unknown)}
    bounds = start_bounds(key, guessed, caps, ceilings or {})
    # the caps restrict the particulates, but their coordinates keep the uncapped range
    scales = [high - low for low, high in start_bounds(key, guessed, {}, ceilings or {})]
    low, high = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    best, losses = None, {}
    for origin, start in {"warm-up": guessed, **(alternatives or {})}.items():
        theta0 = np.clip(np.array([float(start[name]) for name in unknown]), low, high)
        losses[origin] = float(evaluate(jnp.asarray(theta0)))
        found = calibration.local_stage(
            lambda theta: value_and_grad(theta), theta0, bounds, maxiter,
            logarithmic=[name in calibration.BIOMASS for name in unknown], scales=scales)
        theta, value = (found.x, float(found.fun)) if found.fun < losses[origin] else (
            theta0, losses[origin])
        if best is None or value < best[0]:
            best = (value, origin, theta, found)
    value, origin, theta, found = best
    return {
        "start": {name: float(theta[j]) for j, name in enumerate(unknown)},
        "from": origin,
        "guess": guessed,
        "caps": caps,
        "bounds": {name: list(bound) for name, bound in zip(unknown, bounds)},
        "loss": {"guess": losses["warm-up"], "fitted": value, "starts": losses},
        "days": float(days),
        "iterations": int(found.nit),
        "evaluations": int(found.nfev),
        "message": str(found.message),
    }


def fit_fold(benchmarks, split, fold, campaign, variants=VARIANTS, arrhenius=None, days=DAYS,
             maxiter=MAXITER, verbose=False):
    """Fit the start values of ``campaign``'s reactors that fold ``fold`` holds out, for each
    of ``variants``, and write them. Returns the file, or ``None`` when the fold holds none
    of the campaign's reactors out."""
    benchmarks = Path(benchmarks)
    held = sorted(label for label in [*split.validation_labels(fold), *split.test]
                  if campaign_of(label) == campaign)
    if not held:
        return None
    model = benchmark_model(benchmarks)
    if arrhenius is None:
        arrhenius = model["arrhenius"]
    data = benchmarks / f"fold_{fold}" / "data"
    record = {"fold": fold, "campaign": campaign, "days": float(days), "arrhenius": bool(arrhenius),
              "forcing": model["forcing"], "maxiter": int(maxiter),
              "fitted": datetime.now().isoformat(timespec="seconds"), "variants": {}}
    for key in variants:
        spec = get_spec(key)
        # given the forcing the fold's parameters were fitted to
        dataset = spec.dataset(data, data_to_use=[campaign], forcing=model["forcing"])
        labels = list(dataset.labels)
        training = [dataset[i] for i, label in enumerate(labels) if label not in held]
        ranges = campaign_ranges(spec, training)
        ceilings = training_ceilings(key, training)
        alternatives = {"training median": training_medians(key, training)}
        record["variants"][key] = {}
        for label in held:
            began = time.perf_counter()
            x, y = dataset[labels.index(label)]
            result = fit_reactor(key, x, y, ranges, arrhenius, days=days, maxiter=maxiter,
                                 ceilings=ceilings, alternatives=alternatives)
            record["variants"][key][label] = result
            if verbose:
                starts = ", ".join(f"{value:.4f} ({origin})"
                                   for origin, value in result["loss"]["starts"].items())
                print(f"  fold {fold} {key} {label}: loss over the first {days:g} days {starts} "
                      f"-> {result['loss']['fitted']:.4f} from the {result['from']} after "
                      f"{result['iterations']} iterations ({time.perf_counter() - began:.0f} s)",
                      flush=True)
    path = starts_file(benchmarks, fold, campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmarks", type=Path, default=run_dir() / "benchmarks",
                        help="per-fold benchmarks of calibrate_folds.py (default: <runs>/benchmarks)")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="folds to fit (default: all)")
    parser.add_argument("--campaigns", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--days", type=float, default=DAYS,
                        help=f"first days of the record the fit reads (default {DAYS:g}, the "
                             "spin-up the scores leave out)")
    parser.add_argument("--maxiter", type=int, default=MAXITER,
                        help=f"L-BFGS-B iterations per reactor (default {MAXITER})")
    parser.add_argument("--force", action="store_true", help="fit again what is fitted")
    args = parser.parse_args(argv)

    split = DataSplit.from_dict(json.loads((args.benchmarks / "split.json").read_text(
        encoding="utf-8")))
    arrhenius = benchmark_model(args.benchmarks)["arrhenius"]
    folds = args.folds or list(range(1, split.k_folds + 1))
    variants = [key for key in VARIANTS if key in args.variants]
    print(f"{args.benchmarks}: held-out start values of folds {' '.join(map(str, folds))}, "
          f"{', '.join(variants)}, on the first {args.days:g} days"
          + (", with Arrhenius" if arrhenius else ""), flush=True)
    began = time.perf_counter()
    for fold in folds:
        for campaign in args.campaigns:
            path = starts_file(args.benchmarks, fold, campaign)
            if path.exists() and not args.force:
                print(f"fold {fold} {campaign}: fitted before, skipped", flush=True)
                continue
            written = fit_fold(args.benchmarks, split, fold, campaign, variants, arrhenius,
                               args.days, args.maxiter, verbose=True)
            print(f"fold {fold} {campaign}: "
                  + (f"written to {written}" if written else "no reactor held out"), flush=True)
    print(f"done in {time.perf_counter() - began:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
