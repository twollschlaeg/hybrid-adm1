"""Recalibrate the mechanistic benchmarks of one campaign.

The method is the paper's two-stage calibration (Section 2.3 and Appendix C), over all
reactors of a campaign at once. The variants are calibrated in the order R4, R3, R2:
R4 fits the degradability factors and hydrolysis rates, and R3 and R2 keep those and fit
only their own acid kinetics (the paper's Table C.1). Each variant in two stages:

1. **Global.** Differential evolution over the variant's parameters, so the result
   does not depend on where it starts. The initial values stay as they are.
2. **Local.** L-BFGS-B with exact (JAX) gradients from the global optimum, over the
   parameters and every reactor's unmeasured initial values together: its biomass and
   particulates. The better of the two stages on this stage's loss is kept.

Both minimise the loss the hybrids are trained with (``utils/ude_trainer.py``) over all
reactors of the campaign at once, so the campaign gets one parameter set: each
objective's residuals scaled by the reactor's interpercentile range, weighted as in the
registry (water content is not fitted). The global stage leaves out the first
``SPIN_UP_DAYS``, as the hybrids do. The local stage scores the first month too: a start
value that outlasts the spin-up is one more parameter, and without it the fit is free to
make the first month unphysical. The datasets set every state the first measurements
determine (``utils/initial_state.py``).

The particulate starts stay at or below the model's steady state for each reactor's
first-week forcing (the warm-up): a reactor holds no more undegraded substrate than its
own feed builds up. The local stage starts from the stored start values, so what one
calibration leaves in them carries into the next; ``--reset-particulates`` restarts the
particulates at that steady state instead. The local stage works in coordinates that
span [0, 1] for every variable, logarithmically for the biomass starts, so that start
values move alongside steeply curved parameters.

The campaign's parameter file, shared by the variants, and the ``R{2,3,4}-IVs`` files
are updated in place, and ``calibration.json`` records the forcing they were fitted to
(the smoothstep of ``utils/forcing.forcing_coefficients``; without it, the spline of
calibrations made before). A run first copies them to
``data/<campaign>/calibration-backup/<time>/`` and writes ``report.json`` there.

``--exclude`` leaves reactors out of the fit and their files as they are:
``calibrate_folds.py`` uses it to calibrate each cross-validation fold on its training
reactors only, on a copy of the data.

``--arrhenius`` calibrates the temperature-dependent model (``spec.mechanistic(arrhenius=
True)``): hydrolysis and acid uptake scale with the Weinrich activation energies, so the
fitted rates are rates at 35 C. Without it the reduced models do not respond to
temperature, and on LaRA-temp the fit ends in the corner of slow hydrolysis (``k_hyd_ch``
at its lower bound 0.02/d) and complete degradability, which steady data cannot tell from
fast hydrolysis of less; after the +2 to +9 K steps only the latter returns to its level.

``--bound NAME LOW HIGH`` searches a parameter within other bounds than the stored ones,
starting a stored value outside them at the nearer bound, and stores the bounds with the
fit, so the parameter file says what its values were fitted within. ``--arrhenius --bound
k_hyd_ch 0.1 2.88`` is the fast-hydrolysis calibration: k_hyd_ch at least 0.1/d at 35 C,
above the dilution rates (0.03-0.1/d), the regime in which the gas returns to its level
after a temperature step.

The full ADM1 has rows in the tables below too, but no place in ``--variants``:
``base_calibration.py`` runs these two stages for it, after R2.

    python scripts/calibrate/calibrate_benchmarks.py --campaign LaRA-temp
    python scripts/calibrate/calibrate_benchmarks.py --campaign rice-bran --variants r4 --dry-run
    python scripts/calibrate/calibrate_benchmarks.py --campaign LaRA-substrate --reset-particulates
    python scripts/calibrate/calibrate_benchmarks.py --campaign LaRA-temp --arrhenius --bound k_hyd_ch 0.1 2.88
"""

import os
import sys
from pathlib import Path

# Must run before JAX is imported: _bootstrap selects the platform and puts the
# repository on sys.path so the script works without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import configure_jax  # noqa: E402

configure_jax()
if __name__ == "__main__":
    # The global stage tries parameter sets the solver can take a million steps over; a
    # normal solve takes under 2,500. Read by utils/solver.py on import.
    os.environ.setdefault("ADM1_SOLVER_MAX_STEPS", "20000")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import differential_evolution, minimize  # noqa: E402

from adm1_package.experiments.calibrations import CALIBRATION_FILE  # noqa: E402
from adm1_package.experiments.registry import DATASETS, get_spec  # noqa: E402
from adm1_package.models.base.model import STATES as BASE_STATES  # noqa: E402
from adm1_package.models.r2.model import STATES as R2_STATES  # noqa: E402
from adm1_package.models.r3.model import STATES as R3_STATES  # noqa: E402
from adm1_package.models.r4.model import STATES as R4_STATES  # noqa: E402
from adm1_package.utils.datasets import custom_collate_fn  # noqa: E402
from adm1_package.utils.initial_state import warm_start  # noqa: E402
from adm1_package.utils.losses import robust_target_range  # noqa: E402
from adm1_package.utils.parameters import load_params, save_params  # noqa: E402
from adm1_package.utils.ude_trainer import scored_points  # noqa: E402

VARIANTS = ("r4", "r3", "r2")
SPIN_UP_DAYS = 30.0    # as in the hybrids' loss and in the scores
PENALTY = 1e6          # the score of a solve that fails, as in the hybrids' loss

#: Optimiser settings of the two stages.
DE_MAXITER, DE_POPSIZE, LOCAL_MAXITER, SEED = 300, 10, 3000, 42

#: Each variant's state vector, in its dataset's order.
COMPONENTS = {
    "r4": ("S_ch4", "S_IC", "S_TAN", "S_h2o", "X_ch", "X_pr", "X_li", "X_bac",
           "S_gas_ch4", "S_gas_co2"),
    "r3": ("S_ac", "S_ch4", "S_IC", "S_TAN", "S_h2o", "X_ch", "X_pr", "X_li", "X_bac", "X_ac",
           "S_an", "S_cat", "S_ac_i", "S_hco3_i", "S_nh3", "S_gas_ch4", "S_gas_co2"),
    "r2": ("S_va", "S_bu", "S_pro", "S_ac", "S_ch4", "S_IC", "S_TAN", "S_h2o",
           "X_ch", "X_pr", "X_li", "X_bac", "X_va", "X_bu", "X_pro", "X_ac",
           "S_an", "S_cat", "S_va_i", "S_bu_i", "S_pro_i", "S_ac_i", "S_hco3_i", "S_nh3",
           "S_gas_ch4", "S_gas_co2"),
    "base": ("S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC",
             "S_TAN", "S_h2o", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_va", "X_bu",
             "X_pro", "X_ac", "X_h2", "S_an", "S_cat", "S_va_i", "S_bu_i", "S_pro_i", "S_ac_i",
             "S_hco3_i", "S_nh3", "S_gas_h2", "S_gas_ch4", "S_gas_co2"),
}

#: What each variant calibrates (the paper's Table C.1). The full ADM1 keeps every
#: parameter it shares with them and fits the Monod kinetics of the processes they lump
#: away: the uptake of sugars, amino acids, long-chain fatty acids and hydrogen.
PARAMETERS = {
    "r4": ("f_ch", "f_pr", "f_li", "k_hyd_ch", "k_hyd_pr", "k_hyd_li"),
    "r3": ("k_m_ac", "K_ac"),
    "r2": ("k_m_va", "k_m_bu", "k_m_pro", "K_va", "K_bu", "K_pro"),
    "base": ("k_m_su", "k_m_aa", "k_m_fa", "k_m_h2", "K_su", "K_aa", "K_fa", "K_h2"),
}
#: The unmeasured states fitted per reactor: the biology, and in the full ADM1 the products
#: of hydrolysis. The datasets set S_ch4, S_IC and the acid-base states from the first
#: measurements: the dissolved gases in equilibrium with the measured gas phase, and in R2,
#: R3 and the full ADM1 the weak acids and ions at the measured pH
#: (utils/initial_state.measured_chemistry).
INITIAL_VALUES = {
    "r4": ("X_ch", "X_pr", "X_li", "X_bac"),
    "r3": ("X_ch", "X_pr", "X_li", "X_bac", "X_ac"),
    "r2": ("X_ch", "X_pr", "X_li", "X_bac", "X_va", "X_bu", "X_pro", "X_ac"),
    "base": ("X_ch", "X_pr", "X_li", "S_su", "S_aa", "S_fa", "X_su", "X_aa", "X_fa", "X_va",
             "X_bu", "X_pro", "X_ac", "X_h2"),
}
PARTICULATES = ("X_ch", "X_pr", "X_li")
#: The substrates whose starts stay at or below their first-week steady state, and which
#: ``--reset-particulates`` restarts there: the particulates, and in the full ADM1 also
#: what their hydrolysis makes. A reactor holds no more of them than its feed builds up.
SUBSTRATES = {"r4": PARTICULATES, "r3": PARTICULATES, "r2": PARTICULATES,
              "base": PARTICULATES + ("S_su", "S_aa", "S_fa")}
#: Biomass, whose starts the local stage moves on a log scale from ``BIOMASS_FLOOR`` (g/L,
#: the floor R2's right-hand side puts under every state).
BIOMASS = ("X_bac", "X_ac", "X_va", "X_bu", "X_pro", "X_su", "X_aa", "X_fa", "X_h2")
BIOMASS_FLOOR = 1e-12
LAYOUTS = {"r4": R4_STATES, "r3": R3_STATES, "r2": R2_STATES, "base": BASE_STATES}
#: Where each variant's start values live: ``<campaign>/<name>_<reactor>.csv``.
IV_FILES = {"r4": "R4-IVs", "r3": "R3-IVs", "r2": "R2-IVs", "base": "base-IVs"}
#: Parameters both stages search on a log scale. K_h2's bounds span five decades and its
#: calibrated values the lowest two, where a linear search hardly samples.
LOG_PARAMETERS = ("K_h2",)


def weighted_loss(pred, targets, mask, ranges, weights):
    """The hybrids' loss: per objective, the mean squared residual over ``mask`` scaled
    by ``ranges``; then the weighted mean over objectives, and its root.

    Objectives are the last axis; the mean pools all the others (reactors and time).
    An objective without a point in ``mask`` drops out of the weighting.
    """
    squared = jnp.where(mask, ((pred - targets) / ranges) ** 2, 0.0)
    axes = tuple(range(pred.ndim - 1))
    count = jnp.sum(mask, axis=axes)
    mse = jnp.sum(squared, axis=axes) / jnp.maximum(count, 1)
    weights = jnp.where(count > 0, weights, 0.0)
    return jnp.sqrt(jnp.sum(mse * weights) / jnp.sum(weights))


def stage_masks(measured, time, spin_up_days=SPIN_UP_DAYS):
    """The points each stage scores, as ``(global, local)``.

    The global stage fits parameters on the hybrids' loss, which leaves out the first
    ``spin_up_days``. The local stage also fits start values, so it scores every measured
    point but the initial one: behind the spin-up, a start value whose effect outlasts it
    is one more parameter, and the first month whatever helps the score after it.
    """
    measured = measured.astype(float)
    return (scored_points(measured, time, spin_up_days) > 0,
            scored_points(measured, time, 0.0) > 0)


def first_week_particulates(key, x, arrhenius=False):
    """``x``'s start values with every reactor's particulates (``SUBSTRATES``) at the model's
    steady state for its first-week forcing: the warm-up of ``utils/initial_state.py``, run
    on the parameters in ``x`` -- with ``arrhenius``, at the first week's temperature. The
    warm-up reads only the measured states and the first measured pH, so no stored start
    survives in the particulates.

    Restarting the local stage there drops what earlier calibrations left in them, such
    as a large stock of carbohydrate that bursts into gas in the first days.
    """
    layout, model = LAYOUTS[key], get_spec(key).mechanistic(arrhenius=arrhenius)

    def warm(params, y0, inputs):
        return warm_start(layout, model._rhs, lambda forcing: (params, forcing), params, y0, inputs)

    warmed = eqx.filter_jit(jax.vmap(warm))(x["parameters"], x["initial_values"], x["input"])
    columns = jnp.array([layout.index(name) for name in SUBSTRATES[key]])
    return x["initial_values"].at[:, columns].set(warmed[:, columns])


def split_local(theta, n_parameters, n_reactors):
    """The local stage's vector: the parameters, then each reactor's initial values."""
    return theta[:n_parameters], theta[n_parameters:].reshape(n_reactors, -1)


def set_initial_values(y0, positions, values):
    """``y0`` (reactors x states) with ``values`` (reactors x len(positions)) in place."""
    return y0.at[:, positions].set(values)


def parameter_bounds(stored, name):
    """The bounds stored with the calibration, else the defaults below."""
    low, high = stored[name]["min"], stored[name]["max"]
    current = float(stored[name]["value"])
    if name in ("f_ch", "f_pr", "f_li"):
        default = (0.0, 1.0)
    elif name in ("k_hyd_ch", "k_hyd_pr", "k_hyd_li"):
        default = (0.0, 10.0)
    else:
        default = (0.0, max(10.0, 2.0 * current))
    return (default[0] if low is None else float(low), default[1] if high is None else float(high))


def check_bounds(given, variants):
    """``--bound`` as ``{name: (low, high)}``. ``given`` holds ``[name, low, high]`` triples,
    or is ``None``; each names a parameter one of ``variants`` calibrates, once, with finite
    numbers and low < high."""
    calibrated = {name for key in variants for name in PARAMETERS[key]}
    bounds = {}
    for name, low, high in given or ():
        if name not in calibrated:
            raise ValueError(f"no variant in {' '.join(variants)} calibrates {name}; they "
                             f"calibrate {', '.join(sorted(calibrated))}")
        if name in bounds:
            raise ValueError(f"{name} is bounded twice")
        try:
            low_value, high_value = float(low), float(high)
        except ValueError:
            raise ValueError(f"the bounds of {name}, {low} and {high}, are not numbers") from None
        if not (np.isfinite(low_value) and np.isfinite(high_value) and low_value < high_value):
            raise ValueError(f"the bounds of {name}, {low} and {high}, leave nothing to search")
        bounds[name] = (low_value, high_value)
    return bounds


def initial_value_bounds(name, value, cap=None):
    """The bounds of a start value, from 0 to well above ``value``, except that a substrate
    start with a ``cap`` (its first-week steady state) stays below it: a reactor holds no
    more undegraded substrate than its own feed builds up. Unbounded, the local stage can
    give a reactor many times its steady state of carbohydrate, a burst of gas whose TAN
    uptake a long retention carries into the scores."""
    if cap is not None:
        return 0.0, float(cap)
    if name in PARTICULATES:
        return 0.0, max(300.0, 2.0 * value)
    return 0.0, max(10.0, 2.0 * value)


def capped_starts(stored, steady, columns, reset):
    """``stored`` start values with the particulate ``columns`` at their first-week steady
    state (``reset``), or lowered to it where they lie above."""
    starts = np.array(stored, dtype=np.float64)
    starts[:, columns] = steady[:, columns] if reset else np.minimum(stored[:, columns],
                                                                       steady[:, columns])
    return starts


def global_stage(loss, bounds, maxiter, popsize, seed, label="", logarithmic=None):
    """Differential evolution of ``loss(theta)`` over ``bounds``, without polishing. The
    ``logarithmic`` variables are searched in the logarithm of their bounds."""
    began = time.perf_counter()
    log = np.zeros(len(bounds), dtype=bool) if logarithmic is None else np.asarray(logarithmic, bool)
    searched = [(np.log(low), np.log(high)) if is_log else (low, high)
                for (low, high), is_log in zip(bounds, log)]

    def to_x(z):
        x = np.array(z, dtype=np.float64)
        x[log] = np.exp(x[log])
        return x

    def objective(z):
        theta = to_x(z)
        try:
            value = float(loss(jnp.asarray(theta)))
        except Exception:  # the solver gave up on this parameter set
            value = np.nan
        return value if np.isfinite(value) else PENALTY

    def progress(intermediate_result):
        generation = progress.generation = getattr(progress, "generation", 0) + 1
        if label and generation % 10 == 0:
            print(f"    {label} global: generation {generation}, best "
                  f"{intermediate_result.fun:.5f} ({time.perf_counter() - began:.0f} s)", flush=True)

    found = differential_evolution(objective, bounds=searched, maxiter=maxiter, popsize=popsize,
                                   polish=False, updating="deferred", workers=1,
                                   rng=np.random.default_rng(seed), callback=progress)
    found.x = np.clip(to_x(found.x), [b[0] for b in bounds], [b[1] for b in bounds])
    return found


def logarithmic_starts(key, n_parameters, n_reactors):
    """Which entries of the local stage's vector are on a log scale: the parameters in
    ``LOG_PARAMETERS``, which come first, then each reactor's biomass starts among its
    ``INITIAL_VALUES``."""
    if n_parameters != len(PARAMETERS[key]):
        raise ValueError(f"{key} has {len(PARAMETERS[key])} parameters, not {n_parameters}")
    return [name in LOG_PARAMETERS for name in PARAMETERS[key]] + [
        name in BIOMASS for _ in range(n_reactors) for name in INITIAL_VALUES[key]]


def local_stage(value_and_grad, x0, bounds, maxiter, label="", logarithmic=None, scales=None):
    """L-BFGS-B from ``x0`` with exact gradients, in coordinates where every variable spans
    [0, 1] of its scale: linearly over ``scales`` (default: the width of its bounds), or for
    the ``logarithmic`` ones (biomass starts) logarithmically from ``BIOMASS_FLOOR``.

    In the variables' own units the parameters' steep curvature swamps the start values,
    whose gradients are a thousandth of theirs, and the solver's noise (1e-4 of the loss)
    ends the line search before the starts move. A bound tighter than a variable's range,
    such as a cap, belongs in ``bounds`` but not in its scale: scaled to the cap, the
    starts' gradients shrink again and the stage stops early. A biomass that starts at s
    blooms after ln(K / s) / mu days, so its gradient goes as 1/s and diverges at a start
    of 0; on the log scale it is finite.

    A solve that fails scores PENALTY with a zero gradient, so the line search backs off
    instead of stopping the fit. Returns the best point evaluated, in the variables' own
    units: when the line search ends abnormally, scipy reports the value of the last trial,
    which can be a failed solve.
    """
    began = time.perf_counter()
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    log = np.zeros(len(bounds), dtype=bool) if logarithmic is None else np.asarray(logarithmic, bool)
    tiny = np.finfo(float).tiny                           # bounds that coincide (a cap of 0)
    log_lo = np.log(np.maximum(lo, BIOMASS_FLOOR))
    log_span = np.maximum(np.log(np.maximum(hi, BIOMASS_FLOOR)) - log_lo, tiny)
    width = np.maximum(hi - lo if scales is None else np.asarray(scales, np.float64), tiny)
    z_max = np.where(log, 1.0, np.maximum(hi - lo, 0.0) / width)

    def to_x(z):
        return np.where(log, np.exp(log_lo + z * log_span), lo + z * width)

    def to_z(theta):
        return np.where(log, (np.log(np.maximum(theta, BIOMASS_FLOOR)) - log_lo) / log_span,
                        (theta - lo) / width)

    best = {"fun": np.inf, "x": None}

    def fun(z):
        theta = to_x(z)
        try:
            value, grad = value_and_grad(jnp.asarray(theta))
            value, grad = float(value), np.asarray(grad, dtype=np.float64)
        except Exception:
            return PENALTY, np.zeros_like(z)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            return PENALTY, np.zeros_like(z)
        if value < best["fun"]:
            best.update(fun=value, x=theta)
        return value, grad * np.where(log, theta * log_span, width)

    def progress(intermediate_result):
        iteration = progress.iteration = getattr(progress, "iteration", 0) + 1
        if label and iteration % 25 == 0:
            print(f"    {label} local: iteration {iteration}, loss "
                  f"{intermediate_result.fun:.5f} ({time.perf_counter() - began:.0f} s)", flush=True)

    z0 = np.clip(to_z(np.clip(np.asarray(x0, dtype=np.float64), lo, hi)), 0.0, z_max)
    result = minimize(fun, z0, jac=True, bounds=list(zip(np.zeros_like(z_max), z_max)),
                      method="L-BFGS-B", options={"maxiter": maxiter}, callback=progress)
    result.x = to_x(result.x)
    if best["x"] is not None and best["fun"] < result.fun:
        result.x, result.fun = best["x"], best["fun"]
    result.x = np.clip(result.x, lo, hi)                   # to_x rounds across a bound
    return result


def best_of(global_theta, global_loss, local_theta, local_loss):
    """The local result if it improved on the global one, else the global one."""
    return np.asarray(local_theta if local_loss < global_loss else global_theta)


def calibrate_variant(key, campaign, data_dir, values, args):
    """Both stages for one variant. ``values`` carries the parameters fitted so far."""
    spec = get_spec(key)
    dataset = spec.dataset(data_dir, data_to_use=[campaign])
    samples = [dataset[i] for i in range(len(dataset))]
    # A per-fold benchmark (calibrate_folds.py) leaves out the reactors its fold holds
    # out; their start values are neither fitted nor written.
    exclude = list(getattr(args, "exclude", None) or ())
    unknown = sorted(set(exclude) - {y["label"] for _, y in samples})
    if unknown:
        raise ValueError(f"{campaign} has no reactor {unknown}")
    samples = [(x, y) for x, y in samples if y["label"] not in exclude]
    if not samples:
        raise ValueError(f"excluding {exclude} leaves no reactor of {campaign} to calibrate")
    labels = [y["label"] for _, y in samples]
    components = COMPONENTS[key]
    if samples[0][0]["initial_values"].shape[-1] != len(components):
        raise ValueError(f"{key}: the dataset's state vector no longer matches COMPONENTS")
    for x, _ in samples:
        for name in x["parameters"]:
            if name in values:
                x["parameters"][name] = jnp.asarray(values[name])
    stored = load_params(str(data_dir / campaign / f"{campaign}-parameters.json"))

    arrhenius = bool(getattr(args, "arrhenius", False))
    model = spec.mechanistic(initial_state="calibrated", arrhenius=arrhenius)
    objectives = spec.objective_list
    weights = jnp.array([spec.objective_weights[name] for name in objectives])
    x, y = custom_collate_fn(samples)
    columns = [y["object_indices"][name] for name in objectives]
    targets = y["values"][..., columns]
    measured = y["indices"][..., columns] > 0
    scored, whole = stage_masks(measured, x["input"]["time"])
    ranges = robust_target_range(targets, axis=1, where=scored)
    solve = jax.vmap(lambda p, y0, inputs: model(p, y0, inputs)[0])
    names = PARAMETERS[key]
    unknown = INITIAL_VALUES[key]
    positions = jnp.array([components.index(name) for name in unknown])

    def predict(parameters, y0, x):
        params = dict(x["parameters"])
        for i, name in enumerate(names):
            params[name] = jnp.broadcast_to(parameters[i], params[name].shape)
        out = solve(params, y0, x["input"])
        return jnp.stack([out[name] for name in objectives], axis=-1)

    campaign_loss = eqx.filter_jit(lambda theta, x: weighted_loss(
        predict(theta, x["initial_values"], x), targets, scored, ranges, weights))

    def joint_predict(theta, x):
        parameters, initial = split_local(theta, len(names), len(samples))
        return predict(parameters, set_initial_values(x["initial_values"], positions, initial), x)

    joint_value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(lambda theta, x: weighted_loss(
        joint_predict(theta, x), targets, whole, ranges, weights)))
    joint_prediction = eqx.filter_jit(joint_predict)

    def both_losses(theta):
        """The hybrids' loss and the local stage's of ``theta``."""
        pred = joint_prediction(jnp.asarray(theta), x)
        return tuple(float(weighted_loss(pred, targets, mask, ranges, weights))
                     for mask in (scored, whole))

    given = {name: bound for name, bound in check_bounds(getattr(args, "bound", None), VARIANTS).items()
             if name in names}
    if given:
        # A stored value outside a bound given starts at the bound, and so do the steady
        # states that cap the particulate starts below.
        x = {**x, "parameters": {**x["parameters"], **{
            name: jnp.clip(x["parameters"][name], low, high) for name, (low, high) in given.items()}}}
    start = np.array([float(x["parameters"][name][0]) for name in names])
    bounds = [given[name] if name in given else parameter_bounds(stored, name) for name in names]
    start_loss = float(campaign_loss(jnp.asarray(start), x))
    print(f"  {key}: {', '.join(names)} and {len(unknown)} initial values on "
          f"{len(samples)} reactors; loss at the current calibration {start_loss:.5f}", flush=True)
    # The particulates' first-week steady state caps their starts in both stages, on the
    # parameters in effect now: for R3 and R2 those of R4's fit, for R4 the stored ones.
    capped = SUBSTRATES[key]
    columns = [components.index(name) for name in capped]
    stored_start = np.asarray(x["initial_values"])
    steady = np.asarray(first_week_particulates(key, x, arrhenius=arrhenius))
    caps = steady[:, columns]
    starts = capped_starts(stored_start, steady, columns, reset=args.reset_particulates)
    if not np.array_equal(starts, stored_start):
        x = {**x, "initial_values": jnp.asarray(starts)}
        print(f"  {key}: particulates {'restarted from' if args.reset_particulates else 'capped at'}"
              f" the first-week steady state; loss "
              f"{float(campaign_loss(jnp.asarray(start), x)):.5f}", flush=True)
    reset = (stored_start, starts) if args.reset_particulates else None

    found = global_stage(lambda theta: campaign_loss(theta, x), bounds, args.de_maxiter,
                         args.de_popsize, args.seed, label=key,
                         logarithmic=[name in LOG_PARAMETERS for name in names])
    print(f"  {key} global: loss {found.fun:.5f} after {found.nit} generations, "
          f"{found.nfev} evaluations ({found.message})", flush=True)

    initial0 = np.asarray(x["initial_values"][:, positions])
    x0 = np.concatenate([found.x, initial0.ravel()])
    whole_at_global = both_losses(x0)[1]
    cap_of = {name: caps[:, i] for i, name in enumerate(capped)}
    joint_bounds = bounds + [
        initial_value_bounds(name, float(initial0[r, j]),
                             cap=cap_of[name][r] if name in cap_of else None)
        for r in range(len(samples)) for j, name in enumerate(unknown)]
    # the caps restrict the particulates, but their coordinates keep the uncapped range
    ranges_of = bounds + [initial_value_bounds(name, float(initial0[r, j]))
                          for r in range(len(samples)) for j, name in enumerate(unknown)]
    refined = local_stage(lambda theta: joint_value_and_grad(theta, x), x0, joint_bounds,
                          args.local_maxiter, label=key,
                          logarithmic=logarithmic_starts(key, len(names), len(samples)),
                          scales=[high - low for low, high in ranges_of])
    print(f"  {key} local: loss over the whole record {whole_at_global:.5f} -> {refined.fun:.5f} "
          f"after {refined.nit} iterations ({refined.message})", flush=True)

    theta = best_of(x0, whole_at_global, refined.x, refined.fun)
    kept_hybrids, kept_whole = both_losses(theta)
    print(f"  {key}: the hybrids' loss of the result {kept_hybrids:.5f}", flush=True)
    parameters, initial = split_local(theta, len(names), len(samples))
    fitted = dict(zip(names, map(float, parameters)))
    values.update(fitted)

    moved = []
    for r, label in enumerate(labels):
        for j, name in enumerate(unknown):
            before, after = float(initial0[r, j]), float(initial[r, j])
            if max(before, after) > 1e-4 and not 0.1 <= (after + 1e-12) / (before + 1e-12) <= 10:
                moved.append(f"{label} {name} {before:.4g} -> {after:.4g}")
    if moved:
        print(f"  {key}: initial values that moved more than tenfold:\n    " + "\n    ".join(moved))

    report = {
        "objectives": objectives,
        "excluded": exclude,
        # the forcing the parameters were fitted to (utils/forcing.FORCINGS)
        "forcing": dataset.forcing,
        # rates at 35 C of the temperature-dependent model, or at the reactors' own
        "arrhenius": arrhenius,
        # --bound: searched within these, not the stored bounds, which they replace
        "bounds given": {name: list(bound) for name, bound in given.items()},
        # "hybrids": from SPIN_UP_DAYS on, as the hybrids and the scores; "whole record":
        # every point but the initial one, as the local stage
        "loss": {"hybrids": {"current": start_loss, "global": float(found.fun),
                             "kept": kept_hybrids},
                 "whole record": {"global": whole_at_global, "local": float(refined.fun),
                                  "kept": kept_whole},
                 "kept": "local" if refined.fun < whole_at_global else "global"},
        "global": {"generations": int(found.nit), "evaluations": int(found.nfev),
                   "message": str(found.message)},
        "local": {"iterations": int(refined.nit), "evaluations": int(refined.nfev),
                  "message": str(refined.message)},
        "parameters": {name: {"before": float(start[i]), "global": float(found.x[i]),
                              "after": fitted[name], "bounds": list(bounds[i])}
                       for i, name in enumerate(names)},
        "initial_values": {label: {name: [float(initial0[r, j]), float(initial[r, j])]
                                   for j, name in enumerate(unknown)}
                           for r, label in enumerate(labels)},
        "moved_tenfold": moved,
    }
    report["particulate caps"] = {label: {name: float(caps[r, i]) for i, name in enumerate(capped)}
                                  for r, label in enumerate(labels)}
    if reset is not None:
        report["particulates reset"] = {
            label: {name: [float(reset[0][r, c]), float(reset[1][r, c])]
                    for name, c in zip(capped, columns)}
            for r, label in enumerate(labels)}

    if not args.dry_run:
        campaign_dir = data_dir / campaign
        save_parameters(campaign_dir / f"{campaign}-parameters.json", fitted, given)
        y0 = np.asarray(set_initial_values(x["initial_values"], positions, jnp.asarray(initial)))
        for r, label in enumerate(labels):
            reactor = label.split("_", 1)[1]
            pd.Series(y0[r], index=components).to_csv(campaign_dir / f"{IV_FILES[key]}_{reactor}.csv")
        # The forcing the parameters were fitted to, which a model built on them needs.
        (campaign_dir / CALIBRATION_FILE).write_text(
            json.dumps({"forcing": dataset.forcing}), encoding="utf-8")
    return report


def save_parameters(path, fitted, bounds=None):
    """Write the fitted values into the campaign file, keeping everything else but the
    ``bounds`` they were fitted within (``--bound``), which replace the stored ones."""
    params = load_params(str(path))
    for name, value in fitted.items():
        params[name].set(value=float(value))
    for name, (low, high) in (bounds or {}).items():
        params[name].set(min=float(low), max=float(high))
    save_params(params, str(path))


def backup(campaign_dir, campaign, stamp):
    target = campaign_dir / "calibration-backup" / stamp
    target.mkdir(parents=True)
    for path in [campaign_dir / f"{campaign}-parameters.json",
                 *sorted(campaign_dir.glob("R[234]-IVs_*.csv")),
                 *sorted(campaign_dir.glob(CALIBRATION_FILE))]:
        shutil.copy2(path, target / path.name)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--campaign", required=True, choices=DATASETS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS,
                        help="in calibration order; R3 and R2 use R4's parameters "
                             "(default: r4 r3 r2)")
    parser.add_argument("--de-maxiter", type=int, default=DE_MAXITER,
                        help=f"generations of the global stage (default {DE_MAXITER})")
    parser.add_argument("--de-popsize", type=int, default=DE_POPSIZE,
                        help=f"population per parameter (default {DE_POPSIZE})")
    parser.add_argument("--local-maxiter", type=int, default=LOCAL_MAXITER,
                        help=f"L-BFGS-B iterations of the local stage (default {LOCAL_MAXITER})")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reset-particulates", action="store_true",
                        help="start every reactor's particulates (X_ch, X_pr, X_li) from the "
                             "model's steady state at its first-week forcing, not from the "
                             "stored values")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("ADM1_DATA_DIR", "data")))
    parser.add_argument("--exclude", nargs="+", default=[], metavar="LABEL",
                        help="reactors left out of the fit, e.g. a fold's held-out ones "
                             "(rice-bran_R30); their start values are not written")
    parser.add_argument("--arrhenius", action="store_true",
                        help="calibrate the temperature-dependent model: the fitted rates are "
                             "then rates at 35 C")
    parser.add_argument("--bound", nargs=3, action="append", metavar=("NAME", "LOW", "HIGH"),
                        help="search NAME within LOW and HIGH instead of its stored bounds, and "
                             "store them with the fit; repeat for more parameters")
    parser.add_argument("--dry-run", action="store_true",
                        help="fit and report, but write nothing")
    args = parser.parse_args(argv)
    variants = [key for key in VARIANTS if key in args.variants]
    try:
        bounds = check_bounds(args.bound, variants)
    except ValueError as error:
        parser.error(f"--bound: {error}")
    # R3 and R2 keep R4's parameters, which must have been fitted to the forcing they are.
    from adm1_package.experiments.calibrations import campaign_forcing
    from adm1_package.utils.forcing import FORCINGS

    stored = campaign_forcing(args.data_dir, args.campaign)
    if "r4" not in variants and stored != FORCINGS[0]:
        parser.error(f"{args.campaign}'s R4 parameters were fitted to the {stored} forcing; "
                     "calibrate R4 too, which R3 and R2 keep")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = None if args.dry_run else backup(args.data_dir / args.campaign, args.campaign, stamp)
    print(f"{args.campaign}: calibrating {', '.join(variants)}"
          + (" with Arrhenius" if args.arrhenius else "")
          + "".join(f", {name} in [{low:g}, {high:g}]" for name, (low, high) in bounds.items())
          + (f" without {', '.join(args.exclude)}" if args.exclude else "")
          + (" (dry run)" if args.dry_run else f"; backup in {folder}"), flush=True)

    values, report, began = {}, {"campaign": args.campaign, "variants": {}}, time.perf_counter()
    for key in variants:
        report["variants"][key] = calibrate_variant(key, args.campaign, args.data_dir, values, args)
        if folder is not None:
            (folder / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{args.campaign}: done in {time.perf_counter() - began:.0f} s", flush=True)
    if folder is not None:
        print(f"report: {folder / 'report.json'}")


if __name__ == "__main__":
    main()
