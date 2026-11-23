"""The benchmark calibration (scripts/calibrate/calibrate_benchmarks.py).

It has to fit the loss the hybrids are trained with, and the states and parameters
SI Table S1 lists, at the positions the datasets keep them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import diffrax as dx
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from helpers import load_script

from adm1_package.experiments.registry import get_spec
from adm1_package.models.base.model import STATES as BASE_STATES
from adm1_package.models.r2.model import STATES as R2_STATES
from adm1_package.models.r3.model import STATES as R3_STATES
from adm1_package.models.r4.model import STATES as R4_STATES
from adm1_package.utils import initial_state
from adm1_package.utils.datasets import custom_collate_fn

LAYOUTS = {"r4": R4_STATES, "r3": R3_STATES, "r2": R2_STATES, "base": BASE_STATES}

calibration = load_script("calibrate/calibrate_benchmarks.py")


def _trainer_loss(pred, targets, mask, ranges, weights):
    """utils/ude_trainer.py, batched branch: pooled mean over reactors and time."""
    residuals = (pred - targets) / ranges
    mse = jnp.mean(residuals**2, axis=(0, 1), where=mask)
    return jnp.sqrt(jnp.sum(mse * weights) / jnp.sum(weights))


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    pred = jnp.asarray(rng.normal(size=(3, 20, 4)))
    targets = jnp.asarray(rng.normal(size=(3, 20, 4)))
    mask = jnp.asarray(rng.random((3, 20, 4)) > 0.3)
    ranges = jnp.asarray(rng.uniform(0.5, 2.0, size=(3, 1, 4)))
    weights = jnp.array([2.0, 2.0, 1.5, 0.5])
    return pred, targets, mask, ranges, weights


def test_the_loss_is_the_hybrids_loss():
    batch = _batch()
    assert float(calibration.weighted_loss(*batch)) == pytest.approx(
        float(_trainer_loss(*batch)), rel=1e-12)


def test_an_objective_without_points_drops_out_of_the_weighting():
    pred, targets, mask, ranges, weights = _batch()
    mask = mask.at[..., 2].set(False)
    loss = float(calibration.weighted_loss(pred, targets, mask, ranges, weights))
    keep = jnp.array([0, 1, 3])
    expected = float(_trainer_loss(pred[..., keep], targets[..., keep], mask[..., keep],
                                   ranges[..., keep], weights[keep]))
    assert np.isfinite(loss)
    assert loss == pytest.approx(expected, rel=1e-12)


def test_only_the_local_stage_scores_the_first_month():
    # Behind the spin-up a start value that outlasts it is one more parameter: a calibration
    # scored only after the spin-up can buy scores with an implausible start, such as a pH-12
    # start (R2/R3) or a gas burst (R4).
    time = jnp.tile(jnp.arange(60.0), (2, 1))                 # two reactors, daily
    measured = jnp.ones((2, 60, 3), dtype=bool).at[1, 10, 2].set(False)
    hybrids, whole = calibration.stage_masks(measured, time)
    assert not hybrids[:, :30].any() and hybrids[:, 30:].all()
    assert not whole[:, 0].any()                              # the start copies the data there
    assert whole[0, 1:].all() and whole[1, 1:, :2].all()
    assert not whole[1, 10, 2] and whole[1, 11:, 2].all()     # nor is a missing point scored


def test_the_local_vector_is_the_parameters_then_each_reactors_initial_values():
    theta = jnp.arange(2 + 3 * 4, dtype=float)          # 2 parameters, 3 reactors, 4 states
    parameters, initial = calibration.split_local(theta, n_parameters=2, n_reactors=3)
    assert parameters.tolist() == [0.0, 1.0]
    assert initial.shape == (3, 4) and initial[1].tolist() == [6.0, 7.0, 8.0, 9.0]
    y0 = calibration.set_initial_values(jnp.zeros((3, 6)), jnp.array([5, 0, 2, 3]), initial)
    assert y0[1].tolist() == [7.0, 0.0, 8.0, 9.0, 0.0, 6.0]
    assert float(y0[0, 1]) == 0.0 and float(y0[0, 4]) == 0.0


def _bowl(theta):
    return jnp.sum((theta - jnp.array([0.3, 0.7])) ** 2)


def test_the_global_stage_finds_the_minimum_from_any_start():
    first = calibration.global_stage(_bowl, [(0.0, 1.0), (0.0, 1.0)], maxiter=60, popsize=10, seed=1)
    again = calibration.global_stage(_bowl, [(0.0, 1.0), (0.0, 1.0)], maxiter=60, popsize=10, seed=1)
    np.testing.assert_allclose(first.x, [0.3, 0.7], atol=2e-2)
    np.testing.assert_array_equal(first.x, again.x)


def test_the_global_stage_searches_a_logarithmic_parameter_on_a_log_scale():
    # K_h2's bounds span five decades, its calibrated values the lowest two: a linear Latin
    # hypercube puts one member in eighty there
    def loss(theta):
        return (np.log10(theta[0]) + 6.0) ** 2 + (theta[1] - 0.5) ** 2

    found = calibration.global_stage(loss, [(1e-8, 1e-3), (0.0, 1.0)], maxiter=60, popsize=10,
                                     seed=1, logarithmic=[True, False])
    assert found.x[0] == pytest.approx(1e-6, rel=0.1)
    assert found.x[1] == pytest.approx(0.5, abs=2e-2)


def test_the_local_stage_survives_a_solve_that_fails():
    # The minimum (0.9, 0.7) lies where the "solver" fails: the stage must back off, not raise.
    centre = jnp.array([0.9, 0.7])

    def value_and_grad(theta):
        if float(theta[0]) > 0.5:
            raise RuntimeError("max_steps reached")
        return jnp.sum((theta - centre) ** 2), 2 * (theta - centre)

    start = np.array([0.2, 0.2])
    result = calibration.local_stage(value_and_grad, start, [(0.0, 1.0), (0.0, 1.0)], maxiter=200)
    assert np.isfinite(result.fun)
    assert result.fun < float(jnp.sum((jnp.asarray(start) - centre) ** 2))
    assert result.x[0] <= 0.5


def _rates_and_starts(curvature, start_curvature):
    """The local stage in miniature: a rate pushed into its lower bound, five steeply curved
    parameters near their optimum, and 24 start values of g/L scale 15 g/L off theirs, each
    moving the loss little; plus noise the gradient does not see, like the solver's step-size
    switching.

    Returns the loss with its gradient, the start and the start values' optimum.
    """
    rng = np.random.default_rng(0)
    p_opt, x_opt = np.array([0.5, 0.97, 0.53, 0.93, 0.18]), rng.uniform(15, 35, 24)
    coupling = rng.normal(0, 1, (24, 5)) * 0.02

    def loss(theta):
        k, p, X = theta[0], theta[1:6], theta[6:]
        dp, dX = p - p_opt, (X - x_opt) / 300
        value = (5.7 * (k - 0.02) + curvature * dp @ dp + start_curvature * dX @ dX
                 + dX @ coupling @ dp)
        gradient = np.concatenate([[5.7], 2 * curvature * dp + coupling.T @ dX,
                                   (2 * start_curvature * dX + coupling @ dp) / 300])
        return value + 4e-5 * np.sin(3e7 * theta.sum()), gradient

    return loss, np.concatenate([[0.0203], p_opt + 0.003, np.full(24, 39.6)]), x_opt


def test_the_local_stage_moves_start_values_past_steep_parameters():
    # In the variables' own units L-BFGS-B stops after a few iterations with the starts unmoved.
    loss, start, x_opt = _rates_and_starts(curvature=500, start_curvature=0.5)
    bounds = [(0.02, 2.88)] + [(0.0, 1.0)] * 5 + [(0.0, 300.0)] * 24
    result = calibration.local_stage(loss, start, bounds, maxiter=300)
    assert result.x[6:].mean() == pytest.approx(x_opt.mean(), abs=1.0)
    assert result.fun < 1e-3


def test_a_cap_restricts_start_values_without_rescaling_them():
    # A cap at 40 g/L that the optimum (15-35 g/L) never reaches. With coordinates spanning the
    # capped range instead of the starts' own, their gradients shrink 7.5x next to the
    # parameters', and the local stage stops early.
    loss, start, x_opt = _rates_and_starts(curvature=20000, start_curvature=0.1)
    bounds = [(0.02, 2.88)] + [(0.0, 1.0)] * 5 + [(0.0, 40.0)] * 24
    scales = [2.86] + [1.0] * 5 + [300.0] * 24
    result = calibration.local_stage(loss, start, bounds, maxiter=300, scales=scales)
    assert result.x[6:].mean() == pytest.approx(x_opt.mean(), abs=1.0)
    assert np.all(result.x[6:] <= 40.0)


def test_the_local_stage_seeds_a_population_from_nothing():
    # A biomass started at s blooms after ln(K / s) / mu days, so its gradient goes as 1/s: from
    # a start at zero (the right-hand side's floor, 1e-12) it is huge, and the local stage
    # cannot take a step. Here the bloom is wanted on day 20 (s = e^-10, K = 1).
    def loss(theta):
        s, other = theta
        bloom = np.log(1.0 / max(s, 1e-12)) / 0.5
        value = ((bloom - 20.0) / 10.0) ** 2 + (other - 1.0) ** 2
        return value, np.array([-2 * (bloom - 20.0) / (100 * 0.5 * max(s, 1e-12)), 2 * (other - 1.0)])

    result = calibration.local_stage(loss, np.array([0.0, 3.0]), [(0.0, 10.0), (0.0, 10.0)],
                                     maxiter=200, logarithmic=[True, False])
    assert result.x[0] == pytest.approx(np.exp(-10), rel=1e-2)
    assert result.x[1] == pytest.approx(1.0, abs=1e-3)


def test_a_particulate_start_is_bounded_by_its_first_week_steady_state():
    # A reactor holds no more undegraded substrate than its own feed builds up. Unbounded,
    # the local stage can start a reactor with many times that carbohydrate: a gas burst
    # whose TAN uptake the long solids retention carries into the scores.
    assert calibration.initial_value_bounds("X_ch", 14.7, cap=0.52) == (0.0, 0.52)
    assert calibration.initial_value_bounds("X_ch", 14.7) == (0.0, 300.0)
    # so do the full ADM1's products of hydrolysis, which nothing measures either
    assert calibration.initial_value_bounds("S_su", 3.47, cap=0.02) == (0.0, 0.02)


def test_stored_particulate_starts_above_their_steady_state_are_lowered_to_it():
    stored = np.array([[8.2, 0.1, 3.0], [0.3, 0.9, 4.0]])      # X_ch, X_pr, then X_bac
    steady = np.array([[0.52, 0.3, 7.0], [0.75, 0.4, 7.0]])
    capped = calibration.capped_starts(stored, steady, columns=[0, 1], reset=False)
    np.testing.assert_array_equal(capped, [[0.52, 0.1, 3.0], [0.3, 0.4, 4.0]])
    reset = calibration.capped_starts(stored, steady, columns=[0, 1], reset=True)
    np.testing.assert_array_equal(reset, [[0.52, 0.3, 3.0], [0.75, 0.4, 4.0]])


def test_a_variable_whose_bounds_coincide_stays_there():
    # a cap can be a steady state of zero
    def loss(theta):
        return (theta[0] - 1.0) ** 2 + (theta[1] - 0.5) ** 2, 2 * (theta - np.array([1.0, 0.5]))

    result = calibration.local_stage(loss, np.array([0.0, 0.9]), [(0.0, 0.0), (0.0, 1.0)], maxiter=50)
    assert result.x[0] == 0.0
    assert result.x[1] == pytest.approx(0.5, abs=1e-6)


def test_the_local_vector_marks_the_biomass_starts():
    # two parameters, then per reactor R3's starts X_ch, X_pr, X_li, X_bac, X_ac
    assert calibration.logarithmic_starts("r3", n_parameters=2, n_reactors=2) == [
        False, False, False, False, False, True, True, False, False, False, True, True]
    # the full ADM1: its eight parameters, K_h2 last, then its particulates, its products
    # of hydrolysis and its eight biomasses
    assert calibration.logarithmic_starts("base", n_parameters=8, n_reactors=1) == (
        [False] * 7 + [True] + [False] * 6 + [True] * 8)


def test_the_full_adm1_is_no_benchmark_variant():
    # base_calibration.py runs it, after R2
    assert "base" not in calibration.VARIANTS
    with pytest.raises(SystemExit):
        calibration.main(["--campaign", "rice-bran", "--variants", "base", "--dry-run"])


def test_the_better_stage_is_kept():
    start = np.array([0.5, 0.1, 0.2])
    local = np.array([0.4, 0.15, 0.25])
    assert calibration.best_of(start, 1.0, local, 0.8).tolist() == local.tolist()
    assert calibration.best_of(start, 1.0, local, 1.2).tolist() == start.tolist()


def test_the_stages_follow_si_table_s1():
    shared = set(calibration.PARAMETERS["r4"])
    assert not shared & set(calibration.PARAMETERS["r3"])
    assert not shared & set(calibration.PARAMETERS["r2"])
    # measured, or set from the first measurements: the gas phase, the dissolved gases in
    # equilibrium with it, and R2/R3's acid-base chemistry at the measured pH
    from_data = {"S_TAN", "S_h2o", "S_ac", "S_va", "S_bu", "S_pro", "S_ch4", "S_gas_ch4",
                 "S_gas_co2", "S_IC", "S_an", "S_cat", "S_va_i", "S_bu_i", "S_pro_i", "S_ac_i",
                 "S_hco3_i", "S_nh3", "S_h2", "S_gas_h2"}
    for key, names in calibration.INITIAL_VALUES.items():
        assert set(names) <= set(calibration.COMPONENTS[key])
        assert not set(names) & from_data, f"{key} would fit states the data set"
        # and nothing is left as an earlier calibration stored it
        assert set(calibration.COMPONENTS[key]) <= set(names) | from_data, key
    # the full ADM1 comes after them and fits none of their parameters
    assert not set(calibration.PARAMETERS["base"]) & (
        shared | set(calibration.PARAMETERS["r3"]) | set(calibration.PARAMETERS["r2"]))


@pytest.mark.parametrize("key", ["r4", "r3", "r2", "base"])
def test_the_components_are_the_models_states(key):
    # the models call it X_bac; X_aa, the amino-acid degraders, belongs to the full ADM1
    assert calibration.COMPONENTS[key] == LAYOUTS[key].names


def _one_reactor(key, data_dir, label):
    """The collated sample (a batch of one) of reactor ``label``."""
    dataset = get_spec(key).dataset(data_dir, data_to_use=[label.split("_")[0]])
    x, _ = custom_collate_fn([dataset[dataset.labels.index(label)]])
    return x


@pytest.mark.requires_data
@pytest.mark.parametrize("arrhenius", [False, True])
def test_the_r4_particulates_restart_at_the_steady_state_of_the_first_week(data_dir, arrhenius):
    # In R4 the particulates and their decay source X_bac form a linear system at constant
    # forcing: D f_i X_i,in - (D + k_hyd_i) X_i + y_i k_dec X_bac = 0 with the decay yields
    # y = 0.19, 0.77, 0.04, and X_bac grows by 0.1125, 0.1723, 0.2286 per hydrolysed g.
    # With Arrhenius the hydrolysis rates are those at the first week's temperature.
    from adm1_package.utils.corrections import activation_energies, arrhenius_factor

    x = _one_reactor("r4", data_dir, "LaRA-temp_F4")
    p = {name: float(value[0]) for name, value in x["parameters"].items()}
    interp = dx.CubicInterpolation(x["input"]["time"][0], tuple(c[0] for c in x["input"]["coeffs"]))
    forcing = np.asarray(initial_state.first_week_forcing(interp, x["input"]["time"][0, 0]))
    D, feed = forcing[1] / p["V_liq"], forcing[-3:]          # inflow, then X_ch, X_pr, X_li
    k = np.array([p["k_hyd_ch"], p["k_hyd_pr"], p["k_hyd_li"]])
    if arrhenius:
        energies = activation_energies("r4")
        k = k * np.array([float(arrhenius_factor(forcing[0], energies[name]))
                          for name in ("k_hyd_ch", "k_hyd_pr", "k_hyd_li")])
    f = np.array([p["f_ch"], p["f_pr"], p["f_li"]])
    system = np.zeros((4, 4))
    system[:3, :3] = -np.diag(D + k)
    system[:3, 3] = np.array([0.19, 0.77, 0.04]) * p["k_dec_Xbac"]
    system[3, :3] = np.array([0.1125, 0.1723, 0.2286]) * k
    system[3, 3] = -(D + p["k_dec_Xbac"])
    expected = np.linalg.solve(system, np.concatenate([-D * f * feed, [0.0]]))[:3]
    names = LAYOUTS["r4"].names
    reset = np.asarray(calibration.first_week_particulates("r4", x, arrhenius=arrhenius))[0]
    # rtol 1%: Arrhenius at the first week's 37 C moves these steady states by 2.5-4%.
    np.testing.assert_allclose([reset[names.index(n)] for n in ("X_ch", "X_pr", "X_li")],
                               expected, rtol=1e-2)


@pytest.mark.requires_data
@pytest.mark.parametrize("key, label", [("r2", "LaRA-substrate_F2"),   # stored with a large X_ch
                                        ("base", "LaRA-temp_F4")])
def test_the_particulate_reset_does_not_read_the_stored_particulates(key, label, data_dir):
    x = _one_reactor(key, data_dir, label)
    names = LAYOUTS[key].names
    particulates = jnp.array([names.index(n) for n in calibration.SUBSTRATES[key]])
    scrambled = {**x, "initial_values": x["initial_values"].at[:, particulates].set(
        10 * x["initial_values"][:, particulates] + 1)}
    reset = np.asarray(calibration.first_week_particulates(key, x))
    assert np.all(np.isfinite(reset)) and np.all(reset[:, particulates] > 0)
    np.testing.assert_array_equal(
        np.asarray(calibration.first_week_particulates(key, scrambled))[:, particulates], reset[:, particulates])
    rest = [i for i in range(len(names)) if i not in particulates.tolist()]
    np.testing.assert_array_equal(reset[:, rest], np.asarray(x["initial_values"])[:, rest])


def _campaign_copy(data_dir, root, campaign="rice-bran"):
    """R4's files of ``campaign``, copied into ``root``."""
    target = root / campaign
    target.mkdir(parents=True)
    for pattern in (f"{campaign}-parameters.json", "input_*.csv", "output_*.csv", "R4-IVs_*.csv"):
        for path in (Path(data_dir) / campaign).glob(pattern):
            shutil.copy2(path, target / path.name)
    return target


QUICK = {"de_maxiter": 1, "de_popsize": 1, "seed": 42, "reset_particulates": True}


@pytest.fixture(scope="module")
def dry_run(data_dir, tmp_path_factory):
    """R4 on a copy of rice-bran, its particulates reset: one generation, two local
    iterations, nothing written.

    Returns the report and the copy's files before and after.
    """
    campaign = _campaign_copy(data_dir, tmp_path_factory.mktemp("data"))
    files = lambda: {path.name: path.read_bytes() for path in sorted(campaign.iterdir())}  # noqa: E731
    before = files()
    args = SimpleNamespace(**QUICK, local_maxiter=2, dry_run=True)
    report = calibration.calibrate_variant("r4", "rice-bran", campaign.parent, {}, args)
    return report, before, files()


@pytest.mark.requires_data
def test_no_particulate_start_ends_above_its_cap(dry_run):
    report = dry_run[0]
    caps = report["particulate caps"]
    for label, values in caps.items():
        for name, cap in values.items():
            assert report["initial_values"][label][name][1] <= cap, (label, name)


@pytest.mark.requires_data
def test_the_local_stage_starts_from_the_reset_particulates(dry_run):
    report = dry_run[0]
    reset = report["particulates reset"]           # per reactor: [stored, reset]
    assert any(stored != new for values in reset.values() for stored, new in values.values())
    for label, values in reset.items():
        for name, (_, new) in values.items():
            assert report["initial_values"][label][name][0] == new, (label, name)


@pytest.mark.requires_data
def test_what_is_kept_is_judged_on_the_local_stages_loss(dry_run):
    # If the local stage minimised another loss, its result's whole-record loss would not be
    # the one it reports. The gradient's compiled solve and the plain one agree only to the
    # solver's tolerance.
    loss = dry_run[0]["loss"]["whole record"]
    assert loss["kept"] == pytest.approx(min(loss["local"], loss["global"]), rel=1e-3)
    assert dry_run[0]["loss"]["hybrids"]["kept"] != pytest.approx(loss["kept"], rel=1e-3)


@pytest.mark.requires_data
def test_a_dry_run_writes_nothing(dry_run):
    _, before, after = dry_run
    assert after == before


@pytest.mark.requires_data
def test_a_calibration_with_arrhenius_fits_rates_at_35_c(data_dir, tmp_path, dry_run):
    """The same stages on the temperature-dependent model: a different loss at the stored
    rates (rice-bran runs near 37 C, where the factors are 1.03-1.09), and the report says so."""
    _campaign_copy(data_dir, tmp_path)
    args = SimpleNamespace(**QUICK, local_maxiter=2, dry_run=True, arrhenius=True)
    report = calibration.calibrate_variant("r4", "rice-bran", tmp_path, {}, args)
    assert report["arrhenius"] is True and dry_run[0]["arrhenius"] is False
    assert report["loss"]["hybrids"]["current"] != pytest.approx(
        dry_run[0]["loss"]["hybrids"]["current"], rel=1e-6)


@pytest.mark.requires_data
def test_an_excluded_reactor_is_left_out_of_the_fit_and_its_start_values_stay(data_dir, tmp_path):
    # A fold's per-fold benchmark: calibrated without the reactors the fold holds out,
    # which keep the start values the hybrid starts from too.
    campaign = _campaign_copy(data_dir, tmp_path)
    held_out = (campaign / "R4-IVs_R30.csv").read_bytes()
    args = SimpleNamespace(**QUICK, local_maxiter=1, dry_run=False, exclude=["rice-bran_R30"])
    report = calibration.calibrate_variant("r4", "rice-bran", tmp_path, {}, args)
    assert set(report["initial_values"]) == {"rice-bran_R60", "rice-bran_R90"}
    assert report["excluded"] == ["rice-bran_R30"]
    assert (campaign / "R4-IVs_R30.csv").read_bytes() == held_out


@pytest.mark.requires_data
def test_a_bound_given_replaces_the_stored_one_and_is_stored_with_the_fit(data_dir, tmp_path):
    """--bound k_hyd_ch 0.5 2.88 on rice-bran (stored below 0.5): the search starts at the bound and
    stays inside, and the parameter file keeps the bound with the value fitted within it."""
    campaign = _campaign_copy(data_dir, tmp_path)
    args = SimpleNamespace(**QUICK, local_maxiter=1, dry_run=False,
                           bound=[["k_hyd_ch", "0.5", "2.88"]])
    report = calibration.calibrate_variant("r4", "rice-bran", tmp_path, {}, args)
    entry = report["parameters"]["k_hyd_ch"]
    assert entry["bounds"] == [0.5, 2.88] and entry["before"] == pytest.approx(0.5)
    assert 0.5 <= entry["after"] <= 2.88
    stored = json.loads((campaign / "rice-bran-parameters.json").read_text(encoding="utf-8"))
    installed = json.loads((Path(data_dir) / "rice-bran" / "rice-bran-parameters.json")
                           .read_text(encoding="utf-8"))
    assert (stored["k_hyd_ch"]["min"], stored["k_hyd_ch"]["max"]) == (0.5, 2.88)
    assert stored["k_hyd_ch"]["value"] == pytest.approx(entry["after"])
    for name in ("f_ch", "k_hyd_pr"):
        assert (stored[name]["min"], stored[name]["max"]) == (installed[name]["min"],
                                                              installed[name]["max"])
    # the forcing it was fitted to, beside the parameters
    assert report["forcing"] == "smoothstep"
    assert json.loads((campaign / "calibration.json").read_text(encoding="utf-8")) == {
        "forcing": "smoothstep"}


@pytest.mark.parametrize("bound, message", [
    (["k_m_ac", "0.1", "1"], "k_m_ac"),          # R3's, and only R4 is calibrated
    (["k_hyd_ch", "0.3", "0.1"], "0.3"),         # an empty range
    (["k_hyd_ch", "fast", "1"], "fast"),
])
def test_a_bound_on_nothing_calibrated_or_an_empty_range_is_refused(bound, message, capsys):
    with pytest.raises(SystemExit):
        calibration.main(["--campaign", "rice-bran", "--variants", "r4", "--dry-run",
                          "--bound", *bound])
    error = capsys.readouterr().err
    assert "--bound" in error and message in error and "unrecognized" not in error


def test_bounds_are_read_once_per_name():
    assert calibration.check_bounds([["k_hyd_ch", "0.1", "2.88"], ["f_ch", "0", "0.9"]],
                                    ["r4"]) == {"k_hyd_ch": (0.1, 2.88), "f_ch": (0.0, 0.9)}
    assert calibration.check_bounds(None, ["r4"]) == {}
    with pytest.raises(ValueError, match="twice"):
        calibration.check_bounds([["k_hyd_ch", "0.1", "1"], ["k_hyd_ch", "0.2", "1"]], ["r4"])


@pytest.mark.requires_data
@pytest.mark.parametrize("exclude, message", [
    (["rice-bran_R31"], "rice-bran_R31"),
    (["rice-bran_R30", "rice-bran_R60", "rice-bran_R90"], "no reactor"),
])
def test_an_exclusion_that_names_no_reactor_or_leaves_none_is_refused(exclude, message, data_dir,
                                                                       tmp_path):
    _campaign_copy(data_dir, tmp_path)
    args = SimpleNamespace(**QUICK, local_maxiter=1, dry_run=True, exclude=exclude)
    with pytest.raises(ValueError, match=message):
        calibration.calibrate_variant("r4", "rice-bran", tmp_path, {}, args)


@pytest.mark.requires_data
@pytest.mark.parametrize("key", ["r4", "r3", "r2", "base"])
def test_the_components_are_where_the_dataset_keeps_them(key, data_dir, installed_dataset):
    components = calibration.COMPONENTS[key]
    dataset = installed_dataset(key, ["rice-bran"])
    x, y = dataset[dataset.labels.index("rice-bran_R30")]
    assert x["initial_values"].shape == (len(components),)
    stored = pd.read_csv(Path(data_dir) / "rice-bran" / f"{calibration.IV_FILES[key]}_R30.csv",
                         index_col=0).iloc[:, 0]
    for name in calibration.INITIAL_VALUES[key]:
        assert float(x["initial_values"][components.index(name)]) == pytest.approx(
            float(stored[name]), rel=1e-12), name
