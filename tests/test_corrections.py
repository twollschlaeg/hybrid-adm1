"""The mapping from network outputs to model parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from adm1_package.experiments.registry import MODEL_KEYS, get_spec
from adm1_package.utils.corrections import (
    CATALOGUE,
    DEFAULT,
    PRUNED,
    CorrectionOptions,
    layout_for,
)

PARAMS = {"f_ch": 0.7, "f_pr": 0.8, "f_li": 0.9997, "k_hyd_ch": 0.05, "k_m_ac": 0.3,
          "K_I_nh3": 0.026}
#: A reactor without calibrated values: every parameter starts at its reference.
UNCALIBRATED: dict = {}
#: A calibrated parameter set in the mechanistic models' own names.
MECH_PARAMS = {
    "f_ch": 0.51, "f_pr": 0.37, "f_li": 0.9997,
    "k_hyd_ch": 0.023, "k_hyd_pr": 2.03, "k_hyd_li": 0.76,
    "k_m_va": 0.78, "k_m_bu": 0.36, "k_m_pro": 0.33, "k_m_ac": 0.83,
    "k_dec_Xbac": 0.021, "k_dec_Xac": 0.022, "k_dec_Xpro": 0.023, "k_dec_Xc4": 0.024,
    "K_I_nh3": 0.026,
}
#: R2's valerate and butyrate degraders share the mechanistic k_dec_Xc4 and have no
#: parameter of their own, so calibrated R2 holds them at the literature value. The
#: benchmark equivalence therefore needs k_dec_Xc4 = 0.02, as every campaign file has.
WITHOUT_OWN_PARAMETER = {"k_dec_Xva": 0.02, "k_dec_Xbu": 0.02}


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_the_unpruned_network_emits_factors_then_rates(model_key):
    """The split network takes its factor outputs from the front."""
    spec = get_spec(model_key)
    layout = layout_for(model_key, CorrectionOptions(prune=False))
    assert layout.size == spec.nn_out_size
    assert layout.outputs == tuple(spec.factor_names) + tuple(spec.rate_names)
    assert layout.n_factor_outputs == 3
    assert layout_for(model_key, DEFAULT).n_factor_outputs == 2   # f_li is pruned


def test_no_options_means_the_default():
    assert CorrectionOptions.from_dict(None) == DEFAULT
    assert layout_for("r4", None).options == DEFAULT


def test_the_rate_names_match_the_registry():
    """figures.kinetic_trajectories indexes dynamic_rates by these names."""
    for model_key in MODEL_KEYS:
        layout = layout_for(model_key, DEFAULT)
        assert layout.rate_names == tuple(get_spec(model_key).rate_names)
        assert layout.factor_names == tuple(get_spec(model_key).factor_names)


# ---------------------------------------------------------------------------
# Calibrated starts where it claims to
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", MODEL_KEYS)
@pytest.mark.parametrize("prune", [False, True])
def test_calibrated_at_zero_output_uses_every_mechanistic_value(model_key, prune):
    """Held parameters too: otherwise calibrated+pruned is not the benchmark."""
    layout = layout_for(model_key, CorrectionOptions("calibrated", prune=prune))
    values = layout.resolve(jnp.zeros(layout.size), MECH_PARAMS, 310.0)
    for target in CATALOGUE[model_key]:
        if target.name in WITHOUT_OWN_PARAMETER:
            expected = WITHOUT_OWN_PARAMETER[target.name]
        else:
            expected = MECH_PARAMS[target.name]
        # The one exception: an unpruned fraction is kept inside the logit band.
        if target.kind == "fraction" and not prune and target.name == "f_li":
            expected = min(expected, 0.99)
        assert float(values[target.name]) == pytest.approx(expected), target.name


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_without_calibrated_values_zero_output_is_the_literature_model(model_key):
    layout = layout_for(model_key, CorrectionOptions(prune=False))
    values = layout.resolve(jnp.zeros(layout.size), UNCALIBRATED, 310.0)
    for target in CATALOGUE[model_key]:
        assert float(values[target.name]) == pytest.approx(target.reference)


def test_calibrated_falls_back_to_the_reference_for_uncalibrated_parameters():
    """A parameter the campaign file does not carry has no value to anchor on."""
    layout = layout_for("r4", CorrectionOptions("calibrated", prune=False))
    values = layout.resolve(jnp.zeros(layout.size), PARAMS, 310.0)
    assert float(values["k_dec_Xbac"]) == pytest.approx(0.02)


def test_calibrated_keeps_a_near_one_fraction_trainable():
    """A calibrated f_li of 0.9997 would put the logit on sigmoid's flat tail."""
    layout = layout_for("r4", CorrectionOptions("calibrated", prune=False))
    index = layout.outputs.index("f_li")
    grad = jax.grad(lambda d: layout.resolve(
        jnp.zeros(layout.size).at[index].set(d), PARAMS, 310.0)["f_li"])(0.0)
    assert float(grad) > 1e-3


def test_a_reference_override_moves_the_starting_point_of_an_uncalibrated_parameter():
    options = CorrectionOptions(prune=False, references=(("f_pr", 0.75),))
    layout = layout_for("r4", options)
    values = layout.resolve(jnp.zeros(layout.size), UNCALIBRATED, 310.0)
    assert float(values["f_pr"]) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# The new links have no dead zones
# ---------------------------------------------------------------------------
def _grad(layout, name, delta, params=PARAMS):
    index = layout.outputs.index(name)
    return float(jax.grad(lambda d: layout.resolve(
        jnp.zeros(layout.size).at[index].set(d), params, 310.0)[name])(delta))


def test_rates_keep_a_gradient_where_the_original_clip_killed_it():
    """clip(anchor + x, 1e-10, 10) had zero gradient below its floor."""
    assert _grad(layout_for("r4", CorrectionOptions(prune=False)), "k_dec_Xbac", -0.5) > 0


def test_fractions_keep_a_gradient_where_sparse_sigmoid_was_flat():
    assert _grad(layout_for("r4", DEFAULT), "f_ch", 1.5) > 0


def test_rates_stay_positive_and_bounded():
    layout = layout_for("r4", DEFAULT)
    for delta in (-1e3, -3.0, 0.0, 3.0, 1e3):
        value = float(layout.resolve(jnp.full(layout.size, delta), PARAMS, 310.0)["k_hyd_ch"])
        assert 0.0 < value <= PARAMS["k_hyd_ch"] * np.exp(3.0) + 1e-12


def test_a_multiplicative_link_makes_corrections_relative():
    """The same network output moves a 0.02 and a 2.0 anchor by the same factor."""
    layout = layout_for("r4", CorrectionOptions(prune=False))
    values = layout.resolve(jnp.full(layout.size, 0.5), UNCALIBRATED, 310.0)
    assert float(values["k_hyd_ch"]) / 0.02 == pytest.approx(float(values["k_hyd_pr"]) / 2.0)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_pruning_removes_outputs_and_holds_them_at_their_fixed_value(model_key):
    full = layout_for(model_key, CorrectionOptions(prune=False))
    pruned = layout_for(model_key, CorrectionOptions(prune=True))
    assert pruned.size == full.size - len(PRUNED[model_key])

    values = pruned.resolve(jnp.zeros(pruned.size), UNCALIBRATED, 310.0)
    for target in CATALOGUE[model_key]:
        if target.name in PRUNED[model_key]:
            assert float(values[target.name]) == pytest.approx(target.fixed_value)


def test_pruned_lipid_degradability_is_complete_not_the_sigmoid_midpoint():
    values = layout_for("r4", DEFAULT).resolve(jnp.zeros(3), UNCALIBRATED, 310.0)
    assert float(values["f_li"]) == pytest.approx(1.0)
    # calibrated, it is held at the calibration
    values = layout_for("r4", DEFAULT).resolve(jnp.zeros(3), PARAMS, 310.0)
    assert float(values["f_li"]) == pytest.approx(PARAMS["f_li"])


def test_pruning_keeps_the_rates_that_moved_an_observable():
    """Acetate and propionate decay moved S_ac and S_pro; they are not dead weight."""
    outputs = layout_for("r2", DEFAULT).outputs
    for name in ("f_ch", "f_pr", "k_hyd_ch", "k_m_ac", "k_m_pro", "k_dec_Xac", "k_dec_Xpro"):
        assert name in outputs


def test_every_rate_is_still_resolved_whether_or_not_it_is_pruned():
    """dynamic_rates must keep its full width, or the figures misindex it."""
    for model_key in MODEL_KEYS:
        layout = layout_for(model_key, DEFAULT)
        values = layout.resolve(jnp.zeros(layout.size), PARAMS, 310.0)
        assert all(name in values for name in layout.rate_names)


# ---------------------------------------------------------------------------
# Physics (Arrhenius: test_arrhenius.py)
# ---------------------------------------------------------------------------
def test_nh3_inhibition_adds_an_output_for_r2_and_r3():
    for model_key in ("r2", "r3"):
        layout = layout_for(model_key, CorrectionOptions(physics=("nh3_inhibition",)))
        assert layout.outputs[-1] == "K_I_nh3"
        values = layout.resolve(jnp.zeros(layout.size), PARAMS, 310.0)
        assert float(values["K_I_nh3"]) == pytest.approx(0.026)


def test_nh3_inhibition_is_skipped_for_r4_rather_than_breaking_an_array_job():
    """R4 has no inhibition term; a round-robin job passes the same flags to all."""
    layout = layout_for("r4", CorrectionOptions(physics=("nh3_inhibition", "tan_inflow")))
    assert "K_I_nh3" not in layout.outputs
    assert "f_TAN_in" in layout.outputs
    assert layout.skipped_physics == ("nh3_inhibition",)


def test_tan_inflow_starts_at_one():
    layout = layout_for("r4", CorrectionOptions(physics=("tan_inflow",)))
    assert float(layout.resolve(jnp.zeros(layout.size), PARAMS, 310.0)["f_TAN_in"]) == 1.0


def test_extras_are_appended_behind_the_catalogue():
    base = layout_for("r3", CorrectionOptions(prune=False)).outputs
    extended = layout_for("r3", CorrectionOptions(prune=False, physics=("nh3_inhibition",))).outputs
    assert extended[: len(base)] == base


# ---------------------------------------------------------------------------
# Options round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("options", [
    CorrectionOptions("calibrated", prune=False, physics=("tan_inflow", "arrhenius"),
                      references=(("f_pr", 0.7),)),
    CorrectionOptions("calibrated", corrected=("k_hyd_ch", "f_ch"), bound=float(np.log(2.0))),
    CorrectionOptions("calibrated", exact_anchors=True),
], ids=["physics and references", "narrow", "exact anchors"])
def test_options_round_trip_through_a_saved_record(options):
    assert CorrectionOptions.from_dict(options.to_dict()) == options


@pytest.mark.parametrize("field, one, other", [
    ("physics", ("tan_inflow", "arrhenius"), ("arrhenius", "tan_inflow")),
    ("corrected", ("k_hyd_ch", "f_ch"), ("f_ch", "k_hyd_ch")),
])
def test_the_order_of_a_set_of_names_does_not_change_the_options(field, one, other):
    a, b = CorrectionOptions(**{field: one}), CorrectionOptions(**{field: other})
    assert a == b and hash(a) == hash(b)


@pytest.mark.parametrize("bad", [{"parametrization": "nope"}, {"physics": ["magic"]},
                                 {"bound": 0.0}, {"bound": -1.0}, {"corrected": []}])
def test_unknown_options_are_rejected(bad):
    with pytest.raises(ValueError):
        CorrectionOptions.from_dict(bad)


# ---------------------------------------------------------------------------
# A narrow correction: which parameters the network changes, and how far
# ---------------------------------------------------------------------------
NARROW = CorrectionOptions("calibrated", corrected=("k_hyd_ch", "f_ch"), bound=float(np.log(2.0)))


def _logit(value) -> float:
    value = float(value)
    return float(np.log(value) - np.log1p(-value))


def test_corrected_names_the_outputs_in_catalogue_order():
    """Factors first, so the split network still takes them from the front."""
    layout = layout_for("r4", NARROW)
    assert layout.outputs == ("f_ch", "k_hyd_ch")
    assert layout.n_factor_outputs == 1


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_everything_not_corrected_is_held_at_its_calibrated_value(model_key):
    layout = layout_for(model_key, NARROW)
    values = layout.resolve(jnp.full(layout.size, 0.3), MECH_PARAMS, 310.0)
    for target in CATALOGUE[model_key]:
        if target.name in ("f_ch", "k_hyd_ch") or target.name in WITHOUT_OWN_PARAMETER:
            continue
        assert float(values[target.name]) == pytest.approx(MECH_PARAMS[target.name]), target.name
    assert float(values["k_hyd_ch"]) == pytest.approx(MECH_PARAMS["k_hyd_ch"] * np.exp(0.3), rel=0.05)


def test_at_zero_output_a_narrow_correction_is_the_calibrated_model():
    layout = layout_for("r4", NARROW)
    values = layout.resolve(jnp.zeros(layout.size), MECH_PARAMS, 310.0)
    assert float(values["f_ch"]) == pytest.approx(MECH_PARAMS["f_ch"])
    assert float(values["k_hyd_ch"]) == pytest.approx(MECH_PARAMS["k_hyd_ch"])


def test_corrected_overrides_pruning():
    """k_hyd_pr is pruned by default; naming it makes it an output."""
    layout = layout_for("r4", CorrectionOptions("calibrated", prune=True, corrected=("k_hyd_pr",)))
    assert layout.outputs == ("k_hyd_pr",)


@pytest.mark.parametrize("names", [("k_m_ac",), ("f_xx",)])
def test_a_corrected_name_the_variant_does_not_have_fails(names):
    """R4 has no acetate uptake; a typo must not train a model that corrects nothing."""
    with pytest.raises(KeyError, match="ADM1-R4|r4"):
        layout_for("r4", CorrectionOptions("calibrated", corrected=names))


def test_a_bound_keeps_rates_and_fractions_within_it():
    bound = float(np.log(2.0))
    layout = layout_for("r4", NARROW)
    for delta in (-50.0, -1.0, 1.0, 50.0):
        values = layout.resolve(jnp.full(layout.size, delta), MECH_PARAMS, 310.0)
        ratio = float(values["k_hyd_ch"]) / MECH_PARAMS["k_hyd_ch"]
        assert 0.5 - 1e-12 <= ratio <= 2.0 + 1e-12
        shift = float(_logit(values["f_ch"]) - _logit(MECH_PARAMS["f_ch"]))
        assert abs(shift) <= bound + 1e-9


def test_a_bound_keeps_a_gradient_at_the_edge():
    """Squashed smoothly, not clipped: a saturated output can still come back."""
    assert _grad(layout_for("r4", NARROW), "f_ch", 2.0) > 0
    assert _grad(layout_for("r4", NARROW), "k_hyd_ch", 2.0) > 0


def test_without_a_bound_rates_reach_the_wide_default_and_fractions_are_free():
    """The records of every earlier run say nothing about a bound; they must not change."""
    layout = layout_for("r4", CorrectionOptions("calibrated"))
    values = layout.resolve(jnp.full(layout.size, 50.0), MECH_PARAMS, 310.0)
    assert float(values["k_hyd_ch"]) == pytest.approx(MECH_PARAMS["k_hyd_ch"] * np.exp(3.0), rel=1e-6)
    assert float(values["f_ch"]) > 0.999


def test_the_narrow_options_are_recorded_in_catalogue_order():
    record = NARROW.to_dict()
    assert record["corrected"] == ["f_ch", "k_hyd_ch"]
    assert record["bound"] == pytest.approx(np.log(2.0))


def test_a_record_without_the_narrow_options_reads_as_before():
    record = {"parametrization": "calibrated", "prune": True, "physics": [], "references": {}}
    options = CorrectionOptions.from_dict(record)
    assert options.corrected is None and options.bound is None and options.exact_anchors is False
    assert options == CorrectionOptions("calibrated")


# ---------------------------------------------------------------------------
# Exact anchors: a calibrated fraction outside the logit band
# ---------------------------------------------------------------------------
def _saturated():
    """As the per-fold calibrations found them: LaRA-substrate's protein fully degradable
    in four folds, LaRA-temp's carbohydrates in fold 3 nearly so."""
    return {**MECH_PARAMS, "f_pr": 1.0, "f_ch": 0.9977}


def test_a_fraction_outside_the_band_starts_at_the_band_edge_by_default():
    """What every earlier calibrated run did, kept so that their scores reproduce."""
    values = layout_for("r4", CorrectionOptions("calibrated")).resolve(jnp.zeros(3), _saturated(), 310.0)
    assert float(values["f_pr"]) == pytest.approx(0.99) and float(values["f_ch"]) == pytest.approx(0.99)


@pytest.mark.parametrize("options", [CorrectionOptions("calibrated", exact_anchors=True),
                                     CorrectionOptions("calibrated", exact_anchors=True,
                                                       corrected=("f_ch", "k_hyd_ch"),
                                                       bound=float(np.log(2.0)))],
                         ids=["pruned", "narrow"])
def test_exact_anchors_start_every_fraction_at_its_calibrated_value(options):
    layout = layout_for("r4", options)
    values = layout.resolve(jnp.zeros(layout.size), _saturated(), 310.0)
    for name in ("f_ch", "f_pr", "f_li", "k_hyd_ch"):
        assert float(values[name]) == pytest.approx(_saturated()[name], abs=1e-12), name


def test_exact_anchors_keep_the_gradient_and_the_logit_shape():
    """The cut-off part is an offset: the network moves the fraction as before."""
    exact = layout_for("r4", CorrectionOptions("calibrated", exact_anchors=True))
    plain = layout_for("r4", CorrectionOptions("calibrated"))
    assert _grad(exact, "f_pr", -1.0, _saturated()) == pytest.approx(_grad(plain, "f_pr", -1.0, _saturated()))
    shift = jnp.zeros(3).at[exact.outputs.index("f_pr")].set(-1.0)
    moved = float(exact.resolve(shift, _saturated(), 310.0)["f_pr"])
    assert moved == pytest.approx(float(plain.resolve(shift, _saturated(), 310.0)["f_pr"]) + 0.01)
