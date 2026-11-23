"""The command-line flags that select a UDE correction.

scripts/_bootstrap.py repeats the option names so that ``--help`` does not have
to import JAX; the first test keeps that copy honest.
"""

from __future__ import annotations

import argparse
import math

import _bootstrap
import pytest

from adm1_package.utils import corrections
from adm1_package.utils.corrections import DEFAULT, CorrectionOptions


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _bootstrap.add_correction_arguments(parser)
    return parser.parse_args(list(argv))


def _resolve(recipe, *argv):
    return CorrectionOptions.from_dict(_bootstrap.resolve_correction(_parse(*argv), recipe))


def test_the_cli_names_the_same_options_as_the_module():
    assert _bootstrap._PHYSICS == corrections.PHYSICS


def test_no_flags_and_no_recipe_gives_the_current_default():
    assert _resolve(None) == DEFAULT


def test_no_flags_keeps_the_recipe():
    recipe = CorrectionOptions("calibrated", prune=False, physics=("arrhenius",)).to_dict()
    assert _resolve(recipe) == CorrectionOptions.from_dict(recipe)


def test_no_prune_wins():
    assert _resolve(None, "--no-prune").prune is False
    assert _resolve(CorrectionOptions(prune=False).to_dict()).prune is False


def test_physics_flags_replace_the_recipes():
    recipe = CorrectionOptions(physics=("tan_inflow",)).to_dict()
    assert _resolve(recipe, "--physics", "arrhenius").physics == ("arrhenius",)


def test_an_empty_physics_flag_clears_the_recipes():
    recipe = CorrectionOptions(physics=("arrhenius",)).to_dict()
    assert _resolve(recipe, "--physics").physics == ()


def test_the_recipes_reference_overrides_survive_a_flag():
    recipe = CorrectionOptions(references=(("f_pr", 0.7),)).to_dict()
    assert dict(_resolve(recipe, "--physics", "arrhenius").references) == {"f_pr": 0.7}


def test_the_recipes_narrow_options_survive_a_flag():
    """An experiment that names its corrected parameters must not lose them to --physics."""
    recipe = CorrectionOptions("calibrated", corrected=("f_ch", "k_hyd_ch"), bound=0.7).to_dict()
    resolved = _resolve(recipe, "--physics", "arrhenius")
    assert resolved.corrected == ("f_ch", "k_hyd_ch") and resolved.bound == pytest.approx(0.7)
    assert _resolve(recipe) == CorrectionOptions.from_dict(recipe)


def test_unknown_physics_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        _parse("--physics", "magic")


def test_the_banner_names_a_narrow_correction():
    options = CorrectionOptions("calibrated", corrected=("k_hyd_ch", "f_ch"), bound=math.log(2.0),
                                exact_anchors=True)
    text = _bootstrap.describe_correction("r4", options.to_dict())
    assert "corrects only f_ch, k_hyd_ch" in text and "rates within x2" in text
    assert "exact anchors" in text
    assert "exact anchors" not in _bootstrap.describe_correction("r4", CorrectionOptions().to_dict())


@pytest.mark.parametrize("model_key", ["r2", "r3", "r4"])
def test_the_banner_names_what_a_run_uses(model_key):
    text = _bootstrap.describe_correction(
        model_key, CorrectionOptions(physics=("nh3_inhibition",)).to_dict()
    )
    assert "pruned" in text and "network outputs" in text
    if model_key == "r4":
        # R4 has no inhibition term; the banner says so rather than failing.
        assert "does not apply" in text
