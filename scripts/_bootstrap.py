"""Shared start-up for the command-line scripts.

Importing this module makes the repository importable when a script is run
directly (``python scripts/train_cv.py``) without an editable install, and puts
JAX into the configuration the models assume: CPU by default and float64
throughout, because the ADM1 right-hand sides are stiff and lose accuracy in
single precision.

Set ``ADM1_PLATFORM=gpu`` to override the device selection.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Must happen before JAX is imported anywhere.
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("ADM1_PLATFORM", "cpu"))


def configure_jax() -> None:
    """Enable double precision. Call once, at the start of a script."""
    import jax

    jax.config.update("jax_enable_x64", True)


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments every experiment script understands."""
    from adm1_package.experiments.config import data_dir, run_dir
    from adm1_package.experiments.splits import default_split_path

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=data_dir(),
        help="Directory holding the reactor data (default: $ADM1_DATA_DIR or ./data).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=run_dir(),
        help="Where run artifacts are written (default: $ADM1_RUN_DIR or ./runs).",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=default_split_path(),
        help="Split file to use (default: configs/split.json).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser


# Kept in sync with adm1_package.utils.corrections; duplicated so that --help
# does not import JAX. tests/test_corrections_cli.py checks they agree.
_PHYSICS = ("arrhenius", "nh3_inhibition", "tan_inflow")


def add_correction_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """How the network's outputs become model parameters.

    See adm1_package/utils/corrections.py for what each option does and why.
    Every flag is optional: whatever is not given comes from the hyperparameter
    file, or from the current default if there is none.
    """
    group = parser.add_argument_group(
        "UDE correction", "How network outputs become model parameters "
        "(adm1_package/utils/corrections.py)."
    )
    group.add_argument(
        "--no-prune",
        action="store_true",
        help="Let the network correct every parameter, including those a sensitivity "
             "probe found to have no effect (saturated hydrolysis, zero-weight VFAs, "
             "dead decay rates).",
    )
    group.add_argument(
        "--physics",
        nargs="*",
        choices=_PHYSICS,
        default=None,
        help="Additional physics: arrhenius (temperature dependence of the rates), "
             "nh3_inhibition (network-corrected K_I_nh3; R2/R3 only), tan_inflow "
             "(network-corrected influent TAN). Give none of them to clear any set by "
             "the hyperparameter file.",
    )
    return parser


def resolve_correction(arguments: argparse.Namespace, recipe: dict | None) -> dict:
    """Combine the hyperparameter file's correction with the command-line flags.

    Flags override the file field by field; what no flag sets, such as a narrow
    correction's ``corrected`` and ``bound``, is kept.
    """
    from dataclasses import replace

    from adm1_package.utils.corrections import CorrectionOptions

    base = CorrectionOptions.from_dict(recipe)
    prune = False if arguments.no_prune else base.prune
    physics = tuple(arguments.physics) if arguments.physics is not None else base.physics
    return replace(base, prune=prune, physics=physics).to_dict()


def add_experiment_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """``--experiment FILE``: what the model is, on top of the training recipe."""
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Experiment file (configs/experiments/*.json): the network split, the anchors, "
             "the initial state, the spin-up window and the correction. Applied over the "
             "hyperparameters; the correction flags apply over both.",
    )
    return parser


def describe_model(hyperparameters) -> str:
    """One line saying what network a run trains and how it is scored."""
    network = "split (factors from the current feed)" if hyperparameters.split_network else "single"
    text = (f"Network: {network}; initial state: {hyperparameters.initial_state}; scored from "
            f"day {hyperparameters.spin_up_days:g}")
    if hyperparameters.objective_weights:
        changed = ", ".join(f"{name} {value:g}"
                            for name, value in hyperparameters.objective_weights.items())
        text += f"; loss weights: {changed}"
    text += "; built on each fold's per-fold calibration"
    text += "".join(f" fitted with {name} in [{low:g}, {high:g}]"
                    for name, (low, high) in sorted(hyperparameters.calibration_bounds.items()))
    if hyperparameters.correction_penalty > 0:
        text += f"; correction penalty {hyperparameters.correction_penalty:g}"
    if hyperparameters.keep_checkpoints:
        text += "; a checkpoint at every validation"
    return text


def describe_correction(model_key: str, correction: dict) -> str:
    """One line saying which equations a run uses, and what that costs."""
    from adm1_package.experiments.registry import get_spec

    spec = get_spec(model_key)
    layout = spec.correction_layout(correction)
    options = layout.options
    physics = ", ".join(layout.physics) or "none"
    if options.corrected is not None:
        outputs = f"corrects only {', '.join(layout.outputs)}"
    else:
        outputs = "pruned" if options.prune else "unpruned"
    text = (
        f"UDE correction: {outputs}"
        f"; physics: {physics}; network outputs: {layout.size} "
        f"(original model: {spec.nn_out_size})"
    )
    if options.bound is not None:
        text += (f"; changes bounded at +-{options.bound:.3g} "
                 f"(rates within x{math.exp(options.bound):.3g})")
    if options.exact_anchors:
        text += "; exact anchors"
    if layout.skipped_physics:
        text += f"\n  note: {', '.join(layout.skipped_physics)} does not apply to {spec.display_name}"
    return text
