"""Shared fixtures.

Tests that need the experimental reactor data are marked ``requires_data`` and skip
automatically when the data are not present, so the suite is meaningful in a clone of the
public repository. The data are read from ``$ADM1_DATA_DIR``, else ``data/``.

A few tests read the paper's stored runs and per-fold calibrations, which are not
published either, and skip without them. They are read from ``$ADM1_STORED_RUNS``, else
from this checkout's ``runs/``, else, in a linked git worktree, the main working tree's.
The tests only read there. A test that writes copies to ``tmp_path`` first.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
from helpers import ROOT, SCRIPTS, first_days

# The scripts (scripts/*.py) import as modules, and adm1_package from this checkout.
for _path in (SCRIPTS, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Keep JAX on CPU and in double precision for the tests.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

#: Campaign directories the model datasets need.
REQUIRED_CAMPAIGNS = ("LaRA-temp", "LaRA-substrate", "rice-bran")
#: A run of the paper, its per-fold calibrations, under the stored runs.
PAPER_R4 = Path("r4") / "fold_fast_narrow_tan_seed45_final"
FAST_BENCHMARKS = Path("benchmarks-smoothstep-fast")


def _data_dir() -> Path:
    return Path(os.environ.get("ADM1_DATA_DIR", ROOT / "data"))


def pytest_collection_modifyitems(config, items):
    data_dir = _data_dir()
    available = all((data_dir / campaign).is_dir() for campaign in REQUIRED_CAMPAIGNS)
    if available:
        return
    skip = pytest.mark.skip(
        reason=f"experimental data not found under {data_dir} (see data/README.md)"
    )
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return _data_dir()


@pytest.fixture(scope="session", autouse=True)
def _configure_jax():
    import jax

    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# The stored runs, read only
# ---------------------------------------------------------------------------
def _stored_runs() -> Path | None:
    configured = os.environ.get("ADM1_STORED_RUNS")
    if configured:
        return Path(configured)
    candidates = [ROOT / "runs"]
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                cwd=ROOT, capture_output=True, encoding="utf-8", check=True).stdout
        candidates.append(Path(common.strip()).parent / "runs")
    except (OSError, subprocess.CalledProcessError):
        pass
    return next((runs for runs in candidates
                 if (runs / PAPER_R4).is_dir() or (runs / FAST_BENCHMARKS).is_dir()), None)


def _stored(relative: Path) -> Path:
    runs = _stored_runs()
    if runs is None or not (runs / relative).is_dir():
        pytest.skip(f"no stored {relative.as_posix()} (set ADM1_STORED_RUNS to the runs/ "
                    "that holds the paper's runs)")
    return runs / relative


@pytest.fixture(scope="session")
def paper_r4_run() -> Path:
    """Seed 45 of the paper's R4 TAN arm (read only)."""
    return _stored(PAPER_R4)


@pytest.fixture(scope="session")
def fast_benchmarks() -> Path:
    """The paper's per-fold fast-hydrolysis calibrations, held-out starts fitted
    (read only)."""
    return _stored(FAST_BENCHMARKS)


# ---------------------------------------------------------------------------
# The installed data
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _dataset(data_dir: Path, key: str, campaigns: tuple[str, ...] | None, diagnostics: bool):
    from adm1_package.experiments.registry import get_spec

    options = {"diagnostics": True} if diagnostics else {}
    if campaigns is not None:
        options["data_to_use"] = list(campaigns)
    return get_spec(key).dataset(data_dir, **options)


@pytest.fixture(scope="session")
def installed_dataset(data_dir):
    """``dataset(key, campaigns=None, diagnostics=False)``: a variant's dataset of the
    installed data, built once per session and shared, so a test must not change it.
    Mark a test that uses it ``requires_data``."""
    def dataset(key, campaigns=None, diagnostics=False):
        return _dataset(Path(data_dir), key, tuple(campaigns) if campaigns else None, diagnostics)

    return dataset


@pytest.fixture(scope="session")
def short_r4_samples(installed_dataset):
    """Two R4 reactors' first 40 days: ``(dataset, [(x, y), (x, y)])``.

    Long enough for ten days past a 30-day spin-up, short enough that a loss
    evaluation takes seconds. Mark a test that uses it ``requires_data``.
    """
    import numpy as np

    dataset = installed_dataset("r4")
    samples = []
    for label in ("LaRA-temp_F4", "LaRA-substrate_F2"):
        x, y = dataset[dataset.labels.index(label)]
        knots = int(np.searchsorted(np.asarray(x["input"]["time"]), 40.0)) + 1
        samples.append(({**x, "input": first_days(x["input"], 40.0)},
                        {**y, "values": y["values"][:knots], "indices": y["indices"][:knots]}))
    return dataset, samples


@pytest.fixture(scope="session")
def short_r4_batch(short_r4_samples):
    """:func:`short_r4_samples` as one batch: ``(dataset, inputs, targets)``."""
    from adm1_package.utils.datasets import custom_collate_fn

    dataset, samples = short_r4_samples
    inputs, targets = custom_collate_fn(samples)
    return dataset, inputs, targets


@pytest.fixture
def r4_trainer(short_r4_samples, tmp_path):
    """``build(correction=DEFAULT, **overrides)``: a small R4 trainer sized for
    :func:`short_r4_samples`, starting at zero output, logging to ``tmp_path/fold_1``."""
    import jax

    from adm1_package.experiments.registry import get_spec
    from adm1_package.utils.corrections import DEFAULT

    spec = get_spec("r4")
    dataset, _ = short_r4_samples

    def build(correction=DEFAULT, **overrides):
        settings = dict(
            in_size=spec.nn_in_size(dataset), width_size=4,
            out_size=spec.correction_layout(correction).size, depth=2,
            activation=jax.nn.tanh, use_final_bias=True, zero_output=True, batch_size=2,
            ude=spec.ude(correction), objective_weights=dict(spec.objective_weights),
            objective_list=spec.objective_list, spin_up_days=30.0,
            correction=correction.to_dict(), optimizer_hparams={"optimizer": "adam", "lr": 1e-3},
            logger_params={"base_log_dir": str(tmp_path), "log_dir": "fold_1"},
            enable_progress_bar=False, verbose=False,
        )
        settings.update(overrides)
        return spec.trainer_class()(**settings)

    return build


@pytest.fixture
def installed_benchmarks(data_dir):
    """``build(root, split, folds)``: per-fold benchmarks that are the installed
    calibration, copied from the data. Mark a test that uses it ``requires_data``."""
    from adm1_package.experiments.calibrations import installed_forcing
    from adm1_package.experiments.registry import DATASETS

    def build(root: Path, split, folds) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "split.json").write_text(json.dumps(split.to_dict()), encoding="utf-8")
        # the installed calibration's model: without Arrhenius, within the stored bounds,
        # fitted to the forcing it records
        (root / "model.json").write_text(json.dumps({
            "arrhenius": False, "bounds": {}, "forcing": installed_forcing(data_dir)}),
            encoding="utf-8")
        for fold in folds:
            for campaign in DATASETS:
                target = root / f"fold_{fold}" / "data" / campaign
                target.mkdir(parents=True)
                for pattern in (f"{campaign}-parameters.json", "input_*.csv", "output_*.csv",
                                "R4-IVs_*.csv", "R3-IVs_*.csv", "R2-IVs_*.csv"):
                    for path in (Path(data_dir) / campaign).glob(pattern):
                        shutil.copy2(path, target / path.name)
                (root / f"fold_{fold}" / f"{campaign}.json").write_text(
                    json.dumps({"status": "installed"}), encoding="utf-8")
        return root

    return build


# ---------------------------------------------------------------------------
# Searches without training
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_training(monkeypatch):
    """Every fold a search trains is scored without training: the closer its learning
    rate to 2e-3, the better. Returns the folds trained, as the arguments
    :func:`~adm1_package.experiments.training.train_fold` was given."""
    from adm1_package.experiments import training, tuning

    trained = []

    def train_fold(**arguments):
        trained.append(arguments)
        recipe = arguments["hyperparameters"]
        return training.FoldResult(
            fold=arguments["fold"], train_loss=0.0,
            val_loss=abs(math.log(recipe.learning_rate / 2e-3)), train_labels=[],
            val_labels=[], checkpoint_dir="", n_parameters=0, hyperparameters=recipe.to_dict())

    monkeypatch.setattr(tuning, "train_fold", train_fold)
    return trained
