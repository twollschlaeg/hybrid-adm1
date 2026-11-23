"""The full ADM1's calibration (scripts/calibrate/base_calibration.py).

It runs the two stages of calibrate_benchmarks.py after R4, R3 and R2: it keeps every
parameter it shares with them as their calibration left it, fits only the Monod kinetics
of its own processes and the biology of its starts, and writes nothing else.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from helpers import load_script

base_calibration = load_script("calibrate/base_calibration.py")

pytestmark = pytest.mark.requires_data

CAMPAIGN = "rice-bran"         # three reactors once the replicates are averaged: the quickest
PARAMETERS_FILE = f"{CAMPAIGN}-parameters.json"
QUICK = ["--de-maxiter", "1", "--de-popsize", "1", "--local-maxiter", "2"]


def _own(name):
    """Whether the calibration may write this file of the campaign."""
    return name == PARAMETERS_FILE or name.lower().startswith("base-ivs_")


def _copy_campaign(data_dir, root, forcing="smoothstep"):
    """The campaign, its reduced variants calibrated to ``forcing`` (``None``: before the
    forcing was recorded, the spline)."""
    campaign = root / CAMPAIGN
    campaign.mkdir(parents=True)
    for path in (Path(data_dir) / CAMPAIGN).iterdir():
        if path.is_file() and path.suffix in (".csv", ".json"):
            shutil.copy2(path, campaign / path.name)
    if forcing is not None:
        (campaign / "calibration.json").write_text(json.dumps({"forcing": forcing}),
                                                   encoding="utf-8")
    else:
        (campaign / "calibration.json").unlink(missing_ok=True)
    return campaign


def _files(campaign):
    return {path.name: path.read_bytes() for path in sorted(campaign.iterdir()) if path.is_file()}


@pytest.fixture(scope="module")
def calibrated(data_dir, tmp_path_factory):
    """A quick run, one generation and two local iterations, on a copy of the campaign."""
    root = tmp_path_factory.mktemp("data")
    campaign = _copy_campaign(data_dir, root)
    before = _files(campaign)
    base_calibration.main(["--campaign", CAMPAIGN, "--data-dir", str(root), *QUICK])
    return campaign, before, _files(campaign)


def test_only_its_own_parameters_change(calibrated):
    _, before, after = calibrated
    old, new = json.loads(before[PARAMETERS_FILE]), json.loads(after[PARAMETERS_FILE])
    assert old.keys() == new.keys()
    changed = {name for name in old if old[name] != new[name]}
    assert changed <= set(base_calibration.two_stage.PARAMETERS["base"])


def test_it_writes_only_its_parameters_and_its_starts(calibrated):
    # the inputs, the measurements and the reduced variants' starts stay as they are
    _, before, after = calibrated
    assert after.keys() == before.keys()
    written = {name for name in after if after[name] != before[name]}
    assert all(_own(name) for name in written), written


def _report(campaign):
    [report] = (campaign / "calibration-backup").glob("*/report.json")
    return json.loads(report.read_text())["variants"]["base"]


def test_the_written_starts_are_the_fitted_ones(calibrated):
    campaign = calibrated[0]
    starts = _report(campaign)["initial_values"]
    for label, fitted in starts.items():
        stored = pd.read_csv(campaign / f"base-IVs_{label.split('_', 1)[1]}.csv",
                             index_col=0).iloc[:, 0]
        for name, (_, after) in fitted.items():
            assert float(stored[name]) == pytest.approx(after, rel=1e-12), (label, name)


def test_no_substrate_starts_above_what_its_feed_builds_up(calibrated):
    # the products of hydrolysis are capped at their first-week steady state like the
    # particulates: the stored files held up to 3.5 g/L of sugars
    report = _report(calibrated[0])
    for label, caps in report["particulate caps"].items():
        assert set(caps) == {"X_ch", "X_pr", "X_li", "S_su", "S_aa", "S_fa"}, label
        for name, cap in caps.items():
            assert report["initial_values"][label][name][1] <= cap, (label, name)


def test_what_it_replaces_is_backed_up(calibrated):
    campaign, before, _ = calibrated
    [folder] = (campaign / "calibration-backup").iterdir()
    backed_up = {path.name: path.read_bytes() for path in folder.iterdir()
                 if path.name != "report.json"}
    assert backed_up == {name: content for name, content in before.items() if _own(name)}


def test_it_keeps_no_parameters_fitted_to_another_forcing(data_dir, tmp_path, capsys):
    """The reduced variants' parameters, which it keeps, were fitted to the spline: the
    full ADM1 waits for their calibration on the smoothstep."""
    campaign = _copy_campaign(data_dir, tmp_path, forcing=None)
    before = _files(campaign)
    with pytest.raises(SystemExit):
        base_calibration.main(["--campaign", CAMPAIGN, "--data-dir", str(tmp_path), *QUICK])
    assert "spline" in capsys.readouterr().err
    assert _files(campaign) == before


def test_a_dry_run_writes_nothing(data_dir, tmp_path):
    campaign = _copy_campaign(data_dir, tmp_path)
    before = _files(campaign)
    base_calibration.main(["--campaign", CAMPAIGN, "--data-dir", str(tmp_path), "--dry-run",
                           *QUICK])
    assert _files(campaign) == before
    assert not (campaign / "calibration-backup").exists()


@pytest.mark.parametrize("key", ["r4", "r3", "r2"])
def test_no_reduced_variant_reads_what_it_fits(key, installed_dataset):
    # their datasets hand the models only the parameters they read
    x, _ = installed_dataset(key, [CAMPAIGN])[0]
    assert not set(x["parameters"]) & set(base_calibration.two_stage.PARAMETERS["base"])
