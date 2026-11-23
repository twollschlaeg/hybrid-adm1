"""The per-fold benchmarks (scripts/calibrate/calibrate_folds.py).

Each fold's benchmark is calibrated as the installed one is, but without the reactors
the fold holds out, on a copy of the data: the hybrid's fair comparator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import ROOT, canonical_split, load_script

from adm1_package.experiments.registry import DATASETS
from adm1_package.experiments.splits import DataSplit

calibrate_folds = load_script("calibrate/calibrate_folds.py")
SPLIT = canonical_split()


def _fake_data(root: Path, forcing: str | None = "smoothstep") -> Path:
    """Every campaign's directory with one reactor's worth of placeholder files, calibrated
    with ``forcing`` (``None``: before the forcing was recorded, the spline)."""
    for campaign in DATASETS:
        directory = root / campaign
        directory.mkdir(parents=True)
        for name in (f"{campaign}-parameters.json", "input_X.csv", "output_X.csv",
                     "R4-IVs_X.csv", "R3-IVs_X.csv", "R2-IVs_X.csv", "base-IVs_X.csv",
                     "notes.xlsx"):
            (directory / name).write_text(name, encoding="utf-8")
        if forcing is not None:
            (directory / "calibration.json").write_text(json.dumps({"forcing": forcing}),
                                                        encoding="utf-8")
        (directory / "calibration-backup").mkdir()
    return root


def _calibrations(commands):
    """The calibrations among the commands run, without the fits of held-out starts."""
    return [command for command in commands if Path(command[1]).name == "calibrate_benchmarks.py"]


def _main(tmp_path, *arguments, fold="3", data=None, output="out", runner=None):
    """calibrate_folds.py on fold ``fold`` into ``tmp_path/output``, from the fake data
    ``data`` (made in ``tmp_path/data`` if not given): ``(exit code, commands run)``.
    ``runner`` decides each command's exit code (default: every one succeeds)."""
    commands = []

    def record(command, log, env):
        commands.append(command)
        return runner(command) if runner else 0

    data = data if data is not None else _fake_data(tmp_path / "data")
    code = calibrate_folds.main(["--folds", fold, "--output", str(tmp_path / output),
                                 "--data-dir", str(data),
                                 "--split", str(ROOT / "configs" / "split.json"), *arguments],
                                runner=record)
    return code, commands


def test_each_fold_calibrates_each_campaign_without_the_reactors_it_holds_out():
    jobs = calibrate_folds.plan(SPLIT, range(1, 6), DATASETS)
    assert len(jobs) == 5 * len(DATASETS)
    for job in jobs:
        assert job.held_out == tuple(sorted(
            label for label in SPLIT.validation_labels(job.fold)
            if label.startswith(job.campaign + "_")))


def test_the_splits_test_reactors_are_left_out_of_every_fold():
    payload = SPLIT.to_dict()
    payload["folds"]["fold_3"].remove("rice-bran_R30")
    payload["test"] = ["rice-bran_R30"]
    (job,) = calibrate_folds.plan(DataSplit.from_dict(payload), [1], ["rice-bran"])
    assert job.held_out == ("rice-bran_R30",)


def test_a_campaign_a_fold_holds_nothing_out_of_keeps_the_installed_calibration():
    # Folds 1 and 2 hold out no rice-bran reactor: calibrated on all three, which the
    # installed calibration already is.
    (job,) = calibrate_folds.plan(SPLIT, [1], ["rice-bran"])
    assert job.held_out == () and not job.calibrates


def test_only_the_files_the_datasets_read_are_copied(tmp_path):
    data = _fake_data(tmp_path / "data")
    copy = calibrate_folds.copy_campaign(data, tmp_path / "copy", "rice-bran")
    assert sorted(path.name for path in copy.iterdir()) == sorted([
        "rice-bran-parameters.json", "input_X.csv", "output_X.csv",
        "R4-IVs_X.csv", "R3-IVs_X.csv", "R2-IVs_X.csv", "calibration.json"])


def test_the_chain_runs_on_the_folds_copy_as_the_installed_calibration_ran(tmp_path):
    (job,) = calibrate_folds.plan(SPLIT, [3], ["rice-bran"])
    command = calibrate_folds.calibration_command(job, tmp_path / "out", [])
    assert command[command.index("--campaign") + 1] == "rice-bran"
    assert Path(command[command.index("--data-dir") + 1]) == tmp_path / "out" / "fold_3" / "data"
    assert "--reset-particulates" in command
    assert command[command.index("--exclude") + 1:] == ["rice-bran_R30"]


def test_benchmarks_of_another_split_are_refused(tmp_path):
    calibrate_folds.prepare_output(tmp_path, SPLIT)
    calibrate_folds.prepare_output(tmp_path, SPLIT)  # the same split again is fine
    payload = SPLIT.to_dict()
    folds = payload["folds"]
    folds["fold_1"], folds["fold_2"] = folds["fold_2"], folds["fold_1"]
    with pytest.raises(ValueError, match="another split"):
        calibrate_folds.prepare_output(tmp_path, DataSplit.from_dict(payload))


def test_a_run_calibrates_what_is_missing_and_never_writes_data(tmp_path):
    data = _fake_data(tmp_path / "data")
    before = {path: path.read_bytes() for path in data.rglob("*") if path.is_file()}
    code, commands = _main(tmp_path, "--workers", "1", data=data)
    assert code == 0
    # fold 3 holds out LaRA-substrate_F6, LaRA-temp_F5 and rice-bran_R30
    assert sorted(command[command.index("--campaign") + 1]
                  for command in _calibrations(commands)) == sorted(DATASETS)
    for campaign in DATASETS:
        assert (tmp_path / "out" / "fold_3" / "data" / campaign / f"{campaign}-parameters.json").exists()
    record = json.loads((tmp_path / "out" / "fold_3" / "rice-bran.json").read_text(encoding="utf-8"))
    assert record["held_out"] == ["rice-bran_R30"] and record["status"] == "calibrated"

    assert _main(tmp_path, "--workers", "1", data=data) == (0, [])  # fold 3 is done
    assert {path: path.read_bytes() for path in data.rglob("*") if path.is_file()} == before


def test_the_folds_can_be_calibrated_with_arrhenius(tmp_path):
    """Into an output of its own: benchmarks with and without it must not mix."""
    code, commands = _main(tmp_path, "--campaigns", "rice-bran", "--arrhenius")
    assert code == 0
    calibrations = _calibrations(commands)
    assert calibrations and all("--arrhenius" in command for command in calibrations)
    assert calibrations[0].index("--arrhenius") < calibrations[0].index("--exclude")
    model = json.loads((tmp_path / "out" / "model.json").read_text(encoding="utf-8"))
    assert model == {"arrhenius": True, "bounds": {}, "forcing": "smoothstep"}


def test_with_arrhenius_a_campaign_with_nothing_held_out_is_calibrated_too(tmp_path):
    """The installed calibration is the fold's only for the model it was made for."""
    data = _fake_data(tmp_path / "data")
    # fold 1 holds no rice-bran reactor out: installed is the fold's
    assert _main(tmp_path, "--campaigns", "rice-bran", fold="1", data=data) == (0, [])
    code, (command,) = _main(tmp_path, "--campaigns", "rice-bran", "--arrhenius", fold="1",
                             data=data, output="warm")
    assert code == 0
    assert "--arrhenius" in command and "--exclude" not in command
    record = json.loads((tmp_path / "warm" / "fold_1" / "rice-bran.json").read_text(encoding="utf-8"))
    assert record["status"] == "calibrated" and record["held_out"] == []


def test_benchmarks_calibrated_with_another_model_are_refused(tmp_path):
    calibrate_folds.prepare_output(tmp_path / "with", SPLIT, arrhenius=True)
    calibrate_folds.prepare_output(tmp_path / "with", SPLIT, arrhenius=True)  # again is fine
    with pytest.raises(ValueError, match="Arrhenius"):
        calibrate_folds.prepare_output(tmp_path / "with", SPLIT, arrhenius=False)
    # benchmarks from before the option count as calibrated without it
    calibrate_folds.prepare_output(tmp_path / "old", SPLIT)
    (tmp_path / "old" / "model.json").unlink()
    with pytest.raises(ValueError, match="Arrhenius"):
        calibrate_folds.prepare_output(tmp_path / "old", SPLIT, arrhenius=True)
    # and on the spline forcing, which calibrations on the smoothstep do not join
    with pytest.raises(ValueError, match="forcing"):
        calibrate_folds.prepare_output(tmp_path / "old", SPLIT, arrhenius=False)


def test_the_folds_can_be_calibrated_within_bounds(tmp_path):
    """--bound passes on, and like --arrhenius it has every campaign calibrated: the
    installed calibration was fitted within the stored bounds."""
    # fold 1 holds no rice-bran reactor out, and it is calibrated
    code, (command,) = _main(tmp_path, "--campaigns", "rice-bran", "--bound", "k_hyd_ch", "0.1",
                             "2.88", fold="1")
    assert code == 0
    start = command.index("--bound")
    assert command[start:start + 4] == ["--bound", "k_hyd_ch", "0.1", "2.88"]
    model = json.loads((tmp_path / "out" / "model.json").read_text(encoding="utf-8"))
    assert model == {"arrhenius": False, "bounds": {"k_hyd_ch": [0.1, 2.88]},
                     "forcing": "smoothstep"}


def test_a_bound_the_calibration_would_refuse_is_refused_before_anything_is_written(tmp_path,
                                                                                     capsys):
    with pytest.raises(SystemExit):
        _main(tmp_path, "--campaigns", "rice-bran", "--bound", "k_hyd_cx", "0.1", "2.88", fold="1")
    error = capsys.readouterr().err
    assert "k_hyd_cx" in error and "unrecognized" not in error
    assert not (tmp_path / "out").exists()


def test_benchmarks_calibrated_within_other_bounds_are_refused(tmp_path):
    fast = {"k_hyd_ch": (0.1, 2.88)}
    calibrate_folds.prepare_output(tmp_path / "fast", SPLIT, arrhenius=True, bounds=fast)
    calibrate_folds.prepare_output(tmp_path / "fast", SPLIT, arrhenius=True, bounds=fast)
    for other in ({}, {"k_hyd_ch": (0.2, 2.88)}):
        with pytest.raises(ValueError, match="bounds"):
            calibrate_folds.prepare_output(tmp_path / "fast", SPLIT, arrhenius=True, bounds=other)
    # benchmarks from before the option were calibrated within the stored bounds
    calibrate_folds.prepare_output(tmp_path / "old", SPLIT, arrhenius=True)
    (tmp_path / "old" / "model.json").write_text(json.dumps({"arrhenius": True, "forcing": "smoothstep"}),
                                                 encoding="utf-8")
    with pytest.raises(ValueError, match="bounds"):
        calibrate_folds.prepare_output(tmp_path / "old", SPLIT, arrhenius=True, bounds=fast)
    calibrate_folds.prepare_output(tmp_path / "old", SPLIT, arrhenius=True)


def test_benchmarks_say_which_forcing_their_calibrations_were_fitted_to(tmp_path):
    calibrate_folds.prepare_output(tmp_path / "new", SPLIT)
    model = json.loads((tmp_path / "new" / "model.json").read_text(encoding="utf-8"))
    assert model["forcing"] == "smoothstep"
    # benchmarks from before the record were fitted to the spline
    (tmp_path / "new" / "model.json").write_text(json.dumps({"arrhenius": False, "bounds": {}}),
                                                 encoding="utf-8")
    with pytest.raises(ValueError, match="forcing"):
        calibrate_folds.prepare_output(tmp_path / "new", SPLIT)


def test_a_campaign_copied_as_installed_must_have_been_calibrated_to_the_same_forcing(tmp_path):
    """Fold 1 holds no rice-bran reactor out, so it takes the installed calibration, which
    is the fold's only if it was fitted to the forcing these benchmarks are."""
    spline = _fake_data(tmp_path / "spline", forcing=None)
    assert _main(tmp_path, "--campaigns", "rice-bran", fold="1", data=spline)[0] == 1
    assert not (tmp_path / "out" / "fold_1" / "rice-bran.json").exists()
    smooth = _fake_data(tmp_path / "smooth")
    assert _main(tmp_path, "--campaigns", "rice-bran", fold="1", data=smooth)[0] == 0
    record = json.loads((tmp_path / "out" / "fold_1" / "rice-bran.json").read_text(encoding="utf-8"))
    assert record["status"] == "installed"


def test_a_failed_calibration_is_reported_and_left_to_rerun(tmp_path):
    assert _main(tmp_path, "--campaigns", "rice-bran", runner=lambda command: 1)[0] == 1
    assert not (tmp_path / "out" / "fold_3" / "rice-bran.json").exists()


def test_a_calibrated_fold_fits_its_held_out_reactors_start_values(tmp_path):
    """With the fold's own parameters, on their first days only
    (fit_held_out_starts.py); a fresh calibration fits them afresh."""
    code, (calibration, fit) = _main(tmp_path, "--campaigns", "rice-bran")
    assert code == 0
    assert Path(calibration[1]).name == "calibrate_benchmarks.py"
    assert Path(fit[1]).name == "fit_held_out_starts.py"
    assert Path(fit[fit.index("--benchmarks") + 1]) == tmp_path / "out"
    assert fit[fit.index("--folds") + 1:fit.index("--folds") + 2] == ["3"]
    assert fit[fit.index("--campaigns") + 1:fit.index("--campaigns") + 2] == ["rice-bran"]
    assert "--force" in fit
    record = json.loads((tmp_path / "out" / "fold_3" / "rice-bran.json").read_text(encoding="utf-8"))
    assert record["held_out_starts"] == "fitted"


def test_a_fold_whose_held_out_starts_failed_is_left_to_rerun(tmp_path):
    def runner(command):
        return 1 if Path(command[1]).name == "fit_held_out_starts.py" else 0

    assert _main(tmp_path, "--campaigns", "rice-bran", runner=runner)[0] == 1
    assert not (tmp_path / "out" / "fold_3" / "rice-bran.json").exists()
