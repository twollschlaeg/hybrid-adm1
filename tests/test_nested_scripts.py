"""The scripts that run a nested search on a cluster, and train its selections.

Python, sbatch and the environment are stubbed: what is checked is what the scripts
run, with which arguments, in which array task -- the part that otherwise only shows
once the jobs have waited in the queue.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import bash_works, stub_repository

pytestmark = pytest.mark.skipif(not bash_works(), reason="no working bash")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


PYTHON_STUB = r"""#!/bin/bash
# Stands in for Python: reads fold counts from the split files, records what the
# scripts run, passes every check, and lays out what make_nested.py would write.
if [[ "${1:-}" == "-c" ]]; then
    case "$2" in
        *k_folds*) sed -n 's/.*"k_folds": *\([0-9]*\).*/\1/p' "$3" | head -1 ;;
        *check_experiments*) echo "check $3 $4 $(basename "$5")" >> "$FAKE_LOG" ;;
    esac
    exit 0
fi
# Arguments joined by "|", since the paths may hold spaces.
case "${1:-}" in
    *.py) (IFS="|"; echo "$(basename "$1")|${*:2}") >> "$FAKE_LOG" ;;
esac
if [[ "${1:-}" == *make_nested.py ]]; then
    output=""
    while [[ $# -gt 0 ]]; do [[ "$1" == --output ]] && output="$2"; shift; done
    for k in 1 2 3 4 5; do
        mkdir -p "$output/outer_$k/benchmarks"
        echo '{"k_folds": 4}' > "$output/outer_$k/split.json"
    done
fi
exit 0
"""


@pytest.fixture
def cluster(tmp_path):
    """The nested-search scripts in a repository of their own, Python and sbatch stubbed."""
    hpc = ["submit_nested.sh", "slurm_nested.sh", "slurm_nested_collect.sh",
           "submit_experiments.sh", "slurm_experiments.sh"]
    root, run = stub_repository(
        tmp_path,
        copies=[f"scripts/hpc/{name}" for name in hpc]
        + ["scripts/run_experiment.sh", "scripts/run_experiments.sh"],
        files={"scripts/hpc/job_tmpdir.sh": "# nothing to move to scratch here\n",
               "scripts/hpc/activate_env.sh": "# nothing to activate here\n",
               "configs/experiments/fold_fast_narrow_tan.json": "{}\n",
               "configs/experiments/fold_fast_narrow_tan_no_ph.json": "{}\n",
               "configs/split.json": '{"k_folds": 5}\n',
               "runs/benchmarks-smoothstep-fast/split.json": '{"k_folds": 5}\n'},
        stubs={"python": PYTHON_STUB,
               "sbatch": '#!/bin/bash\necho "sbatch $*" >> "$FAKE_LOG"\necho 12345\n'},
        # Git's bash on Windows, started from Python, expands braces in its arguments,
        # so {fold} would arrive as fold; Linux passes them through as they are.
        environment={"MSYS": "noglob"})
    return SimpleNamespace(root=root, run=run)


def _option(call: str, name: str) -> str:
    """The value after option ``name`` in a recorded call."""
    words = call.split("|")
    return words[words.index(name) + 1]


# ---------------------------------------------------------------------------
# Submitting the search
# ---------------------------------------------------------------------------
def test_a_nested_submission_prepares_and_checks_every_outer_fold_first(cluster):
    result, calls = cluster.run("scripts/hpc/submit_nested.sh", "--model", "r4", "--name", "nested-x",
                                "--tasks-per-fold", "3", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    prepare = [call for call in calls if call.startswith("make_nested.py|")]
    assert len(prepare) == 1
    assert _option(prepare[0], "--benchmarks").endswith("runs/benchmarks-smoothstep-fast")
    assert _option(prepare[0], "--output") == "runs/nested-smoothstep-fast"
    assert sorted(call for call in calls if call.startswith("check")) == \
        [f"check fold_fast_narrow_tan r4 outer_{k}" for k in range(1, 6)]
    assert "15 tasks (0-14), 3 per outer fold" in result.stdout
    assert "--array=0-14" in result.stdout and "sbatch" not in "".join(calls)
    assert "tuning_nested-x_outer{fold}" in result.stdout  # the final training's pattern


@pytest.mark.parametrize("model", ["r2", "r3"])
def test_r2_and_r3_search_the_recipe_that_does_not_fit_ph(cluster, model):
    result, calls = cluster.run("scripts/hpc/submit_nested.sh", "--model", model, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"check fold_fast_narrow_tan_no_ph {model} outer_1" in calls


@pytest.mark.parametrize("arguments, message", [
    ((), "--model is required"),
    (("--model", "r5"), "unknown model"),
    (("--model", "r4", "--trials", "0"), "positive whole number"),
    (("--model", "r4", "--name", "a b"), "--name"),
    (("--model", "r4", "--outer-folds", "1 6"), "outer fold '6' does not exist"),
    (("--model", "r4", "--benchmarks", "runs/none"), "no per-fold calibrations"),
])
def test_a_request_that_cannot_run_is_refused_before_anything_is_queued(cluster, arguments, message):
    result, calls = cluster.run("scripts/hpc/submit_nested.sh", *arguments, "--dry-run")
    assert result.returncode != 0 and message in result.stderr, result.stdout + result.stderr
    assert not any(call.startswith("sbatch") for call in calls)


def test_the_collection_waits_for_the_array_and_names_the_searches_after_it(cluster):
    result, calls = cluster.run("scripts/hpc/submit_nested.sh", "--model", "r4", "--outer-folds", "2 4")
    assert result.returncode == 0, result.stdout + result.stderr
    array, collect = [call for call in calls if call.startswith("sbatch")]
    assert "--array=0-3" in array and array.endswith("scripts/hpc/slurm_nested.sh")
    assert "--dependency=afterany:12345" in collect and collect.endswith("slurm_nested_collect.sh")
    assert "ADM1_NESTED_NAME=nested-12345" in collect
    assert "runs/{model}/tuning_nested-12345_outer{fold}/best_hyperparameters.json" in result.stdout


# ---------------------------------------------------------------------------
# The array tasks and the collection
# ---------------------------------------------------------------------------
def _nested_layout(root: Path) -> Path:
    nested = root / "runs" / "nested-fast"
    for k in range(1, 6):
        (nested / f"outer_{k}" / "benchmarks").mkdir(parents=True)
        _write(nested / f"outer_{k}" / "split.json", '{"k_folds": 4}\n')
    return nested


def test_the_tasks_of_one_outer_fold_share_its_study_and_only_one_starts_it(cluster):
    nested = _nested_layout(cluster.root)
    searches = {}
    for task in range(10):
        result, calls = cluster.run("scripts/hpc/slurm_nested.sh", ADM1_MODEL="r4",
                                    ADM1_EXPERIMENT="configs/experiments/fold_fast_narrow_tan.json",
                                    ADM1_NESTED_DIR=str(nested), ADM1_OUTER_FOLDS="1 2 3 4 5",
                                    SLURM_ARRAY_TASK_ID=str(task), SLURM_ARRAY_JOB_ID="7",
                                    SLURM_CPUS_PER_TASK="16")
        assert result.returncode == 0, result.stdout + result.stderr
        (call,) = [call for call in calls if call.startswith("tune_hyperparameters.py|")]
        outer = task % 5 + 1
        assert Path(_option(call, "--split")) == nested / f"outer_{outer}" / "split.json"
        assert Path(_option(call, "--benchmarks")) == nested / f"outer_{outer}" / "benchmarks"
        assert _option(call, "--study-name") == f"nested_r4_outer{outer}"
        assert "--start-from-defaults" in call.split("|")
        # four inner folds per trial, four trials at once on sixteen cores
        assert _option(call, "--parallel-trials") == "4" and _option(call, "--fold-workers") == "16"
        searches.setdefault(_option(call, "--tuning-name"), []).append(int(_option(call, "--worker-id")))
    assert searches == {f"nested-7_outer{k}": [0, 1] for k in range(1, 6)}


def test_the_collection_selects_from_every_outer_folds_study_with_its_epochs(cluster):
    nested = _nested_layout(cluster.root)
    result, calls = cluster.run("scripts/hpc/slurm_nested_collect.sh", ADM1_MODEL="r3",
                                ADM1_EXPERIMENT="configs/experiments/fold_fast_narrow_tan_no_ph.json",
                                ADM1_NESTED_DIR=str(nested), ADM1_NESTED_NAME="nested-7",
                                ADM1_OUTER_FOLDS="1 2 3 4 5", EPOCHS="150")
    assert result.returncode == 0, result.stdout + result.stderr
    collected = [call for call in calls if call.startswith("tune_hyperparameters.py|")]
    assert [_option(call, "--tuning-name") for call in collected] == [f"nested-7_outer{k}" for k in range(1, 6)]
    assert all(_option(call, "--n-trials") == "0" and _option(call, "--epochs") == "150"
               and _option(call, "--model") == "r3" for call in collected)
    (summary,) = [call for call in calls if call.startswith("nested_summary.py|")]
    assert _option(summary, "--pattern") == "runs/{model}/tuning_nested-7_outer{fold}/best_hyperparameters.json"


# ---------------------------------------------------------------------------
# Training the selections
# ---------------------------------------------------------------------------
PATTERN = "runs/{model}/tuning_nested-x_outer{fold}/best_hyperparameters.json"


def _selections(root: Path, model: str = "r4", folds=range(1, 6)) -> None:
    for fold in folds:
        _write(root / PATTERN.replace("{model}", model).replace("{fold}", str(fold)), "{}\n")


def test_the_final_training_is_submitted_with_every_folds_selection(cluster):
    _selections(cluster.root)
    result, _ = cluster.run("scripts/hpc/submit_experiments.sh", "--experiments", "fold_fast_narrow_tan",
                            "--models", "r4", "--seeds", "45 46 47", "--fold-hyperparameters", PATTERN,
                            "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "per fold: " in result.stdout and "tuning_nested-x_outer{fold}" in result.stdout


@pytest.mark.parametrize("script", ["scripts/hpc/submit_experiments.sh", "scripts/run_experiments.sh"])
def test_a_missing_selection_is_refused_before_anything_runs(cluster, script):
    _selections(cluster.root, folds=[1, 2, 3, 5])
    result, calls = cluster.run(script, "--experiments", "fold_fast_narrow_tan", "--models", "r4",
                                "--fold-hyperparameters", PATTERN,
                                *(["--dry-run"] if "submit" in script else []))
    assert result.returncode != 0 and "outer4" in result.stderr, result.stdout + result.stderr
    assert not any(call.startswith(("sbatch", "train_cv.py|")) for call in calls)


def test_each_fold_trains_with_its_selection(cluster):
    result, calls = cluster.run("scripts/run_experiment.sh", "fold_fast_narrow_tan", "r4", "tuned",
                                FOLD_HYPERPARAMETERS=PATTERN, SEED="45")
    assert result.returncode == 0, result.stdout + result.stderr
    (train,) = [call for call in calls if call.startswith("train_cv.py|")]
    assert _option(train, "--fold-hyperparameters") == PATTERN and _option(train, "--seed") == "45"
