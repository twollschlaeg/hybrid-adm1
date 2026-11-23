"""The shell scripts that run the experiments on a cluster or server.

They are written on Windows and copied to Linux, where a carriage return breaks
them (``$'\\r': command not found``) only once they run there -- for a batch job,
after the queue wait.
"""

import subprocess
from types import SimpleNamespace

import pytest
from helpers import ROOT, bash_works, stub_repository

SCRIPTS = sorted((ROOT / "scripts").rglob("*.sh"))
needs_bash = pytest.mark.skipif(not bash_works(), reason="no working bash")


def test_the_experiment_scripts_are_found():
    names = {path.name for path in SCRIPTS}
    assert {"run_experiment.sh", "run_experiments.sh", "submit_experiments.sh",
            "slurm_experiments.sh", "slurm_combine.sh"} <= names


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_shell_scripts_have_unix_line_endings(script):
    assert b"\r" not in script.read_bytes()


@needs_bash
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_shell_scripts_parse(script):
    result = subprocess.run(["bash", "-n", script.name], cwd=script.parent,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# What the batch scripts run: each experiment and variant, once per seed
# ---------------------------------------------------------------------------
RUN_EXPERIMENT_STUB = """#!/bin/bash
# Records what it is asked to run and leaves the results a merge looks for.
echo "$1 $2 $3 SEED=${SEED:-}" >> "$FAKE_LOG"
mkdir -p "${ADM1_RUN_DIR:-$PWD/runs}/$2/$3/results"
touch "${ADM1_RUN_DIR:-$PWD/runs}/$2/$3/results/cv_fold_metrics.csv"
"""
PYTHON_STUB = """#!/bin/bash
# Stands in for Python: answers the fold count, records merges, passes every check.
if [[ "${1:-}" == "-c" && "${2:-}" == *k_folds* ]]; then echo 5; exit 0; fi
if [[ "${1:-}" == *combine_results.py ]]; then shift; echo "combine $*" >> "$FAKE_LOG"; fi
exit 0
"""


@pytest.fixture
def batch(tmp_path):
    """The batch scripts in a repository of their own, with training, Python, sbatch and
    the environment stubbed out; run_experiment.sh records what it is asked to run."""
    root, run = stub_repository(
        tmp_path,
        copies=["scripts/run_experiments.sh", "scripts/hpc/slurm_experiments.sh",
                "scripts/hpc/slurm_combine.sh", "scripts/hpc/submit_experiments.sh"],
        files={"scripts/run_experiment.sh": RUN_EXPERIMENT_STUB,
               "scripts/hpc/job_tmpdir.sh": "# nothing to move to scratch here\n",
               "scripts/hpc/activate_env.sh": "# nothing to activate here\n",
               "configs/experiments/a.json": "{}\n", "configs/experiments/b.json": "{}\n",
               "configs/split.json": '{"k_folds": 5}\n'},
        stubs={"python": PYTHON_STUB, "sbatch": "#!/bin/bash\necho 12345\n"})
    return SimpleNamespace(root=root, run=run)


def _runs(calls):
    return sorted(call for call in calls if not call.startswith("combine"))


def _merged(calls):
    return sorted(call.rsplit("--output ", 1)[1] for call in calls if call.startswith("combine"))


@needs_bash
def test_a_server_batch_runs_every_pair_once_per_seed_and_merges_each_seed(batch):
    result, calls = batch.run("scripts/run_experiments.sh", "--experiments", "a b", "--models", "r4",
                              "--seeds", "43 44", "--tag", "T")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _runs(calls) == ["a r4 a_seed43_T SEED=43", "a r4 a_seed44_T SEED=44",
                            "b r4 b_seed43_T SEED=43", "b r4 b_seed44_T SEED=44"]
    assert _merged(calls) == ["evaluations/a_seed43_T", "evaluations/a_seed44_T",
                              "evaluations/b_seed43_T", "evaluations/b_seed44_T"]


@needs_bash
def test_without_seeds_a_server_batch_runs_and_names_as_before(batch):
    result, calls = batch.run("scripts/run_experiments.sh", "--experiments", "a", "--models", "r3 r4",
                              "--tag", "T")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _runs(calls) == ["a r3 a_T SEED=", "a r4 a_T SEED="]
    assert _merged(calls) == ["evaluations/a_T"]


@needs_bash
@pytest.mark.parametrize("script, arguments", [
    ("scripts/run_experiments.sh", ("--experiments", "a", "--models", "r4", "--seeds", "43 4x")),
    ("scripts/hpc/submit_experiments.sh", ("--experiments", "a", "--models", "r4", "--seeds", "-1",
                                           "--dry-run")),
])
def test_a_seed_that_is_not_a_whole_number_is_refused_before_anything_runs(batch, script, arguments):
    result, calls = batch.run(script, *arguments)
    output = result.stdout + result.stderr
    assert result.returncode != 0 and "seed" in output.lower() and "unknown option" not in output
    assert _runs(calls) == []


@needs_bash
@pytest.mark.parametrize("seeds, tasks", [("43 44", 8), ("", 4)])
def test_the_array_is_sized_for_every_experiment_variant_and_seed(batch, seeds, tasks):
    arguments = ["--experiments", "a b", "--models", "r3 r4", "--dry-run"]
    result, _ = batch.run("scripts/hpc/submit_experiments.sh", *arguments,
                          *(["--seeds", seeds] if seeds else []))
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{tasks} tasks (0-{tasks - 1})" in result.stdout
    assert f"--array=0-{tasks - 1}" in result.stdout


@needs_bash
@pytest.mark.parametrize("seeds", ["43 44", ""])
def test_the_array_tasks_cover_every_experiment_variant_and_seed_once(batch, seeds):
    listed = seeds.split() or [""]
    expected = sorted(f"{e} {m} {e}{'_seed' + s if s else ''}_7 SEED={s}"
                      for e in ("a", "b") for m in ("r3", "r4") for s in listed)
    seen = []
    for task in range(len(expected)):
        result, calls = batch.run("scripts/hpc/slurm_experiments.sh", ADM1_EXPERIMENTS="a b",
                                  ADM1_MODELS="r3 r4", ADM1_SEEDS=seeds,
                                  SLURM_ARRAY_TASK_ID=str(task), SLURM_ARRAY_JOB_ID="7")
        assert result.returncode == 0, result.stdout + result.stderr
        seen += _runs(calls)
    assert sorted(seen) == expected


@needs_bash
def test_the_merge_job_merges_each_seed_of_an_experiment_on_its_own(batch):
    for seed in ("43", "44"):
        results = batch.root / "runs" / "r4" / f"a_seed{seed}_7" / "results"
        results.mkdir(parents=True)
        (results / "cv_fold_metrics.csv").write_text("", encoding="utf-8")
    result, calls = batch.run("scripts/hpc/slurm_combine.sh", ADM1_EXPERIMENTS="a", ADM1_MODELS="r4",
                              ADM1_SEEDS="43 44", ADM1_ARRAY_JOB="7")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _merged(calls) == ["evaluations/a_seed43_7", "evaluations/a_seed44_7"]
