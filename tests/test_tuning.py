"""The hyperparameter search (adm1_package/experiments/tuning.py).

Spread over the tasks of a SLURM array, a search breaks silently in three ways: a storage
backend that cannot take concurrent writers, tasks that propose the same trials, and a
study resumed with other settings. The searches here score their folds without training
(``fake_training``), so what is checked is the search around the training.
"""

from __future__ import annotations

import json
import re

import pytest
from helpers import EXPERIMENTS, canonical_split

from adm1_package.experiments import tuning
from adm1_package.experiments.retired import REFUSAL
from adm1_package.experiments.splits import nested_split
from adm1_package.experiments.training import Hyperparameters, apply_experiment, load_experiment

optuna = pytest.importorskip("optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Outer fold 1's inner split: four folds per trial.
INNER = nested_split(canonical_split(), 1)


def _search(tmp_path, n_trials, **options):
    """Outer fold 1's nested search, in ``tmp_path/runs``, resumed if it exists."""
    return tuning.run_study("r4", INNER, data_dir=tmp_path, run_root=tmp_path / "runs",
                            n_trials=n_trials, num_epochs=7, storage="journal",
                            tuning_name="nested-x_outer1", study_name="nested_r4_outer1", **options)


def _default_r4() -> dict:
    """Where R4's nested search starts: the paper's network under its experiment."""
    return tuning.default_search_values(
        "r4", load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json"))


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def test_default_storage_is_sqlite_beside_the_trials(tmp_path):
    resolved = tuning._resolve_storage(None, tmp_path, "adm1_r4_cv")
    assert resolved.startswith("sqlite:///") and resolved.endswith("adm1_r4_cv.db")


def test_a_database_url_is_passed_through_untouched(tmp_path):
    url = "postgresql://user@host/optuna"
    assert tuning._resolve_storage(url, tmp_path, "adm1_r4_cv") == url


def test_journal_keyword_builds_a_journal_storage(tmp_path):
    resolved = tuning._resolve_storage("journal", tmp_path, "adm1_r4_cv")
    assert isinstance(resolved, optuna.storages.JournalStorage)


def test_an_explicit_log_path_builds_a_journal_storage(tmp_path):
    target = tmp_path / "nested" / "study.log"
    resolved = tuning._resolve_storage(str(target), tmp_path, "adm1_r4_cv")
    assert isinstance(resolved, optuna.storages.JournalStorage)
    assert target.parent.is_dir()


def test_two_workers_share_one_journal_study(tmp_path):
    """The reason the journal backend is the default on a cluster.

    Two processes against one SQLite file on a shared filesystem hit ``database is
    locked``; the journal backend appends. Here both workers run in one process, so this
    checks the wiring rather than the locking, but a regression to a non-shareable backend
    would still show up.
    """
    def worker(n_trials: int):
        storage = tuning._resolve_storage("journal", tmp_path, "shared")
        study = optuna.create_study(study_name="shared", storage=storage, load_if_exists=True)
        study.optimize(lambda t: (t.suggest_float("x", 0.0, 1.0) - 0.3) ** 2, n_trials=n_trials)
        return study

    worker(3)
    assert len(worker(2).trials) == 5
    assert (tmp_path / "shared.log").exists()


# ---------------------------------------------------------------------------
# The search directory's name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("explicit, name", [
    ("sweep-a", "tuning_sweep-a"),
    ("tuning_sweep-a", "tuning_sweep-a"),       # not prefixed twice
    ("nice", "tuning_nice"),                    # removeprefix, not lstrip ("ce")
    ("gun", "tuning_gun"),
])
def test_an_explicit_name_is_used_as_it_is(explicit, name, monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "999")
    assert tuning.tuning_run_name(explicit) == name


def test_an_empty_explicit_name_is_rejected():
    with pytest.raises(ValueError):
        tuning.tuning_run_name("tuning_")


def test_array_tasks_of_one_submission_agree(monkeypatch):
    """They share one Optuna study through a file in that directory.

    SLURM_ARRAY_JOB_ID is constant across an array; SLURM_JOB_ID is not. Taking the
    latter would give every task its own directory, its own journal file and therefore its
    own private study -- each task exploring alone while appearing to cooperate.
    """
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
    names = set()
    for task_job_id in ("12346", "12347", "12348"):
        monkeypatch.setenv("SLURM_JOB_ID", task_job_id)
        names.add(tuning.tuning_run_name())
    assert names == {"tuning_12345"}


def test_separate_submissions_get_separate_directories(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1000")
    first = tuning.tuning_run_name()
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1001")
    assert tuning.tuning_run_name() != first


def test_a_non_array_job_falls_back_to_the_job_id(monkeypatch):
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    assert tuning.tuning_run_name() == "tuning_555"


@pytest.mark.parametrize("array_job, job", [(None, None), ("", "  ")],
                         ids=["off a cluster", "blank SLURM variables"])
def test_without_a_job_id_it_is_a_timestamp(array_job, job, monkeypatch):
    """SLURM exports the variables as empty strings in some configurations."""
    for name, value in (("SLURM_ARRAY_JOB_ID", array_job), ("SLURM_JOB_ID", job)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert re.fullmatch(r"tuning_\d{8}_\d{6}", tuning.tuning_run_name())


# ---------------------------------------------------------------------------
# What a trial draws
# ---------------------------------------------------------------------------
def test_a_search_draws_over_a_fixed_model():
    """The experiment fixes the model; a trial varies the network and its training."""
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    base = apply_experiment(Hyperparameters(),
                            load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json"))
    drawn = tuning.suggest_hyperparameters(study.ask(), "r4", num_epochs=10, base=base)
    assert drawn.correction == base.correction and drawn.split_network and drawn.num_epochs == 10
    width = tuning.SEARCH_SPACES["r4"]["width_size"]
    assert width[1] <= drawn.width_size <= width[2]


@pytest.mark.parametrize("model_key", sorted(tuning.SEARCH_SPACES))
def test_the_lambda_range_reaches_where_the_penalty_bites(model_key):
    """A range that stops at 1e-3 can only ever report that L2 does not matter.

    On a trained R4 fold the penalty gradient ``2*lambda*||W||`` is a fraction of a percent
    of the data gradient at lambda=1e-4 and a few percent at 1e-3. It first shapes the fit
    around 1e-2.
    """
    kind, low, high = tuning.SEARCH_SPACES[model_key]["regularization_lambda"]
    assert kind == "log" and low <= 1e-4 and high >= 1e-2


# ---------------------------------------------------------------------------
# The array tasks of one search
# ---------------------------------------------------------------------------
def _openings(tmp_path, model_key, worker_id, n_trials=4):
    """The configurations a fresh task proposes before it has any results."""
    study = tuning.run_study(model_key, INNER, data_dir=tmp_path,
                             run_root=tmp_path / f"{model_key}-{worker_id}-{n_trials}",
                             n_trials=n_trials, num_epochs=2, storage="journal",
                             tuning_name="openings", worker_id=worker_id)
    return [trial.params for trial in study.trials]


@pytest.mark.parametrize("model_key", sorted(tuning.SEARCH_SPACES))
def test_different_tasks_propose_different_openings(model_key, tmp_path, fake_training):
    """TPE draws its first trials at random from a seeded generator, so tasks sharing a
    sampler seed would open with identical configurations."""
    first, second, third = (_openings(tmp_path, model_key, worker) for worker in (0, 1, 2))
    assert first != second and first != third and second != third


def test_the_same_task_proposes_the_same_openings(tmp_path, fake_training):
    assert _openings(tmp_path, "r4", 0) == _openings(tmp_path, "r4", 0, n_trials=5)[:4]


def test_every_fold_trains_from_the_search_seed_whichever_task_samples(tmp_path, fake_training):
    """Only the sampler seed moves with the task. If the training seed did, two tasks would
    score the same hyperparameters differently and the study would compare noise. The
    folds are trained by the fold worker from the jobs the search hands it, so a key it
    reads that the jobs lack fails here too."""
    _search(tmp_path, 2, seed=5, worker_id=3)
    assert [arguments["fold"] for arguments in fake_training] == [1, 2, 3, 4] * 2
    assert {arguments["seed"] for arguments in fake_training} == {5}


def test_only_the_first_task_starts_from_the_default_values(tmp_path, fake_training):
    study = _search(tmp_path, 1, enqueue=_default_r4(), worker_id=3)
    assert study.trials[0].params != _default_r4()


def test_a_resumed_search_does_not_start_from_the_default_values_twice(tmp_path, fake_training):
    start = _default_r4()
    _search(tmp_path, 1, enqueue=start)
    study = _search(tmp_path, 2, enqueue=start)
    assert sum(trial.params == start for trial in study.trials) == 1


def test_a_resumed_search_proposes_new_configurations(tmp_path, fake_training):
    """The sampler seeded as at the start would propose the opening trials again."""
    _search(tmp_path, 2)
    study = _search(tmp_path, 2)
    assert len({tuple(sorted(trial.params.items())) for trial in study.trials}) == 4


def test_a_study_is_not_resumed_with_other_settings(tmp_path, fake_training):
    """A resubmission under the same name adds trials to the same study; trained for
    another number of epochs, or on another model, they would not be comparable."""
    _search(tmp_path, 1)
    with pytest.raises(ValueError, match="num_epochs"):
        tuning.run_study("r4", INNER, data_dir=tmp_path, run_root=tmp_path / "runs", n_trials=1,
                         num_epochs=8, storage="journal", tuning_name="nested-x_outer1",
                         study_name="nested_r4_outer1")
    base = apply_experiment(Hyperparameters(),
                            load_experiment(EXPERIMENTS / "fold_fast_narrow_tan.json"))
    with pytest.raises(ValueError, match="correction"):
        _search(tmp_path, 1, base=base)


def test_a_study_started_before_the_retirement_resumes(tmp_path, fake_training):
    """Its signature records the retired fields at the values that changed nothing. With one
    that changed the model, it is refused as retired before this release."""
    _search(tmp_path, 1)
    storage = tuning._resolve_storage("journal", tmp_path / "runs" / "r4" / "tuning_nested-x_outer1",
                                      "nested_r4_outer1")
    study = optuna.load_study(study_name="nested_r4_outer1", storage=storage)
    signature = study.user_attrs["signature"]
    signature["base"].update(memory_days=0.0, synthetic_ratio=0.0, campaign_balance=False,
                             validation_every=0, loss_scale="reactor")
    study.set_user_attr("signature", signature)
    _search(tmp_path, 1)
    signature["base"]["memory_days"] = 10.0
    study.set_user_attr("signature", signature)
    with pytest.raises(ValueError, match=REFUSAL):
        _search(tmp_path, 1)


def test_no_trials_rewrites_the_selection_from_the_trials_so_far(tmp_path, fake_training):
    """How the selection is collected once every task of an array has ended, killed or not.
    The collection is not told where the search started; the study remembers."""
    _search(tmp_path, 2, enqueue=_default_r4())
    best = tmp_path / "runs" / "r4" / "tuning_nested-x_outer1" / "best_hyperparameters.json"
    best.unlink()
    study = _search(tmp_path, 0)
    assert len(study.trials) == 2 and best.exists()
    assert json.loads(best.read_text(encoding="utf-8"))["started_from"] == _default_r4()
