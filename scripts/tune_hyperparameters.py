"""Cross-validated hyperparameter search for one ADM1-UDE variant.

    python scripts/tune_hyperparameters.py --model r4 --n-trials 40 --epochs 40
    python scripts/tune_hyperparameters.py --model r3 --fold-workers 4

Each trial trains one model per fold and is scored by the mean fold-validation
loss, so validation data drive the selection. The hold-out test reactors in the
split file are never touched here; they are what ``scripts/evaluate.py`` reports
at the end.

On a cluster, point several array tasks at one shared study::

    python scripts/tune_hyperparameters.py --model r4 \\
        --storage postgresql://user@host/optuna --study-name adm1_r4_cv --n-trials 10

The search writes ``<run-dir>/<model>/tuning_<id>/best_hyperparameters.json``,
which ``scripts/train_cv.py --hyperparameters`` consumes directly. The directory
name is unique per search (the array job id on a cluster, else a timestamp), so a new
search does not overwrite the last.

A nested search runs this once per outer fold, on that fold's inner split and
calibrations (``scripts/make_nested.py``, ``scripts/hpc/submit_nested.sh``)::

    python scripts/tune_hyperparameters.py --model r4 \\
        --experiment configs/experiments/fold_fast_narrow_tan.json \\
        --split runs/nested-fast/outer_1/split.json \\
        --benchmarks runs/nested-fast/outer_1/benchmarks \\
        --tuning-name nested-fast_outer1 --start-from-defaults --epochs 200

``--n-trials 0`` trains nothing and writes the selection from the trials a study
holds, e.g. after an array task was killed at the wall clock.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _bootstrap import (  # noqa: E402
    add_common_arguments,
    add_correction_arguments,
    add_experiment_argument,
    configure_jax,
    describe_correction,
    describe_model,
    resolve_correction,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    add_correction_arguments(parser)
    add_experiment_argument(parser)
    parser.add_argument("--model", required=True, choices=["r2", "r3", "r4"])
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Trials this process runs. 0 trains nothing and writes the selection from the "
             "trials the study holds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Epochs per fold during the search. Keep below the final training budget; "
             "the final models are retrained with scripts/train_cv.py.",
    )
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Defaults to a SQLite file inside the run directory.",
    )
    parser.add_argument(
        "--fold-workers",
        type=int,
        default=1,
        help="Folds per trial trained in parallel. Parallel folds turn the pruner off "
             "(a trial can only be judged once every fold has finished), so this trades "
             "wall-clock time for wasted compute on bad configurations. Use 1 on a "
             "single machine; use >1 on a cluster where the cores are already reserved.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="Offsets the sampler seed so array tasks sharing one study do not all "
             "propose the same opening trials (default: $SLURM_ARRAY_TASK_ID, else 0). "
             "The training seed is unaffected.",
    )
    parser.add_argument(
        "--tuning-name",
        default=None,
        help="Directory this search writes into, under <run-dir>/<model>/. Defaults "
             "to tuning_$SLURM_ARRAY_JOB_ID on a cluster (the same value for every "
             "task of one array, which is what lets them share a study) and to a "
             "timestamp otherwise. Pass an existing name to add trials to that "
             "search rather than starting a new one.",
    )
    parser.add_argument("--parallel-trials", type=int, default=1)
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=None,
        help="Per-fold calibrations an experiment with calibration 'per-fold' is built on "
             "(default: $ADM1_BENCHMARKS_DIR, else <run-dir>/benchmarks). A nested search "
             "passes its outer fold's, <nested>/outer_<k>/benchmarks (scripts/make_nested.py).",
    )
    parser.add_argument(
        "--start-from-defaults",
        action="store_true",
        help="Try the default hyperparameters first, so the search selects them unless a "
             "trial does better. Only worker 0 enqueues them, once per study.",
    )
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    if arguments.n_trials < 0 or arguments.epochs < 1:
        parser.error("--n-trials must not be negative and --epochs must be positive")

    configure_jax()

    from adm1_package.experiments.config import benchmarks_dir, checkpoint_safe_dir
    from adm1_package.experiments.splits import load_split
    from adm1_package.experiments.training import (
        Hyperparameters,
        apply_experiment,
        fold_data_dir,
        load_experiment,
        loss_weights,
    )
    from adm1_package.experiments.registry import get_spec
    from adm1_package.experiments.tuning import default_search_values, run_study

    split = load_split(arguments.split)
    # As train_cv.py's runs: checkpoints cannot be written below a path with spaces or
    # non-ASCII characters on Windows, so an explicit --run-dir gets its short form.
    run_root = checkpoint_safe_dir(arguments.run_dir)
    # What the model is stays fixed across the whole search; a study mixing
    # models would compare different models by one loss.
    base = Hyperparameters()
    experiment = None if arguments.experiment is None else load_experiment(arguments.experiment)
    if experiment is not None:
        base = apply_experiment(base, experiment)
    base.correction = resolve_correction(arguments, base.correction)
    # Before any trial: a pH weight on R4 fails here, and so do per-fold calibrations
    # that were made for another split or model, or never copied here.
    loss_weights(get_spec(arguments.model), base)
    benchmarks = arguments.benchmarks or benchmarks_dir(run_root)
    if base.calibration == "per-fold":
        for fold in range(1, split.k_folds + 1):
            fold_data_dir(base, arguments.data_dir, split, fold, benchmarks)
    print(describe_correction(arguments.model, base.correction))
    print(describe_model(base))
    print(split.summary())
    print(
        f"\nSelecting on the mean validation loss of {split.k_folds} folds; "
        f"the {len(split.test)} test reactors are excluded from the search."
    )
    if base.calibration == "per-fold":
        print(f"The folds are built on the calibrations in {benchmarks}.")

    run_study(
        model_key=arguments.model,
        split=split,
        data_dir=arguments.data_dir,
        run_root=run_root,
        study_name=arguments.study_name,
        storage=arguments.storage,
        n_trials=arguments.n_trials,
        num_epochs=arguments.epochs,
        seed=arguments.seed,
        worker_id=arguments.worker_id,
        tuning_name=arguments.tuning_name,
        base=base,
        fold_workers=arguments.fold_workers,
        parallel_trials=arguments.parallel_trials,
        verbose=arguments.verbose,
        benchmarks=benchmarks,
        enqueue=(default_search_values(arguments.model, experiment)
                 if arguments.start_from_defaults else None),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
