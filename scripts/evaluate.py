"""Score a cross-validated run and emit the manuscript tables.

    python scripts/evaluate.py --model r4 --run runs/r4/<run>
    python scripts/evaluate.py --model r4 --ensemble runs/r4/<experiment>_seed42_<tag> \
        runs/r4/<experiment>_seed43_<tag> runs/r4/<experiment>_seed44_<tag>

Writes to ``<run>/results``:

* ``cv_fold_metrics.csv`` / ``cv_fold_objective_metrics.csv`` -- one row per
  fold, split, variant (and objective);
* ``cv_aggregated_metrics.csv`` / ``cv_aggregated_objective_metrics.csv`` --
  mean, standard deviation and standard error over folds;
* ``cv_paired_comparisons.csv`` -- paired hybrid-vs-mechanistic tests;
* ``evaluation.json`` -- the checkpoint scored and its epoch per fold;
* ``*.tex`` -- the tables in the form the manuscript uses.

The split is read from the run directory, not recomputed, so the reactors scored
as "held out" really are the ones the fold was not trained on.

Every fold is scored at its final epoch. ``--checkpoint best`` scores the epoch with
the lowest loss on the fold's held-out reactors instead: chosen by the reactors it
is then scored on, so optimistic, and written to ``<run>/results-best``.

The hybrid is compared with the installed benchmark (MECH, calibrated on every reactor,
so in-sample on the held-out ones) and with the per-fold benchmarks (MECH-CV, calibrated
on each fold's training reactors only by ``scripts/calibrate/calibrate_folds.py``), which
every run is built on: its hybrid is scored on its fold's, so ``--benchmarks`` (or
``$ADM1_BENCHMARKS_DIR``) has to hold them.

On those calibrations a fold's held-out reactors start from values fitted on their first
days (``scripts/calibrate/fit_held_out_starts.py``), MECH-CV and the hybrid alike. Three
options score otherwise, each into a results directory of its own, for comparison:
``--held-out-starts installed`` starts them from the installed calibration, fitted on their
whole record (``results-installed-starts``); ``--spin-up-days 45`` scores from day 45 on
(``results-from-day45``); ``--initial-state warmup`` starts every model from its own warm-up
at the first week's forcing, which reads no start values (``results-warmup``).

``--ensemble`` scores the mean prediction of runs of one experiment trained with different
seeds. It makes an ensemble directory beside them, ``<experiment>_ensemble_s<seeds>`` (or
``--name``), which lists the members and takes their results; ``--run`` scores it again,
and ``scripts/manuscript_figures.py`` draws it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_common_arguments, configure_jax  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--model", required=True, choices=["r2", "r3", "r4"])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path,
                        help="Run directory produced by train_cv.py, or an ensemble directory.")
    target.add_argument("--ensemble", type=Path, nargs="+", metavar="RUN",
                        help="two or more runs of one experiment with different seeds: score "
                             "their mean prediction, in an ensemble directory beside them")
    parser.add_argument("--name", default=None,
                        help="with --ensemble: the directory's name "
                             "(default: <experiment>_ensemble_s<seeds>)")
    parser.add_argument("--no-test", action="store_true", help="Skip the hold-out test set.")
    parser.add_argument("--output", type=Path, default=None, help="Override the results directory.")
    parser.add_argument("--checkpoint", choices=["last", "best"], default="last",
                        help="last: the final epoch (default); best: the epoch the held-out "
                             "reactors chose, optimistic, written to results-best")
    parser.add_argument("--benchmarks", type=Path, default=None,
                        help="per-fold benchmarks of scripts/calibrate/calibrate_folds.py, which "
                             "the run is built on (default: $ADM1_BENCHMARKS_DIR, else "
                             "<run-dir>/benchmarks)")
    parser.add_argument("--held-out-starts", choices=["fitted", "installed"], default="fitted",
                        help="where a fold's held-out reactors start on its per-fold "
                             "calibration: fitted on their first days (default), or as installed, "
                             "fitted on their whole record (results-installed-starts)")
    parser.add_argument("--spin-up-days", type=float, default=None, metavar="DAYS",
                        help="score from this day on instead of the fold's spin-up "
                             "(results-from-day<DAYS>)")
    parser.add_argument("--initial-state", choices=["calibrated", "warmup"], default=None,
                        help="start every model so instead of as the fold trained "
                             "(warmup: results-warmup)")
    arguments = parser.parse_args()
    if arguments.ensemble is not None and len(arguments.ensemble) < 2:
        parser.error("--ensemble needs two or more runs")

    from adm1_package.experiments.config import benchmarks_dir

    benchmarks = arguments.benchmarks or benchmarks_dir(arguments.run_dir)

    configure_jax()

    from adm1_package.experiments.ensembles import (
        create_ensemble,
        default_ensemble_name,
        is_ensemble,
    )
    from adm1_package.experiments.evaluation import evaluate_ensemble, evaluate_run, results_name
    from adm1_package.experiments.reporting import write_tables
    from adm1_package.experiments.splits import load_run_split, load_split

    run = arguments.run
    if arguments.ensemble is not None:
        members = list(arguments.ensemble)
        try:
            name = arguments.name or default_ensemble_name(members)
            run = create_ensemble(members[0].parent / name, arguments.model, members)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        print(f"Ensemble of {len(members)} runs: {run}")

    try:
        split = load_run_split(run)
    except FileNotFoundError as exc:
        print(f"{exc}\nFalling back to {arguments.split}; verify this is the split the run used.")
        split = load_split(arguments.split)

    evaluate = evaluate_ensemble if is_ensemble(run) else evaluate_run
    result = evaluate(
        arguments.model,
        run,
        split=split,
        data_dir=arguments.data_dir,
        include_test=not arguments.no_test,
        checkpoint=arguments.checkpoint,
        benchmarks=benchmarks,
        held_out_starts=arguments.held_out_starts,
        spin_up_days=arguments.spin_up_days,
        initial_state=arguments.initial_state,
    )
    print(f"\nScored at the {'final' if arguments.checkpoint == 'last' else 'selected'} "
          f"checkpoint, epochs {result.settings['epochs']}; per-fold benchmarks: "
          f"{benchmarks}; held-out starts: "
          f"{result.settings.get('held_out_starts')}; scored from day "
          f"{result.settings.get('spin_up_days')}")

    output = arguments.output or (Path(run) / results_name(
        arguments.checkpoint, held_out_starts=arguments.held_out_starts,
        spin_up_days=arguments.spin_up_days, initial_state=arguments.initial_state))
    written = result.write(output)
    tables = write_tables(result, output)

    print("\nAggregated metrics:")
    columns = ["model", "variant", "split", "nrmse_mean", "nrmse_std", "rmse_mean", "r2_mean", "n_folds"]
    print(result.aggregated[columns].to_string(index=False))

    print("\nPaired hybrid vs mechanistic (NRMSE):")
    comparison = result.comparisons
    comparison = comparison[(comparison["metric"] == "nrmse") & (comparison["objective"] == "all")]
    print(
        comparison[
            ["model", "split", "benchmark", "mean_mechanistic", "mean_hybrid",
             "mean_relative_change_percent", "wins", "n_folds", "sign_test_p", "wilcoxon_p"]
        ].to_string(index=False)
    )

    print("\nWrote:")
    for path in list(written.values()) + tables:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
