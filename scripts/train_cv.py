"""Cross-validated training of one ADM1-UDE variant.

    python scripts/train_cv.py --model r4 --experiment configs/experiments/fold_fast_narrow_tan.json
    python scripts/train_cv.py --model r4 --experiment configs/experiments/fold_fast_narrow_tan.json \
        --hyperparameters runs/r4/tuning_<id>/best_hyperparameters.json
    python scripts/train_cv.py --model r3 --experiment configs/experiments/fold_fast_narrow_tan_no_ph.json \
        --fold-workers 4 --epochs 100
    python scripts/train_cv.py --model r4 --experiment configs/experiments/fold_fast_narrow_tan.json \\
        --fold-hyperparameters "runs/{model}/tuning_nested-<name>_outer{fold}/best_hyperparameters.json"

The hyperparameters (by default, each variant's defaults) say how to
train; an experiment file says what model to train; the correction flags apply
last. ``--fold-hyperparameters`` gives every fold its own, e.g. the selection of a
nested search (scripts/hpc/submit_nested.sh) for that outer fold.

Every hybrid is built on its fold's per-fold calibration
(scripts/calibrate/calibrate_folds.py), which ``--benchmarks`` points at.

Each fold trains on the fold's training reactors and validates on its held-out
reactors. The split used is copied into the run directory so the run can be
evaluated later without guessing which reactors were held out.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import (
    add_common_arguments,
    add_correction_arguments,
    add_experiment_argument,
    configure_jax,
    describe_correction,
    describe_model,
    resolve_correction,
)  # noqa: E402


def _worker(job):
    """Top-level so it can be pickled by ProcessPoolExecutor's spawn start method."""
    from adm1_package.experiments.tuning import _fold_worker

    return _fold_worker(job)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    add_correction_arguments(parser)
    add_experiment_argument(parser)
    parser.add_argument("--model", required=True, choices=["r2", "r3", "r4"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=None,
        help="JSON file from the tuning run. Omit to use the variant's defaults.",
    )
    parser.add_argument(
        "--fold-hyperparameters",
        default=None,
        metavar="PATTERN",
        help="Every fold's own JSON file, e.g. a nested search's "
             "'runs/{model}/tuning_nested-<name>_outer{fold}/best_hyperparameters.json': "
             "{fold} becomes the fold number, {model} the variant. Instead of --hyperparameters.",
    )
    parser.add_argument("--run-name", default=None, help="Run directory name (default: timestamp).")
    parser.add_argument(
        "--fold-workers",
        type=int,
        default=1,
        help="Folds trained in parallel processes. Each fold needs its own JAX process.",
    )
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="Subset of folds to train.")
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=None,
        help="Per-fold calibrations the folds are built on "
             "(scripts/calibrate/calibrate_folds.py; default: $ADM1_BENCHMARKS_DIR, else "
             "<run-dir>/benchmarks).",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable JAX debug checks (slower).")
    arguments = parser.parse_args()
    if arguments.fold_hyperparameters is not None and arguments.hyperparameters is not None:
        parser.error("give --hyperparameters (one recipe for every fold) or --fold-hyperparameters "
                     "(one per fold), not both")

    configure_jax()

    from adm1_package.experiments.config import RunPaths, benchmarks_dir
    from adm1_package.experiments.registry import get_spec
    from adm1_package.experiments.splits import load_split
    from adm1_package.experiments.training import fold_data_dir, loss_weights

    spec = get_spec(arguments.model)
    split = load_split(arguments.split)
    fold_numbers = arguments.folds or list(range(1, split.k_folds + 1))
    recipes, source = _recipes(arguments, fold_numbers)
    for recipe in {id(recipe): recipe for recipe in recipes.values()}.values():
        recipe.correction = resolve_correction(arguments, recipe.correction)
        # Before a run directory exists: a pH weight on R4, say, fails here, and so do
        # per-fold calibrations that were never made or copied here.
        loss_weights(spec, recipe)
    benchmarks = arguments.benchmarks or benchmarks_dir(arguments.run_dir)
    for fold in fold_numbers:
        fold_data_dir(recipes[fold], arguments.data_dir, split, fold, benchmarks)

    paths = RunPaths.create(arguments.model, arguments.run_name, root=arguments.run_dir)
    # A run directory holds one recipe for every fold or one per fold, never both: a
    # hyperparameters.json beside a run trained per fold would misstate most of its folds.
    per_fold = arguments.fold_hyperparameters is not None
    written, other = (("fold_hyperparameters.json", "hyperparameters.json") if per_fold
                      else ("hyperparameters.json", "fold_hyperparameters.json"))
    if (paths.base / other).exists():
        parser.error(f"{paths.base} holds a run trained with {other}; give another --run-name")
    (paths.base / "split.json").write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    recorded = ({str(fold): recipe.to_dict() for fold, recipe in recipes.items()} if per_fold
                else recipes[fold_numbers[0]].to_dict())
    (paths.base / written).write_text(json.dumps(recorded, indent=2), encoding="utf-8")
    paths.write_metadata(
        model=spec.display_name,
        model_key=spec.key,
        hyperparameter_source=source,
        experiment=None if arguments.experiment is None else str(arguments.experiment),
        seed=arguments.seed,
        split_file=str(arguments.split),
        **({"fold_hyperparameters": arguments.fold_hyperparameters} if per_fold else {}),
    )

    print(f"{spec.display_name}: training folds {fold_numbers} into {paths.base}")
    first = recipes[fold_numbers[0]]
    print(describe_correction(arguments.model, first.correction))
    print(describe_model(first))
    if per_fold:
        for fold in fold_numbers:
            recipe = recipes[fold]
            print(f"  fold {fold}: width {recipe.width_size}, depth {recipe.depth}, dropout "
                  f"{recipe.dropout_rate:g}, learning rate {recipe.learning_rate:.3g}, lambda "
                  f"{recipe.regularization_lambda:.3g}, batch {recipe.batch_size}, warmup "
                  f"{recipe.warmup_steps}")
    print(split.summary())

    jobs = [
        {
            "model_key": arguments.model,
            "split": split.to_dict(),
            "data_dir": str(arguments.data_dir),
            "run_root": str(arguments.run_dir),
            "run_name": paths.name,
            "fold": fold,
            "hyperparameters": recipes[fold].to_dict(),
            "seed": arguments.seed,
            "verbose": arguments.verbose,
            "debug": arguments.debug,
            "benchmarks": str(benchmarks),
        }
        for fold in fold_numbers
    ]
    results = _train(jobs, arguments.fold_workers)
    if not results:
        print("no fold completed")
        return 1
    summary_path = _write_summary(paths.base / "cv_results.csv", results)

    import numpy as np

    train_losses = np.array([r["train_loss"] for r in results], dtype=float)
    val_losses = np.array([r["val_loss"] for r in results], dtype=float)
    print(f"\n{len(results)}/{len(jobs)} folds completed. Summary written to {summary_path}")
    print(f"train loss {np.nanmean(train_losses):.4f} +/- {np.nanstd(train_losses):.4f}")
    print(f"val   loss {np.nanmean(val_losses):.4f} +/- {np.nanstd(val_losses):.4f}")
    print(f"\nEvaluate with:\n  python scripts/evaluate.py --model {arguments.model} --run {paths.base}")
    return 0


def _recipes(arguments: argparse.Namespace, fold_numbers: list[int]):
    """Every fold's recipe and where they came from: ``({fold: recipe}, source)``.

    One recipe for every fold (the same object), from ``--hyperparameters`` or the
    variant's defaults, or one per fold from ``--fold-hyperparameters``; the experiment
    goes over each and ``--epochs`` sets the epochs.
    """
    from adm1_package.experiments.training import (
        Hyperparameters,
        apply_experiment,
        default_hyperparameters,
        fold_recipes,
        load_experiment,
    )

    experiment = None if arguments.experiment is None else load_experiment(arguments.experiment)
    if arguments.fold_hyperparameters is not None:
        recipes = fold_recipes(arguments.fold_hyperparameters, arguments.model, fold_numbers,
                               experiment, arguments.epochs)
        return recipes, f"per fold: {arguments.fold_hyperparameters}"
    if arguments.hyperparameters is not None:
        payload = json.loads(arguments.hyperparameters.read_text(encoding="utf-8"))
        # Accept either the tuning output or a bare hyperparameter dict.
        hyperparameters = Hyperparameters.from_dict(payload.get("hyperparameters", payload))
        source = str(arguments.hyperparameters)
    else:
        hyperparameters = default_hyperparameters(arguments.model)
        source = "defaults (adm1_package.experiments.training.default_hyperparameters)"
    if experiment is not None:
        hyperparameters = apply_experiment(hyperparameters, experiment)
    hyperparameters.num_epochs = arguments.epochs
    return {fold: hyperparameters for fold in fold_numbers}, source


def _train(jobs: list[dict], fold_workers: int) -> list[dict]:
    """Every fold's job trained (:func:`_worker`), in parallel processes if
    ``fold_workers`` > 1; the results of the folds that completed, in fold order. One bad
    fold must not lose the rest, so a failure is reported and skipped."""
    results = []
    if fold_workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        with ProcessPoolExecutor(max_workers=fold_workers) as executor:
            futures = {executor.submit(_worker, job): job["fold"] for job in jobs}
            for future in as_completed(futures):
                fold = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 - one bad fold must not lose the rest
                    print(f"fold {fold} failed: {type(exc).__name__}: {exc}")
    else:
        for job in jobs:
            try:
                results.append(_worker(job))
            except Exception as exc:  # noqa: BLE001
                print(f"fold {job['fold']} failed: {type(exc).__name__}: {exc}")
    results.sort(key=lambda item: item["fold"])
    return results


def _write_summary(path: Path, results: list[dict]) -> Path:
    """One row per trained fold: its losses, held-out reactors, size and checkpoint."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "train_loss", "val_loss", "val_labels", "n_parameters", "checkpoint_dir"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "fold": result["fold"],
                    "train_loss": result["train_loss"],
                    "val_loss": result["val_loss"],
                    "val_labels": ";".join(result["val_labels"]),
                    "n_parameters": result["n_parameters"],
                    "checkpoint_dir": result["checkpoint_dir"],
                }
            )
    return path


if __name__ == "__main__":
    raise SystemExit(main())
