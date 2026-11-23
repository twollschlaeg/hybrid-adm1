# Running on a SLURM cluster

The paper's nested search and hybrid models were run with these scripts. Every submit
script checks the environment, the split, the data, the experiment files and the
calibrations before it submits anything, and `--dry-run` prints the plan and a time
estimate without submitting.

## Setup

The code, `data/` and the fold-wise calibrations have to be on the cluster. Commit
first, because every run records the git revision. From the repository root:

```bash
rsync -av --exclude runs/ --exclude evaluations/ --exclude logs/ ./ cluster:hybrid-adm1/
rsync -a runs/benchmarks-smoothstep-fast/ cluster:hybrid-adm1/runs/benchmarks-smoothstep-fast/
```

Copy the other calibration sets (`runs/benchmarks-smoothstep*`) the same way for the
comparison runs. On the cluster, build the environment once:

```bash
bash scripts/hpc/setup_env.sh                       # ADM1_PYTHON=... picks the interpreter
export ADM1_VENV=/path/it/printed/envs/hybrid-adm1
```

It builds a pip virtualenv from `requirements-hpc.txt` (conda's solver needs more memory
than login nodes often allow) and checks that float64 works. Nodes behind a proxy need
`https_proxy`.

## The nested search

Every outer fold gets a search of its own on inner folds of its twelve training
reactors, so its three held-out reactors inform neither the selection nor the training.
The inner folds are built on the outer fold's calibration. The paper's searches:

```bash
bash scripts/hpc/submit_nested.sh --model r4 --name nested-fast --dry-run
bash scripts/hpc/submit_nested.sh --model r4 --name nested-fast
bash scripts/hpc/submit_nested.sh --model r3 --name nested-acids --experiment fold_fast_acids_r3_no_ph
bash scripts/hpc/submit_nested.sh --model r2 --name nested-acids --experiment fold_fast_acids_r2_no_ph
```

A submission prepares the inner folds (`scripts/make_nested.py`), submits the trials to
one Optuna study per outer fold, started from the default hyperparameters, and queues a
job that writes each outer fold's selection to
`runs/<model>/tuning_<name>_outer<k>/best_hyperparameters.json`. Resubmitting under the
same name adds trials, e.g. after tasks hit the wall clock. With the defaults (24 trials
of 200 epochs per outer fold, 10 tasks of 16 cores per variant):

| Variant | Per trial | Per task | Core-hours |
|---|---|---|---|
| R4 | ~2.7 h | ~8 h | ~1300 |
| R3 | ~3.8 h | ~11 h | ~1800 |
| R2 | ~5 h | ~15 h | ~2400 |

## The final hybrid models

Every outer fold is trained with its own selection, with three seeds:

```bash
export ADM1_BENCHMARKS_DIR=$PWD/runs/benchmarks-smoothstep-fast
bash scripts/hpc/submit_experiments.sh --experiments fold_fast_narrow_tan --models r4 \
    --seeds "45 46 47" --epochs 200 \
    --fold-hyperparameters "runs/{model}/tuning_nested-fast_outer{fold}/best_hyperparameters.json"
bash scripts/hpc/submit_experiments.sh --experiments fold_fast_acids_r3_no_ph --models r3 \
    --seeds "45 46 47" --epochs 200 \
    --fold-hyperparameters "runs/{model}/tuning_nested-acids_outer{fold}/best_hyperparameters.json"
bash scripts/hpc/submit_experiments.sh --experiments fold_fast_acids_r2_no_ph --models r2 \
    --seeds "45 46 47" --epochs 200 \
    --fold-hyperparameters "runs/{model}/tuning_nested-acids_outer{fold}/best_hyperparameters.json"
```

One array task trains the five folds of one experiment, variant and seed in parallel and
evaluates them, which takes 1–3 h. A job queued behind the array copies the results off
`$SCRATCH` into `runs/<model>/<experiment>_seed<N>_<job>/`. It leaves the checkpoints
there, so copy the runs whole from `$SCRATCH/runs/<model>/` before scoring the seeds as
an ensemble:

```bash
python scripts/evaluate.py --model r4 --ensemble runs/r4/fold_fast_narrow_tan_seed4{5,6,7}_<job>
```

## The comparison runs

The other experiments of the paper use the default hyperparameters, each built on its
calibration set (main README, step 5):

```bash
export ADM1_BENCHMARKS_DIR=$PWD/runs/benchmarks-smoothstep
bash scripts/hpc/submit_experiments.sh --models "r4" --epochs 200 --seeds "45 46 47" \
    --experiments "fold_calibrated fold_calibrated_narrow"
bash scripts/hpc/submit_experiments.sh --models "r2 r3" --epochs 200 --seeds "45 46 47" \
    --experiments "fold_calibrated_narrow_no_ph"
export ADM1_BENCHMARKS_DIR=$PWD/runs/benchmarks-smoothstep-arrhenius
bash scripts/hpc/submit_experiments.sh --models "r4" --epochs 200 --seeds "45 46 47" \
    --experiments "fold_arrhenius_narrow"
export ADM1_BENCHMARKS_DIR=$PWD/runs/benchmarks-smoothstep-fast
bash scripts/hpc/submit_experiments.sh --models "r4" --epochs 200 --seeds "45 46 47" \
    --experiments "fold_fast_narrow fold_fast_narrow_tan fold_fast_narrow_tan_linear"
bash scripts/hpc/submit_experiments.sh --models "r2 r3" --epochs 200 --seeds "45 46 47" \
    --experiments "fold_fast_narrow_tan_no_ph"
bash scripts/hpc/submit_experiments.sh --models "r3" --epochs 200 --seeds "45 46 47" \
    --experiments "fold_fast_acids_r3_no_ph fold_fast_pruned_r3_no_ph"
bash scripts/hpc/submit_experiments.sh --models "r2" --epochs 200 --seeds "45 46 47" \
    --experiments "fold_fast_acids_r2_no_ph fold_fast_pruned_r2_no_ph"
```

A run is scored on the calibration it was built on, and the scripts refuse calibrations
of another model or fitted within other bounds.

## When something fails

| Symptom | Cause |
|---|---|
| `missing: .../results -- that task failed or did not finish` in `logs/adm1-merge-*.out` | See that task's `logs/adm1-exp-<job>_<task>.err`, then resubmit just that experiment and variant. |
| `outer folds without a selection` in `logs/adm1-nested-collect-*.out` | No trial of those folds completed. Resubmit with the same `--name` and `--outer-folds`. |
| `study nested_<model>_outer<k> ... was started with other settings` | A resubmission under an earlier `--name` with other epochs, experiment or split. Give it another name. |
| `... was retired before this release` | The run, recipe or calibration uses an option this release no longer has. |
| `ADM1_VENV: parameter null or not set` or `activation did not take` | `ADM1_VENV` is not exported, or is not the directory containing `bin/python`. |
| `float64 is not available` | The run would be quietly wrong. Do not proceed. |
| `database is locked` | SQLite on a shared filesystem. Use the default `journal` storage. |
| Torch pulls GBs of CUDA wheels | The `--extra-index-url` line of `requirements-hpc.txt` was dropped. |
| pip hangs with no output | No `https_proxy` on a cluster that needs one. |
| Killed at the wall clock | Safe. Resubmit, and Optuna resumes the study. |
