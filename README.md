# hybrid-adm1

Code for the paper *Universal Differential Equation Models for the Anaerobic Digestion
Process under Dynamic Operating Conditions* by Tim Wollschläger, Tim Römer, Christiane
Herrmann and Andreas Focks, submitted to *Chemical Engineering Journal*.

The hybrid models couple reduced variants of the Anaerobic Digestion Model No. 1
(ADM1-R2, ADM1-R3, ADM1-R4) with small neural networks that correct their kinetic
parameters and influent degradability for the operating conditions. The code is Python
with JAX: [diffrax](https://github.com/patrick-kidger/diffrax) integrates the ODEs, the
networks are [Equinox](https://github.com/patrick-kidger/equinox) modules, and the
gradients are backpropagated through the solver.

## Installation

```bash
git clone https://github.com/twollschlaeg/hybrid-adm1.git
cd hybrid-adm1
python -m venv .venv && . .venv/bin/activate      # or: conda env create -f environment.yml
pip install -e ".[tuning,dev]"
```

Python 3.11 or newer. The models run on the CPU in double precision. Set
`ADM1_PLATFORM=gpu` to use an accelerator.

## Data and trained runs

The reactor data are not part of this repository. They come from laboratory-scale CSTR
campaigns at the Leibniz Institute for Agricultural Engineering and Bioeconomy (ATB) and
are available from the corresponding author on request (twollschlaeg@uos.de). So are the
trained runs and fold-wise calibrations behind the paper's tables and figures. Copy the
data into `data/` ([`data/README.md`](data/README.md) describes the files) and the runs
into `runs/`. On Windows, keep the path of the repository below about 110 characters,
because the checkpoint paths inside `runs/` are deep.

`pytest` runs the tests. Those that need the data or the runs skip without them.

## Names in the paper and in the code

The code keeps the laboratory names of the campaigns and reactors:

| Paper | Campaign | Reactors |
|---|---|---|
| Dataset A, A1–A3 | `rice-bran` | `rice-bran_R30`, `_R60`, `_R90`, each the mean of three replicates (R10–R30, R40–R60, R70–R90) |
| Dataset B, B1–B6 | `LaRA-temp` | `LaRA-temp_F1` … `_F6` |
| Dataset C, C1–C6 | `LaRA-substrate` | `LaRA-substrate_F1` … `_F6` |

The mechanistic benchmarks of Table 3:

| Paper | In the code | Calibrated with |
|---|---|---|
| MECH | the installed calibration in `data/` | `scripts/calibrate/calibrate_benchmarks.py` |
| MECH-CV | `runs/benchmarks-smoothstep` | `scripts/calibrate/calibrate_folds.py` |
| Arr. | `runs/benchmarks-smoothstep-arrhenius` | the same, with `--arrhenius` |
| Arr.⁺, the base model of the final hybrid models | `runs/benchmarks-smoothstep-fast` | the same, with `--arrhenius --bound k_hyd_ch 0.1 2.88` |

The evaluation of a run calls the fold-wise calibration it was built on `MECH-CV`. The
table and figure scripts call it `Base`.

## Reproducing the paper

Steps 3 and 4 are the expensive ones. The nested search took about 5,500 core-hours on a
cluster, and every final run trains 200 epochs per fold.

**1. The split.** The five folds are fixed in [`configs/split.json`](configs/split.json):
three held-out reactors per fold, stratified by dataset, without a separate test set.
Every later step reads this file. `python scripts/make_splits.py --show` prints it.

**2. The mechanistic benchmarks.** The data contain the installed calibration (MECH). It
was made campaign by campaign, the reduced variants first and then the full ADM1:

```bash
for campaign in rice-bran LaRA-temp LaRA-substrate; do
    python scripts/calibrate/calibrate_benchmarks.py --campaign $campaign --reset-particulates
    python scripts/calibrate/base_calibration.py --campaign $campaign --reset-particulates
done
```

The three fold-wise sets also fit the start values of each fold's held-out reactors to
their first 30 days:

```bash
python scripts/calibrate/calibrate_folds.py --output runs/benchmarks-smoothstep --workers 6
python scripts/calibrate/calibrate_folds.py --arrhenius --output runs/benchmarks-smoothstep-arrhenius --workers 6
python scripts/calibrate/calibrate_folds.py --arrhenius --bound k_hyd_ch 0.1 2.88 \
    --output runs/benchmarks-smoothstep-fast --workers 6
```

**3. The nested hyperparameter search**, on a SLURM cluster set up as in
[`scripts/hpc/README.md`](scripts/hpc/README.md#the-nested-search):

```bash
bash scripts/hpc/submit_nested.sh --model r4 --name nested-fast
bash scripts/hpc/submit_nested.sh --model r3 --name nested-acids --experiment fold_fast_acids_r3_no_ph
bash scripts/hpc/submit_nested.sh --model r2 --name nested-acids --experiment fold_fast_acids_r2_no_ph
```

The selection of each outer fold lands in
`runs/<model>/tuning_<name>_outer<k>/best_hyperparameters.json`.

**4. The final hybrid models**, with three seeds each, built on Arr.⁺:

```bash
export ADM1_BENCHMARKS_DIR=$PWD/runs/benchmarks-smoothstep-fast
mkdir -p logs
EPOCHS=200 bash scripts/run_experiments.sh --experiments fold_fast_narrow_tan --models r4 \
    --seeds "45 46 47" --fold-hyperparameters "runs/{model}/tuning_nested-fast_outer{fold}/best_hyperparameters.json"
EPOCHS=200 bash scripts/run_experiments.sh --experiments fold_fast_acids_r3_no_ph --models r3 \
    --seeds "45 46 47" --fold-hyperparameters "runs/{model}/tuning_nested-acids_outer{fold}/best_hyperparameters.json"
EPOCHS=200 bash scripts/run_experiments.sh --experiments fold_fast_acids_r2_no_ph --models r2 \
    --seeds "45 46 47" --fold-hyperparameters "runs/{model}/tuning_nested-acids_outer{fold}/best_hyperparameters.json"
```

On a cluster, `scripts/hpc/submit_experiments.sh` takes the same options. The three seeds
of a variant are scored together, as the mean prediction of an ensemble:

```bash
python scripts/evaluate.py --model r4 --benchmarks runs/benchmarks-smoothstep-fast --ensemble \
    runs/r4/fold_fast_narrow_tan_seed45_<tag> runs/r4/fold_fast_narrow_tan_seed46_<tag> \
    runs/r4/fold_fast_narrow_tan_seed47_<tag>
```

This writes `runs/r4/fold_fast_narrow_tan_tuned_ensemble_s45-46-47`, the name the table
and figure scripts read. R3 and R2 follow the same pattern.

**5. The comparison runs.** The other hybrid models of the paper are trained like the
final ones, but with the default hyperparameters (no `--fold-hyperparameters`) and with
`ADM1_BENCHMARKS_DIR` pointing at the calibration they are built on. They are scored as
`<experiment>_ensemble_s45-46-47`.

| Experiment ([`configs/experiments/`](configs/experiments)) | What the network corrects | Built on |
|---|---|---|
| `fold_calibrated` | every parameter the pruning keeps | MECH-CV |
| `fold_calibrated_narrow` | `f_ch` and `k_hyd_ch`, within a factor of two | MECH-CV |
| `fold_arrhenius_narrow` | the same, with Arrhenius rates | Arr. |
| `fold_fast_narrow` | the same | Arr.⁺ |
| `fold_fast_narrow_tan` | also the influent TAN (the final ADM1-R4) | Arr.⁺ |
| `fold_fast_narrow_tan_linear` | the same, with linear networks | Arr.⁺ |
| `fold_calibrated_narrow_no_ph`, `fold_fast_narrow_tan_no_ph` | as their arm, for R2 and R3, with pH scored but not fitted | as their arm |
| `fold_fast_acids_r3_no_ph`, `fold_fast_acids_r2_no_ph` | also the growth of the acid degraders (the final ADM1-R3 and ADM1-R2) | Arr.⁺ |
| `fold_fast_pruned_r3_no_ph`, `fold_fast_pruned_r2_no_ph` | every parameter the pruning keeps, within a factor of two | Arr.⁺ |

The scripts name the ensembles they read at their top: `FINAL`, `PLAIN`, `UNTUNED` and
`LINEAR` in `scripts/manuscript_figures.py`, `ROWS` in `scripts/development_table.py` and
`scripts/change_windows_table.py`, and `VARIANTS` in `scripts/si_tables.py`.

**6. Tables and figures.**

| Output | Script |
|---|---|
| Table 1, loading rates | `scripts/loading_rates.py` |
| Section 2.1, replicate variation of dataset A | `scripts/replicate_variation.py` |
| Section 3.1 and Table 3 | `scripts/benchmark_numbers.py` |
| Tables 4 and 5 | `scripts/heldout_tables.py --latex` |
| Table 6 and SI Table S5 | `scripts/change_windows_table.py --latex` |
| Figures 3–6 | `scripts/manuscript_figures.py` |
| Table D.1 | `scripts/nested_summary.py` |
| SI Table S4 | `scripts/development_table.py --latex` |
| SI Tables S1–S3 and S6–S9 | `scripts/si_tables.py --latex`, after `scripts/benchmark_numbers.py --json plots/manuscript/benchmark_numbers.json` (S3) |
| SI Figures S1–S4 | `scripts/si_figures.py` |

`scripts/nested_summary.py` takes the selections of step 3, e.g.
`--models r4 --pattern "runs/{model}/tuning_nested-fast_outer{fold}/best_hyperparameters.json"`.

## Where to find the methods

The module docstrings explain the methods in more detail than the paper.

| | |
|---|---|
| The models of each variant | `adm1_package/models/{base,r2,r3,r4}/` |
| How network outputs become parameters | `adm1_package/utils/corrections.py`, `adm1_package/utils/nn_architectures.py` |
| Forcing between the daily records | `adm1_package/utils/forcing.py` |
| Initial states | `adm1_package/utils/initial_state.py`, `scripts/calibrate/fit_held_out_starts.py` |
| Solver and biogas output | `adm1_package/utils/solver.py`, `adm1_package/utils/gas.py` |
| Calibration of the mechanistic models | `scripts/calibrate/` |
| Nested search | `adm1_package/experiments/nested.py` |
| Training, evaluation and metrics | `scripts/train_cv.py`, `scripts/evaluate.py`, `adm1_package/experiments/` |

Options the paper's models do not use were removed before this release. A run or request
that needs one is refused with a message saying so
([`adm1_package/experiments/retired.py`](adm1_package/experiments/retired.py)).

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `ADM1_DATA_DIR` | reactor data | `./data` |
| `ADM1_RUN_DIR` | runs (checkpoints, metrics, Optuna storage) | `./runs` |
| `ADM1_BENCHMARKS_DIR` | the fold-wise calibrations a run is built on and scored against | `<runs>/benchmarks` |
| `ADM1_STORED_RUNS` | where the tests find the stored runs | `./runs` |
| `ADM1_PLATFORM` | JAX platform | `cpu` |
| `ADM1_SOLVER_RTOL`, `ADM1_SOLVER_ATOL` | solver tolerances | `1e-4`, `1e-5` |
| `ADM1_VENV` | the virtualenv the cluster scripts activate | — |

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
