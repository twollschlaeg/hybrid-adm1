"""Recalibrate the full ADM1 of one campaign, after the reduced variants.

The full ADM1 is fitted with the two stages of ``calibrate_benchmarks.py``, on R2's
objectives: differential evolution over its parameters, then L-BFGS-B with exact (JAX)
gradients over those parameters and every reactor's unmeasured start values together.
It comes after R4, R3 and R2 (the paper's Section 2.3): it keeps every parameter it shares
with them as their calibration left it in the campaign file, and fits only the Monod kinetics
of the four processes they lump away, the uptake of sugars, amino acids, long-chain fatty
acids and hydrogen. Of the starts it fits what nothing measures: its eight biomasses, the
particulates and the products of their hydrolysis (sugars, amino acids, fatty acids), these
six at or below their steady state for the reactor's first-week feed. The dataset sets the
gas phase and the chemistry from the first measurements, as R2's and R3's do.

Those eight entries of the campaign's parameter file and the ``base-IVs`` files are
updated in place. A run first copies them to ``data/<campaign>/calibration-backup/<time>/``
and writes ``report.json`` there.

    python scripts/calibrate/base_calibration.py --campaign LaRA-temp
    python scripts/calibrate/base_calibration.py --campaign rice-bran --dry-run
    python scripts/calibrate/base_calibration.py --campaign LaRA-substrate --reset-particulates
"""

import os
import sys
from pathlib import Path

# Must run before JAX is imported: _bootstrap selects the platform and puts the
# repository on sys.path so the script works without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # calibrate_benchmarks
from _bootstrap import configure_jax  # noqa: E402

configure_jax()
if __name__ == "__main__":
    # As in calibrate_benchmarks.py: the global stage tries parameter sets the solver can
    # take a million steps over. Read by utils/solver.py on import.
    os.environ.setdefault("ADM1_SOLVER_MAX_STEPS", "20000")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import calibrate_benchmarks as two_stage  # noqa: E402
from adm1_package.experiments.registry import DATASETS  # noqa: E402

KEY = "base"


def own_files(campaign_dir, campaign):
    """What a run replaces: the campaign's parameter file and the full ADM1's starts.
    LaRA-substrate's are named ``Base-IVs``, which a case-sensitive glob would miss."""
    starts = [path for path in campaign_dir.iterdir()
              if path.name.lower().startswith("base-ivs_") and path.suffix == ".csv"]
    return [campaign_dir / f"{campaign}-parameters.json", *sorted(starts)]


def backup(campaign_dir, campaign, stamp):
    target = campaign_dir / "calibration-backup" / stamp
    target.mkdir(parents=True)
    for path in own_files(campaign_dir, campaign):
        shutil.copy2(path, target / path.name)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--campaign", required=True, choices=DATASETS)
    parser.add_argument("--de-maxiter", type=int, default=two_stage.DE_MAXITER,
                        help=f"generations of the global stage (default {two_stage.DE_MAXITER})")
    parser.add_argument("--de-popsize", type=int, default=two_stage.DE_POPSIZE,
                        help=f"population per parameter (default {two_stage.DE_POPSIZE})")
    parser.add_argument("--local-maxiter", type=int, default=two_stage.LOCAL_MAXITER,
                        help="L-BFGS-B iterations of the local stage "
                             f"(default {two_stage.LOCAL_MAXITER})")
    parser.add_argument("--seed", type=int, default=two_stage.SEED)
    parser.add_argument("--reset-particulates", action="store_true",
                        help="start every reactor's particulates (X_ch, X_pr, X_li) and their "
                             "products of hydrolysis (S_su, S_aa, S_fa) from the model's "
                             "steady state at its first-week forcing, not from the stored "
                             "values")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("ADM1_DATA_DIR", "data")))
    parser.add_argument("--dry-run", action="store_true",
                        help="fit and report, but write nothing")
    args = parser.parse_args(argv)
    # The full ADM1 keeps every parameter it shares with the reduced variants, which must
    # have been fitted to the forcing it is given.
    from adm1_package.experiments.calibrations import campaign_forcing
    from adm1_package.utils.forcing import FORCINGS

    stored = campaign_forcing(args.data_dir, args.campaign)
    if stored != FORCINGS[0]:
        parser.error(f"{args.campaign}'s reduced variants were calibrated to the {stored} forcing; "
                     "calibrate them first (calibrate_benchmarks.py), whose parameters the full "
                     "ADM1 keeps")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = args.data_dir / args.campaign
    folder = None if args.dry_run else backup(campaign_dir, args.campaign, stamp)
    print(f"{args.campaign}: calibrating the full ADM1"
          + (" (dry run)" if args.dry_run else f"; backup in {folder}"), flush=True)

    began = time.perf_counter()
    # no values from an earlier step: the shared parameters are the campaign file's
    report = {"campaign": args.campaign, "variants": {
        KEY: two_stage.calibrate_variant(KEY, args.campaign, args.data_dir, {}, args)}}
    print(f"{args.campaign}: done in {time.perf_counter() - began:.0f} s", flush=True)
    if folder is not None:
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        print(f"report: {folder / 'report.json'}")


if __name__ == "__main__":
    main()
