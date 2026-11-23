"""Prepare a nested hyperparameter search: inner splits and their calibrations.

    python scripts/make_nested.py --benchmarks runs/benchmarks-smoothstep-fast \\
        --output runs/nested-fast
    python scripts/make_nested.py --benchmarks runs/benchmarks-smoothstep-fast \\
        --output runs/nested-fast --show

For every outer fold k of the split, the outer fold's training reactors are dealt
into inner folds, stratified by campaign; its held-out reactors become the inner
split's test set, which the search never touches. Each inner fold is built on outer
fold k's calibration from --benchmarks, which was fitted on the same reactors the
inner folds divide (adm1_package/experiments/nested.py says what that costs).

It writes <output>/outer_<k>/split.json and <output>/outer_<k>/benchmarks/, which
are what scripts/tune_hyperparameters.py takes as --split and --benchmarks.
scripts/hpc/submit_nested.sh runs this before it submits. The copies hold reactor
data, so keep <output> under runs/, which git ignores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_common_arguments  # noqa: E402  (path setup)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--benchmarks", type=Path, required=True,
                        help="The outer folds' per-fold calibrations (calibrate_folds.py), "
                             "e.g. runs/benchmarks-smoothstep-fast.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where the inner splits and calibrations go, e.g. runs/nested-fast.")
    parser.add_argument("--inner-folds", type=int, default=4,
                        help="Inner folds per outer fold (default 4: three reactors each).")
    parser.add_argument("--show", action="store_true",
                        help="Only print the inner splits an existing preparation holds.")
    arguments = parser.parse_args()

    from adm1_package.experiments.nested import (
        MANIFEST,
        inner_benchmarks,
        inner_split_path,
        prepare_nested,
    )
    from adm1_package.experiments.splits import load_split

    if arguments.show:
        manifest = json.loads((arguments.output / MANIFEST).read_text(encoding="utf-8"))
        for outer in sorted(manifest["outer"], key=int):
            print(f"outer fold {outer}:")
            print(load_split(inner_split_path(arguments.output, int(outer))).summary())
        return 0

    split = load_split(arguments.split)
    inner_splits = prepare_nested(split, arguments.benchmarks, arguments.output,
                                  k_folds=arguments.inner_folds)
    for outer, inner in inner_splits.items():
        print(f"outer fold {outer}: the search never sees {', '.join(inner.test)}")
        for index, fold in enumerate(inner.folds, start=1):
            print(f"  inner fold {index} validation: {', '.join(fold)}")
    print(f"\nWrote {arguments.output}: per outer fold k, --split "
          f"{inner_split_path(arguments.output, 'k')} and --benchmarks "
          f"{inner_benchmarks(arguments.output, 'k')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
