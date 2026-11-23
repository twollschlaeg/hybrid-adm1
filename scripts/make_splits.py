"""Create the canonical train / validation / test split.

Run this once; training, hyperparameter search and evaluation all read the file
it writes, so the fold assignment can never be re-derived differently on another
machine.

    python scripts/make_splits.py                  # write configs/split.json
    python scripts/make_splits.py --show           # print the existing split
    python scripts/make_splits.py --k-folds 5 --force

The split is checked against all three ADM1 variants: they must enumerate the
same reactors, otherwise a fold would mean different things per variant.
"""

from __future__ import annotations

import argparse

from _bootstrap import add_common_arguments, configure_jax  # noqa: E402  (path setup)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--k-folds", type=int, default=5, help="Cross-validation folds (default: 5).")
    parser.add_argument(
        "--test-per-campaign",
        type=int,
        default=0,
        help="Reactors held out per campaign as the test set (default: 0, no test set).",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Shuffle globally instead of spreading each campaign across the folds.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing split file.")
    parser.add_argument("--show", action="store_true", help="Print the existing split and exit.")
    arguments = parser.parse_args()

    configure_jax()

    from adm1_package.experiments.registry import MODEL_KEYS, get_spec
    from adm1_package.experiments.splits import load_split, make_split, save_split

    if arguments.show:
        print(load_split(arguments.split).summary())
        return 0

    if arguments.split.exists() and not arguments.force:
        print(f"{arguments.split} already exists. Re-run with --force to replace it.")
        print("Replacing the split invalidates every existing run.")
        return 1

    label_sets = {}
    for key in MODEL_KEYS:
        dataset = get_spec(key).dataset(arguments.data_dir)
        label_sets[key] = sorted(dataset.labels)
        print(f"{key}: {len(label_sets[key])} reactors")

    reference = label_sets[MODEL_KEYS[0]]
    mismatched = {key: labels for key, labels in label_sets.items() if labels != reference}
    if mismatched:
        for key, labels in mismatched.items():
            print(f"  {key}: {labels}")
        raise SystemExit(
            "the model variants do not enumerate the same reactors; a shared split would be "
            "meaningless. Check the dataset constructors in adm1_package/experiments/registry.py."
        )

    split = make_split(
        reference,
        k_folds=arguments.k_folds,
        seed=arguments.seed,
        test_per_campaign=arguments.test_per_campaign,
        stratified=not arguments.no_stratify,
        # The note travels with the split into every run directory, so it has to
        # describe the split that was actually made. A fixed sentence about
        # held-out test reactors would be wrong whenever --test-per-campaign is 0.
        note=(
            "Canonical split for the ADM1-UDE study. The folds are used for both "
            "cross-validated training and hyperparameter selection."
            + (
                " The test reactors are held out from both."
                if arguments.test_per_campaign > 0
                else " There is no hold-out test set: the same folds inform hyperparameter"
                     " selection and the reported scores, so those scores are a mildly"
                     " optimistic estimate of performance on an unseen reactor."
            )
        ),
    )

    path = save_split(split, arguments.split)
    print()
    print(split.summary())
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
