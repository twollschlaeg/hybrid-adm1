"""What a nested search selected, per variant and outer fold.

    python scripts/nested_summary.py \\
        --pattern "runs/{model}/tuning_nested-fast_outer{fold}/best_hyperparameters.json"
    python scripts/nested_summary.py --models r4 --pattern "..." --output selections.csv

For every outer fold: how many trials completed, the selected values, the inner
validation loss of the selection and of the default values (when the search
started from them, --start-from-defaults), and their difference. The default values
are the ones the study recorded at its start, so a search space changed since does not
move them. The inner
losses compare configurations within one outer fold; across outer folds they rest on
different reactors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _bootstrap import add_common_arguments  # noqa: E402  (path setup)

COLUMNS = ("width_size", "depth", "dropout_rate", "learning_rate", "regularization_lambda",
           "batch_size", "correction_penalty", "warmup_steps", "num_epochs")


def _default_loss(trials: Path, defaults: dict | None) -> float:
    """The inner loss of the trial that ran the default values, NaN if none did."""
    import pandas as pd

    if defaults is None or not trials.exists():
        return math.nan
    frame = pd.read_csv(trials)
    frame = frame[frame["state"] == "COMPLETE"]
    for name, value in defaults.items():
        if f"params_{name}" not in frame:
            return math.nan
        column = frame[f"params_{name}"]
        frame = frame[column.astype(str) == str(value)] if isinstance(value, str) else \
            frame[(column - value).abs() <= 1e-12 * max(1.0, abs(value))]
    return float(frame["value"].iloc[0]) if len(frame) else math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--pattern", required=True,
                        help="The selections, with {model} and {fold}, as train_cv.py "
                             "--fold-hyperparameters takes them.")
    parser.add_argument("--models", nargs="+", default=["r4", "r3", "r2"])
    parser.add_argument("--output", type=Path, default=None, help="Also write the table as CSV.")
    arguments = parser.parse_args()

    import pandas as pd

    from adm1_package.experiments.splits import load_split

    split = load_split(arguments.split)
    rows = []
    for model in arguments.models:
        for fold in range(1, split.k_folds + 1):
            path = Path(arguments.pattern.replace("{model}", model).replace("{fold}", str(fold)))
            if not path.exists():
                print(f"{model} outer fold {fold}: no selection at {path}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            selected = payload["hyperparameters"]
            defaults = payload.get("started_from")
            reference = _default_loss(path.parent / "trials.csv", defaults)
            rows.append({
                "model": model,
                "outer_fold": fold,
                "trials": payload.get("n_trials_completed"),
                "best_trial": payload.get("best_trial"),
                "inner_loss": payload.get("mean_val_loss"),
                "inner_loss_default": reference,
                "change_percent": 100 * (payload.get("mean_val_loss") / reference - 1),
                "default_selected": None if defaults is None else
                all(selected[name] == value for name, value in defaults.items()),
                **{name: selected[name] for name in COLUMNS},
            })
    if not rows:
        print("no selection found")
        return 1
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    if arguments.output is not None:
        table.to_csv(arguments.output, index=False)
        print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
