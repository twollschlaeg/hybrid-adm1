"""LaTeX tables, generated from the evaluation output.

Every table is written from the same dataframes the CSVs come from, so a table
cannot drift apart from the results it reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "objective_rmse_table",
    "overall_metrics_table",
    "comparison_table",
    "write_tables",
]

#: Display names and units of the reported objectives.
OBJECTIVE_LABELS = {
    "q_gasout": (r"\makecell{Biogas \\ production}", "NL/d"),
    "ch4_out": (r"\makecell{Methane \\ production}", "NL/d"),
    "S_ac": (r"\makecell{Acetic \\ acid}", "g/L"),
    "S_pro": (r"\makecell{Propionic \\ acid}", "g/L"),
    "S_TAN": ("TAN", "g/L"),
    "S_va": (r"\makecell{Valeric \\ acid}", "g/L"),
    "S_bu": (r"\makecell{Butyric \\ acid}", "g/L"),
    "pH": ("pH", "-"),
    "S_h2o": ("Water", "g/L"),
}

SPLIT_LABELS = {
    "train": "training",
    "validation": "cross-validated held-out",
    "test": "hold-out test",
}


def _mean_std(mean: float, std: float, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def _tabular(header: Sequence[str], rows: Sequence[Sequence[str]], column_spec: str) -> str:
    lines = [
        r"\begin{tabular}{" + column_spec + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def objective_rmse_table(
    objectives_aggregated: pd.DataFrame,
    split: str = "validation",
    objectives: Sequence[str] | None = None,
    label: str = "tab:rmse_objectives_cv",
) -> str:
    """Per-objective RMSE as mean +/- std over folds."""
    frame = objectives_aggregated[objectives_aggregated["split"] == split]
    if frame.empty:
        raise ValueError(f"no rows for split {split!r}")

    if objectives is None:
        ordered = [name for name in OBJECTIVE_LABELS if name in set(frame["objective"])]
        objectives = [name for name in ordered if frame.loc[frame["objective"] == name, "reported"].any()]

    header = ["Model"] + [OBJECTIVE_LABELS.get(name, (name, ""))[0] for name in objectives]

    rows = []
    for (model, variant), group in frame.groupby(["model", "variant"], sort=False):
        indexed = group.set_index("objective")
        cells = [f"{model} {variant}"]
        for name in objectives:
            if name not in indexed.index:
                cells.append("--")
                continue
            cells.append(_mean_std(indexed.loc[name, "rmse_mean"], indexed.loc[name, "rmse_std"]))
        rows.append(cells)

    rows.sort(key=lambda row: (row[0].split()[0], VARIANT_ORDER.get(row[0].split()[-1], 9)))

    caption = (
        f"Root mean squared error per objective on the {SPLIT_LABELS.get(split, split)} data, "
        r"reported as mean $\pm$ standard deviation over the cross-validation folds "
        "(L/d for gas production rates, g/L for concentrations). "
        r"ADM1: Anaerobic Digestion Model No.~1; R2/R3/R4: reduced ADM1 variants with "
        r"decreasing complexity; MECH: purely mechanistic model, calibrated on all reactors; "
        r"MECH-CV: the same, calibrated on each fold's training reactors; UDE: Universal "
        r"Differential Equation hybrid model; TAN: total ammoniacal nitrogen. "
        r"Dashes (--) indicate that the respective variable is not part of the model structure."
    )
    body = _tabular(header, rows, "l" + "r" * len(objectives))
    return (
        "\\begin{table*}[!h]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\n\\end{{table*}}\n"
    )


def overall_metrics_table(
    aggregated: pd.DataFrame,
    split: str = "validation",
    label: str = "tab:overall_metrics_cv",
) -> str:
    """Pooled NRMSE / MAPE / R^2 table."""
    frame = aggregated[aggregated["split"] == split].copy()
    if frame.empty:
        raise ValueError(f"no rows for split {split!r}")

    frame["sort_key"] = frame["model"] + frame["variant"].map(VARIANT_ORDER).astype(str)
    frame = frame.sort_values("sort_key")

    header = ["Model", "NRMSE", "MAPE (\\%)", "sMAPE (\\%)", "$R^2$", "Params"]
    rows = [
        [
            f"{row['model']} {row['variant']}",
            _mean_std(row["nrmse_mean"], row["nrmse_std"]),
            _mean_std(row["mape_mean"], row["mape_std"], digits=1),
            _mean_std(row["smape_mean"], row["smape_std"], digits=1),
            _mean_std(row["r2_mean"], row["r2_std"], digits=3),
            f"{int(row['n_parameters'])}" if "n_parameters" in frame.columns else "--",
        ]
        for _, row in frame.iterrows()
    ]

    caption = (
        f"Aggregated prediction metrics on the {SPLIT_LABELS.get(split, split)} data, "
        r"mean $\pm$ standard deviation over the cross-validation folds. Lower is better "
        r"except for $R^2$. NRMSE: range-normalized root mean squared error "
        r"(Eq.~\ref{eq:normalized-rmse}); MAPE: mean absolute percentage error; "
        r"sMAPE: symmetric mean absolute percentage error, which stays bounded when "
        r"measurements approach the detection limit; Params: number of trainable "
        r"neural-network parameters. MECH: purely mechanistic, calibrated on all reactors; "
        r"MECH-CV: purely mechanistic, calibrated on each fold's training reactors; "
        r"UDE: hybrid."
    )
    body = _tabular(header, rows, "lrrrrr")
    return (
        "\\begin{table}[!h]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\n\\end{{table}}\n"
    )


#: What each benchmark is, for the captions.
BENCHMARK_DESCRIPTIONS = {
    "MECH": "the purely mechanistic variant (MECH), calibrated on every reactor of each "
            "campaign, the held-out reactors included",
    "MECH-CV": "the purely mechanistic variant calibrated anew for every fold on that "
               "fold's training reactors only (MECH-CV), which learned from the same "
               "reactors as the fold's hybrid model",
}

#: Table order of the variants: the benchmarks, then the hybrid.
VARIANT_ORDER = {"MECH": 0, "MECH-CV": 1, "UDE": 2}


def comparison_table(
    comparisons: pd.DataFrame,
    split: str = "validation",
    metric: str = "nrmse",
    label: str = "tab:paired_comparison",
    benchmark: str = "MECH",
) -> str:
    """Paired hybrid-vs-mechanistic statistics.

    ``benchmark`` picks the mechanistic variant the hybrid is paired with; results
    written before there was a choice compare with ``MECH``.
    """
    if comparisons.empty:
        raise ValueError("no paired comparisons available (at least two folds are needed)")

    against = (comparisons["benchmark"] if "benchmark" in comparisons
               else pd.Series("MECH", index=comparisons.index))
    frame = comparisons[
        (comparisons["split"] == split)
        & (comparisons["metric"] == metric)
        & (comparisons["objective"] == "all")
        & (against == benchmark)
    ]
    if frame.empty:
        raise ValueError(f"no comparison rows for split={split!r}, metric={metric!r}, "
                         f"benchmark={benchmark!r}")

    header = [
        "Model",
        benchmark,
        "UDE",
        r"$\Delta$ (95\% CI)",
        r"Change (95\% CI)",
        "Folds won",
        "$p$ (sign)",
        "$p$ (Wilcoxon)",
    ]

    def interval(value):
        return value if isinstance(value, tuple) else (float("nan"), float("nan"))

    rows = []
    for _, row in frame.iterrows():
        low, high = interval(row["ci95_difference"])
        relative_low, relative_high = interval(row.get("ci95_relative_change_percent"))
        rows.append(
            [
                str(row["model"]),
                f"{row['mean_mechanistic']:.4f}",
                f"{row['mean_hybrid']:.4f}",
                f"${row['mean_difference']:.4f}$ (${low:.4f}$, ${high:.4f}$)",
                f"${row['mean_relative_change_percent']:.1f}\\%$ "
                f"(${relative_low:.1f}$, ${relative_high:.1f}$)",
                f"{int(row['wins'])}/{int(row['n_folds'])}",
                f"{row['sign_test_p']:.4f}",
                f"{row['wilcoxon_p']:.4f}",
            ]
        )

    caption = (
        f"Paired comparison of the hybrid (UDE) with "
        f"{BENCHMARK_DESCRIPTIONS.get(benchmark, benchmark)}, on the "
        f"{SPLIT_LABELS.get(split, split)} data, using the {metric.upper()} of each "
        r"cross-validation fold. Each fold contributes one paired observation, evaluated on "
        f"the same held-out reactors. $\\Delta$ is the mean UDE~$-$~{benchmark} difference "
        r"and Change the mean of the folds' relative changes, each with a 95\% confidence "
        r"interval from the $t$ distribution over the folds; negative favors the hybrid "
        r"model. $p$ values are one-sided exact sign and Wilcoxon signed-rank tests."
    )
    body = _tabular(header, rows, "lrrlrrrr")
    return (
        "\\begin{table*}[!h]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\n\\end{{table*}}\n"
    )


def write_tables(result, directory: Path, splits: Sequence[str] = ("validation", "test", "train")) -> list[Path]:
    """Emit every table for the available splits."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    available = set(result.aggregated["split"])
    for split in splits:
        if split not in available:
            continue
        for name, builder in (
            ("rmse_objectives", lambda: objective_rmse_table(
                result.objectives_aggregated, split, label=f"tab:rmse_objectives_{split}")),
            ("overall_metrics", lambda: overall_metrics_table(
                result.aggregated, split, label=f"tab:overall_metrics_{split}")),
            ("paired_comparison", lambda: comparison_table(
                result.comparisons, split, label=f"tab:paired_comparison_{split}")),
            *([("paired_comparison_cv", lambda: comparison_table(
                result.comparisons, split, label=f"tab:paired_comparison_cv_{split}",
                benchmark="MECH-CV"))]
              if "benchmark" in result.comparisons
              and (result.comparisons["benchmark"] == "MECH-CV").any() else []),
        ):
            try:
                content = builder()
            except (ValueError, KeyError) as exc:
                print(f"  skipped {name} ({split}): {exc}")
                continue
            path = directory / f"{name}_{split}.tex"
            path.write_text(content, encoding="utf-8")
            written.append(path)
    return written
