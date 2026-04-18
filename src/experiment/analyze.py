"""Few-shot voice cloning experiment analysis.

Loads results CSV, computes summary statistics and significance tests,
adds ASR-calibrated WER diagnostics, and generates publication-ready plots.

Usage:
    python scripts/analyze_fewshot.py
    python scripts/analyze_fewshot.py --results-csv outputs/fewshot/phase3/results.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, avoid Type 3 fonts in PDFs
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True  # avoid embedded font subsets entirely
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

METRICS = [
    {"key": "utmos", "label": "UTMOS", "higher_is_better": True, "fmt": ".3f", "arrow": "^"},
    {"key": "speaker_sim", "label": "Speaker Sim", "higher_is_better": True, "fmt": ".4f", "arrow": "^"},
    {"key": "wer", "label": "WER", "higher_is_better": False, "fmt": ".1%", "arrow": "v"},
]

APPROACH_ORDER = ["single_baseline", "concat_audio", "concat_code", "embed_avg"]
APPROACH_LABELS = {
    "single_baseline": "Single Baseline",
    "concat_audio": "Concat Audio",
    "concat_code": "Concat Code",
    "embed_avg": "Embed Avg",
}
APPROACH_COLORS = {
    "single_baseline": "#636363",
    "concat_audio": "#1f77b4",
    "concat_code": "#d62728",
    "embed_avg": "#2ca02c",
}
STRATEGY_MARKERS = {"random": "o", "longest": "s"}

GROUP_COLS = ["approach", "n_refs", "strategy"]

REQUIRED_COLUMNS = {
    "speaker_id",
    "target_id",
    "approach",
    "n_refs",
    "strategy",
    "seed",
    "utmos",
    "speaker_sim",
    "wer",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze few-shot voice cloning experiment results.")
    parser.add_argument("--results-csv", default="outputs/fewshot/results.csv", help="Path to results CSV.")
    parser.add_argument("--output-dir", default="outputs/fewshot/analysis/", help="Directory for analysis outputs.")
    parser.add_argument("--format", default="table", choices=["table", "json", "latex"], help="Output format for summary tables.")
    parser.add_argument("--baseline", default="single_baseline", help="Approach name to use as baseline.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold.")
    parser.add_argument("--drop-approaches", nargs="*", default=["concat_code"], help="Approaches to exclude from analysis.")
    parser.add_argument("--save-formats", nargs="+", default=["png", "pdf"], help="File formats for plots.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--manifest", default="data/libritts_r_aligned/manifest.json", help="Manifest for ASR calibration.")
    parser.add_argument("--whisper-model", default="turbo", help="Whisper model for GT calibration.")
    parser.add_argument("--skip-wer-calibration", action="store_true", help="Skip Whisper-on-ground-truth calibration.")
    parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.5, 1.0], help="WER thresholds for failure rate.")
    return parser


def load_and_validate(csv_path: str | Path, drop_approaches: list[str]) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for metric in METRICS:
        df[metric["key"]] = pd.to_numeric(df[metric["key"]], errors="coerce")

    df = df.dropna(subset=[m["key"] for m in METRICS], how="all").copy()

    if drop_approaches:
        before = len(df)
        df = df[~df["approach"].isin(drop_approaches)].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows from approaches: {drop_approaches}")

    df["n_refs"] = df["n_refs"].astype(int)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Speakers: {sorted(df['speaker_id'].astype(str).unique())}")
    print(f"  Approaches: {sorted(df['approach'].unique())}")
    print(f"  n_refs: {sorted(df['n_refs'].unique())}")
    print(f"  Strategies: {sorted(df['strategy'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(GROUP_COLS, sort=False)
    rows = []
    for name, group in grouped:
        approach, n_refs, strategy = name
        row = {"approach": approach, "n_refs": n_refs, "strategy": strategy}
        for m in METRICS:
            key = m["key"]
            vals = group[key].dropna()
            row[f"{key}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{key}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            row[f"{key}_median"] = vals.median() if len(vals) else np.nan
            row[f"{key}_q25"] = vals.quantile(0.25) if len(vals) else np.nan
            row[f"{key}_q75"] = vals.quantile(0.75) if len(vals) else np.nan
            row[f"{key}_count"] = len(vals)
        rows.append(row)

    summary = pd.DataFrame(rows)
    rank = {name: idx for idx, name in enumerate(APPROACH_ORDER)}
    summary["_rank"] = summary["approach"].map(rank).fillna(99)
    summary = summary.sort_values(["_rank", "n_refs", "strategy"]).drop(columns=["_rank"]).reset_index(drop=True)
    return summary


def add_failure_metrics(summary: pd.DataFrame, df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    out = summary.copy()
    grouped = df.groupby(GROUP_COLS, sort=False)
    for threshold in thresholds:
        col = f"failure_rate_wer_gt_{str(threshold).replace('.', '_')}"
        values = []
        for _, row in out.iterrows():
            g = grouped.get_group((row["approach"], row["n_refs"], row["strategy"]))
            wer = g["wer"].dropna()
            values.append(float((wer > threshold).mean()) if len(wer) else np.nan)
        out[col] = values
    return out


def compute_per_speaker_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["speaker_id"] + GROUP_COLS
    rows = []
    for name, group in df.groupby(cols, sort=False):
        speaker_id, approach, n_refs, strategy = name
        row = {
            "speaker_id": speaker_id,
            "approach": approach,
            "n_refs": n_refs,
            "strategy": strategy,
            "utmos_mean": group["utmos"].mean(),
            "speaker_sim_mean": group["speaker_sim"].mean(),
            "wer_mean": group["wer"].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("speaker_sim_mean", ascending=False).reset_index(drop=True)


def compute_significance_tests(df: pd.DataFrame, baseline: str, alpha: float) -> pd.DataFrame:
    baseline_df = df[df["approach"] == baseline].copy()
    if baseline_df.empty:
        warnings.warn(f"No baseline rows for approach='{baseline}'.")
        return pd.DataFrame()

    non_baseline = df[df["approach"] != baseline].copy()
    configs = non_baseline.groupby(GROUP_COLS).size().reset_index(name="n")
    rows = []

    for _, config in configs.iterrows():
        approach = config["approach"]
        n_refs = config["n_refs"]
        strategy = config["strategy"]

        cfg = non_baseline[
            (non_baseline["approach"] == approach)
            & (non_baseline["n_refs"] == n_refs)
            & (non_baseline["strategy"] == strategy)
        ]

        paired = cfg.merge(
            baseline_df,
            on=["speaker_id", "target_id", "seed"],
            suffixes=("_exp", "_bl"),
        )

        for metric in METRICS:
            key = metric["key"]
            exp = paired[f"{key}_exp"].dropna()
            bl = paired[f"{key}_bl"].dropna()
            valid = exp.index.intersection(bl.index)
            exp_vals = exp.loc[valid].values
            bl_vals = bl.loc[valid].values

            row = {
                "approach": approach,
                "n_refs": n_refs,
                "strategy": strategy,
                "metric": key,
                "n_pairs": len(exp_vals),
            }

            if len(exp_vals) < 3:
                row.update(
                    {
                        "test": "skipped",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect_size": np.nan,
                        "note": f"Only {len(exp_vals)} pair(s), need >=3",
                    }
                )
                rows.append(row)
                continue

            diffs = exp_vals - bl_vals
            std = diffs.std(ddof=1)
            row["effect_size"] = diffs.mean() / std if std > 0 else 0.0

            if len(exp_vals) < 10:
                stat, p = stats.ttest_rel(exp_vals, bl_vals)
                row["test"] = "paired_t"
            else:
                try:
                    stat, p = stats.wilcoxon(exp_vals, bl_vals)
                    row["test"] = "wilcoxon"
                except ValueError:
                    row.update({"test": "wilcoxon", "statistic": 0.0, "p_value": 1.0, "note": "all differences zero"})
                    rows.append(row)
                    continue

            row["statistic"] = stat
            row["p_value"] = p
            row.setdefault("note", "")
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    sig = pd.DataFrame(rows)
    valid = sig["p_value"].notna() & (sig["test"] != "skipped")
    n_tests = int(valid.sum())
    sig["p_corrected"] = np.nan
    sig["significant"] = False
    if n_tests > 0:
        sig.loc[valid, "p_corrected"] = (sig.loc[valid, "p_value"] * n_tests).clip(upper=1.0)
        sig.loc[valid, "significant"] = sig.loc[valid, "p_corrected"] < alpha
    return sig


def _fmt_metric(mean: float, std: float, count: int, fmt: str) -> str:
    if pd.isna(mean):
        return "-"
    mean_str = format(mean, fmt)
    if count > 1 and not pd.isna(std) and std > 0:
        return f"{mean_str} +/- {format(std, fmt)}"
    return mean_str


def _fmt_metric_median_iqr(median: float, q25: float, q75: float, fmt: str) -> str:
    if pd.isna(median):
        return "-"
    return f"{format(median, fmt)} [{format(q25, fmt)}, {format(q75, fmt)}]"


def format_summary_table(summary: pd.DataFrame, fmt: str = "table") -> str:
    if fmt == "json":
        return json.dumps(summary.to_dict(orient="records"), indent=2)
    if fmt == "latex":
        return format_latex_table(summary)

    lines = []
    header = f"{'Approach':<20} {'n':>3} {'Strategy':<10} {'UTMOS ^':>20} {'Speaker Sim ^':>20} {'WER v (median[IQR])':>26}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in summary.iterrows():
        label = APPROACH_LABELS.get(row["approach"], row["approach"])
        utmos = _fmt_metric(row["utmos_mean"], row["utmos_std"], row["utmos_count"], ".3f")
        sim = _fmt_metric(row["speaker_sim_mean"], row["speaker_sim_std"], row["speaker_sim_count"], ".4f")
        wer = _fmt_metric_median_iqr(row["wer_median"], row["wer_q25"], row["wer_q75"], ".1%")
        lines.append(f"{label:<20} {int(row['n_refs']):>3} {row['strategy']:<10} {utmos:>20} {sim:>20} {wer:>26}")
    return "\n".join(lines)


def format_latex_table(summary: pd.DataFrame) -> str:
    best_sim = summary["speaker_sim_mean"].max()
    best_utmos = summary["utmos_mean"].max()
    lines = []
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Approach} & \textbf{Strategy} & \textbf{\#Refs} & \textbf{UTMOS} $\uparrow$ & \textbf{Speaker SIM} $\uparrow$ \\")
    lines.append(r"\midrule")
    for _, row in summary.iterrows():
        label = APPROACH_LABELS.get(row["approach"], row["approach"])
        utmos = _fmt_metric(row["utmos_mean"], row["utmos_std"], row["utmos_count"], ".3f")
        sim = _fmt_metric(row["speaker_sim_mean"], row["speaker_sim_std"], row["speaker_sim_count"], ".4f")
        if abs(row["utmos_mean"] - best_utmos) < 1e-12:
            utmos = rf"\textbf{{{utmos}}}"
        if abs(row["speaker_sim_mean"] - best_sim) < 1e-12:
            sim = rf"\textbf{{{sim}}}"
        lines.append(f"{label} & {row['strategy']} & {int(row['n_refs'])} & {utmos} & {sim} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def format_significance_table(sig_df: pd.DataFrame) -> str:
    if sig_df.empty:
        return "No significance tests computed."

    header = (
        f"{'Approach':<15} {'n':>3} {'Strategy':<10} {'Metric':<12} {'Test':<10} "
        f"{'Pairs':>5} {'p':>10} {'p_corr':>10} {'Sig':>5} {'d':>8} {'Note'}"
    )
    lines = [header, "-" * len(header)]
    for _, row in sig_df.iterrows():
        p = f"{row['p_value']:.4f}" if not pd.isna(row["p_value"]) else "-"
        pc = f"{row['p_corrected']:.4f}" if not pd.isna(row.get("p_corrected")) else "-"
        d = f"{row['effect_size']:.3f}" if not pd.isna(row.get("effect_size")) else "-"
        sig = "yes" if bool(row.get("significant", False)) else "no"
        lines.append(
            f"{row['approach']:<15} {int(row['n_refs']):>3} {row['strategy']:<10} {row['metric']:<12} "
            f"{row['test']:<10} {int(row['n_pairs']):>5} {p:>10} {pc:>10} {sig:>5} {d:>8} {row.get('note','')}"
        )
    return "\n".join(lines)


def format_top_n(df: pd.DataFrame, n: int = 5) -> str:
    summary = compute_summary_stats(df)
    top = summary.nlargest(n, "speaker_sim_mean")
    lines = [f"Top {n} configurations by speaker similarity:", ""]
    header = f"{'Rank':>4} {'Approach':<20} {'n':>3} {'Strategy':<10} {'SpkSim':>12} {'UTMOS':>12} {'WER med[IQR]':>20}"
    lines.extend([header, "-" * len(header)])
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        label = APPROACH_LABELS.get(row["approach"], row["approach"])
        sim = _fmt_metric(row["speaker_sim_mean"], row["speaker_sim_std"], row["speaker_sim_count"], ".4f")
        utmos = _fmt_metric(row["utmos_mean"], row["utmos_std"], row["utmos_count"], ".3f")
        wer = _fmt_metric_median_iqr(row["wer_median"], row["wer_q25"], row["wer_q75"], ".1%")
        lines.append(f"{rank:>4} {label:<20} {int(row['n_refs']):>3} {row['strategy']:<10} {sim:>12} {utmos:>12} {wer:>20}")
    return "\n".join(lines)


def _get_plot_approaches(df: pd.DataFrame) -> list[str]:
    present = set(df["approach"].unique())
    return [a for a in APPROACH_ORDER if a in present]


def _save_plot(fig: plt.Figure, output_dir: Path, name: str, formats: list[str]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(plots_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_tradeoff(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    summary = compute_summary_stats(df)
    # Column-friendly size: avoid generating a huge canvas that will be heavily downscaled in LaTeX.
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    # Encode configuration metadata visually so we don't need unreadable per-point labels.
    size_map = {1: 26, 2: 34, 3: 44, 5: 62}

    for _, row in summary.iterrows():
        approach = row["approach"]
        strategy = row["strategy"]
        n_refs = int(row["n_refs"])
        color = APPROACH_COLORS.get(approach, "#777777")
        marker = STRATEGY_MARKERS.get(strategy, "o")
        x = row["speaker_sim_mean"]
        y = row["utmos_mean"]
        xerr = row["speaker_sim_std"]
        yerr = row["utmos_std"]

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=1.5,
            alpha=0.35,
            zorder=1,
        )
        ax.scatter(
            [x],
            [y],
            s=size_map.get(n_refs, 40),
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
            zorder=2,
        )

    # Annotate only the key reference points.
    baseline = summary[(summary["approach"] == "single_baseline")].head(1)
    if not baseline.empty:
        bx, by = float(baseline["speaker_sim_mean"].iloc[0]), float(baseline["utmos_mean"].iloc[0])
        ax.scatter([bx], [by], s=80, marker="D", c=APPROACH_COLORS["single_baseline"], edgecolors="black", linewidths=0.7, zorder=3)
        ax.annotate("Baseline", (bx, by), textcoords="offset points", xytext=(6, -10), fontsize=8)

    best_sim = summary.loc[summary["speaker_sim_mean"].idxmax()]
    ax.scatter(
        [best_sim["speaker_sim_mean"]],
        [best_sim["utmos_mean"]],
        s=110,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )
    ax.annotate(
        "Best SIM",
        (best_sim["speaker_sim_mean"], best_sim["utmos_mean"]),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=8,
    )

    best_utmos = summary.loc[summary["utmos_mean"].idxmax()]
    ax.scatter(
        [best_utmos["speaker_sim_mean"]],
        [best_utmos["utmos_mean"]],
        s=110,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
    )
    ax.annotate(
        "Best UTMOS",
        (best_utmos["speaker_sim_mean"], best_utmos["utmos_mean"]),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=8,
    )

    # Legend: approach (color), strategy (marker), n_refs (size).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    approach_handles = [
        Patch(facecolor=APPROACH_COLORS[a], edgecolor="none", label=APPROACH_LABELS.get(a, a))
        for a in APPROACH_ORDER
        if a in set(summary["approach"].unique())
    ]
    strategy_handles = [
        Line2D([0], [0], marker=STRATEGY_MARKERS[s], color="black", linestyle="none", markersize=5, label=s)
        for s in ["random", "longest"]
        if s in set(summary["strategy"].unique())
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="none", markersize=np.sqrt(size_map[n]) * 0.50, label=f"n={n}")
        for n in sorted(set(int(v) for v in summary["n_refs"].unique()))
        if n in size_map
    ]
    handles = approach_handles + strategy_handles + size_handles
    ax.legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        frameon=True,
        framealpha=0.9,
        fontsize=6,
        borderpad=0.25,
        labelspacing=0.25,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    ax.set_xlabel("Speaker Similarity")
    ax.set_ylabel("UTMOS")
    ax.grid(alpha=0.22, linewidth=0.6)
    _save_plot(fig, output_dir, "pareto_tradeoff", formats)


def plot_stability_fail_rate(summary: pd.DataFrame, output_dir: Path, formats: list[str], threshold: float = 0.5) -> None:
    col = f"failure_rate_wer_gt_{str(threshold).replace('.', '_')}"
    if col not in summary.columns:
        return
    plot_df = summary.copy()
    plot_df[col] = (plot_df[col] * 100.0).fillna(0.0)
    nonzero = plot_df[plot_df[col] > 0].sort_values(col, ascending=True).copy()

    # Compact label formatting for readability at column width.
    def _label(row: pd.Series) -> str:
        a = row["approach"]
        s = row["strategy"]
        n = int(row["n_refs"])
        if a == "single_baseline":
            return "baseline"
        a_short = {"concat_audio": "concat", "embed_avg": "embed"}.get(a, a)
        s_short = {"random": "rand", "longest": "long"}.get(s, s)
        return f"{a_short}/{s_short}/n={n}"

    labels = [_label(r) for _, r in nonzero.iterrows()]
    values = nonzero[col].values
    colors = [APPROACH_COLORS.get(a, "#777777") for a in nonzero["approach"]]

    # Column-friendly size: keep canvas close to final rendered size to preserve font readability.
    height = max(1.9, 0.28 * len(values) + 0.95)
    fig, ax = plt.subplots(figsize=(3.6, height))
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f"Failure rate (WER > {threshold}) [%]")
    ax.grid(axis="x", alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="x", labelsize=8)

    # Value labels
    for yi, v in zip(y, values):
        ax.text(v + 0.15, yi, f"{v:.1f}%", va="center", fontsize=8)

    ax.set_xlim(0, max(1.0, float(np.max(values)) * 1.15))
    ax.set_title("Stability (Non-Zero Failure Rates)", fontsize=9)

    name = "stability_fail_rate" if abs(threshold - 0.5) < 1e-9 else f"stability_fail_rate_wer_gt_{str(threshold).replace('.', '_')}"
    _save_plot(fig, output_dir, name, formats)


def plot_speaker_sim_by_approach(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    summary = compute_summary_stats(df)
    approaches = _get_plot_approaches(df)
    n_refs_vals = sorted(df["n_refs"].unique())

    fig, ax = plt.subplots(figsize=(9.4, 5))
    width = 0.8 / max(len(n_refs_vals), 1)

    for i, n_refs in enumerate(n_refs_vals):
        x = []
        y = []
        yerr = []
        for j, approach in enumerate(approaches):
            subset = summary[(summary["approach"] == approach) & (summary["n_refs"] == n_refs)]
            if subset.empty:
                continue
            x.append(j + i * width)
            y.append(subset["speaker_sim_mean"].mean())
            yerr.append(subset["speaker_sim_std"].mean())

        ax.bar(x, y, width=width, yerr=yerr if any(v > 0 for v in yerr) else None, label=f"{n_refs} refs", alpha=0.85, capsize=2)

    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in approaches], rotation=15)
    ax.set_ylabel("Speaker Similarity")
    ax.set_title("Speaker Similarity by Approach x Number of References")
    ax.legend(title="#Refs")
    _save_plot(fig, output_dir, "speaker_sim_by_approach", formats)


def plot_utmos_by_approach(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    summary = compute_summary_stats(df)
    approaches = _get_plot_approaches(df)
    n_refs_vals = sorted(df["n_refs"].unique())

    fig, ax = plt.subplots(figsize=(9.4, 5))
    width = 0.8 / max(len(n_refs_vals), 1)

    for i, n_refs in enumerate(n_refs_vals):
        x = []
        y = []
        yerr = []
        for j, approach in enumerate(approaches):
            subset = summary[(summary["approach"] == approach) & (summary["n_refs"] == n_refs)]
            if subset.empty:
                continue
            x.append(j + i * width)
            y.append(subset["utmos_mean"].mean())
            yerr.append(subset["utmos_std"].mean())

        ax.bar(x, y, width=width, yerr=yerr if any(v > 0 for v in yerr) else None, label=f"{n_refs} refs", alpha=0.85, capsize=2)

    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in approaches], rotation=15)
    ax.set_ylabel("UTMOS")
    ax.set_title("UTMOS by Approach x Number of References")
    ax.legend(title="#Refs")
    _save_plot(fig, output_dir, "utmos_by_approach", formats)


def plot_scaling_curve(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    approaches = _get_plot_approaches(df)
    strategies = sorted(df["strategy"].unique())

    # Column-friendly layout: stack strategies vertically so fonts remain readable at column width.
    fig, axes = plt.subplots(len(strategies), 1, figsize=(3.6, 3.8), sharex=True, squeeze=False)
    for idx, strategy in enumerate(strategies):
        ax = axes[idx][0]
        strat_df = df[df["strategy"] == strategy]
        ymins: list[float] = []
        ymaxs: list[float] = []
        for approach in approaches:
            app = strat_df[strat_df["approach"] == approach]
            if app.empty:
                continue
            grouped = app.groupby("n_refs")["speaker_sim"]
            means = grouped.mean()
            stds = grouped.std().fillna(0)
            x = means.index.values
            y = means.values
            yerr = stds.values
            if len(y):
                ymins.append(float((y - yerr).min()))
                ymaxs.append(float((y + yerr).max()))
            color = APPROACH_COLORS.get(approach, "#777777")
            label = APPROACH_LABELS.get(approach, approach)
            ax.plot(x, y, "o-", color=color, linewidth=1.8, markersize=4.5, label=label)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12, linewidth=0)

        # Zoom y-axis so small deltas are visible in the paper.
        if ymins and ymaxs:
            margin = 0.004
            lo = max(0.0, min(ymins) - margin)
            hi = min(1.0, max(ymaxs) + margin)
            # Avoid pathological near-zero ranges.
            if hi - lo < 0.02:
                mid = (hi + lo) / 2.0
                lo = max(0.0, mid - 0.01)
                hi = min(1.0, mid + 0.01)
            ax.set_ylim(lo, hi)

        ax.set_ylabel("Speaker Sim")
        ax.set_title(f"Scaling ({strategy})", fontsize=9)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    axes[-1][0].set_xticks(sorted(df["n_refs"].unique()))
    axes[-1][0].set_xlabel("# References")
    fig.tight_layout(pad=0.6)
    _save_plot(fig, output_dir, "scaling_curve", formats)


def plot_strategy_heatmap(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    target_n = 2 if (df["n_refs"] == 2).any() else int(df["n_refs"].mode().iloc[0])
    sub = df[df["n_refs"] == target_n]
    if sub.empty:
        return

    pivot = sub.groupby(["approach", "strategy"])["speaker_sim"].mean().unstack()
    pivot = pivot.reindex([a for a in APPROACH_ORDER if a in pivot.index])
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(pivot.columns)), max(4, 1.2 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([APPROACH_LABELS.get(a, a) for a in pivot.index])

    avg = np.nanmean(pivot.values)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > avg else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title(f"Speaker Similarity Heatmap (n_refs={target_n})")
    fig.colorbar(im, ax=ax, label="Speaker Similarity")
    fig.tight_layout()
    _save_plot(fig, output_dir, "strategy_heatmap", formats)


def plot_per_speaker_breakdown(df: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    speakers = sorted(df["speaker_id"].astype(str).unique())
    if len(speakers) <= 1:
        print("  Skipping per-speaker breakdown (single speaker).")
        return

    rows = []
    for speaker in speakers:
        sub = df[df["speaker_id"].astype(str) == speaker]
        baseline = sub[sub["approach"] == "single_baseline"]["speaker_sim"].mean()
        best = sub.groupby(GROUP_COLS)["speaker_sim"].mean().max()
        rows.append({"speaker": speaker, "baseline": baseline, "best": best})

    plot_df = pd.DataFrame(rows)
    x = np.arange(len(plot_df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 1.2), 5))
    ax.bar(x - width / 2, plot_df["baseline"], width, label="Baseline", color="#636363")
    ax.bar(x + width / 2, plot_df["best"], width, label="Best Config", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["speaker"])
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Speaker Similarity")
    ax.set_title("Baseline vs Best Config per Speaker")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_plot(fig, output_dir, "per_speaker_breakdown", formats)


def _resolve_manifest_entry(manifest: list[dict], target_id: str) -> dict | None:
    for entry in manifest:
        if str(entry.get("id")) == str(target_id):
            return entry
    return None


def compute_wer_calibration(df: pd.DataFrame, manifest_path: str | Path, whisper_model: str) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {str(entry["id"]): entry for entry in manifest}

    import whisper
    from jiwer import wer

    print(f"Loading Whisper model for GT calibration: {whisper_model}")
    model = whisper.load_model(whisper_model)

    unique_targets = sorted(df["target_id"].astype(str).unique())
    rows = []
    for idx, target_id in enumerate(unique_targets, start=1):
        entry = by_id.get(target_id)
        if entry is None:
            rows.append({"target_id": target_id, "wer_gt": np.nan, "note": "missing_manifest_entry"})
            continue

        audio_path = Path(entry["path"])
        if not audio_path.exists():
            rows.append({"target_id": target_id, "wer_gt": np.nan, "note": "missing_audio"})
            continue

        ref_text = str(entry.get("text_normalized") or entry.get("text_original") or "").strip()
        if not ref_text:
            rows.append({"target_id": target_id, "wer_gt": np.nan, "note": "missing_text"})
            continue

        result = model.transcribe(str(audio_path), language="en")
        hypothesis = (result.get("text") or "").strip()
        wer_value = float(wer(ref_text, hypothesis))

        rows.append(
            {
                "target_id": target_id,
                "wer_gt": wer_value,
                "target_audio": str(audio_path),
                "target_text": ref_text,
                "transcript_gt": hypothesis,
                "note": "",
            }
        )

        if idx % 5 == 0 or idx == len(unique_targets):
            print(f"  Calibrated {idx}/{len(unique_targets)} targets")

    calib = pd.DataFrame(rows)

    baseline = (
        df[df["approach"] == "single_baseline"]
        .groupby("target_id", as_index=False)["wer"]
        .mean()
        .rename(columns={"wer": "wer_single_baseline_mean"})
    )
    calib = calib.merge(baseline, on="target_id", how="left")
    calib["delta_single_minus_gt"] = calib["wer_single_baseline_mean"] - calib["wer_gt"]
    return calib


def summarize_wer_calibration(calib: pd.DataFrame) -> pd.DataFrame:
    valid = calib["wer_gt"].dropna()
    if valid.empty:
        return pd.DataFrame([{"metric": "wer_gt", "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "max": np.nan}])
    return pd.DataFrame(
        [
            {
                "metric": "wer_gt",
                "mean": valid.mean(),
                "median": valid.median(),
                "q25": valid.quantile(0.25),
                "q75": valid.quantile(0.75),
                "max": valid.max(),
                "n_targets": len(valid),
            },
            {
                "metric": "delta_single_minus_gt",
                "mean": calib["delta_single_minus_gt"].mean(),
                "median": calib["delta_single_minus_gt"].median(),
                "q25": calib["delta_single_minus_gt"].quantile(0.25),
                "q75": calib["delta_single_minus_gt"].quantile(0.75),
                "max": calib["delta_single_minus_gt"].max(),
                "n_targets": calib["delta_single_minus_gt"].notna().sum(),
            },
        ]
    )


def main() -> int:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Few-Shot Voice Cloning - Experiment Analysis")
    print("=" * 60)

    df = load_and_validate(args.results_csv, args.drop_approaches)
    print()

    summary = compute_summary_stats(df)
    summary = add_failure_metrics(summary, df, args.failure_thresholds)

    print("Summary Statistics")
    print("-" * 40)
    print(format_summary_table(summary, args.format))
    print()

    summary.to_csv(output_dir / "summary_stats.csv", index=False)

    print(format_top_n(df))
    print()

    if len(df["speaker_id"].unique()) > 1:
        per_speaker = compute_per_speaker_stats(df)
        per_speaker.to_csv(output_dir / "per_speaker_stats.csv", index=False)
        print(f"Per-speaker stats saved to {output_dir / 'per_speaker_stats.csv'}")
        print()

    print("Significance Tests")
    print("-" * 40)
    sig_df = compute_significance_tests(df, baseline=args.baseline, alpha=args.alpha)
    if sig_df.empty:
        print("No significance tests computed.")
    else:
        print(format_significance_table(sig_df))
        sig_df.to_csv(output_dir / "significance_tests.csv", index=False)
    print()

    latex = format_latex_table(summary)
    (output_dir / "main_results.tex").write_text(latex, encoding="utf-8")

    if not args.skip_wer_calibration:
        print("WER Calibration (Whisper on ground-truth targets)")
        print("-" * 40)
        calib = compute_wer_calibration(df, args.manifest, args.whisper_model)
        calib.to_csv(output_dir / "wer_calibration_targets.csv", index=False)
        calib_summary = summarize_wer_calibration(calib)
        calib_summary.to_csv(output_dir / "wer_calibration_summary.csv", index=False)
        print(calib_summary.to_string(index=False))
        print()
    else:
        calib = pd.DataFrame()

    if not args.no_plots:
        print("Generating plots...")
        plot_speaker_sim_by_approach(df, output_dir, args.save_formats)
        print("  speaker_sim_by_approach")
        plot_utmos_by_approach(df, output_dir, args.save_formats)
        print("  utmos_by_approach")
        plot_scaling_curve(df, output_dir, args.save_formats)
        print("  scaling_curve")
        plot_strategy_heatmap(df, output_dir, args.save_formats)
        print("  strategy_heatmap")
        plot_per_speaker_breakdown(df, output_dir, args.save_formats)
        print("  per_speaker_breakdown")
        plot_pareto_tradeoff(df, output_dir, args.save_formats)
        print("  pareto_tradeoff")
        threshold = args.failure_thresholds[0] if args.failure_thresholds else 0.5
        plot_stability_fail_rate(summary, output_dir, args.save_formats, threshold=threshold)
        print("  stability_fail_rate")
        print()

    print("=" * 60)
    print(f"Analysis complete. Outputs in {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
