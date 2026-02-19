"""Few-shot voice cloning experiment analysis.

Loads results CSV, computes summary statistics and significance tests,
generates publication-ready tables (text/JSON/LaTeX) and plots.

Usage:
    python scripts/analyze_fewshot.py
    python scripts/analyze_fewshot.py --results-csv outputs/fewshot/results.csv
    python scripts/analyze_fewshot.py --format latex --no-plots
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS = [
    {"key": "utmos", "label": "UTMOS", "higher_is_better": True, "fmt": ".3f", "arrow": "↑"},
    {"key": "speaker_sim", "label": "Speaker Sim", "higher_is_better": True, "fmt": ".4f", "arrow": "↑"},
    {"key": "wer", "label": "WER", "higher_is_better": False, "fmt": ".1%", "arrow": "↓"},
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
    "concat_audio": "#3182bd",
    "concat_code": "#e6550d",
    "embed_avg": "#31a354",
}

GROUP_COLS = ["approach", "n_refs", "strategy"]

REQUIRED_COLUMNS = {"speaker_id", "target_id", "approach", "n_refs", "strategy", "seed",
                    "utmos", "speaker_sim", "wer"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze few-shot voice cloning experiment results.",
    )
    parser.add_argument(
        "--results-csv",
        default="outputs/fewshot/results.csv",
        help="Path to results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/fewshot/analysis/",
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--format",
        default="table",
        choices=["table", "json", "latex"],
        help="Output format for summary tables.",
    )
    parser.add_argument(
        "--baseline",
        default="single_baseline",
        help="Approach name to use as the baseline for significance tests.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--drop-approaches",
        nargs="*",
        default=["concat_code"],
        help="Approaches to exclude from analysis.",
    )
    parser.add_argument(
        "--save-formats",
        nargs="+",
        default=["png", "pdf"],
        help="File formats for plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_validate(csv_path: str | Path, drop_approaches: list[str]) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Coerce metrics to numeric
    for m in METRICS:
        df[m["key"]] = pd.to_numeric(df[m["key"]], errors="coerce")

    # Drop rows where all three core metrics are NaN
    metric_keys = [m["key"] for m in METRICS]
    df = df.dropna(subset=metric_keys, how="all").copy()

    # Filter out dropped approaches
    if drop_approaches:
        before = len(df)
        df = df[~df["approach"].isin(drop_approaches)].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows from approaches: {drop_approaches}")

    # Ensure n_refs is int
    df["n_refs"] = df["n_refs"].astype(int)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Speakers: {sorted(df['speaker_id'].unique())}")
    print(f"  Approaches: {sorted(df['approach'].unique())}")
    print(f"  n_refs: {sorted(df['n_refs'].unique())}")
    print(f"  Strategies: {sorted(df['strategy'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")

    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    metric_keys = [m["key"] for m in METRICS]

    agg_dict = {}
    for mk in metric_keys:
        agg_dict[(mk, "mean")] = (mk, "mean")
        agg_dict[(mk, "std")] = (mk, "std")
        agg_dict[(mk, "count")] = (mk, "count")

    grouped = df.groupby(GROUP_COLS, sort=False)

    records = []
    for name, group in grouped:
        approach, n_refs, strategy = name
        row = {"approach": approach, "n_refs": n_refs, "strategy": strategy}
        for mk in metric_keys:
            vals = group[mk].dropna()
            row[f"{mk}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[f"{mk}_std"] = vals.std() if len(vals) > 1 else 0.0
            row[f"{mk}_count"] = len(vals)
        records.append(row)

    summary = pd.DataFrame(records)

    # Sort: by approach order, then n_refs, then strategy
    approach_rank = {a: i for i, a in enumerate(APPROACH_ORDER)}
    summary["_rank"] = summary["approach"].map(approach_rank).fillna(99)
    summary = summary.sort_values(["_rank", "n_refs", "strategy"]).drop(columns=["_rank"])
    summary = summary.reset_index(drop=True)

    return summary


def compute_per_speaker_stats(df: pd.DataFrame) -> pd.DataFrame:
    metric_keys = [m["key"] for m in METRICS]
    group_cols = ["speaker_id"] + GROUP_COLS

    records = []
    for name, group in df.groupby(group_cols, sort=False):
        speaker_id, approach, n_refs, strategy = name
        row = {"speaker_id": speaker_id, "approach": approach,
               "n_refs": n_refs, "strategy": strategy}
        for mk in metric_keys:
            vals = group[mk].dropna()
            row[f"{mk}_mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[f"{mk}_std"] = vals.std() if len(vals) > 1 else 0.0
            row[f"{mk}_count"] = len(vals)
        records.append(row)

    per_speaker = pd.DataFrame(records)
    per_speaker = per_speaker.sort_values("speaker_sim_mean", ascending=False)
    return per_speaker.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def compute_significance_tests(
    df: pd.DataFrame,
    baseline: str = "single_baseline",
    alpha: float = 0.05,
) -> pd.DataFrame:
    metric_keys = [m["key"] for m in METRICS]

    # Baseline rows — for single_baseline, n_refs=1
    bl_df = df[df["approach"] == baseline].copy()
    if bl_df.empty:
        warnings.warn(f"No baseline rows for approach='{baseline}'. Skipping significance tests.")
        return pd.DataFrame()

    # Non-baseline configs
    non_bl = df[df["approach"] != baseline].copy()
    configs = non_bl.groupby(GROUP_COLS).size().reset_index(name="_n")

    records = []
    for _, cfg_row in configs.iterrows():
        approach = cfg_row["approach"]
        n_refs = cfg_row["n_refs"]
        strategy = cfg_row["strategy"]

        cfg_df = non_bl[
            (non_bl["approach"] == approach)
            & (non_bl["n_refs"] == n_refs)
            & (non_bl["strategy"] == strategy)
        ]

        # Pair by (speaker_id, target_id, seed)
        merge_keys = ["speaker_id", "target_id", "seed"]
        paired = cfg_df.merge(bl_df, on=merge_keys, suffixes=("_exp", "_bl"))

        n_pairs = len(paired)

        for m in METRICS:
            mk = m["key"]
            exp_vals = paired[f"{mk}_exp"].dropna()
            bl_vals = paired[f"{mk}_bl"].dropna()

            # Need matching indices
            valid_idx = exp_vals.index.intersection(bl_vals.index)
            exp_vals = exp_vals.loc[valid_idx].values
            bl_vals = bl_vals.loc[valid_idx].values
            n_valid = len(exp_vals)

            row = {
                "approach": approach, "n_refs": n_refs, "strategy": strategy,
                "metric": mk, "n_pairs": n_valid,
            }

            if n_valid < 3:
                row.update({"test": "skipped", "statistic": np.nan,
                            "p_value": np.nan, "p_corrected": np.nan,
                            "significant": False, "effect_size": np.nan,
                            "note": f"Only {n_valid} pair(s), need >=3"})
                records.append(row)
                continue

            diffs = exp_vals - bl_vals

            # Cohen's d
            d_std = diffs.std(ddof=1)
            cohens_d = diffs.mean() / d_std if d_std > 0 else 0.0

            if n_valid < 10:
                # Paired t-test fallback
                t_stat, p_val = stats.ttest_rel(exp_vals, bl_vals)
                row["test"] = "paired_t"
                row["statistic"] = t_stat
            else:
                # Wilcoxon signed-rank
                try:
                    w_stat, p_val = stats.wilcoxon(exp_vals, bl_vals)
                    row["test"] = "wilcoxon"
                    row["statistic"] = w_stat
                except ValueError:
                    # All differences are zero
                    row.update({"test": "wilcoxon", "statistic": 0.0,
                                "p_value": 1.0, "p_corrected": 1.0,
                                "significant": False, "effect_size": cohens_d,
                                "note": "all differences zero"})
                    records.append(row)
                    continue

            row["p_value"] = p_val
            row["effect_size"] = cohens_d
            records.append(row)

    if not records:
        return pd.DataFrame()

    sig_df = pd.DataFrame(records)

    # Bonferroni correction (only for rows that have valid p-values)
    valid_mask = sig_df["p_value"].notna() & (sig_df["test"] != "skipped")
    n_tests = valid_mask.sum()
    if n_tests > 0:
        sig_df.loc[valid_mask, "p_corrected"] = (
            sig_df.loc[valid_mask, "p_value"] * n_tests
        ).clip(upper=1.0)
        sig_df.loc[valid_mask, "significant"] = sig_df.loc[valid_mask, "p_corrected"] < alpha
    else:
        sig_df["p_corrected"] = np.nan
        sig_df["significant"] = False

    # Fill note for non-skipped
    if "note" not in sig_df.columns:
        sig_df["note"] = ""
    sig_df["note"] = sig_df["note"].fillna("")

    return sig_df


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_metric(mean: float, std: float, count: int, fmt_spec: str) -> str:
    """Format mean +/- std, omitting std if count <= 1."""
    if pd.isna(mean):
        return "—"
    mean_str = format(mean, fmt_spec)
    if count > 1 and not pd.isna(std) and std > 0:
        std_str = format(std, fmt_spec)
        return f"{mean_str} ± {std_str}"
    return mean_str


def format_summary_table(summary: pd.DataFrame, fmt: str = "table") -> str:
    if fmt == "json":
        return _format_json(summary)
    elif fmt == "latex":
        return format_latex_table(summary)
    else:
        return _format_text_table(summary)


def _format_text_table(summary: pd.DataFrame) -> str:
    lines = []
    header = f"{'Approach':<20} {'n_refs':>6} {'Strategy':<10}"
    for m in METRICS:
        header += f" {m['label'] + ' ' + m['arrow']:>20}"
    lines.append(header)
    lines.append("─" * len(header))

    for _, row in summary.iterrows():
        line = f"{APPROACH_LABELS.get(row['approach'], row['approach']):<20} {row['n_refs']:>6} {row['strategy']:<10}"
        for m in METRICS:
            mk = m["key"]
            val = _fmt_metric(row[f"{mk}_mean"], row[f"{mk}_std"],
                              row[f"{mk}_count"], m["fmt"])
            line += f" {val:>20}"
        lines.append(line)

    return "\n".join(lines)


def _format_json(summary: pd.DataFrame) -> str:
    records = []
    for _, row in summary.iterrows():
        rec = {
            "approach": row["approach"],
            "n_refs": int(row["n_refs"]),
            "strategy": row["strategy"],
        }
        for m in METRICS:
            mk = m["key"]
            rec[mk] = {
                "mean": round(row[f"{mk}_mean"], 6) if not pd.isna(row[f"{mk}_mean"]) else None,
                "std": round(row[f"{mk}_std"], 6) if not pd.isna(row[f"{mk}_std"]) else None,
                "count": int(row[f"{mk}_count"]),
            }
        records.append(rec)
    return json.dumps(records, indent=2)


def format_latex_table(summary: pd.DataFrame) -> str:
    """Publication-ready LaTeX tabular for Interspeech."""
    metric_keys = [m["key"] for m in METRICS]

    # Find best value per metric (across all configs)
    best = {}
    for m in METRICS:
        mk = m["key"]
        col = summary[f"{mk}_mean"]
        if m["higher_is_better"]:
            best[mk] = col.max()
        else:
            best[mk] = col.min()

    lines = []
    lines.append(r"\begin{tabular}{llc" + "c" * len(METRICS) + "}")
    lines.append(r"\toprule")

    # Header row
    header_parts = [r"\textbf{Approach}", r"\textbf{Strategy}", r"\textbf{\#Refs}"]
    for m in METRICS:
        header_parts.append(rf"\textbf{{{m['label']}}} {m['arrow']}")
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    for _, row in summary.iterrows():
        label = APPROACH_LABELS.get(row["approach"], row["approach"])
        parts = [label, row["strategy"], str(int(row["n_refs"]))]

        for m in METRICS:
            mk = m["key"]
            mean_val = row[f"{mk}_mean"]
            std_val = row[f"{mk}_std"]
            count = row[f"{mk}_count"]

            if pd.isna(mean_val):
                parts.append("---")
                continue

            cell = _fmt_metric(mean_val, std_val, count, m["fmt"])

            # Bold if best
            if not pd.isna(best[mk]) and abs(mean_val - best[mk]) < 1e-9:
                cell = rf"\textbf{{{cell}}}"

            parts.append(cell)

        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def format_significance_table(sig_df: pd.DataFrame) -> str:
    if sig_df.empty:
        return "No significance tests computed."

    lines = []
    header = (f"{'Approach':<15} {'n':>3} {'Strategy':<10} {'Metric':<12} "
              f"{'Test':<10} {'Pairs':>5} {'p-value':>10} {'p-corr':>10} "
              f"{'Sig?':>5} {'Cohen d':>8} {'Note'}")
    lines.append(header)
    lines.append("─" * len(header))

    for _, row in sig_df.iterrows():
        p_str = f"{row['p_value']:.4f}" if not pd.isna(row["p_value"]) else "—"
        pc_str = f"{row['p_corrected']:.4f}" if not pd.isna(row.get("p_corrected")) else "—"
        sig_str = "yes" if row.get("significant", False) else "no"
        d_str = f"{row['effect_size']:.3f}" if not pd.isna(row.get("effect_size")) else "—"
        note = row.get("note", "")

        line = (f"{row['approach']:<15} {row['n_refs']:>3} {row['strategy']:<10} "
                f"{row['metric']:<12} {row['test']:<10} {row['n_pairs']:>5} "
                f"{p_str:>10} {pc_str:>10} {sig_str:>5} {d_str:>8} {note}")
        lines.append(line)

    return "\n".join(lines)


def format_top_n(df: pd.DataFrame, n: int = 5) -> str:
    """Top N configs by speaker_sim."""
    summary = compute_summary_stats(df)
    top = summary.nlargest(n, "speaker_sim_mean")

    lines = [f"Top {n} Configurations by Speaker Similarity:"]
    lines.append("")
    header = f"{'Rank':>4}  {'Approach':<20} {'n_refs':>6} {'Strategy':<10} {'Spk Sim':>10} {'UTMOS':>8} {'WER':>8}"
    lines.append(header)
    lines.append("─" * len(header))

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        label = APPROACH_LABELS.get(row["approach"], row["approach"])
        sim = _fmt_metric(row["speaker_sim_mean"], row["speaker_sim_std"],
                          row["speaker_sim_count"], ".4f")
        utmos = _fmt_metric(row["utmos_mean"], row["utmos_std"],
                            row["utmos_count"], ".3f")
        wer = _fmt_metric(row["wer_mean"], row["wer_std"],
                          row["wer_count"], ".1%")
        lines.append(f"{rank:>4}  {label:<20} {row['n_refs']:>6} {row['strategy']:<10} {sim:>10} {utmos:>8} {wer:>8}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _get_plot_approaches(df: pd.DataFrame) -> list[str]:
    """Return approaches present in data, in canonical order."""
    present = set(df["approach"].unique())
    return [a for a in APPROACH_ORDER if a in present]


def _save_plot(fig: plt.Figure, output_dir: Path, name: str, formats: list[str]):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(plots_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speaker_sim_by_approach(
    df: pd.DataFrame, output_dir: Path, formats: list[str],
):
    summary = compute_summary_stats(df)
    approaches = _get_plot_approaches(df)
    n_refs_vals = sorted(df["n_refs"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / max(len(n_refs_vals), 1)

    for i, nr in enumerate(n_refs_vals):
        x_positions = []
        heights = []
        errs = []
        colors = []
        for j, approach in enumerate(approaches):
            row = summary[(summary["approach"] == approach)
                          & (summary["n_refs"] == nr)]
            if row.empty:
                continue
            # Average across strategies for the grouped bar
            mean_val = row["speaker_sim_mean"].mean()
            std_val = row["speaker_sim_std"].mean()
            x_positions.append(j + i * bar_width)
            heights.append(mean_val)
            errs.append(std_val)
            colors.append(APPROACH_COLORS.get(approach, "#999999"))

        ax.bar(x_positions, heights, bar_width, yerr=errs if any(e > 0 for e in errs) else None,
               label=f"{nr} refs", alpha=0.85, capsize=3)

    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in approaches], rotation=15)
    ax.set_ylabel("Speaker Similarity ↑")
    ax.set_title("Speaker Similarity by Approach × Num References")
    ax.legend(title="# Refs")
    ax.set_ylim(bottom=min(0.9, ax.get_ylim()[0]))

    _save_plot(fig, output_dir, "speaker_sim_by_approach", formats)


def plot_utmos_by_approach(
    df: pd.DataFrame, output_dir: Path, formats: list[str],
):
    summary = compute_summary_stats(df)
    approaches = _get_plot_approaches(df)
    n_refs_vals = sorted(df["n_refs"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / max(len(n_refs_vals), 1)

    for i, nr in enumerate(n_refs_vals):
        x_positions = []
        heights = []
        errs = []
        for j, approach in enumerate(approaches):
            row = summary[(summary["approach"] == approach)
                          & (summary["n_refs"] == nr)]
            if row.empty:
                continue
            mean_val = row["utmos_mean"].mean()
            std_val = row["utmos_std"].mean()
            x_positions.append(j + i * bar_width)
            heights.append(mean_val)
            errs.append(std_val)

        ax.bar(x_positions, heights, bar_width, yerr=errs if any(e > 0 for e in errs) else None,
               label=f"{nr} refs", alpha=0.85, capsize=3)

    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels([APPROACH_LABELS.get(a, a) for a in approaches], rotation=15)
    ax.set_ylabel("UTMOS ↑")
    ax.set_title("UTMOS by Approach × Num References")
    ax.legend(title="# Refs")

    _save_plot(fig, output_dir, "utmos_by_approach", formats)


def plot_scaling_curve(
    df: pd.DataFrame, output_dir: Path, formats: list[str],
):
    approaches = _get_plot_approaches(df)
    strategies = sorted(df["strategy"].unique())

    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 5),
                             squeeze=False)

    for s_idx, strategy in enumerate(strategies):
        ax = axes[0][s_idx]
        strat_df = df[df["strategy"] == strategy]

        for approach in approaches:
            app_df = strat_df[strat_df["approach"] == approach]
            if app_df.empty:
                continue

            grouped = app_df.groupby("n_refs")["speaker_sim"]
            means = grouped.mean()
            stds = grouped.std().fillna(0)

            x = means.index.values
            y = means.values
            yerr = stds.values

            color = APPROACH_COLORS.get(approach, "#999999")
            label = APPROACH_LABELS.get(approach, approach)
            ax.plot(x, y, "o-", color=color, label=label, linewidth=2, markersize=6)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

        ax.set_xlabel("Number of References")
        ax.set_ylabel("Speaker Similarity ↑")
        ax.set_title(f"Scaling Curve — {strategy}")
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(df["n_refs"].unique()))
        ax.set_ylim(bottom=min(0.9, ax.get_ylim()[0]))

    fig.tight_layout()
    _save_plot(fig, output_dir, "scaling_curve", formats)


def plot_strategy_heatmap(
    df: pd.DataFrame, output_dir: Path, formats: list[str],
):
    """Heatmap: approach x strategy at fixed n_refs=2 (or max available)."""
    # Pick n_refs=2 if available, otherwise the most common
    if 2 in df["n_refs"].values:
        target_nrefs = 2
    else:
        target_nrefs = df["n_refs"].mode().iloc[0]

    sub = df[df["n_refs"] == target_nrefs]
    if sub.empty:
        return

    approaches = _get_plot_approaches(sub)
    strategies = sorted(sub["strategy"].unique())

    pivot = sub.groupby(["approach", "strategy"])["speaker_sim"].mean().unstack(fill_value=np.nan)
    # Reorder rows
    pivot = pivot.reindex([a for a in approaches if a in pivot.index])
    pivot = pivot.reindex(columns=[s for s in strategies if s in pivot.columns])

    fig, ax = plt.subplots(figsize=(max(6, len(strategies) * 2), max(4, len(approaches) * 1.2)))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([APPROACH_LABELS.get(a, a) for a in pivot.index])

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=10,
                        color="white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black")

    ax.set_title(f"Speaker Similarity — Approach × Strategy (n_refs={target_nrefs})")
    fig.colorbar(im, ax=ax, label="Speaker Sim ↑")
    fig.tight_layout()

    _save_plot(fig, output_dir, "strategy_heatmap", formats)


def plot_per_speaker_breakdown(
    df: pd.DataFrame, output_dir: Path, formats: list[str],
):
    """Baseline vs best config per speaker. Only generated if >1 speaker."""
    speakers = sorted(df["speaker_id"].unique())
    if len(speakers) <= 1:
        print("  Skipping per-speaker breakdown (only 1 speaker).")
        return

    # For each speaker: baseline mean speaker_sim and best config mean speaker_sim
    records = []
    for spk in speakers:
        spk_df = df[df["speaker_id"] == spk]
        bl = spk_df[spk_df["approach"] == "single_baseline"]["speaker_sim"].mean()
        best = spk_df.groupby(GROUP_COLS)["speaker_sim"].mean().max()
        records.append({"speaker": str(spk), "baseline": bl, "best": best})

    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(max(8, len(speakers) * 1.5), 5))
    x = np.arange(len(plot_df))
    width = 0.35

    ax.bar(x - width / 2, plot_df["baseline"], width, label="Baseline", color="#636363")
    ax.bar(x + width / 2, plot_df["best"], width, label="Best Config", color="#31a354")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["speaker"])
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Speaker Similarity ↑")
    ax.set_title("Baseline vs Best Config per Speaker")
    ax.legend()
    ax.set_ylim(bottom=min(0.9, ax.get_ylim()[0]))

    fig.tight_layout()
    _save_plot(fig, output_dir, "per_speaker_breakdown", formats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print("=" * 60)
    print("Few-Shot Voice Cloning — Experiment Analysis")
    print("=" * 60)
    print()

    df = load_and_validate(args.results_csv, args.drop_approaches)
    print()

    # --- Summary stats ---
    print("Summary Statistics")
    print("─" * 40)
    summary = compute_summary_stats(df)
    table_str = format_summary_table(summary, args.format)
    print(table_str)
    print()

    # Save summary
    summary.to_csv(output_dir / "summary_stats.csv", index=False)

    # --- Top N ---
    print(format_top_n(df))
    print()

    # --- Per-speaker stats ---
    speakers = df["speaker_id"].unique()
    if len(speakers) > 1:
        print("Per-Speaker Statistics")
        print("─" * 40)
        per_speaker = compute_per_speaker_stats(df)
        per_speaker.to_csv(output_dir / "per_speaker_stats.csv", index=False)
        print(f"  Saved to {output_dir / 'per_speaker_stats.csv'}")
        print()

    # --- Significance tests ---
    print("Significance Tests")
    print("─" * 40)
    sig_df = compute_significance_tests(df, baseline=args.baseline, alpha=args.alpha)
    if not sig_df.empty:
        sig_str = format_significance_table(sig_df)
        print(sig_str)
        sig_df.to_csv(output_dir / "significance_tests.csv", index=False)
    else:
        print("  No significance tests computed.")
    print()

    # --- LaTeX table ---
    latex_str = format_latex_table(summary)
    latex_path = output_dir / "main_results.tex"
    latex_path.write_text(latex_str, encoding="utf-8")
    print(f"LaTeX table saved to {latex_path}")
    print()

    # --- Plots ---
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
        print("  Done.")
        print()

    print("=" * 60)
    print(f"Analysis complete. Outputs in {output_dir}")
    print("=" * 60)

    return 0
