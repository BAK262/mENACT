"""
validation_fnirs_quality: SCI signal quality summary for Section IV.

Reads the preprocessing-matched quality table:
  data/fnirs_signals/AEPO_001filt02/quality.csv

Outputs under results/validation_fnirs_quality/:
  figures/fig_fnirs_quality_SCI_by_task_trialchannel.{pdf,png}
  tables/table_fnirs_quality_by_subject_task.csv
  tables/table_fnirs_quality_sci_wilcoxon.csv
  tables/table_fnirs_quality_sci_global_summary.csv
  logs/run_*.log

Usage (from project root):
  python code/validation_fnirs_quality.py --quick
  python code/validation_fnirs_quality.py --full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.validation_fnirs.common import get_fnirs_preproc_root, setup_logging
from utils.validation_paths import ensure_result_layout, get_project_root, get_result_dir
from utils.validation_style import (
    REGION_COLORS,
    TASK_COLORS,
    TASKS,
    add_ygrid,
    apply_style,
    fig_single,
    filter_fnirs_three_task_rows,
    normalize_task_name,
    save_fig,
    strip_spines,
    task_display_name,
)
from utils.validation_vocab import TASK_DISPLAY, TASK_KEYS

LOGGER = logging.getLogger("validation_fnirs_quality")


def _load_channel_regions_from_latex(tex_path: Path) -> pd.DataFrame:
    """
    Parse tab_appendix_fnirs_channels_tabular.tex and return channel->region map.

    Expected row format:
      Ch. & S & D & x & y & z & Hem. & Lob. \\
      19 & S10 & D9 & ... & M & F \\
    """
    if not tex_path.is_file():
        raise FileNotFoundError(f"Missing channel table: {tex_path}")

    rows: list[dict[str, str | int]] = []
    for raw in tex_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if "&" not in line or "\\\\" not in line:
            continue
        if line.startswith("Ch.") or line.startswith("\\") or "toprule" in line or "midrule" in line or "bottomrule" in line:
            continue
        parts = [p.strip() for p in line.replace("\\\\", "").split("&")]
        if len(parts) < 8:
            continue
        ch_idx, s, d, *_xyz, hem, lob = parts[:8]
        if not ch_idx.isdigit():
            continue
        channel = f"{s}-{d}"
        hem = hem.strip()
        lob = lob.strip()
        region = f"{hem}-{lob}"
        rows.append(
            {
                "channel": channel,
                "hem": hem,
                "lob": lob,
                "region": region,
                "channel_index_1based": int(ch_idx),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"Failed to parse any rows from: {tex_path}")
    return out


def _plot_sci_by_task_region(
    df: pd.DataFrame,
    tasks: list[str],
    result_dir: Path,
    channel_region_map: pd.DataFrame,
    region_order: list[str],
) -> None:
    """Grouped bar chart: x = region, grouped/colored by task (TASK_COLORS).

    Bars show mean SCI ± SEM (between-subject); scatter shows every
    trial-channel observation so the full data density is visible.
    """
    metric_col = "SCI"
    plot_df = df.dropna(subset=[metric_col, "task", "subject", "channel"]).copy()
    plot_df = plot_df[plot_df["task"].isin(tasks)].copy()
    if plot_df.empty:
        LOGGER.warning("Skip SCI region plot (no data)")
        return

    plot_df = plot_df.merge(
        channel_region_map[["channel", "region"]], on="channel", how="left"
    )
    plot_df = plot_df.dropna(subset=["region"]).copy()
    if plot_df.empty:
        LOGGER.warning("Skip SCI region plot (no channels mapped to regions)")
        return

    plot_df["task"] = pd.Categorical(plot_df["task"], categories=tasks, ordered=True)
    plot_df["region"] = pd.Categorical(plot_df["region"], categories=region_order, ordered=True)
    plot_df = plot_df.dropna(subset=["task", "region"]).copy()

    by_sub_reg = (
        plot_df.groupby(["task", "subject", "region"], observed=True, as_index=False)
        .agg(**{metric_col: (metric_col, "mean")})
        .copy()
    )

    stats = (
        by_sub_reg.groupby(["task", "region"], as_index=False, observed=True)
        .agg(
            mean=(metric_col, "mean"),
            sd=(metric_col, "std"),
            n=("subject", "nunique"),
        )
        .copy()
    )
    stats["sem"] = stats["sd"] / np.sqrt(stats["n"].clip(lower=1))

    fig, ax = fig_single(h=2.2)

    x_labels = region_order
    n_groups = len(tasks)
    x = np.arange(len(x_labels))
    total_w = 0.78
    bar_w = total_w / max(1, n_groups)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_w
    rng = np.random.default_rng(42)

    for j, task in enumerate(tasks):
        display_name = task_display_name(task)
        color = TASK_COLORS.get(display_name, "#888888")
        sub = stats[stats["task"] == task]
        means = [
            float(sub.loc[sub["region"] == r, "mean"].iloc[0])
            if (sub["region"] == r).any()
            else np.nan
            for r in region_order
        ]
        sems = [
            float(sub.loc[sub["region"] == r, "sem"].iloc[0])
            if (sub["region"] == r).any()
            else np.nan
            for r in region_order
        ]
        xx = x + offsets[j]
        ax.bar(
            xx,
            means,
            yerr=sems,
            capsize=2,
            width=bar_w * 0.92,
            color=color,
            alpha=0.90,
            linewidth=0.0,
            label=display_name,
        )
        pts = by_sub_reg[by_sub_reg["task"] == task]
        for i, r in enumerate(region_order):
            y = pts.loc[pts["region"] == r, metric_col].to_numpy()
            if y.size == 0:
                continue
            jitter = (rng.random(y.size) - 0.5) * (bar_w * 0.55)
            ax.scatter(
                np.full_like(y, xx[i]) + jitter,
                y,
                s=8,
                alpha=0.35,
                color="black",
                linewidths=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("SCI")
    strip_spines(ax)
    add_ygrid(ax)

    ax.legend(
        ncols=n_groups,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.88))

    save_fig(fig, "fig_fnirs_quality_SCI_by_task_trialchannel", result_dir)


def _subject_region_sci_means(
    df: pd.DataFrame,
    channel_region_map: pd.DataFrame,
    region_order: list[str],
) -> pd.DataFrame:
    """Subject-level mean SCI per task and montage region."""
    plot_df = df.dropna(subset=["SCI", "task", "subject", "channel"]).copy()
    plot_df = plot_df.merge(
        channel_region_map[["channel", "region"]], on="channel", how="left"
    )
    plot_df = plot_df.dropna(subset=["region"]).copy()
    by_sub_reg = (
        plot_df.groupby(["task", "subject", "region"], observed=True, as_index=False)
        .agg(sci_mean=("SCI", "mean"))
        .copy()
    )
    by_sub_reg["region"] = pd.Categorical(
        by_sub_reg["region"], categories=region_order, ordered=True
    )
    return by_sub_reg


def _subject_task_sci_means(df: pd.DataFrame) -> pd.DataFrame:
    """Subject-level global SCI per task (mean across all trials and analysis channels)."""
    return (
        df.groupby(["subject", "task"], as_index=False)
        .agg(sci_mean=("SCI", "mean"))
        .copy()
    )


def _wilcoxon_greater(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """One-sided Wilcoxon signed-rank: x > y. Returns (p, n_pairs)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        return float("nan"), n
    try:
        res = stats.wilcoxon(x, y, alternative="greater", zero_method="wilcox")
        return float(res.pvalue), n
    except ValueError:
        return float("nan"), n


def _build_sci_wilcoxon_tests(
    by_sub_reg: pd.DataFrame,
    by_sub_task: pd.DataFrame,
    region_order: list[str],
    tasks: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regional and global pairwise Wilcoxon tests (Perception > Production > Performance)."""
    pairs = [(tasks[0], tasks[1]), (tasks[1], tasks[2])]
    pair_labels = [
        ("perception_gt_production", f"{tasks[0]}>{tasks[1]}"),
        ("production_gt_performance", f"{tasks[1]}>{tasks[2]}"),
    ]

    regional_rows: list[dict[str, object]] = []
    for region in region_order:
        sub_reg = by_sub_reg[by_sub_reg["region"] == region]
        for (hi, lo), (key, label) in zip(pairs, pair_labels):
            merged = sub_reg.pivot(index="subject", columns="task", values="sci_mean")
            if hi not in merged.columns or lo not in merged.columns:
                continue
            p, n = _wilcoxon_greater(
                merged[hi].to_numpy(dtype=float),
                merged[lo].to_numpy(dtype=float),
            )
            regional_rows.append(
                {
                    "region": region,
                    "comparison": key,
                    "comparison_label": label,
                    "n_subjects": n,
                    "p_value": p,
                    "method": "wilcoxon_signed_rank_greater",
                }
            )

    global_rows: list[dict[str, object]] = []
    merged_g = by_sub_task.pivot(index="subject", columns="task", values="sci_mean")
    for (hi, lo), (key, label) in zip(pairs, pair_labels):
        if hi not in merged_g.columns or lo not in merged_g.columns:
            continue
        p, n = _wilcoxon_greater(
            merged_g[hi].to_numpy(dtype=float),
            merged_g[lo].to_numpy(dtype=float),
        )
        global_rows.append(
            {
                "comparison": key,
                "comparison_label": label,
                "n_subjects": n,
                "p_value": p,
                "method": "wilcoxon_signed_rank_greater",
            }
        )

    cohort_rows: list[dict[str, object]] = []
    for task in tasks:
        vals = by_sub_task.loc[by_sub_task["task"] == task, "sci_mean"]
        cohort_rows.append(
            {
                "task": task,
                "n_subjects": int(vals.notna().sum()),
                "cohort_mean_sci": float(vals.mean()) if vals.notna().any() else float("nan"),
            }
        )

    regional_df = pd.DataFrame(regional_rows)
    global_df = pd.DataFrame(global_rows)
    cohort_df = pd.DataFrame(cohort_rows)
    return regional_df, global_df, cohort_df


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run validation_fnirs_quality (SCI signal quality)."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run on a small subject subset (default)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="All subjects (N=53 when data complete)",
    )
    p.add_argument(
        "--quick-subjects",
        type=int,
        default=5,
        help="Subject cap for --quick (default: 5)",
    )
    p.add_argument(
        "--preproc",
        type=str,
        default="AEPO_001filt02",
        help="Preprocessing folder under data/fnirs_signals/",
    )
    args = p.parse_args(argv)
    if not args.quick and not args.full:
        args.quick = True
    return args


def _maybe_filter_subjects(df: pd.DataFrame, subject_limit: Optional[int]) -> pd.DataFrame:
    if subject_limit is None:
        return df
    subs = sorted(pd.unique(df["subject"]).tolist())
    keep = set(subs[: int(subject_limit)])
    return df[df["subject"].isin(keep)].copy()


def main(argv: Optional[list[str]] = None) -> None:
    apply_style()

    args = parse_args(argv)
    root = get_project_root()
    result_dir = get_result_dir("fnirs_quality")
    ensure_result_layout(result_dir)
    setup_logging(result_dir / "logs", logger_name="validation_fnirs_quality")

    LOGGER.info(
        "Task display names: %s",
        ", ".join(TASK_DISPLAY[k] for k in TASK_KEYS),
    )

    preproc_root = get_fnirs_preproc_root(args.preproc)
    quality_csv = preproc_root / "quality.csv"
    if not quality_csv.is_file():
        raise FileNotFoundError(f"Missing quality.csv: {quality_csv}")
    df = pd.read_csv(quality_csv)
    required = {"subject", "task", "SCI"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"quality.csv missing columns: {sorted(missing)}")

    raw_task = df["task"].astype(str)
    df["task"] = raw_task.map(normalize_task_name)

    tasks = list(TASKS)
    df = filter_fnirs_three_task_rows(df)

    subject_limit = int(args.quick_subjects) if args.quick else None
    df = _maybe_filter_subjects(df, subject_limit)

    channel_table_tex = root / "manuscript" / "tables" / "tab_appendix_fnirs_channels_tabular.tex"
    ch_map = _load_channel_regions_from_latex(channel_table_tex)
    if "channel" in df.columns:
        ch_map = ch_map[ch_map["channel"].isin(set(pd.unique(df["channel"]).tolist()))].copy()
    desired_region_order = list(REGION_COLORS.keys())
    present_regions = set(pd.unique(ch_map["region"]).tolist())
    region_order = [r for r in desired_region_order if r in present_regions]

    by_sub_task = _subject_task_sci_means(df)
    out_csv = result_dir / "tables" / "table_fnirs_quality_by_subject_task.csv"
    by_sub_task.to_csv(out_csv, index=False)
    LOGGER.info("Wrote: %s (rows=%d)", out_csv, len(by_sub_task))

    if "channel" not in df.columns:
        LOGGER.error("quality.csv has no 'channel' column; cannot produce SCI region plot")
        return

    by_sub_reg = _subject_region_sci_means(df, ch_map, region_order)
    regional_tests, global_tests, cohort_means = _build_sci_wilcoxon_tests(
        by_sub_reg, by_sub_task, region_order, tasks
    )
    wilcoxon_path = result_dir / "tables" / "table_fnirs_quality_sci_wilcoxon.csv"
    regional_tests.to_csv(wilcoxon_path, index=False)
    LOGGER.info("Wrote: %s (rows=%d)", wilcoxon_path, len(regional_tests))

    global_path = result_dir / "tables" / "table_fnirs_quality_sci_global_summary.csv"
    summary_parts = []
    if not global_tests.empty:
        g = global_tests.copy()
        g.insert(0, "row_type", "wilcoxon_global")
        summary_parts.append(g)
    if not cohort_means.empty:
        c = cohort_means.copy()
        c.insert(0, "row_type", "cohort_mean")
        summary_parts.append(c)
    if summary_parts:
        pd.concat(summary_parts, ignore_index=True).to_csv(global_path, index=False)
        LOGGER.info("Wrote: %s", global_path)

    _plot_sci_by_task_region(
        df=df,
        tasks=tasks,
        result_dir=result_dir,
        channel_region_map=ch_map,
        region_order=region_order,
    )


if __name__ == "__main__":
    main()
