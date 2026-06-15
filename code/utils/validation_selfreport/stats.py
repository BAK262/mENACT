"""Manuscript-aligned inferential tests for trial-level self-reports."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from utils.validation_selfreport.performance_dual_ratings import (
    PERFORMANCE_EXPERIENCE_PREFIX,
    PERFORMANCE_INTENTION_PREFIX,
)
from utils.validation_style import EMOTIONS_8
from utils.validation_traits.load import GROUP_ORDER
from utils.validation_vocab import GROUP_DISPLAY, TASK_KEYS

logger = logging.getLogger(__name__)

NON_NEUTRAL_TARGETS: tuple[str, ...] = tuple(EMOTIONS_8)


def _one_sided_paired_greater(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Return (mean_diff, one-sided p, n_pairs) for x > y."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        return float("nan"), float("nan"), n
    diff = x - y
    mean_diff = float(np.mean(diff))
    try:
        res = stats.ttest_rel(x, y, alternative="greater", nan_policy="omit")
        p = float(res.pvalue)
    except Exception:
        p = float("nan")
    return mean_diff, p, n


def _dominance_contrasts_for_task(
    ratings: pd.DataFrame,
    task_key: str,
) -> pd.DataFrame:
    """One-sided paired dominance: matching dimension > each non-matching dimension."""
    df = ratings[ratings["task_key"] == task_key].copy()
    df = df[df["targetEmotion"].isin(NON_NEUTRAL_TARGETS)].copy()
    rows: list[dict[str, object]] = []

    for target in NON_NEUTRAL_TARGETS:
        sub = df[df["targetEmotion"] == target]
        if sub.empty:
            continue
        match_vals = sub[target].to_numpy(dtype=float)
        for other in NON_NEUTRAL_TARGETS:
            if other == target:
                continue
            other_vals = sub[other].to_numpy(dtype=float)
            mean_diff, p_raw, n_pairs = _one_sided_paired_greater(match_vals, other_vals)
            rows.append(
                {
                    "task_key": task_key,
                    "target_emotion": target,
                    "contrast_dimension": other,
                    "mean_matching_minus_other": mean_diff,
                    "n_pairs": n_pairs,
                    "p_raw": p_raw,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    finite = out["p_raw"].notna() & np.isfinite(out["p_raw"])
    out["q_bh"] = np.nan
    if finite.any():
        _, q, _, _ = multipletests(out.loc[finite, "p_raw"].to_numpy(), method="fdr_bh")
        out.loc[finite, "q_bh"] = q
    return out


def build_dominance_tests(ratings: pd.DataFrame) -> pd.DataFrame:
    """Dominance contrasts with Benjamini--Hochberg FDR within each task."""
    frames = [_dominance_contrasts_for_task(ratings, tk) for tk in TASK_KEYS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_intention_exceeds_experience_tests(
    performance_dual: pd.DataFrame,
) -> pd.DataFrame:
    """On matching dimensions, expressive intention > emotional experience (Performance)."""
    rows: list[dict[str, object]] = []
    for target in NON_NEUTRAL_TARGETS:
        sub = performance_dual[performance_dual["targetEmotion"] == target]
        if sub.empty:
            continue
        exp_col = f"{PERFORMANCE_EXPERIENCE_PREFIX}{target}"
        int_col = f"{PERFORMANCE_INTENTION_PREFIX}{target}"
        intention = sub[int_col].to_numpy(dtype=float)
        experience = sub[exp_col].to_numpy(dtype=float)
        mean_gap, p_raw, n_pairs = _one_sided_paired_greater(intention, experience)
        rows.append(
            {
                "target_emotion": target,
                "mean_intention_minus_experience": mean_gap,
                "n_pairs": n_pairs,
                "p_raw": p_raw,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    finite = out["p_raw"].notna() & np.isfinite(out["p_raw"])
    out["q_bh"] = np.nan
    if finite.any():
        _, q, _, _ = multipletests(out.loc[finite, "p_raw"].to_numpy(), method="fdr_bh")
        out.loc[finite, "q_bh"] = q
    return out


def _subject_matching_gap(performance_dual: pd.DataFrame) -> pd.DataFrame:
    """Subject-level mean (intention - experience) on the matching dimension."""
    rows: list[dict[str, object]] = []
    for target in NON_NEUTRAL_TARGETS:
        sub = performance_dual[performance_dual["targetEmotion"] == target]
        if sub.empty:
            continue
        exp_col = f"{PERFORMANCE_EXPERIENCE_PREFIX}{target}"
        int_col = f"{PERFORMANCE_INTENTION_PREFIX}{target}"
        for sid, grp in sub.groupby("subID"):
            gap = grp[int_col].to_numpy(dtype=float) - grp[exp_col].to_numpy(dtype=float)
            gap = gap[np.isfinite(gap)]
            if gap.size == 0:
                continue
            rows.append(
                {
                    "subID": int(sid),
                    "target_emotion": target,
                    "mean_gap": float(np.mean(gap)),
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return pd.DataFrame(columns=["subID", "mean_gap"])
    return (
        long.groupby("subID", as_index=False)["mean_gap"]
        .mean()
        .rename(columns={"mean_gap": "mean_intention_minus_experience_gap"})
    )


def build_experience_intention_gap_group_tests(
    performance_dual: pd.DataFrame,
    sub_info: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-way ANOVA and Tukey HSD on subject-level experience--intention gaps."""
    gaps = _subject_matching_gap(performance_dual)
    if gaps.empty:
        return pd.DataFrame(), pd.DataFrame()

    info = sub_info.copy()
    if "id" in info.columns and "subID" not in info.columns:
        info = info.rename(columns={"id": "subID"})
    info["group"] = info["group"].map(GROUP_DISPLAY)
    merged = gaps.merge(info[["subID", "group"]], on="subID", how="inner")
    merged = merged[merged["group"].isin(GROUP_ORDER)].copy()

    group_data = [
        merged.loc[merged["group"] == g, "mean_intention_minus_experience_gap"].to_numpy(dtype=float)
        for g in GROUP_ORDER
        if (merged["group"] == g).any()
    ]
    if len(group_data) < 2:
        return pd.DataFrame(), merged

    f_stat, p_omnibus = stats.f_oneway(*group_data)
    group_means = (
        merged.groupby("group", observed=True)["mean_intention_minus_experience_gap"]
        .agg(["mean", "std", "count"])
        .reindex(GROUP_ORDER)
    )

    tukey_rows: list[dict[str, object]] = []
    if merged["group"].nunique() >= 2:
        tuk = pairwise_tukeyhsd(
            merged["mean_intention_minus_experience_gap"],
            merged["group"],
            alpha=0.05,
        )
        for row in tuk.summary().data[1:]:
            g1, g2, meandiff, p_adj, lower, upper, reject = row
            tukey_rows.append(
                {
                    "group_high": str(g2) if float(meandiff) > 0 else str(g1),
                    "group_low": str(g1) if float(meandiff) > 0 else str(g2),
                    "mean_diff": float(meandiff),
                    "p_adj": float(p_adj),
                    "ci_low": float(lower),
                    "ci_high": float(upper),
                    "significant": str(reject) == "True",
                }
            )

    summary = pd.DataFrame(
        [
            {
                "test": "one_way_anova",
                "n_subjects": int(len(merged)),
                "df_between": len(GROUP_ORDER) - 1,
                "df_within": int(len(merged) - len(GROUP_ORDER)),
                "F": float(f_stat),
                "p": float(p_omnibus),
                **{
                    f"mean_{g}": float(group_means.loc[g, "mean"]) if g in group_means.index else float("nan")
                    for g in GROUP_ORDER
                },
            }
        ]
    )
    return summary, pd.DataFrame(tukey_rows)


def write_selfreport_stats(
    *,
    ratings: pd.DataFrame,
    performance_dual: pd.DataFrame,
    sub_info: pd.DataFrame,
    data_dir: Path,
) -> None:
    """Write manuscript-referenced self-report inferential summaries to CSV."""
    data_dir.mkdir(parents=True, exist_ok=True)

    dominance = build_dominance_tests(ratings)
    dominance_path = data_dir / "selfreport_dominance_fdr_by_task.csv"
    dominance.to_csv(dominance_path, index=False)
    logger.info("Wrote %s (%d rows)", dominance_path, len(dominance))

    intention = build_intention_exceeds_experience_tests(performance_dual)
    intention_path = data_dir / "selfreport_performance_intention_gt_experience_fdr.csv"
    intention.to_csv(intention_path, index=False)
    logger.info("Wrote %s (%d rows)", intention_path, len(intention))

    gap_anova, gap_tukey = build_experience_intention_gap_group_tests(performance_dual, sub_info)
    gap_anova_path = data_dir / "selfreport_performance_gap_group_anova.csv"
    gap_tukey_path = data_dir / "selfreport_performance_gap_group_tukey.csv"
    gap_anova.to_csv(gap_anova_path, index=False)
    gap_tukey.to_csv(gap_tukey_path, index=False)
    logger.info("Wrote %s and %s", gap_anova_path, gap_tukey_path)
