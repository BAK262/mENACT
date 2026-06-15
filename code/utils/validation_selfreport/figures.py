"""Figures for trial-level self-report validation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from utils.validation_selfreport.performance_dual_ratings import (
    GROUP_ORDER,
    LAYER_DIFFERENCE,
    LAYER_EXPERIENCE,
    LAYER_INTENTION,
    LAYER_ORDER,
)
from utils.validation_selfreport.ratings import TARGET_LEVELS
from utils.validation_style import (
    ACCENT_DARK,
    ACCENT_LIGHT,
    EMOTION_ABBR,
    EMOTION_FULL,
    EMOTIONS_8,
    PAGE_W,
    TASK_ORDER,
    add_ygrid,
    fig_double_multi,
    save_fig,
    strip_spines,
    task_caption_label,
)
from utils.validation_vocab import GROUP_DISPLAY


def plot_target_structure_figure(
    bar_all: pd.DataFrame,
    result_dir: Path,
) -> None:
    """Faceted bar chart: target vs non-target intensities across three tasks."""
    fig, axes = fig_double_multi(3, 1, h=3.55)
    facet_order = list(TASK_ORDER)

    dodge_w = 0.72
    n_emos = len(EMOTIONS_8)
    target_abbr = {k: EMOTION_ABBR[k] for k in TARGET_LEVELS}

    for ax_idx, (ax, facet) in enumerate(zip(axes, facet_order)):
        sub = bar_all[bar_all["task"] == facet].copy()
        target_cats = TARGET_LEVELS
        emo_dims = EMOTIONS_8

        for ti, tcat in enumerate(target_cats):
            sub_t = sub[sub["targetEmotion"] == tcat]
            for di, dim in enumerate(emo_dims):
                row = sub_t[sub_t["emotion"] == dim]
                if row.empty:
                    continue

                x = ti + (di - (n_emos - 1) / 2) * (dodge_w / n_emos)
                is_target = dim == tcat
                color = ACCENT_DARK if is_target else ACCENT_LIGHT
                mean_val = row["mean"].values[0]
                se_val = row["se"].values[0]

                ax.bar(
                    x,
                    mean_val,
                    width=dodge_w / n_emos * 0.92,
                    color=color,
                    edgecolor="none",
                    zorder=3,
                )
                ax.errorbar(
                    x,
                    mean_val,
                    yerr=se_val,
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.35,
                    capsize=1.2,
                    capthick=0.35,
                    zorder=4,
                )

        ax.set_xticks(range(len(target_cats)))
        ax.set_xticklabels(
            [target_abbr[c] for c in target_cats], rotation=40, ha="right"
        )
        ax.set_ylabel("Self-report ratings" if ax_idx == 1 else "")
        ax.set_title(task_caption_label(facet), fontweight="bold", fontsize=7.5, pad=2)
        strip_spines(ax)
        add_ygrid(ax)
        ax.set_xlim(-0.6, len(target_cats) - 0.4)

    axes[-1].set_xlabel("Target category")

    legend_handles = [
        Patch(facecolor=ACCENT_DARK, label="Target dimension"),
        Patch(facecolor=ACCENT_LIGHT, label="Non-target dimension"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.subplots_adjust(hspace=0.58, top=0.90, bottom=0.10, left=0.06, right=0.98)
    save_fig(fig, "fig_trial_selfreport_target_structure", result_dir)


def plot_performance_experience_intention_grid(
    experience_by_group: pd.DataFrame,
    intention_by_group: pd.DataFrame,
    diff_by_group: pd.DataFrame,
    result_dir: Path,
) -> None:
    """Appendix 3x3 heatmap grid: groups x experience/intention/difference layers."""
    target_full = {k: EMOTION_FULL[k] for k in TARGET_LEVELS}
    layer_data = {
        LAYER_EXPERIENCE: experience_by_group,
        LAYER_INTENTION: intention_by_group,
        LAYER_DIFFERENCE: diff_by_group,
    }

    fig3 = plt.figure(figsize=(PAGE_W, 7.0))
    gs = fig3.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=[1, 1, 1.25],
        hspace=0.20,
        wspace=0.08,
        top=0.95,
        bottom=0.22,
    )
    axes3 = np.array([[fig3.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)])

    diff_max = diff_by_group["mean_rating"].abs().max()
    if not np.isfinite(diff_max) or diff_max < 1e-6:
        diff_max = 1.0

    im_bounded = None
    im_diff = None

    for ri, grp in enumerate(GROUP_ORDER):
        for ci, lyr in enumerate(LAYER_ORDER):
            ax = axes3[ri, ci]
            src = layer_data[lyr]
            dcell = src[src["group"] == grp].copy()

            matrix = np.full((len(TARGET_LEVELS), len(EMOTIONS_8)), np.nan)
            for _, row in dcell.iterrows():
                if row["targetEmotion"] in TARGET_LEVELS and row["emotion"] in EMOTIONS_8:
                    ri_idx = TARGET_LEVELS.index(row["targetEmotion"])
                    ci_idx = EMOTIONS_8.index(row["emotion"])
                    matrix[ri_idx, ci_idx] = row["mean_rating"]

            if lyr in (LAYER_EXPERIENCE, LAYER_INTENTION):
                im = ax.imshow(
                    matrix,
                    aspect="auto",
                    cmap="magma",
                    vmin=0,
                    vmax=100,
                    interpolation="nearest",
                )
                if im_bounded is None:
                    im_bounded = im
            else:
                im = ax.imshow(
                    matrix,
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-diff_max,
                    vmax=diff_max,
                    interpolation="nearest",
                )
                if im_diff is None:
                    im_diff = im

            ax.set_xticks(range(len(EMOTIONS_8)))
            ax.set_yticks(range(len(TARGET_LEVELS)))

            if ri == len(GROUP_ORDER) - 1:
                ax.set_xticklabels(
                    [EMOTION_FULL[e] for e in EMOTIONS_8],
                    rotation=45,
                    ha="right",
                    fontsize=6,
                )
                if ci == 1:
                    ax.set_xlabel("Dimension", fontsize=7)
            else:
                ax.set_xticklabels([])

            if ci == 0:
                ax.set_yticklabels(
                    [target_full[t] for t in TARGET_LEVELS],
                    fontsize=6,
                )
                ax.set_ylabel(GROUP_DISPLAY[grp], fontsize=7, fontweight="bold")
            else:
                ax.set_yticklabels([])

            if ri == 0:
                ax.set_title(lyr, fontsize=8, fontweight="bold", pad=4)

    if im_bounded is not None:
        cax1 = fig3.add_axes([0.10, 0.08, 0.42, 0.018])
        cb1 = fig3.colorbar(im_bounded, cax=cax1, orientation="horizontal")
        cb1.set_label("Mean rating (0–100)", fontsize=6.5)
        cb1.ax.tick_params(labelsize=6)
    if im_diff is not None:
        cax2 = fig3.add_axes([0.60, 0.08, 0.30, 0.018])
        cb2 = fig3.colorbar(im_diff, cax=cax2, orientation="horizontal")
        cb2.set_label("Experience − Intention", fontsize=6.5)
        cb2.ax.tick_params(labelsize=6)
    save_fig(fig3, "fig_appendix_performance_experience_intention_diff_grid", result_dir)
