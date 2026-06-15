"""Appendix figures for trait scales and ECS heatmaps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.validation_style import (
    EMOTION_FULL,
    GROUP_COLORS,
    fig_double_multi,
    save_fig,
)
from utils.validation_traits.load import (
    ECS_TAXONOMY,
    FAMILY_DIMS,
    GROUP_ORDER,
    INSTRUMENT_ORDER,
)
from utils.validation_traits.tables import _pretty_dim


def plot_trait_violins(traits: pd.DataFrame, result_dir: Path) -> None:
    """15-facet violin+jitter plot (3 rows x 5 cols, one row per family)."""
    rng = np.random.default_rng(42)
    fig, axes = fig_double_multi(
        nrows=3, ncols=5, h=6.0, gridspec_kw={"hspace": 0.70, "wspace": 0.35},
    )
    colors = [GROUP_COLORS[g] for g in GROUP_ORDER]

    for row_idx, family in enumerate(INSTRUMENT_ORDER):
        dims = FAMILY_DIMS[family]
        for col_idx, dim in enumerate(dims):
            ax = axes[row_idx, col_idx]
            for g_idx, group in enumerate(GROUP_ORDER):
                vals = traits.loc[traits["group"] == group, dim].dropna().values
                if len(vals) >= 2:
                    parts = ax.violinplot(
                        [vals],
                        positions=[g_idx],
                        widths=0.85,
                        showmeans=False,
                        showmedians=False,
                        showextrema=False,
                    )
                    for pc in parts["bodies"]:
                        pc.set_facecolor(colors[g_idx])
                        pc.set_alpha(0.35)
                        pc.set_edgecolor("none")
                if len(vals) > 0:
                    jitter = rng.uniform(-0.07, 0.07, size=len(vals))
                    ax.scatter(
                        g_idx + jitter,
                        vals,
                        s=5,
                        alpha=0.35,
                        color=colors[g_idx],
                        edgecolors="none",
                        zorder=3,
                    )

            ax.set_title(
                f"{family}\n{_pretty_dim(dim)}", fontsize=7, pad=3,
            )
            ax.set_xticks(range(len(GROUP_ORDER)))
            ax.set_xticklabels(
                GROUP_ORDER, rotation=30, ha="right", fontsize=6,
            )
            if col_idx == 0:
                ax.set_ylabel("Score")

    save_fig(fig, "fig_appendix_trait_scales_by_experience", result_dir)


def plot_ecs_heatmaps(
    traits: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    result_dir: Path,
) -> None:
    """Three side-by-side 9x9 heatmaps for extreme ECS cases."""
    ecs_labels = [EMOTION_FULL[e] for e in ECS_TAXONOMY]

    i_high = int(traits.loc[traits["mean_pairwise"].idxmax(), "subID"])
    i_low = int(traits.loc[traits["mean_pairwise"].idxmin(), "subID"])
    i_mixed = int(traits.loc[traits["sd_pairwise"].idxmax(), "subID"])

    cases = [
        (i_high, "High mean pairwise"),
        (i_low, "Low mean pairwise"),
        (i_mixed, "High dispersion"),
    ]

    fig, axes = fig_double_multi(1, 3, h=2.8)
    im = None

    for idx, (sub_id, label) in enumerate(cases):
        ax = axes[idx]
        mat = matrices[str(sub_id)]
        im = ax.imshow(mat, cmap="magma", vmin=0, vmax=10, aspect="equal")

        for edge in range(1, 9):
            ax.axhline(edge - 0.5, color="white", lw=0.3)
            ax.axvline(edge - 0.5, color="white", lw=0.3)

        ax.set_xticks(range(9))
        ax.set_xticklabels(ecs_labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(9))
        ax.set_yticklabels(ecs_labels, fontsize=6)
        ax.set_title(
            f"{label} (sub {sub_id})", fontweight="bold", fontsize=8,
        )

    fig.colorbar(
        im,
        ax=axes.tolist(),
        orientation="horizontal",
        shrink=0.35,
        pad=0.22,
        aspect=25,
        label="Similarity",
    )

    save_fig(fig, "fig_appendix_ecs_three_case_heatmaps", result_dir)
