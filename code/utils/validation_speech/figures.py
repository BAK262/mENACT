"""Speech validation bar charts (cross-speaker | cross-mode)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from utils.validation_speech.cross_mode import CROSS_MODE_CONDITION_ORDER, CROSS_MODE_DISPLAY
from utils.validation_speech.cross_speaker import CROSS_SPEAKER_CONDITION_ORDER, CROSS_SPEAKER_DISPLAY
from utils.validation_style import (
    COL_W,
    MODEL_DISPLAY,
    SPEECH_FEATURE_COLORS,
    add_ygrid,
    apply_style,
    save_fig,
    strip_spines,
)
from utils.validation_vocab import FEATURE_SETS, LABEL_MODES

_XTICK_ROTATION = 22.0


def _read_macro_f1_ci(summary_df: pd.DataFrame, test_id: str) -> Tuple[float, float, float]:
    hit = summary_df[summary_df["test_id"] == test_id]
    if hit.empty:
        return float("nan"), float("nan"), float("nan")
    row = hit.iloc[0]
    return float(row["macro_f1"]), float(row["macro_f1_ci_low"]), float(row["macro_f1_ci_high"])


def _speech_decoding_ylim(
    summary_by_model: Dict[str, pd.DataFrame],
    model_order: Sequence[str],
    test_ids: Sequence[str],
    chance_f1: float,
) -> Tuple[float, float]:
    hi_vals = [chance_f1]
    lo_vals = [chance_f1]
    for model_name in model_order:
        s_df = summary_by_model.get(model_name, pd.DataFrame())
        for test_id in test_ids:
            f1, lo, hi = _read_macro_f1_ci(s_df, test_id)
            if np.isfinite(hi):
                hi_vals.append(hi)
            if np.isfinite(lo):
                lo_vals.append(lo)
            elif np.isfinite(f1):
                lo_vals.append(f1)
    y_bot = max(0.0, min(lo_vals) - 0.04, chance_f1 - 0.10)
    max_hi = max(hi_vals)
    legend_clear_top = (max_hi + 0.012 - 0.10 * y_bot) / 0.90
    y_top = max(max_hi + 0.03, legend_clear_top)
    return y_bot, y_top


def _plot_validation_region(
    ax: plt.Axes,
    summary_by_model: Dict[str, pd.DataFrame],
    model_order: Sequence[str],
    condition_specs: Sequence[Tuple[str, str]],
    chance_f1: float,
    ylim: Tuple[float, float],
    *,
    xtick_rotation: float = 0.0,
) -> None:
    n_models = len(model_order)
    n_slices = len(condition_specs)
    bar_width = 0.72 / n_models

    for slice_idx, (test_id, _slice_label) in enumerate(condition_specs):
        x_center = float(slice_idx)
        for model_idx, model_name in enumerate(model_order):
            offset = (model_idx - (n_models - 1) / 2.0) * bar_width
            x_pos = x_center + offset
            f1, lo, hi = _read_macro_f1_ci(summary_by_model[model_name], test_id)
            if not np.isfinite(f1):
                continue
            err_lo = max(f1 - lo, 0.0) if np.isfinite(lo) else 0.0
            err_hi = max(hi - f1, 0.0) if np.isfinite(hi) else 0.0
            ax.bar(
                x_pos,
                f1,
                width=bar_width * 0.88,
                color=SPEECH_FEATURE_COLORS.get(model_name, "#888888"),
                edgecolor="white",
                linewidth=0.4,
                yerr=np.array([[err_lo], [err_hi]]),
                capsize=1.8,
                error_kw={"linewidth": 0.7, "capthick": 0.7, "ecolor": "#333333"},
                zorder=3,
            )

    ax.axhline(chance_f1, color="grey", linestyle="--", linewidth=0.7, zorder=1)
    ax.set_xticks(np.arange(n_slices, dtype=float))
    tick_labels = [label for _, label in condition_specs]
    ax.set_xticklabels(
        tick_labels,
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
    )
    ax.set_xlim(-0.55, n_slices - 0.45)
    ax.set_ylim(*ylim)
    strip_spines(ax)
    add_ygrid(ax)


def plot_speech_decoding_validation_figure(
    summary_by_model: Dict[str, pd.DataFrame],
    label_mode: str,
    result_dir: Path,
) -> Path:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    apply_style()
    stem = f"fig_speech_decoding_{label_mode}_validation"
    model_order = [m for m in FEATURE_SETS if m in summary_by_model]
    if not model_order:
        return result_dir / "figures" / f"{stem}.pdf"

    chance_f1 = 0.333 if label_mode == "valence3" else 0.111
    cs_specs = [(tid, CROSS_SPEAKER_DISPLAY[tid]) for tid in CROSS_SPEAKER_CONDITION_ORDER]
    cm_specs = [(tid, CROSS_MODE_DISPLAY[tid]) for tid in CROSS_MODE_CONDITION_ORDER]
    all_test_ids = [tid for tid, _ in cs_specs + cm_specs]
    ylim = _speech_decoding_ylim(summary_by_model, model_order, all_test_ids, chance_f1)

    fig = plt.figure(figsize=(COL_W, 2.8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 3], wspace=0.18)
    ax_cs = fig.add_subplot(gs[0, 0])
    ax_cm = fig.add_subplot(gs[0, 1], sharey=ax_cs)

    _plot_validation_region(ax_cs, summary_by_model, model_order, cs_specs, chance_f1, ylim, xtick_rotation=_XTICK_ROTATION)
    _plot_validation_region(ax_cm, summary_by_model, model_order, cm_specs, chance_f1, ylim, xtick_rotation=_XTICK_ROTATION)

    ax_cs.set_ylabel("Macro-F1")
    ax_cs.set_title("Cross-speaker", fontweight="bold", pad=2)
    ax_cm.set_title("Cross-mode", fontweight="bold", pad=2)
    plt.setp(ax_cm.get_yticklabels(), visible=False)

    legend_handles = [
        Patch(
            facecolor=SPEECH_FEATURE_COLORS[m],
            edgecolor="white",
            linewidth=0.4,
            label=MODEL_DISPLAY[m],
        )
        for m in model_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(model_order),
        frameon=False,
        bbox_to_anchor=(0.5, 0.815),
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.1,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.14, right=0.98, top=0.74, bottom=0.24)
    save_fig(fig, stem, result_dir)
    return result_dir / "figures" / f"{stem}.pdf"
