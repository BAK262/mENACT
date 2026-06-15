"""Shared visual style module for IEEE TAFFC validation figures and tables.

Single source of truth for physical constants, color palettes, label
dictionaries, and helper functions used across all validation scripts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.validation_vocab import LEGACY_TASK_ALIASES, TASK_DISPLAY, TASK_KEYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IEEE TAFFC physical constants
# ---------------------------------------------------------------------------
COL_W: float = 3.5      # single-column width in inches (88 mm)
PAGE_W: float = 7.16    # double-column width in inches (181 mm)
MAX_H: float = 8.5      # maximum figure depth in inches (216 mm)
DPI: int = 300

# ---------------------------------------------------------------------------
# Semantic color palettes
# ---------------------------------------------------------------------------
TASK_COLORS: dict[str, str] = {
    "Perception": "#4C72B0",
    "Production": "#DD8452",
    "Performance": "#55A868",
}

GROUP_COLORS: dict[str, str] = {
    "Professional": "#7570B3",
    "Amateur": "#E7298A",
    "Non-actor": "#A6761D",
}

# Coordinated dark/light accent pair for non-task contrasts (self-report target
# structure, speech feature-set bars, and similar binary emphasis patterns).
ACCENT_DARK: str = "#3b528b"
ACCENT_LIGHT: str = "#b8c4d6"

# Speech validation: eGeMAPS (lighter) vs HuBERT (darker), aligned with self-report.
SPEECH_FEATURE_COLORS: dict[str, str] = {
    "egemaps": ACCENT_LIGHT,
    "hubert": ACCENT_DARK,
}

VALENCE_COLORS: dict[str, str] = {
    "Positive": "#E24A33",
    "Neutral": "#999999",
    "Negative": "#348ABD",
}

REGION_COLORS: dict[str, str] = {
    "L-F": "#66C2A5",
    "M-F": "#FC8D62",
    "R-F": "#8DA0CB",
    "L-T": "#E78AC3",
    "R-T": "#A6D854",
}

MODEL_DISPLAY: dict[str, str] = {
    "egemaps": "eGeMAPS",
    "hubert": "HuBERT",
}

# ---------------------------------------------------------------------------
# Task identifiers (manuscript display order)
# ---------------------------------------------------------------------------
TASKS: list[str] = [TASK_DISPLAY[k] for k in TASK_KEYS]
TASK_ORDER: list[str] = TASKS

TASK_CONCRETE: dict[str, str] = {
    "Perception": "video clip viewing",
    "Production": "personal narrative",
    "Performance": "scripted enactment",
}

TASK_CONCRETE_DATASET_ROW: str = (
    "Video clip viewing; personal narrative; scripted enactment"
)

EMOTION_FULL: dict[str, str] = {
    "anger": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "sadness": "Sadness",
    "amusement": "Amusement",
    "inspiration": "Inspiration",
    "joy": "Joy",
    "tenderness": "Tenderness",
    "neutral": "Neutral",
}

EMOTION_ABBR: dict[str, str] = {
    "anger": "Ang",
    "disgust": "Dis",
    "fear": "Fear",
    "sadness": "Sad",
    "amusement": "Amu",
    "inspiration": "Ins",
    "joy": "Joy",
    "tenderness": "Ten",
    "neutral": "Neu",
}

EMOTIONS_9: list[str] = [
    "anger", "disgust", "fear", "sadness",
    "amusement", "inspiration", "joy", "tenderness", "neutral",
]
EMOTIONS_8: list[str] = EMOTIONS_9[:-1]


# ---------------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------------
def apply_style() -> None:
    """Set matplotlib rcParams for IEEE TAFFC conformance. Call once per script."""
    params: dict[str, Any] = {
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    mpl.rcParams.update(params)


# ---------------------------------------------------------------------------
# Figure creation helpers
# ---------------------------------------------------------------------------
def fig_single(h: float = 2.5, **subplots_kw: Any) -> tuple[plt.Figure, plt.Axes]:
    """Create a single-column figure (COL_W x h inches)."""
    fig, ax = plt.subplots(figsize=(COL_W, h), **subplots_kw)
    return fig, ax


def fig_double(h: float = 2.5, **subplots_kw: Any) -> tuple[plt.Figure, plt.Axes]:
    """Create a double-column figure (PAGE_W x h inches)."""
    fig, ax = plt.subplots(figsize=(PAGE_W, h), **subplots_kw)
    return fig, ax


def fig_single_multi(
    nrows: int, ncols: int, h: float = 2.5, **kw: Any
) -> tuple[plt.Figure, np.ndarray]:
    """Create a multi-panel single-column figure."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL_W, h), **kw)
    return fig, axes


def fig_double_multi(
    nrows: int, ncols: int, h: float = 2.5, **kw: Any
) -> tuple[plt.Figure, np.ndarray]:
    """Create a multi-panel double-column figure."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_W, h), **kw)
    return fig, axes


# ---------------------------------------------------------------------------
# Axes formatting helpers
# ---------------------------------------------------------------------------
def strip_spines(ax: plt.Axes) -> None:
    """Remove top and right spines from an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_ygrid(ax: plt.Axes) -> None:
    """Add subtle y-axis grid lines."""
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, linestyle="--", zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def normalize_task_name(name: str) -> str:
    """Map legacy or canonical task identifier to manuscript task display name."""
    key = str(name).strip()
    if key in TASK_ORDER:
        return key
    lower = key.lower()
    if lower in TASK_DISPLAY:
        return TASK_DISPLAY[lower]
    if lower in LEGACY_TASK_ALIASES:
        return TASK_DISPLAY[LEGACY_TASK_ALIASES[lower]]
    return key


def task_display_name(internal_name: str) -> str:
    """Return manuscript task label for plots and tables."""
    return normalize_task_name(internal_name)


def include_task_in_three_task_analysis(task: str) -> bool:
    """Return True when task maps to Perception, Production, or Performance."""
    return normalize_task_name(task) in TASK_ORDER


def filter_fnirs_three_task_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose task maps to the three manuscript tasks."""
    if df.empty or "task" not in df.columns:
        return df.copy()
    work = df.copy()
    work["task"] = work["task"].map(normalize_task_name)
    out = work.loc[work["task"].isin(TASK_ORDER)].copy()
    dropped = len(work) - len(out)
    if dropped:
        logger.info("filter_fnirs_three_task_rows: dropped %d rows", dropped)
    return out.reset_index(drop=True)


def task_concrete_gloss(abstract_name: str) -> str:
    """Return the concrete task gloss for an abstract task label."""
    return TASK_CONCRETE.get(abstract_name, abstract_name)


def task_caption_label(abstract_name: str) -> str:
    """Return abstract (concrete) label for figure/table captions."""
    return f"{abstract_name} ({task_concrete_gloss(abstract_name)})"


# ---------------------------------------------------------------------------
# Save / export helpers
# ---------------------------------------------------------------------------
def save_fig(fig: plt.Figure, stem: str, result_dir: Path, close: bool = True) -> None:
    """Save figure as PDF+PNG under ``result_dir/figures/``.

    Parameters
    ----------
    fig : matplotlib Figure
    stem : filename without extension
    result_dir : validation result root (parent of ``figures/``)
    close : whether to close figure after saving
    """
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = fig_dir / f"{stem}.pdf"
    png_path = fig_dir / f"{stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    logger.info("Saved %s and %s", pdf_path, png_path)

    if close:
        plt.close(fig)


def write_tabular(lines: list[str], path: Path) -> None:
    """Write LaTeX tabular lines to a file.

    Parameters
    ----------
    lines : list of LaTeX source lines
    path : output .tex file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote tabular to %s", path)
