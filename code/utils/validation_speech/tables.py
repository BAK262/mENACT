"""LaTeX tables for speech validation (cross-speaker and cross-mode)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from utils.validation_speech.cross_mode import CROSS_MODE_CONDITION_ORDER, CROSS_MODE_DISPLAY
from utils.validation_speech.cross_speaker import CROSS_SPEAKER_CONDITION_ORDER, CROSS_SPEAKER_DISPLAY
from utils.validation_style import MODEL_DISPLAY, write_tabular
from utils.validation_vocab import FEATURE_SETS, LABEL_MODES


def _metric_cell(summary_df: pd.DataFrame, test_id: str, metric_col: str) -> str:
    if summary_df.empty:
        return "N/A"
    hit = summary_df[summary_df["test_id"].astype(str) == str(test_id)]
    if hit.empty or metric_col not in hit.columns:
        return "N/A"
    v = pd.to_numeric(hit.iloc[0][metric_col], errors="coerce")
    if not np.isfinite(v):
        return "N/A"
    ci_lo_col = f"{metric_col}_ci_low"
    ci_hi_col = f"{metric_col}_ci_high"
    lo = pd.to_numeric(hit.iloc[0].get(ci_lo_col, float("nan")), errors="coerce")
    hi = pd.to_numeric(hit.iloc[0].get(ci_hi_col, float("nan")), errors="coerce")
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{float(v):.2f} ({float(lo):.2f}, {float(hi):.2f})"
    return f"{float(v):.2f}"


def _build_transposed_table(
    model_order: Sequence[str],
    summary_by_model: Dict[str, pd.DataFrame],
    first_col_header: str,
    col_specs: Sequence[tuple[str, str]],
) -> list[str]:
    n_cols = len(col_specs)
    col_spec = "l" + "cc" * n_cols
    col_headers = [rf"\multicolumn{{2}}{{c}}{{{label}}}" for label, _ in col_specs]
    top_line = f"{first_col_header} & " + " & ".join(col_headers) + r" \\"
    sub_cells = [""] + ["Acc & F1"] * n_cols
    sub_line = " & ".join(sub_cells) + r" \\"
    cmidrules = " ".join(rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(n_cols))
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        top_line,
        cmidrules,
        sub_line,
        r"\midrule",
    ]
    for m in model_order:
        cells: list[str] = [MODEL_DISPLAY[m]]
        s_df = summary_by_model.get(m, pd.DataFrame())
        for _, test_id in col_specs:
            cells.append(_metric_cell(s_df, test_id, "accuracy"))
            cells.append(_metric_cell(s_df, test_id, "macro_f1"))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def _first_model_summary(summary_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    for df in summary_by_model.values():
        if not df.empty:
            return df
    return pd.DataFrame()


def _row_sample_size(summary_df: pd.DataFrame, test_id: str) -> tuple[int, int]:
    hit = summary_df[summary_df["test_id"].astype(str) == str(test_id)]
    if hit.empty:
        return 0, 0
    row = hit.iloc[0]
    return int(row.get("n_subjects", 0) or 0), int(row.get("n_trials", 0) or 0)


def _cross_speaker_sample_note(summary_by_model: Dict[str, pd.DataFrame]) -> str:
    ref = _first_model_summary(summary_by_model)
    n_prod, _ = _row_sample_size(ref, "cross_speaker_production")
    n_perf, _ = _row_sample_size(ref, "cross_speaker_performance")
    if n_prod <= 0 or n_perf <= 0:
        return ""
    return (
        f"Spontaneous Production $N={n_prod}$; Deliberate Performance $N={n_perf}$ "
        "(participants with released speech recordings per task)."
    )


def _cross_mode_sample_note(summary_by_model: Dict[str, pd.DataFrame]) -> str:
    ref = _first_model_summary(summary_by_model)
    n_train, _ = _row_sample_size(ref, "cross_speaker_performance")
    n_test, n_trials = _row_sample_size(ref, "cross_mode_all")
    if n_test <= 0:
        return ""
    train_clause = f"trained on Deliberate Performance ($N={n_train}$)" if n_train > 0 else "trained on Deliberate Performance"
    trial_clause = f", {n_trials} trials" if n_trials > 0 else ""
    return (
        f"{train_clause} and evaluated on Spontaneous Production test recordings "
        f"($N={n_test}$ participants{trial_clause}; within-subject pairing where both tasks are released)."
    )


def _note_head(label_mode: str) -> str:
    cls_desc = (
        "3-class valence (negative, neutral, positive)"
        if label_mode == "valence3"
        else "9-category discrete emotion"
    )
    return (
        r"\par\smallskip\noindent\scriptsize\textit{Note.} "
        "Acc = pooled accuracy; F1 = pooled Macro-$F_1$; "
        "95\\% bootstrap CI in parentheses (trial-level resampling). "
        f"Classification: {cls_desc}. "
    )


def build_acc_f1_tex_tables(
    summary_by_model: Dict[str, pd.DataFrame],
    tables_dir: Path,
    label_mode: str,
) -> Dict[str, Path]:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    tables_dir.mkdir(parents=True, exist_ok=True)
    model_order = [m for m in FEATURE_SETS if m in summary_by_model]
    if not model_order:
        return {}

    path_map: Dict[str, Path] = {}

    cs_specs = [
        (CROSS_SPEAKER_DISPLAY[tid], tid) for tid in CROSS_SPEAKER_CONDITION_ORDER
    ]
    cs_lines = _build_transposed_table(model_order, summary_by_model, "Feature", cs_specs)
    cs_lines.append(
        _note_head(label_mode)
        + "Evaluation: cross-speaker evaluation with 5-fold subject-grouped cross-validation within each expressive mode. "
        + _cross_speaker_sample_note(summary_by_model)
    )
    path_cs = tables_dir / f"tab_speech_cross_speaker_{label_mode}_acc_f1_tabular.tex"
    write_tabular(cs_lines, path_cs)
    path_map["cross_speaker"] = path_cs

    cm_specs = [(CROSS_MODE_DISPLAY[tid], tid) for tid in CROSS_MODE_CONDITION_ORDER]
    cm_lines = _build_transposed_table(model_order, summary_by_model, "Feature", cm_specs)
    cm_lines.append(
        _note_head(label_mode)
        + "Evaluation: cross-mode evaluation (Deliberate Performance $\\rightarrow$ Spontaneous Production). "
        + _cross_mode_sample_note(summary_by_model)
        + " Actors and Non-actors subdivide the pooled Production test predictions without retraining."
    )
    path_cm = tables_dir / f"tab_speech_cross_mode_{label_mode}_acc_f1_tabular.tex"
    write_tabular(cm_lines, path_cm)
    path_map["cross_mode"] = path_cm

    return path_map
