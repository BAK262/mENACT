"""Manuscript tables and figures for validation_fnirs_decoding."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from utils.validation_fnirs.features import VALENCE3_CLASSES
from utils.validation_fnirs.task_wise_loso import (
    BOOTSTRAP_N_MANUSCRIPT,
    bootstrap_ci_pooled_from_trial_preds,
    compute_metrics_str,
    load_trial_predictions,
    multiclass_per_class_f1_map,
    stable_int_hash,
)
from utils.validation_style import (
    EMOTION_ABBR,
    EMOTIONS_9,
    TASK_COLORS,
    TASKS,
    add_ygrid,
    fig_double,
    fig_single,
    save_fig,
    strip_spines,
    task_display_name,
    write_tabular,
)
from utils.validation_fnirs.stats import write_fnirs_decoding_stats

LOGGER = logging.getLogger("validation_fnirs_decoding.report")


def _data_dir(result_dir: Path) -> Path:
    return result_dir / "data"


def _tables_dir(result_dir: Path) -> Path:
    return result_dir / "tables"


def valence3_main_row_from_predictions(pred_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if pred_df.empty or "y_true" not in pred_df.columns or "y_pred" not in pred_df.columns:
        return None
    yt = pred_df["y_true"].to_numpy(dtype=object)
    yp = pred_df["y_pred"].to_numpy(dtype=object)
    met = compute_metrics_str(yt, yp)
    f1m = multiclass_per_class_f1_map(yt, yp, labels=list(VALENCE3_CLASSES))
    return {
        "n_trials": int(len(pred_df)),
        "overall_acc": float(met["accuracy"]),
        "overall_f1": float(met["macro_f1_present"]),
        "f1_positive": float(f1m["positive"]),
        "f1_neutral": float(f1m["neutral"]),
        "f1_negative": float(f1m["negative"]),
    }


def collect_valence3_main_rows(result_dir: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for task in TASKS:
        pred = load_trial_predictions(result_dir, task=str(task), label_mode="valence3")
        pred = pred.drop(columns=["task", "label_mode"], errors="ignore") if not pred.empty else pred
        if pred.empty:
            continue
        r = valence3_main_row_from_predictions(pred)
        if not r:
            continue
        out.append({"task": str(task), **r})
    return out


def emotion9_appendix_row_from_predictions(pred_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if pred_df.empty or "y_true" not in pred_df.columns or "y_pred" not in pred_df.columns:
        return None
    yt = pred_df["y_true"].to_numpy(dtype=object)
    yp = pred_df["y_pred"].to_numpy(dtype=object)
    met = compute_metrics_str(yt, yp)
    f1m = multiclass_per_class_f1_map(yt, yp, labels=list(EMOTIONS_9))
    row: Dict[str, object] = {
        "n_trials": int(len(pred_df)),
        "overall_acc": float(met["accuracy"]),
        "overall_f1": float(met["macro_f1_present"]),
    }
    for e in EMOTIONS_9:
        row[f"f1_{e}"] = float(f1m[str(e)])
    return row


def collect_emotion9_appendix_rows(result_dir: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for task in TASKS:
        pred = load_trial_predictions(result_dir, task=str(task), label_mode="emotion9")
        pred = pred.drop(columns=["task", "label_mode"], errors="ignore") if not pred.empty else pred
        if pred.empty:
            continue
        r = emotion9_appendix_row_from_predictions(pred)
        if not r:
            continue
        out.append({"task": str(task), **r})
    return out


def _fmt_ci(v: object, ci_lo: object, ci_hi: object) -> str:
    try:
        x = float(v)
    except Exception:
        return "--"
    if not np.isfinite(x):
        return "--"
    try:
        lo = float(ci_lo)
        hi = float(ci_hi)
    except Exception:
        return f"{x:.3f}"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{x:.3f} ({lo:.3f}, {hi:.3f})"
    return f"{x:.3f}"


def write_decoding_summary_tex(
    path: Path,
    rows: Sequence[Dict[str, object]],
    *,
    label_mode: str,
) -> None:
    if not rows:
        return

    task_order = [t for t in TASKS if any(str(r.get("task", "")) == t for r in rows)]
    task_display = [task_display_name(t) for t in task_order]

    col_spec = "l" + (r">{\centering\arraybackslash}X" * 2)
    lines = [
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
        r"\toprule",
        r"Task & Accuracy & Macro-F1 \\",
        r"\midrule",
    ]

    row_by_task = {str(r["task"]): r for r in rows}
    for t, display in zip(task_order, task_display):
        rr = row_by_task.get(t, {})
        acc_cell = _fmt_ci(rr.get("overall_acc"), rr.get("acc_ci_low"), rr.get("acc_ci_high"))
        f1_cell = _fmt_ci(rr.get("overall_f1"), rr.get("macro_f1_ci_low"), rr.get("macro_f1_ci_high"))
        lines.append(f"{display} & {acc_cell} & {f1_cell}" + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabularx}"])

    if label_mode == "valence3":
        cls_desc = "3-class valence (negative, neutral, positive)"
    else:
        cls_desc = "9-category discrete emotion"

    lines.append(
        r"\par\smallskip\noindent\scriptsize\textit{Note.} "
        "Accuracy = pooled accuracy; Macro-F1 = macro-averaged $F_1$; "
        "95\\% bootstrap CI in parentheses (subject-level resampling). "
        f"Classification: {cls_desc}. "
        "Evaluation: leave-one-subject-out cross-validation ($N=53$)."
    )
    write_tabular(lines, path)


def write_decoding_perclass_tex(
    path: Path,
    rows: Sequence[Dict[str, object]],
    *,
    label_mode: str,
) -> None:
    if not rows:
        return

    task_order = [t for t in TASKS if any(str(r.get("task", "")) == t for r in rows)]
    task_display = [task_display_name(t) for t in task_order]
    n_tasks = len(task_order)

    if label_mode == "valence3":
        classes = list(VALENCE3_CLASSES)
        class_labels = [c.capitalize() for c in classes]
        cls_desc = "3-class valence (negative, neutral, positive)"
    else:
        classes = list(EMOTIONS_9)
        class_labels = [EMOTION_ABBR.get(e, e.capitalize()) for e in classes]
        cls_desc = "9-category discrete emotion"

    col_spec = "l" + (r">{\centering\arraybackslash}X" * n_tasks)
    hdr_cells = ["Class"] + [f"F1 {name}" for name in task_display]

    lines = [
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(hdr_cells) + r" \\",
        r"\midrule",
    ]

    row_by_task = {str(r["task"]): r for r in rows}
    for cls, cls_lbl in zip(classes, class_labels):
        cells: list[str] = [cls_lbl]
        for t in task_order:
            rr = row_by_task.get(t, {})
            cells.append(_fmt_ci(
                rr.get(f"f1_{cls}"),
                rr.get(f"f1_{cls}_ci_low"),
                rr.get(f"f1_{cls}_ci_high"),
            ))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    lines.append(
        r"\par\smallskip\noindent\scriptsize\textit{Note.} "
        "F1 = class-specific $F_1$; "
        "95\\% bootstrap CI in parentheses (subject-level resampling). "
        f"Classification: {cls_desc}. "
        "Evaluation: leave-one-subject-out cross-validation ($N=53$)."
    )
    write_tabular(lines, path)


def plot_fnirs_decoding_grouped_bars(
    rows: List[Dict[str, object]],
    label_mode: str,
    result_dir: Path,
) -> None:
    if not rows:
        return

    if label_mode == "valence3":
        per_class_keys = ["positive", "neutral", "negative"]
        category_labels = ["Overall", "Positive", "Neutral", "Negative"]
        chance = 1.0 / 3.0
        fig, ax = fig_single(h=2.2)
        stem = "fig_fnirs_decoding_valence3_grouped_bars"
    elif label_mode == "emotion9":
        per_class_keys = list(EMOTIONS_9)
        category_labels = ["Overall"] + [EMOTION_ABBR.get(e, e.capitalize()) for e in EMOTIONS_9]
        chance = 1.0 / 9.0
        fig, ax = fig_double(h=2.2)
        stem = "fig_fnirs_decoding_emotion9_grouped_bars"
    else:
        return

    tasks_present = [str(r["task"]) for r in rows]
    tasks_ordered = [t for t in TASKS if t in tasks_present]
    if not tasks_ordered:
        plt.close(fig)
        return

    n_categories = 1 + len(per_class_keys)
    n_tasks = len(tasks_ordered)

    bar_width = 0.7 / n_tasks
    x_base = np.arange(n_categories, dtype=float)

    for ti, task in enumerate(tasks_ordered):
        rr = next((r for r in rows if str(r["task"]) == task), None)
        if rr is None:
            continue

        vals: List[float] = []
        err_lo: List[float] = []
        err_hi: List[float] = []

        ov_f1 = float(rr.get("overall_f1", float("nan")))
        vals.append(ov_f1)
        mf1_lo = float(rr.get("macro_f1_ci_low", float("nan")))
        mf1_hi = float(rr.get("macro_f1_ci_high", float("nan")))
        if np.isfinite(ov_f1) and np.isfinite(mf1_lo) and np.isfinite(mf1_hi):
            err_lo.append(max(0.0, ov_f1 - mf1_lo))
            err_hi.append(max(0.0, mf1_hi - ov_f1))
        else:
            err_lo.append(0.0)
            err_hi.append(0.0)

        for ck in per_class_keys:
            v = float(rr.get(f"f1_{ck}", float("nan")))
            vals.append(v)
            cl = float(rr.get(f"f1_{ck}_ci_low", float("nan")))
            ch = float(rr.get(f"f1_{ck}_ci_high", float("nan")))
            if np.isfinite(v) and np.isfinite(cl) and np.isfinite(ch):
                err_lo.append(max(0.0, v - cl))
                err_hi.append(max(0.0, ch - v))
            else:
                err_lo.append(0.0)
                err_hi.append(0.0)

        offset = (ti - (n_tasks - 1) / 2.0) * bar_width
        x_pos = x_base + offset
        display_name = task_display_name(task)
        color = TASK_COLORS.get(display_name, "#888888")
        yerr = np.vstack([err_lo, err_hi])

        ax.bar(
            x_pos,
            vals,
            width=bar_width * 0.88,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=display_name,
            yerr=yerr,
            capsize=1.8,
            error_kw={"linewidth": 0.7, "capthick": 0.7},
            zorder=3,
        )

    ax.axhline(chance, color="grey", linewidth=0.7, linestyle="--", zorder=2)

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        category_labels,
        rotation=30 if label_mode == "emotion9" else 0,
        ha="right" if label_mode == "emotion9" else "center",
    )
    ax.set_ylabel("F1")
    ax.set_xlim(x_base[0] - 0.5, x_base[-1] + 0.5)
    ax.set_ylim(0, None)

    strip_spines(ax)
    add_ygrid(ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, frameon=False,
        loc="upper center", bbox_to_anchor=(0.5, 1.15),
        ncol=len(handles), handlelength=1.0, handletextpad=0.3,
        borderpad=0.2, columnspacing=0.8,
    )

    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.90))
    save_fig(fig, stem, result_dir)


def bootstrap_per_class_f1_ci(
    result_dir: Path,
    label_mode: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    if label_mode == "valence3":
        classes = list(VALENCE3_CLASSES)
    elif label_mode == "emotion9":
        classes = list(EMOTIONS_9)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    nan_ci: Dict[str, Tuple[float, float]] = {c: (float("nan"), float("nan")) for c in classes}
    result: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for task in TASKS:
        pred = load_trial_predictions(result_dir, task=str(task), label_mode=str(label_mode))
        pred = pred.drop(columns=["task", "label_mode"], errors="ignore") if not pred.empty else pred

        if pred.empty or "y_true" not in pred.columns or "y_pred" not in pred.columns:
            result[task] = dict(nan_ci)
            continue

        if "test_subject" not in pred.columns:
            result[task] = dict(nan_ci)
            continue

        pred["test_subject"] = pred["test_subject"].astype(int)
        subjects = sorted(pd.unique(pred["test_subject"]).astype(int).tolist())
        if len(subjects) < 2:
            result[task] = dict(nan_ci)
            continue

        rng = np.random.RandomState(int(seed) + stable_int_hash(f"{task}_{label_mode}", mod=997))
        labs = np.asarray(classes, dtype=object)
        boot_f1: Dict[str, List[float]] = {c: [] for c in classes}

        by_subject_true: Dict[int, np.ndarray] = {}
        by_subject_pred: Dict[int, np.ndarray] = {}
        for sid, sd in pred.groupby("test_subject"):
            by_subject_true[int(sid)] = sd["y_true"].astype(str).to_numpy(dtype=object)
            by_subject_pred[int(sid)] = sd["y_pred"].astype(str).to_numpy(dtype=object)

        for _ in range(int(n_boot)):
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            yt = np.concatenate([by_subject_true[int(s)] for s in draw], axis=0)
            yp = np.concatenate([by_subject_pred[int(s)] for s in draw], axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
                _, _, f1, _ = precision_recall_fscore_support(
                    yt, yp, labels=labs, average=None, zero_division=0,
                )
            for i, c in enumerate(classes):
                boot_f1[c].append(float(f1[i]))

        task_ci: Dict[str, Tuple[float, float]] = {}
        for c in classes:
            vals = np.asarray(boot_f1[c], dtype=float)
            if vals.size == 0:
                task_ci[c] = (float("nan"), float("nan"))
            else:
                task_ci[c] = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
        result[task] = task_ci
        LOGGER.info("Bootstrap per-class CI done | task=%s label=%s n_boot=%d", task, label_mode, n_boot)

    return result


def enrich_rows_with_per_class_ci(
    rows: List[Dict[str, object]],
    ci_map: Dict[str, Dict[str, Tuple[float, float]]],
    classes: Sequence[str],
) -> List[Dict[str, object]]:
    for rr in rows:
        task = str(rr.get("task", ""))
        task_ci = ci_map.get(task, {})
        for c in classes:
            lo, hi = task_ci.get(c, (float("nan"), float("nan")))
            rr[f"f1_{c}_ci_low"] = lo
            rr[f"f1_{c}_ci_high"] = hi
    return rows


def enrich_rows_with_overall_ci(
    rows: List[Dict[str, object]],
    result_dir: Path,
    label_mode: str,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> List[Dict[str, object]]:
    for rr in rows:
        if np.isfinite(float(rr.get("acc_ci_low", float("nan")))):
            continue
        task = str(rr.get("task", ""))
        pred = load_trial_predictions(result_dir, task=str(task), label_mode=str(label_mode))
        pred = pred.drop(columns=["task", "label_mode"], errors="ignore") if not pred.empty else pred
        if pred.empty:
            continue
        ci = bootstrap_ci_pooled_from_trial_preds(
            pred,
            task=task,
            n_boot=n_boot,
            seed=int(seed) + stable_int_hash(f"{task}_{label_mode}", mod=997),
        )
        rr.update(ci)
    return rows


def write_fnirs_decoding_manuscript_tables(result_dir: Path, *, n_boot: int = 2000, seed: int = 42) -> None:
    tables_dir = _tables_dir(result_dir)
    data_dir = _data_dir(result_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rows_v3 = collect_valence3_main_rows(result_dir)
    if rows_v3:
        enrich_rows_with_overall_ci(rows_v3, result_dir, "valence3", n_boot=n_boot, seed=seed)
        ci_v3 = bootstrap_per_class_f1_ci(result_dir, "valence3", n_boot=n_boot, seed=seed)
        enrich_rows_with_per_class_ci(rows_v3, ci_v3, list(VALENCE3_CLASSES))

        pd.DataFrame(rows_v3).to_csv(data_dir / "table_fnirs_decoding_valence3_main_breakdown.csv", index=False)
        LOGGER.info("Wrote: %s", data_dir / "table_fnirs_decoding_valence3_main_breakdown.csv")

        write_decoding_summary_tex(
            tables_dir / "table_fnirs_decoding_valence3_summary_main.tex",
            rows_v3,
            label_mode="valence3",
        )
        LOGGER.info("Wrote: %s", tables_dir / "table_fnirs_decoding_valence3_summary_main.tex")

        write_decoding_perclass_tex(
            tables_dir / "table_fnirs_decoding_valence3_perclass_main.tex",
            rows_v3,
            label_mode="valence3",
        )
        LOGGER.info("Wrote: %s", tables_dir / "table_fnirs_decoding_valence3_perclass_main.tex")

        plot_fnirs_decoding_grouped_bars(rows_v3, label_mode="valence3", result_dir=result_dir)

    rows_e9 = collect_emotion9_appendix_rows(result_dir)
    if rows_e9:
        enrich_rows_with_overall_ci(rows_e9, result_dir, "emotion9", n_boot=n_boot, seed=seed)
        ci_e9 = bootstrap_per_class_f1_ci(result_dir, "emotion9", n_boot=n_boot, seed=seed)
        enrich_rows_with_per_class_ci(rows_e9, ci_e9, list(EMOTIONS_9))

        pd.DataFrame(rows_e9).to_csv(data_dir / "table_fnirs_decoding_emotion9_appendix_multiclass.csv", index=False)
        LOGGER.info("Wrote: %s", data_dir / "table_fnirs_decoding_emotion9_appendix_multiclass.csv")

        write_decoding_summary_tex(
            tables_dir / "table_fnirs_decoding_emotion9_summary_appendix.tex",
            rows_e9,
            label_mode="emotion9",
        )
        LOGGER.info("Wrote: %s", tables_dir / "table_fnirs_decoding_emotion9_summary_appendix.tex")

        write_decoding_perclass_tex(
            tables_dir / "table_fnirs_decoding_emotion9_perclass_appendix.tex",
            rows_e9,
            label_mode="emotion9",
        )
        LOGGER.info("Wrote: %s", tables_dir / "table_fnirs_decoding_emotion9_perclass_appendix.tex")

        plot_fnirs_decoding_grouped_bars(rows_e9, label_mode="emotion9", result_dir=result_dir)


def write_root_aggregates(
    result_dir: Path,
    *,
    args,
    n_perm: int = 10_000,
) -> None:
    _data_dir(result_dir).mkdir(parents=True, exist_ok=True)

    write_fnirs_decoding_manuscript_tables(
        result_dir,
        n_boot=int(getattr(args, "bootstrap_n_manuscript", BOOTSTRAP_N_MANUSCRIPT)),
        seed=int(args.random_state),
    )
    write_fnirs_decoding_stats(
        result_dir,
        n_perm=int(n_perm),
        seed=int(args.random_state),
    )
