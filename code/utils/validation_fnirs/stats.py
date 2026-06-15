"""Manuscript-aligned inferential tests for fNIRS decoding validation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_recall_fscore_support

from utils.validation_fnirs.features import VALENCE3_CLASSES
from utils.validation_fnirs.task_wise_loso import (
    load_trial_predictions,
    multiclass_per_class_f1_map,
)
from utils.validation_style import EMOTIONS_9, TASKS

logger = logging.getLogger("validation_fnirs_decoding.stats")

CHANCE_MACRO: Dict[str, float] = {"valence3": 1.0 / 3.0, "emotion9": 1.0 / 9.0}
DEFAULT_N_PERM_FULL = 10_000
DEFAULT_N_PERM_QUICK = 200


def _permutation_p_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    labels_list = [str(x) for x in labels]
    obs = float(
        f1_score(y_true, y_pred, average="macro", labels=labels_list, zero_division=0)
    )
    count = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(y_true)
        null = float(
            f1_score(perm, y_pred, average="macro", labels=labels_list, zero_division=0)
        )
        if np.isfinite(null) and null >= obs:
            count += 1
    p = (count + 1) / (int(n_perm) + 1)
    return obs, float(p)


def _permutation_p_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_label: str,
    labels: Sequence[str],
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    labs = np.asarray([str(x) for x in labels], dtype=object)
    cls = str(class_label)
    idx = list(labs).index(cls)
    yt = y_true.astype(str)
    yp = y_pred.astype(str)
    _, _, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=labs, average=None, zero_division=0
    )
    obs = float(f1[idx])
    count = 0
    for _ in range(int(n_perm)):
        perm = rng.permutation(y_true)
        _, _, f1p, _ = precision_recall_fscore_support(
            perm.astype(str), yp, labels=labs, average=None, zero_division=0
        )
        if float(f1p[idx]) >= obs:
            count += 1
    p = (count + 1) / (int(n_perm) + 1)
    return obs, float(p)


def _labels_for_mode(label_mode: str) -> List[str]:
    if label_mode == "valence3":
        return list(VALENCE3_CLASSES)
    if label_mode == "emotion9":
        return list(EMOTIONS_9)
    raise ValueError(label_mode)


def build_decoding_permutation_tests(
    result_dir: Path,
    *,
    label_mode: str,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    """Trial-level label permutation vs chance for pooled macro-F1 and per-class F1."""
    labels = _labels_for_mode(label_mode)
    chance = CHANCE_MACRO[label_mode]
    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, object]] = []

    for task in TASKS:
        pred = load_trial_predictions(result_dir, task=str(task), label_mode=label_mode)
        pred = pred.drop(columns=["task", "label_mode"], errors="ignore")
        if pred.empty:
            continue
        yt = pred["y_true"].to_numpy(dtype=object)
        yp = pred["y_pred"].to_numpy(dtype=object)
        macro_f1, p_macro = _permutation_p_macro_f1(yt, yp, labels, n_perm, rng)
        rows.append(
            {
                "label_mode": label_mode,
                "task": task,
                "metric": "macro_f1",
                "class": "all",
                "n_trials": int(len(pred)),
                "value": macro_f1,
                "chance": chance,
                "p_value": p_macro,
                "n_perm": int(n_perm),
                "method": "trial_label_permutation",
            }
        )
        f1_map = multiclass_per_class_f1_map(yt, yp, labels=labels)
        for cls in labels:
            f1_cls, p_cls = _permutation_p_class_f1(yt, yp, cls, labels, n_perm, rng)
            rows.append(
                {
                    "label_mode": label_mode,
                    "task": task,
                    "metric": "class_f1",
                    "class": cls,
                    "n_trials": int(len(pred)),
                    "value": f1_cls,
                    "chance": 1.0 / len(labels),
                    "p_value": p_cls,
                    "n_perm": int(n_perm),
                    "method": "trial_label_permutation",
                }
            )
            if cls in f1_map and not np.isfinite(f1_cls):
                rows[-1]["value"] = float(f1_map[cls])

    return pd.DataFrame(rows)


def build_friedman_task_macro_f1_tests(
    result_dir: Path,
    *,
    label_mode: str,
) -> pd.DataFrame:
    """Friedman test on subject-level macro-F1 across the three tasks (LOSO folds)."""
    by_task: Dict[str, pd.Series] = {}
    for task in TASKS:
        fold_path = result_dir / "data" / f"{task}__{label_mode}" / "table_mil_loso_fold_summary.csv"
        if not fold_path.is_file():
            continue
        folds = pd.read_csv(fold_path)
        if "test_subject" not in folds.columns or "macro_f1_present" not in folds.columns:
            continue
        use = folds.dropna(subset=["test_subject", "macro_f1_present"]).copy()
        use = use[~use.get("skipped", False).astype(bool)] if "skipped" in use.columns else use
        by_task[task] = use.set_index("test_subject")["macro_f1_present"].astype(float)

    if len(by_task) < 3:
        return pd.DataFrame()

    common = None
    for task in TASKS:
        if task not in by_task:
            return pd.DataFrame()
        idx = set(by_task[task].index.astype(int).tolist())
        common = idx if common is None else common & idx
    if not common:
        return pd.DataFrame()

    ordered_subs = sorted(common)
    samples = [by_task[t].reindex(ordered_subs).to_numpy(dtype=float) for t in TASKS]
    if any(np.isnan(s).any() for s in samples):
        return pd.DataFrame()

    chi2, p = stats.friedmanchisquare(*samples)
    task_means = {t: float(np.mean(by_task[t].reindex(ordered_subs))) for t in TASKS}
    return pd.DataFrame(
        [
            {
                "label_mode": label_mode,
                "test": "friedman",
                "n_subjects": len(ordered_subs),
                "chi2": float(chi2),
                "p": float(p),
                **{f"mean_macro_f1_{t}": task_means[t] for t in TASKS},
            }
        ]
    )


def write_fnirs_decoding_stats(
    result_dir: Path,
    *,
    n_perm: int,
    seed: int,
) -> None:
    """Write permutation and Friedman summaries referenced in Section IV."""
    data_dir = result_dir / "data"
    tables_dir = result_dir / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    perm_frames: List[pd.DataFrame] = []
    friedman_frames: List[pd.DataFrame] = []
    for label_mode in ("valence3", "emotion9"):
        perm_frames.append(
            build_decoding_permutation_tests(
                result_dir, label_mode=label_mode, n_perm=n_perm, seed=seed + (0 if label_mode == "valence3" else 99)
            )
        )
        friedman_frames.append(build_friedman_task_macro_f1_tests(result_dir, label_mode=label_mode))

    perm_df = pd.concat([f for f in perm_frames if not f.empty], ignore_index=True)
    friedman_df = pd.concat([f for f in friedman_frames if not f.empty], ignore_index=True)

    perm_path = data_dir / "fnirs_decoding_permutation_summary.csv"
    perm_df.to_csv(perm_path, index=False)
    logger.info("Wrote %s (%d rows)", perm_path, len(perm_df))

    friedman_path = tables_dir / "table_fnirs_decoding_friedman_macro_f1.csv"
    friedman_df.to_csv(friedman_path, index=False)
    logger.info("Wrote %s (%d rows)", friedman_path, len(friedman_df))
