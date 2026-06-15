"""Task-wise leave-one-subject-out cross-validation for fNIRS decoding."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

from utils.validation_fnirs.features import TrialBag
from utils.validation_fnirs.model import fit_loso_fold
from utils.validation_style import TASKS

LOGGER = logging.getLogger("validation_fnirs_decoding.task_wise_loso")

BOOTSTRAP_N_MANUSCRIPT = 2_000

warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    category=UserWarning,
    module=r"sklearn\.metrics\._classification",
)


def _data_dir(result_dir: Path) -> Path:
    return result_dir / "data"


def _cell_out_dir(result_dir: Path, task: str, label_mode: str) -> Path:
    return _data_dir(result_dir) / f"{task}__{label_mode}"


def stable_int_hash(s: str, mod: int = 10_000) -> int:
    import zlib

    v = int(zlib.crc32(str(s).encode("utf-8")) & 0xFFFFFFFF)
    return int(v % int(mod))


def safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    if labels.size < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_metrics_str(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    labels_present = sorted(set(y_true.tolist()))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        macro_f1_present = float(f1_score(y_true, y_pred, average="macro", labels=labels_present, zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": safe_balanced_accuracy(y_true, y_pred),
        "macro_f1_present": macro_f1_present,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def multiclass_per_class_f1_map(y_true: np.ndarray, y_pred: np.ndarray, *, labels: Sequence[str]) -> Dict[str, float]:
    labs = [str(x) for x in labels]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true.astype(str),
            y_pred.astype(str),
            labels=np.asarray(labs, dtype=object),
            average=None,
            zero_division=0,
        )
    return {c: float(f1[i]) for i, c in enumerate(labs)}


def load_trial_predictions(result_dir: Path, task: str, label_mode: str) -> pd.DataFrame:
    p = _cell_out_dir(result_dir, task, label_mode) / "table_mil_loso_trial_predictions.csv"
    if not p.is_file():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d.insert(0, "task", task)
    d.insert(1, "label_mode", label_mode)
    return d


def bootstrap_ci_pooled_from_trial_preds(
    pred_df: pd.DataFrame,
    *,
    task: str,
    n_boot: int,
    seed: int,
    restrict_subjects: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    need = {"test_subject", "y_true", "y_pred"}
    if pred_df.empty or not need.issubset(set(pred_df.columns)):
        return {
            "acc_ci_low": float("nan"),
            "acc_ci_high": float("nan"),
            "macro_f1_ci_low": float("nan"),
            "macro_f1_ci_high": float("nan"),
        }

    d = pred_df.copy()
    d["test_subject"] = d["test_subject"].astype(int)
    d["y_true"] = d["y_true"].astype(str)
    d["y_pred"] = d["y_pred"].astype(str)
    if restrict_subjects is not None:
        keep = set(int(s) for s in restrict_subjects)
        d = d[d["test_subject"].astype(int).isin(keep)].copy()
    subjects = sorted(pd.unique(d["test_subject"]).astype(int).tolist())
    if len(subjects) < 2:
        return {
            "acc_ci_low": float("nan"),
            "acc_ci_high": float("nan"),
            "macro_f1_ci_low": float("nan"),
            "macro_f1_ci_high": float("nan"),
        }

    rng = np.random.RandomState(int(seed))
    task_name = str(task).strip()
    accs: List[float] = []
    mfs: List[float] = []

    if task_name == "Perception":
        by_subject_true: Dict[int, np.ndarray] = {}
        by_subject_pred: Dict[int, np.ndarray] = {}
        for sid, sd in d.groupby("test_subject"):
            by_subject_true[int(sid)] = sd["y_true"].to_numpy(dtype=object)
            by_subject_pred[int(sid)] = sd["y_pred"].to_numpy(dtype=object)
    else:
        y_all = d["y_true"].to_numpy(dtype=object)
        p_all = d["y_pred"].to_numpy(dtype=object)
        n_all = int(len(d))

    log_every = max(1, int(n_boot) // 5)
    for b in range(int(n_boot)):
        if task_name == "Perception":
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            yt = np.concatenate([by_subject_true[int(s)] for s in draw], axis=0)
            yp = np.concatenate([by_subject_pred[int(s)] for s in draw], axis=0)
        else:
            idx = rng.choice(np.arange(n_all, dtype=np.int64), size=n_all, replace=True)
            yt = y_all[idx]
            yp = p_all[idx]
        met = compute_metrics_str(yt, yp)
        accs.append(float(met["accuracy"]))
        mfs.append(float(met["macro_f1_present"]))
        if (b + 1) % int(log_every) == 0 or (b + 1) == int(n_boot):
            LOGGER.info("Bootstrap %d/%d", int(b + 1), int(n_boot))

    if not accs or not mfs:
        return {
            "acc_ci_low": float("nan"),
            "acc_ci_high": float("nan"),
            "macro_f1_ci_low": float("nan"),
            "macro_f1_ci_high": float("nan"),
        }

    return {
        "acc_ci_low": float(np.quantile(accs, 0.025)),
        "acc_ci_high": float(np.quantile(accs, 0.975)),
        "macro_f1_ci_low": float(np.quantile(mfs, 0.025)),
        "macro_f1_ci_high": float(np.quantile(mfs, 0.975)),
    }


def run_task_wise_loso(
    bags: List[TrialBag],
    classes: List[str],
    class_to_idx: Dict[str, int],
    label_mode: str,
    task: str,
    result_dir: Path,
    args,
    device,
) -> Dict[str, object]:
    out_sub = _cell_out_dir(result_dir, task, label_mode)
    out_sub.mkdir(parents=True, exist_ok=True)
    labels_str = np.asarray([b.label for b in bags], dtype=object)
    subjects = np.asarray([b.subject for b in bags], dtype=np.int64)
    source_tasks = np.asarray([b.source_task for b in bags], dtype=object)
    y = np.asarray([class_to_idx[str(v)] for v in labels_str], dtype=np.int64)
    X_bags = [b.features for b in bags]
    input_dim = int(X_bags[0].shape[1])
    n_classes = len(classes)

    uniq_subjects = sorted(int(s) for s in np.unique(subjects))
    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[Dict[str, object]] = []
    pooled_y_true: List[str] = []
    pooled_y_pred: List[str] = []

    cv = "loso"
    folds: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for rank, test_sid in enumerate(uniq_subjects, start=1):
        te_mask = subjects == test_sid
        tr_mask = ~te_mask
        te_idx = np.flatnonzero(te_mask)
        tr_idx = np.flatnonzero(tr_mask)
        if len(te_idx) == 0:
            continue
        folds.append((rank, tr_idx, te_idx))

    for fold_id, tr_idx, te_idx in folds:
        x_tr = [X_bags[i] for i in tr_idx]
        y_tr = y[tr_idx]
        x_te = [X_bags[i] for i in te_idx]
        y_te = y[te_idx]

        test_subjects = sorted(set(int(s) for s in subjects[te_idx].tolist()))
        fold_seed = int(args.random_state) + int(fold_id) * 101 + stable_int_hash(task, mod=997) + (0 if label_mode == "valence3" else 123)

        y_pred_te, aux, ok = fit_loso_fold(
            train_bags=x_tr,
            train_labels=y_tr,
            held_out_bags=x_te,
            held_out_labels=y_te,
            input_dim=input_dim,
            n_classes=n_classes,
            args=args,
            device=device,
            seed=int(fold_seed),
        )

        if not ok:
            LOGGER.warning(
                "CV skip | cv=%s | task=%s | label=%s | fold=%d | test_subjects=%s | train classes < 2",
                cv,
                task,
                label_mode,
                int(fold_id),
                ",".join([str(s) for s in test_subjects]),
            )
            test_subject_val = int(test_subjects[0]) if len(test_subjects) == 1 else None
            fold_rows.append(
                {
                    "cv": cv,
                    "fold": int(fold_id),
                    **({"test_subject": int(test_subject_val)} if test_subject_val is not None else {}),
                    "test_subjects": ",".join([str(s) for s in test_subjects]),
                    "n_train_trials": int(len(tr_idx)),
                    "n_test_trials": int(len(te_idx)),
                    "n_classes_train": int(np.unique(y_tr).size),
                    "n_classes_test": int(np.unique(y_te).size),
                    "macro_f1_present": float("nan"),
                    "balanced_acc": float("nan"),
                    "weighted_f1": float("nan"),
                    "accuracy": float("nan"),
                    "es_val_loss": float("nan"),
                    "best_epoch": float("nan"),
                    "stopped_epoch": float("nan"),
                    "train_s": float("nan"),
                    "skipped": True,
                }
            )
            continue

        hist = aux.get("val_loss_history", None)
        if isinstance(hist, list) and len(hist) > 0:
            curve_dir = out_sub / "diagnostics"
            curve_dir.mkdir(parents=True, exist_ok=True)
            curve_path = curve_dir / f"val_loss_curve_fold_{int(fold_id):02d}.csv"
            pd.DataFrame(
                {
                    "cv": [cv] * len(hist),
                    "task": [str(task)] * len(hist),
                    "label_mode": [str(label_mode)] * len(hist),
                    "fold": [int(fold_id)] * len(hist),
                    "epoch": list(range(1, len(hist) + 1)),
                    "val_loss": [float(x) for x in hist],
                }
            ).to_csv(curve_path, index=False)

        y_true_str = np.asarray([classes[int(i)] for i in y_te], dtype=object)
        y_pred_str = np.asarray([classes[int(i)] for i in y_pred_te], dtype=object)
        met = compute_metrics_str(y_true_str, y_pred_str)

        fold_rows.append(
            {
                "cv": cv,
                "fold": int(fold_id),
                "test_subject": int(test_subjects[0]),
                "test_subjects": ",".join([str(s) for s in test_subjects]),
                "n_train_trials": int(len(tr_idx)),
                "n_test_trials": int(len(te_idx)),
                "n_classes_train": int(np.unique(y_tr).size),
                "n_classes_test": int(np.unique(y_te).size),
                "macro_f1_present": met["macro_f1_present"],
                "balanced_acc": met["balanced_acc"],
                "weighted_f1": met["weighted_f1"],
                "accuracy": met["accuracy"],
                "es_val_loss": aux.get("es_val_loss", float("nan")),
                "best_epoch": aux.get("best_epoch", float("nan")),
                "stopped_epoch": aux.get("stopped_epoch", float("nan")),
                "train_s": aux.get("train_s", float("nan")),
                "skipped": False,
            }
        )
        pooled_y_true.extend(y_true_str.tolist())
        pooled_y_pred.extend(y_pred_str.tolist())

        for local_j, bag_idx in enumerate(te_idx):
            pred_rows.append(
                {
                    "cv": cv,
                    "fold": int(fold_id),
                    "test_subject": int(bags[bag_idx].subject),
                    "test_subjects": ",".join([str(s) for s in test_subjects]),
                    "bag_id": bags[bag_idx].bag_id,
                    "source_task": str(source_tasks[bag_idx]),
                    "y_true": str(y_true_str[local_j]),
                    "y_pred": str(y_pred_str[local_j]),
                }
            )

        LOGGER.info(
            "CV %d/%d | cv=%s | task=%s | label=%s | test_subjects=%s | test_trials=%d | macroF1=%.4f | BAcc=%.4f | best_epoch=%s | train=%.2fs",
            int(fold_id),
            int(len(folds)),
            cv,
            task,
            label_mode,
            ",".join([str(s) for s in test_subjects]),
            int(len(te_idx)),
            float(met["macro_f1_present"]),
            float(met["balanced_acc"]),
            str(aux.get("best_epoch", "NA")),
            float(aux.get("train_s", float("nan"))),
        )

    df_folds = pd.DataFrame(fold_rows)
    out_folds = out_sub / "table_mil_loso_fold_summary.csv"
    df_folds.to_csv(out_folds, index=False)
    LOGGER.info("Wrote: %s", out_folds)

    pooled_metrics: Dict[str, float] = {}
    if pooled_y_true:
        yt = np.asarray(pooled_y_true, dtype=object)
        yp = np.asarray(pooled_y_pred, dtype=object)
        pooled_metrics = compute_metrics_str(yt, yp)
        pooled_metrics["n_trials"] = float(len(pooled_y_true))
    else:
        pooled_metrics = {
            "n_trials": 0.0,
            "accuracy": float("nan"),
            "balanced_acc": float("nan"),
            "macro_f1_present": float("nan"),
            "weighted_f1": float("nan"),
        }

    pooled_row = {"task": task, "label_mode": label_mode, **pooled_metrics}
    pd.DataFrame([pooled_row]).to_csv(out_sub / "table_mil_loso_pooled_metrics.csv", index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_sub / "table_mil_loso_trial_predictions.csv", index=False)

    if "skipped" in df_folds.columns:
        use_fold = df_folds[df_folds["skipped"].astype(bool) == False].copy()
    else:
        use_fold = df_folds.copy()

    if len(use_fold) and "macro_f1_present" in use_fold.columns:
        mean_folds = float(use_fold["macro_f1_present"].mean())
    else:
        mean_folds = float("nan")
    return {
        "task": task,
        "label_mode": label_mode,
        "n_bags": len(bags),
        "n_subjects": len(uniq_subjects),
        "pooled": pooled_row,
        "mean_loso_macro_f1": float(mean_folds) if np.isfinite(mean_folds) else float("nan"),
    }

