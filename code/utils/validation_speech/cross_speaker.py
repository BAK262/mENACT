"""Cross-speaker speech decoding evaluation (SGKF within expressive mode)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from utils.validation_style import EMOTIONS_9
from utils.validation_speech.trials import BagSample, bag_label, subset_bags
from utils.validation_vocab import LABEL_MODES, TASK_DISPLAY

SVM_C = 1.0
SVM_GAMMA = 0.001

CROSS_SPEAKER_CONDITION_ORDER: Tuple[str, ...] = (
    "cross_speaker_production",
    "cross_speaker_performance",
)

CROSS_SPEAKER_EXP_PREFIX_MAP: Dict[str, str] = {
    "cross_speaker_production": "exp2",
    "cross_speaker_performance": "exp3",
}

CROSS_SPEAKER_DISPLAY: Dict[str, str] = {
    "cross_speaker_production": TASK_DISPLAY["production"],
    "cross_speaker_performance": TASK_DISPLAY["performance"],
}


def labels_for_mode(label_mode: str) -> List[str]:
    if label_mode == "valence3":
        return ["negative", "neutral", "positive"]
    if label_mode == "emotion9":
        return list(EMOTIONS_9)
    raise ValueError(f"Unsupported label mode: {label_mode}")


def bags_to_matrix_and_labels(
    bags: Sequence[BagSample],
    label_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not bags:
        return np.asarray([]), np.asarray([]), np.asarray([])
    x = np.stack([b.instances.mean(axis=0) for b in bags], axis=0).astype(np.float64)
    y = np.asarray([bag_label(b, label_mode) for b in bags], dtype=object)
    groups = np.asarray([int(b.subject_id) for b in bags], dtype=np.int64)
    return x, y, groups


def build_subject_stratify_labels(
    bags: Sequence[BagSample],
    *,
    exp_prefix: str,
    subject_ids: Sequence[int],
    label_mode: str,
) -> np.ndarray:
    subject_set = {int(s) for s in subject_ids}
    labels: List[str] = []
    for sid in sorted(subject_set):
        subject_bags = [b for b in bags if b.exp_prefix == exp_prefix and int(b.subject_id) == int(sid)]
        if not subject_bags:
            raise RuntimeError(f"No bags found for exp_prefix={exp_prefix!r}, subject_id={sid}")
        hard = np.asarray([str(bag_label(b, label_mode)) for b in subject_bags], dtype=object)
        values, counts = np.unique(hard, return_counts=True)
        labels.append(str(values[int(np.argmax(counts))]))
    le = LabelEncoder()
    return le.fit_transform(np.asarray(labels, dtype=object)).astype(np.int64)


def iter_stratified_group_kfold_subject_folds(
    *,
    subject_ids: Sequence[int],
    subject_labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    subjects = np.asarray(sorted({int(s) for s in subject_ids}), dtype=np.int64)
    if subjects.size < 2:
        return []
    labels = np.asarray(subject_labels, dtype=np.int64)
    if int(labels.size) != int(subjects.size):
        raise RuntimeError(f"subject_labels size mismatch: labels={labels.size} subjects={subjects.size}")
    max_splits = max(2, min(int(n_splits), int(subjects.size)))
    x_dummy = np.zeros((int(subjects.size), 1), dtype=np.float32)
    for split_n in range(max_splits, 1, -1):
        try:
            sgkf = StratifiedGroupKFold(n_splits=split_n, shuffle=True, random_state=int(random_state))
            folds: List[Tuple[np.ndarray, np.ndarray]] = []
            for tr_idx, te_idx in sgkf.split(x_dummy, labels, groups=subjects):
                folds.append((subjects[np.asarray(tr_idx, dtype=np.int64)], subjects[np.asarray(te_idx, dtype=np.int64)]))
            if folds:
                return folds
        except ValueError:
            continue
    for split_n in range(max_splits, 1, -1):
        try:
            gkf = GroupKFold(n_splits=split_n)
            y_dummy = np.zeros(int(subjects.size), dtype=np.int64)
            folds = []
            for tr_idx, te_idx in gkf.split(x_dummy, y_dummy, groups=subjects):
                folds.append((subjects[np.asarray(tr_idx, dtype=np.int64)], subjects[np.asarray(te_idx, dtype=np.int64)]))
            if folds:
                return folds
        except ValueError:
            continue
    raise RuntimeError("Unable to build subject folds with StratifiedGroupKFold or GroupKFold fallback.")


def _compute_class_weights(y_train_encoded: np.ndarray, n_classes: int) -> Dict[int, float]:
    labels = np.asarray(y_train_encoded, dtype=np.int64)
    labels = labels[(labels >= 0) & (labels < int(n_classes))]
    if labels.size == 0:
        return {i: 1.0 for i in range(int(n_classes))}
    counts = np.bincount(labels, minlength=int(n_classes)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / inv.sum() * float(n_classes)
    return {i: float(weights[i]) for i in range(int(n_classes))}


def fit_predict_rbf_vectors(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_test: np.ndarray,
    random_state: int,
    c_value: float,
    gamma: float | str = SVM_GAMMA,
) -> np.ndarray:
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    if np.unique(y_train).size < 2:
        return np.asarray([])

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    class_weight = _compute_class_weights(y_train, n_classes=int(np.max(y_train)) + 1)
    clf = SVC(
        kernel="rbf",
        C=float(c_value),
        gamma=gamma,
        class_weight=class_weight,
        cache_size=2000,
        random_state=int(random_state) % (2**31 - 1),
    )
    clf.fit(x_train_s, y_train)
    pred_idx = clf.predict(x_test_s)
    return le.inverse_transform(pred_idx.astype(np.int64))


def predict_train_fixed_c_then_test(
    *,
    train_bags: Sequence[BagSample],
    test_bags: Sequence[BagSample],
    label_mode: str,
    random_state: int,
    c_value: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if not train_bags or not test_bags:
        return np.asarray([]), np.asarray([]), float(c_value)
    x_train, y_train, _ = bags_to_matrix_and_labels(train_bags, label_mode)
    x_test, y_test, _ = bags_to_matrix_and_labels(test_bags, label_mode)
    if x_train.size == 0 or x_test.size == 0:
        return np.asarray([]), np.asarray([]), float(c_value)
    y_pred = fit_predict_rbf_vectors(
        x_train=x_train,
        y_train_raw=y_train,
        x_test=x_test,
        random_state=random_state + 777,
        c_value=float(c_value),
    )
    return y_test, y_pred, float(c_value)


def make_prediction_rows(
    test_bags: Sequence[BagSample],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mode: str,
    test_id: str,
    train_desc: str,
    c_value: float,
    model_name: str,
    fold_id: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if y_true.size == 0 or y_pred.size == 0:
        return rows
    for bag, yt, yp in zip(test_bags, y_true.tolist(), y_pred.tolist()):
        rows.append(
            {
                "model": model_name,
                "label_mode": label_mode,
                "test_id": test_id,
                "train_desc": train_desc,
                "fold_id": int(fold_id),
                "trial_key": bag.trial_key,
                "subject_id": int(bag.subject_id),
                "subject_group": bag.subject_group,
                "task_key": bag.task_key,
                "task": bag.task,
                "y_true": str(yt),
                "y_pred": str(yp),
                "best_c": float(c_value),
            }
        )
    return rows


def confusion_matrix_full(y_true: np.ndarray, y_pred: np.ndarray, labels_full: Sequence[str]) -> np.ndarray:
    if y_true.size == 0 or y_pred.size == 0:
        return np.zeros((len(labels_full), len(labels_full)), dtype=np.float64)
    return confusion_matrix(y_true, y_pred, labels=list(labels_full)).astype(np.float64, copy=False)


def classification_metrics_from_confusion(conf: np.ndarray) -> Dict[str, float]:
    if conf.size == 0 or float(conf.sum()) <= 0.0:
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "macro_f1_present": float("nan"),
        }
    true_counts = conf.sum(axis=1)
    pred_counts = conf.sum(axis=0)
    tp = np.diag(conf)
    total = float(conf.sum())
    accuracy = float(tp.sum() / total) if total > 0.0 else float("nan")

    present_mask = true_counts > 0.0

    fp = pred_counts - tp
    fn = true_counts - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.zeros_like(tp, dtype=np.float64)
    valid = denom > 0.0
    f1[valid] = (2.0 * tp[valid]) / denom[valid]
    macro_f1 = float(f1.mean()) if f1.size > 0 else float("nan")
    macro_f1_present = float(f1[present_mask].mean()) if np.any(present_mask) else float("nan")
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
    }


def pooled_metrics_pair(
    y_true: np.ndarray, y_pred: np.ndarray, labels_full: Sequence[str]
) -> Tuple[float, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan"), float("nan")
    conf = confusion_matrix_full(y_true, y_pred, labels_full=labels_full)
    m = classification_metrics_from_confusion(conf)
    return float(m["macro_f1_present"]), float(m["accuracy"])


def fmt_metric_ci(value: float, ci_low: float, ci_high: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return f"{value:.3f} [N/A]"
    return f"{value:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def bootstrap_ci_by_trial(
    df: pd.DataFrame,
    labels_full: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    nan_block: Dict[str, object] = {
        "macro_f1": float("nan"),
        "macro_f1_ci_low": float("nan"),
        "macro_f1_ci_high": float("nan"),
        "macro_f1_ci_text": "N/A",
        "accuracy": float("nan"),
        "accuracy_ci_low": float("nan"),
        "accuracy_ci_high": float("nan"),
        "accuracy_ci_text": "N/A",
    }
    if df.empty:
        return nan_block

    y_true_all = df["y_true"].to_numpy(dtype=object)
    y_pred_all = df["y_pred"].to_numpy(dtype=object)
    pooled_mf1, pooled_acc = pooled_metrics_pair(y_true_all, y_pred_all, labels_full)
    n = int(len(df))
    rng = np.random.default_rng(int(seed))

    mf1_samples: List[float] = []
    acc_samples: List[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(low=0, high=n, size=n, endpoint=False)
        m_b, a_b = pooled_metrics_pair(y_true_all[idx], y_pred_all[idx], labels_full)
        if np.isfinite(m_b):
            mf1_samples.append(float(m_b))
        if np.isfinite(a_b):
            acc_samples.append(float(a_b))

    def _ci(samples: List[float], pooled: float) -> Tuple[float, float, str]:
        if not samples:
            return float("nan"), float("nan"), fmt_metric_ci(pooled, float("nan"), float("nan"))
        lo, hi = np.quantile(np.asarray(samples), [0.025, 0.975]).tolist()
        return float(lo), float(hi), fmt_metric_ci(pooled, float(lo), float(hi))

    m_lo, m_hi, m_txt = _ci(mf1_samples, pooled_mf1)
    a_lo, a_hi, a_txt = _ci(acc_samples, pooled_acc)

    return {
        **nan_block,
        "macro_f1": pooled_mf1,
        "macro_f1_ci_low": m_lo,
        "macro_f1_ci_high": m_hi,
        "macro_f1_ci_text": m_txt,
        "accuracy": pooled_acc,
        "accuracy_ci_low": a_lo,
        "accuracy_ci_high": a_hi,
        "accuracy_ci_text": a_txt,
    }


def evaluate_cross_speaker(
    bags: Sequence[BagSample],
    label_mode: str,
    random_state: int,
    model_name: str,
) -> pd.DataFrame:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    pred_rows: List[Dict[str, object]] = []

    for test_id, exp_prefix in zip(CROSS_SPEAKER_CONDITION_ORDER, CROSS_SPEAKER_EXP_PREFIX_MAP.values()):
        mode_bags = subset_bags(bags, exp_prefix=exp_prefix)
        subjects = sorted({int(b.subject_id) for b in mode_bags})
        if len(subjects) < 2:
            continue
        subj_labels = build_subject_stratify_labels(
            mode_bags,
            exp_prefix=exp_prefix,
            subject_ids=subjects,
            label_mode=label_mode,
        )
        outer_folds = iter_stratified_group_kfold_subject_folds(
            subject_ids=subjects,
            subject_labels=subj_labels,
            n_splits=5,
            random_state=42,
        )
        for outer_idx, (outer_train_subj, outer_test_subj) in enumerate(outer_folds, start=1):
            tr_set = set(int(v) for v in outer_train_subj.tolist())
            te_set = set(int(v) for v in outer_test_subj.tolist())
            outer_train_bags = [b for b in mode_bags if int(b.subject_id) in tr_set]
            outer_test_bags = [b for b in mode_bags if int(b.subject_id) in te_set]
            if not outer_train_bags or not outer_test_bags:
                continue

            y_true, y_pred, best_c = predict_train_fixed_c_then_test(
                train_bags=outer_train_bags,
                test_bags=outer_test_bags,
                label_mode=label_mode,
                random_state=random_state + outer_idx,
                c_value=float(SVM_C),
            )
            pred_rows.extend(
                make_prediction_rows(
                    test_bags=outer_test_bags,
                    y_true=y_true,
                    y_pred=y_pred,
                    label_mode=label_mode,
                    test_id=test_id,
                    train_desc=f"cross_speaker_SGKF_{exp_prefix}",
                    c_value=float(best_c),
                    model_name=model_name,
                    fold_id=int(outer_idx),
                )
            )

    return pd.DataFrame(pred_rows)


def summarize_cross_speaker(
    pred_df: pd.DataFrame,
    label_mode: str,
    model_name: str,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    labels_full = labels_for_mode(label_mode)
    empty_metrics = bootstrap_ci_by_trial(pd.DataFrame(), labels_full, n_bootstrap=n_bootstrap, seed=0)
    rows: List[Dict[str, object]] = []
    for idx, test_id in enumerate(CROSS_SPEAKER_CONDITION_ORDER):
        cond_df = pred_df[pred_df["test_id"] == test_id].copy()
        if cond_df.empty:
            row: Dict[str, object] = {
                "model": model_name,
                "label_mode": label_mode,
                "test_id": test_id,
                "n_subjects": 0,
                "n_trials": 0,
            }
            row.update({k: v for k, v in empty_metrics.items()})
            rows.append(row)
            continue
        m = bootstrap_ci_by_trial(
            cond_df,
            labels_full,
            n_bootstrap=n_bootstrap,
            seed=random_state + idx + 3000,
        )
        rows.append(
            {
                "model": model_name,
                "label_mode": label_mode,
                "test_id": test_id,
                "n_subjects": int(cond_df["subject_id"].nunique()),
                "n_trials": int(len(cond_df)),
                **m,
            }
        )
    return pd.DataFrame(rows)
