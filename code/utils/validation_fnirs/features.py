"""Window-level fNIRS features and trial bags for validation_fnirs_decoding."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.validation_fnirs.common import (
    discover_subject_dirs,
    load_trialsdc_mat,
    parse_measure_label,
)
from utils.validation_style import EMOTIONS_9, TASKS, include_task_in_three_task_analysis, normalize_task_name
from utils.validation_vocab import LABEL_MODES

LOGGER = logging.getLogger("validation_fnirs_decoding.features")

FS = 11.0
NEGATIVE_4 = {"anger", "disgust", "fear", "sadness"}
POSITIVE_4 = {"amusement", "inspiration", "joy", "tenderness"}
VALENCE3_CLASSES = ("negative", "neutral", "positive")
STAT_COLS = ["mean", "median", "std", "peak", "latency", "slope"]


def map_emotion_to_valence_3(emotion: str) -> str:
    e = str(emotion).strip().lower()
    if e == "neutral":
        return "neutral"
    if e in NEGATIVE_4:
        return "negative"
    if e in POSITIVE_4:
        return "positive"
    raise ValueError(f"Unexpected emotion label for 3-class mapping: {emotion!r}")


def _data_dir(result_dir: Path) -> Path:
    return result_dir / "data"


def _cell_peak_latency(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    mask = np.isfinite(x)
    if int(mask.sum()) == 0:
        return float("nan"), float("nan")
    xx = x[mask].astype(np.float64, copy=False)
    dmax = float(np.max(xx))
    dmin = float(np.min(xx))
    if abs(dmax) >= abs(dmin):
        peak = dmax
        idx = int(np.argmax(xx))
    else:
        peak = dmin
        idx = int(np.argmin(xx))
    return peak, float(idx)


def _linear_slope(x: np.ndarray) -> float:
    mask = np.isfinite(x)
    if int(mask.sum()) < 2:
        return float("nan")
    y = x[mask].astype(np.float64, copy=False)
    t = np.arange(y.size, dtype=np.float64)
    t = t - t.mean()
    y = y - y.mean()
    den = float(np.sum(t * t))
    if den <= 0:
        return 0.0
    return float(np.sum(t * y) / den)


def _summarize_window(x: np.ndarray, fs: float, window_seconds: float) -> Dict[str, float]:
    mask = np.isfinite(x)
    if int(mask.sum()) == 0:
        return {c: float("nan") for c in STAT_COLS}
    xx = x[mask].astype(np.float64, copy=False)
    peak, lat = _cell_peak_latency(xx)
    return {
        "mean": float(np.mean(xx)),
        "median": float(np.median(xx)),
        "std": float(np.std(xx, ddof=1)) if xx.size > 1 else 0.0,
        "peak": float(peak),
        "latency": float(lat),
        "slope": float(_linear_slope(xx)),
    }


@dataclass
class WindowMeta:
    subject: int
    experiment: str
    task: str
    target_emotion: str
    trial_index: int
    window_index: int


@dataclass(frozen=True)
class TrialBag:
    bag_id: str
    subject: int
    source_task: str
    label: str
    features: np.ndarray


def _keep_mat_trial(raw_task: str) -> bool:
    return include_task_in_three_task_analysis(normalize_task_name(raw_task))


def _window_slices(n_time: int, window_n: int, stride_n: int) -> List[Tuple[int, int]]:
    if n_time <= 0 or window_n <= 1 or stride_n <= 0:
        return []
    slices: List[Tuple[int, int]] = []
    start = 0
    while start + window_n <= n_time:
        slices.append((start, start + window_n))
        start += stride_n
    return slices


def _iter_trials_for_subject(subject_dir: Path, fs: float, fixation_seconds: float) -> List[Tuple[WindowMeta, List[str], np.ndarray]]:
    out: List[Tuple[WindowMeta, List[str], np.ndarray]] = []
    sid = int(subject_dir.name)
    fix_n = int(round(fixation_seconds * fs))
    for exp_no in (1, 2, 3):
        mat_path = subject_dir / f"trialsDC_exp{exp_no}.mat"
        if not mat_path.is_file():
            continue
        td = load_trialsdc_mat(mat_path)
        ti = td.trialinfo
        for t_idx, trial in enumerate(td.trials):
            trial_use = trial[:, fix_n:].astype(np.float32, copy=False) if trial.shape[1] > fix_n else trial.astype(np.float32, copy=False)
            raw_task = str(ti[t_idx, 0])
            if not _keep_mat_trial(raw_task):
                continue
            task = normalize_task_name(raw_task)
            if exp_no == 1:
                target = str(ti[t_idx, 2])
            else:
                target = str(ti[t_idx, 1])
            meta = WindowMeta(
                subject=sid,
                experiment=f"exp{exp_no}",
                task=task,
                target_emotion=target,
                trial_index=int(t_idx + 1),
                window_index=-1,
            )
            out.append((meta, td.labels, trial_use))
    return out


def _feature_cache_dir(result_dir: Path) -> Path:
    return _data_dir(result_dir) / "features"


def _per_task_long_path(result_dir: Path, task: str) -> Path:
    return _feature_cache_dir(result_dir) / f"feature_by_window_long__{str(task)}.csv"


def _legacy_long_path(result_dir: Path) -> Path:
    return _data_dir(result_dir) / "feature_by_window_long.csv"


def _clean_reused_long_df(long_df: pd.DataFrame, window_n: int) -> pd.DataFrame:
    d = long_df.copy()
    if "last10s" in d.columns:
        d = d.drop(columns=["last10s"]).copy()
    if "window_start_sample" in d.columns and "window_end_sample" in d.columns:
        wlen = pd.to_numeric(d["window_end_sample"], errors="coerce") - pd.to_numeric(d["window_start_sample"], errors="coerce")
        keep = (wlen >= float(window_n)) & np.isfinite(wlen.to_numpy(dtype=float))
        dropped = int((~keep).sum())
        if dropped > 0:
            d = d.loc[keep].copy()
            LOGGER.warning(
                "Dropped %d window rows with window_len < %d while reusing long features (full-window policy).",
                dropped,
                int(window_n),
            )
    return d


def _apply_subject_limit(long_df: pd.DataFrame, subject_limit: Optional[int]) -> pd.DataFrame:
    if subject_limit is None:
        return long_df
    subs = sorted(pd.unique(long_df["subject"].astype(int)).tolist())[: int(subject_limit)]
    d = long_df[long_df["subject"].astype(int).isin(subs)].copy()
    LOGGER.info("Applied subject_limit=%d -> subjects=%s rows=%d", int(subject_limit), subs, len(d))
    return d


def _build_long_feature_table_for_task(
    preproc_root: Path,
    subject_limit: Optional[int],
    fs: float,
    fixation_seconds: float,
    window_seconds: float,
    stride_seconds: float,
    task: str,
) -> pd.DataFrame:
    subs = discover_subject_dirs(preproc_root, subject_limit=subject_limit)
    rows: List[Dict[str, object]] = []
    window_n = int(round(window_seconds * fs))
    stride_n = int(round(stride_seconds * fs))
    for s_i, sd in enumerate(subs, start=1):
        trials = _iter_trials_for_subject(sd, fs=fs, fixation_seconds=fixation_seconds)
        for trial_meta, labels, trial_mat in trials:
            if str(trial_meta.task) != str(task):
                continue
            slices = _window_slices(trial_mat.shape[1], window_n=window_n, stride_n=stride_n)
            for win_idx, (st, ed) in enumerate(slices, start=1):
                trial_win = trial_mat[:, st:ed]
                for m_i, label in enumerate(labels):
                    ch_id, hb = parse_measure_label(label)
                    if hb not in {"HbO", "HbR"}:
                        continue
                    vec = np.asarray(trial_win[m_i, :], dtype=np.float32).reshape(-1)
                    feats = _summarize_window(vec, fs=fs, window_seconds=window_seconds)
                    rows.append(
                        {
                            "subject": trial_meta.subject,
                            "experiment": trial_meta.experiment,
                            "task": trial_meta.task,
                            "targetEmotion": trial_meta.target_emotion,
                            "trial_index": trial_meta.trial_index,
                            "window_index": win_idx,
                            "window_start_sample": st,
                            "window_end_sample": ed,
                            "channel": ch_id,
                            "hb_type": hb,
                            **feats,
                        }
                    )
        if s_i % 5 == 0:
            LOGGER.info("Built window features for task=%s on %d/%d subjects", str(task), s_i, len(subs))
    df = pd.DataFrame(rows)
    if df.empty:
        LOGGER.warning("No window-level feature rows built for task=%s.", str(task))
    return df


def _migrate_legacy_long_to_per_task(result_dir: Path, tasks_to_write: Sequence[str]) -> None:
    legacy = _legacy_long_path(result_dir)
    if not legacy.is_file():
        return
    features_dir = _feature_cache_dir(result_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    pending = [t for t in tasks_to_write if not _per_task_long_path(result_dir, t).is_file()]
    if not pending:
        return
    t0 = time.perf_counter()
    long_df = pd.read_csv(legacy)
    LOGGER.info("Migrating legacy long cache %s -> per-task files=%s", legacy, pending)
    for task in pending:
        out = _per_task_long_path(result_dir, task)
        sub = long_df[long_df["task"].astype(str) == str(task)].copy()
        sub.to_csv(out, index=False)
        LOGGER.info("Wrote migrated per-task cache: %s (rows=%d)", out, len(sub))
    LOGGER.info("Legacy cache migration done in %.2fs", time.perf_counter() - t0)


def ensure_per_task_feature_caches(
    *,
    result_dir: Path,
    preproc_root: Path,
    tasks_to_write: Sequence[str],
    subject_limit: Optional[int],
    fs: float,
    fixation_seconds: float,
    window_seconds: float,
    stride_seconds: float,
    reuse_long: bool,
) -> None:
    features_dir = _feature_cache_dir(result_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_long_to_per_task(result_dir=result_dir, tasks_to_write=tasks_to_write)

    expected_full_n = len(discover_subject_dirs(preproc_root, subject_limit=None)) if subject_limit is None else None
    for task in tasks_to_write:
        cache_csv = _per_task_long_path(result_dir, task)
        need_build = (not bool(reuse_long)) or (not cache_csv.is_file())
        if (not need_build) and (expected_full_n is not None):
            try:
                have_sub = pd.read_csv(cache_csv, usecols=["subject"])
                n_have = int(pd.unique(have_sub["subject"].astype(int)).size)
            except Exception:
                n_have = 0
            if n_have < int(expected_full_n):
                LOGGER.warning(
                    "Per-task cache %s covers only %d/%d subjects (likely from sanity). Rebuilding.",
                    cache_csv,
                    n_have,
                    int(expected_full_n),
                )
                need_build = True
        if not need_build:
            continue
        t0 = time.perf_counter()
        d = _build_long_feature_table_for_task(
            preproc_root=preproc_root,
            subject_limit=subject_limit,
            fs=fs,
            fixation_seconds=fixation_seconds,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            task=task,
        )
        d.to_csv(cache_csv, index=False)
        LOGGER.info("Wrote per-task long features: %s (rows=%d, %.2fs)", cache_csv, len(d), time.perf_counter() - t0)


def load_long_for_task(
    *,
    result_dir: Path,
    task: str,
    subject_limit: Optional[int],
    fs: float,
    window_seconds: float,
) -> pd.DataFrame:
    cache_csv = _per_task_long_path(result_dir, task)
    if not cache_csv.is_file():
        raise FileNotFoundError(f"Missing per-task cache for task={task}: {cache_csv}")
    t0 = time.perf_counter()
    d = pd.read_csv(cache_csv)
    if "task" in d.columns:
        d["task"] = d["task"].map(lambda x: normalize_task_name(str(x)))
    LOGGER.info("Loaded per-task cache: %s (rows=%d, %.2fs)", cache_csv, len(d), time.perf_counter() - t0)
    window_n = int(round(float(window_seconds) * float(fs)))
    d = _clean_reused_long_df(d, window_n=window_n)
    return _apply_subject_limit(d, subject_limit=subject_limit)


def make_design_matrix(long_df: pd.DataFrame, hb_family: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    if hb_family == "HbO":
        d = long_df[long_df["hb_type"] == "HbO"].copy()
    elif hb_family == "HbR":
        d = long_df[long_df["hb_type"] == "HbR"].copy()
    elif hb_family == "HbO+HbR":
        d = long_df.copy()
    else:
        raise ValueError(f"Unknown hb_family: {hb_family}")
    d["window_key"] = (
        d["subject"].astype(str)
        + "|"
        + d["experiment"].astype(str)
        + "|"
        + d["task"].astype(str)
        + "|"
        + d["trial_index"].astype(str)
        + "|"
        + d["window_index"].astype(str)
    )
    wide = d.pivot_table(
        index=["window_key", "subject", "experiment", "task", "targetEmotion", "trial_index", "window_index"],
        columns=["hb_type", "channel"],
        values=STAT_COLS,
        aggfunc="mean",
    )
    wide.columns = [f"{stat}__{hb}__{ch}" for stat, hb, ch in wide.columns]
    wide = wide.reset_index()
    y = wide["targetEmotion"].astype(str)
    groups = wide["subject"].astype(int)
    meta = wide[["window_key", "subject", "experiment", "task", "targetEmotion", "trial_index", "window_index"]].copy()
    X = wide.drop(columns=["window_key", "subject", "experiment", "task", "targetEmotion", "trial_index", "window_index"])
    return X, y, groups, meta


def build_trial_bags_for_task(long_df: pd.DataFrame, task: str, label_mode: str) -> List[TrialBag]:
    use = long_df[long_df["task"].astype(str) == str(task)].copy()
    if use.empty:
        return []
    if label_mode == "emotion9":
        use = use[use["targetEmotion"].astype(str).isin(EMOTIONS_9)].copy()
        if use.empty:
            return []

    X_df, y_raw, _groups, meta = make_design_matrix(use, hb_family="HbO+HbR")
    y_raw_list = y_raw.astype(str).tolist()
    if label_mode == "valence3":
        y_series = [map_emotion_to_valence_3(v) for v in y_raw_list]
        label_col = "label3"
    else:
        y_series = [str(v).strip().lower() for v in y_raw_list]
        label_col = "label9"

    X = X_df.to_numpy(dtype=np.float32)
    meta = meta.reset_index(drop=True).copy()
    meta[label_col] = y_series

    bags: List[TrialBag] = []
    key_cols = ["subject", "experiment", "trial_index", "task"]
    for key, idx in meta.groupby(key_cols).groups.items():
        d = meta.loc[idx].sort_values("window_index")
        arr = X[d.index.to_numpy()]
        if arr.size == 0:
            continue
        label_values = pd.unique(d[label_col])
        if label_values.size != 1:
            raise RuntimeError(f"Inconsistent labels within one trial bag: {key} -> {label_values.tolist()}")
        sid = int(key[0])
        exp = str(key[1])
        tri = int(key[2])
        src_task = str(key[3])
        bag_id = f"{sid}|{exp}|{tri}"
        bags.append(TrialBag(bag_id=bag_id, subject=sid, source_task=src_task, label=str(label_values[0]), features=arr))

    return bags


def label_classes_and_strings(label_mode: str) -> Tuple[List[str], Dict[str, int]]:
    if label_mode == "valence3":
        classes = list(VALENCE3_CLASSES)
    elif label_mode == "emotion9":
        classes = list(EMOTIONS_9)
    else:
        raise ValueError(f"Unknown label_mode: {label_mode!r}; allowed: {list(LABEL_MODES)}")
    return classes, {c: i for i, c in enumerate(classes)}
