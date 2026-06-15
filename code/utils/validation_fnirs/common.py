from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.validation_paths import get_project_root

LOGGER = logging.getLogger("validation_fnirs")


def setup_logging(log_dir: Path, logger_name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"run_{ts}.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    logging.getLogger(logger_name).info("Log file: %s", log_path)
    return log_path


def discover_subject_dirs(fnirs_root: Path, subject_limit: Optional[int]) -> List[Path]:
    if not fnirs_root.exists():
        raise FileNotFoundError(f"Missing fNIRS directory: {fnirs_root}")
    subs = sorted(
        [p for p in fnirs_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if subject_limit is not None:
        subs = subs[: int(subject_limit)]
    return subs


@dataclass(frozen=True)
class TrialsDC:
    labels: List[str]
    trials: List[np.ndarray]  # list of (n_measures, n_time)
    times: List[np.ndarray]  # list of (n_time,)
    trialinfo: np.ndarray  # object array (n_trials, n_cols)


def load_trialsdc_mat(mat_path: Path) -> TrialsDC:
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise RuntimeError("scipy is required to load MATLAB .mat files") from exc

    d = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "trialsDC" not in d:
        raise RuntimeError(f"Missing trialsDC in {mat_path}")
    td = d["trialsDC"]

    labels = [str(x) for x in np.atleast_1d(td.label).tolist()]

    trials_raw = np.atleast_1d(td.trial).tolist()
    trials: List[np.ndarray] = []
    for t in trials_raw:
        arr = np.asarray(t)
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected trial ndim={arr.ndim} in {mat_path}")
        trials.append(arr)

    times_raw = np.atleast_1d(td.time).tolist()
    times: List[np.ndarray] = []
    for tt in times_raw:
        arr = np.asarray(tt).reshape(-1)
        times.append(arr)

    if "trialinfo" not in d:
        raise RuntimeError(f"Missing trialinfo in {mat_path}")
    trialinfo = np.asarray(d["trialinfo"], dtype=object)
    if trialinfo.ndim != 2:
        raise RuntimeError(f"Unexpected trialinfo shape {trialinfo.shape} in {mat_path}")

    return TrialsDC(labels=labels, trials=trials, times=times, trialinfo=trialinfo)


def parse_measure_label(label: str) -> Tuple[str, str]:
    """
    Parse a trialsDC.label entry like "S1-D1 O2Hb" or "S1-D1 HHb".
    Returns (channel_id, hb_type) where hb_type is one of {"HbO","HbR","other"}.
    """
    parts = label.strip().split()
    if not parts:
        return "", "other"
    channel_id = parts[0]
    hb_raw = parts[1] if len(parts) > 1 else ""
    hb_raw = hb_raw.strip().lower().strip("[]()")
    if hb_raw in {"o2hb", "hbo"}:
        return channel_id, "HbO"
    if hb_raw in {"hhb", "hbr"}:
        return channel_id, "HbR"
    return channel_id, "other"


def get_fnirs_preproc_root(preproc_name: str) -> Path:
    return get_project_root() / "data" / "fnirs_signals" / preproc_name
