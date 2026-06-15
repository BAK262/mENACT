"""Trial discovery and bag assembly for speech validation."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.validation_style import EMOTIONS_9
from utils.validation_vocab import LEGACY_TASK_ALIASES, TASK_DISPLAY

LOGGER = logging.getLogger(__name__)

NEGATIVE = {"anger", "disgust", "fear", "sadness"}
POSITIVE = {"amusement", "inspiration", "joy", "tenderness"}


@dataclass
class TrialRecord:
    subject_id: int
    exp_prefix: str
    timestamp: str
    emotion: str
    audio_path: Path

    @property
    def task_key(self) -> str:
        return LEGACY_TASK_ALIASES[self.exp_prefix]

    @property
    def task(self) -> str:
        return TASK_DISPLAY[self.task_key]

    @property
    def trial_key(self) -> str:
        return f"{self.subject_id}_{self.exp_prefix}_{self.timestamp}_{self.emotion}"


@dataclass
class BagSample:
    trial_key: str
    subject_id: int
    exp_prefix: str
    task_key: str
    task: str
    subject_group: str
    label_9: str
    label_3: str
    instances: np.ndarray


def parse_trial_from_filename(path: Path) -> Optional[Tuple[str, str, str]]:
    m = re.match(r"^(exp[23])_(\d{14})_([a-z]+)\.mp4$", path.name)
    if m is None:
        return None
    exp_prefix, timestamp, emotion = m.group(1), m.group(2), m.group(3)
    if emotion not in EMOTIONS_9:
        return None
    return exp_prefix, timestamp, emotion


def pick_rating_file(subject_dir: Path, exp_prefix: str, timestamp: str) -> Optional[Path]:
    expected = subject_dir / f"{exp_prefix}_{timestamp}_rating.csv"
    if expected.exists():
        return expected
    all_csv = sorted(subject_dir.glob(f"{exp_prefix}_*_rating.csv"))
    if not all_csv:
        return None
    return all_csv[-1]


def load_completion_map(subject_dir: Path, exp_prefix: str, timestamp: str) -> Dict[str, bool]:
    fp = pick_rating_file(subject_dir, exp_prefix, timestamp)
    if fp is None:
        return {}
    try:
        df = pd.read_csv(fp)
    except Exception as exc:
        LOGGER.warning("Failed reading rating file %s: %s", fp, exc)
        return {}
    if "targetEmotion" not in df.columns:
        return {}
    flag = "tellCompletedPerc" if exp_prefix == "exp2" else "actCompletedPerc"
    if flag not in df.columns:
        return {}

    out: Dict[str, bool] = {}
    for _, row in df.iterrows():
        emo = str(row["targetEmotion"]).strip().lower()
        if emo not in EMOTIONS_9:
            continue
        try:
            out[emo] = float(row[flag]) > 1.0
        except Exception:
            out[emo] = False
    return out


def discover_trials(root: Path, subject_limit: Optional[int]) -> List[TrialRecord]:
    all_raw = root / "data" / "all_raw"
    if not all_raw.is_dir():
        raise FileNotFoundError(f"Missing directory: {all_raw}")

    subject_dirs = sorted(
        [p for p in all_raw.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    if subject_limit is not None:
        subject_dirs = subject_dirs[:subject_limit]

    trials: List[TrialRecord] = []
    for sub_dir in subject_dirs:
        sid = int(sub_dir.name)
        completion_cache: Dict[Tuple[str, str], Dict[str, bool]] = {}
        for mp4 in sorted(sub_dir.glob("exp*_*.mp4")):
            parsed = parse_trial_from_filename(mp4)
            if parsed is None:
                continue
            exp_prefix, timestamp, emotion = parsed
            key = (exp_prefix, timestamp)
            if key not in completion_cache:
                completion_cache[key] = load_completion_map(sub_dir, exp_prefix, timestamp)
            c_map = completion_cache[key]
            if c_map and not c_map.get(emotion, False):
                continue
            trials.append(
                TrialRecord(
                    subject_id=sid,
                    exp_prefix=exp_prefix,
                    timestamp=timestamp,
                    emotion=emotion,
                    audio_path=mp4,
                )
            )
    return trials


def load_subject_group_map(root: Path) -> Dict[int, str]:
    fp = root / "data" / "all_raw" / "subject_info.csv"
    if not fp.is_file():
        raise FileNotFoundError(f"Missing subject info CSV: {fp}")
    df = pd.read_csv(fp)
    need_cols = {"id", "group"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"subject_info.csv missing required columns: {need_cols}")
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        try:
            sid = int(row["id"])
        except Exception:
            continue
        grp = str(row["group"]).strip().lower()
        if grp not in {"professional", "amateur", "general"}:
            continue
        out[sid] = grp
    if not out:
        raise RuntimeError("No valid subject->group mappings loaded from subject_info.csv")
    return out


def emotion_to_valence3(emo: str) -> str:
    if emo in NEGATIVE:
        return "negative"
    if emo == "neutral":
        return "neutral"
    if emo in POSITIVE:
        return "positive"
    raise ValueError(f"Unknown emotion label: {emo}")


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("f_")]
    cols.sort(key=lambda x: int(x.split("_", 1)[1]))
    return cols


def to_bags(
    df: pd.DataFrame,
    label_mode: str,
    subject_group_map: Optional[Dict[int, str]] = None,
) -> List[BagSample]:
    label_col = "emotion_3" if label_mode == "valence3" else "emotion_9"
    feat_cols = feature_columns(df)
    bags: List[BagSample] = []
    for trial_key, grp in df.groupby("trial_key", sort=False):
        grp = grp.sort_values("segment_id")
        x = grp[feat_cols].to_numpy(dtype=np.float32)
        if x.shape[0] == 0:
            continue
        first = grp.iloc[0]
        sid = int(first["subject_id"])
        task_key = str(first["task_key"])
        bags.append(
            BagSample(
                trial_key=str(trial_key),
                subject_id=sid,
                exp_prefix=str(first["exp_prefix"]),
                task_key=task_key,
                task=str(first["task"]),
                subject_group=(
                    subject_group_map.get(sid, "unknown") if subject_group_map is not None else "unknown"
                ),
                label_9=str(first["emotion_9"]),
                label_3=str(first["emotion_3"]),
                instances=x,
            )
        )
    return bags


def bag_label(bag: BagSample, label_mode: str) -> str:
    return bag.label_3 if label_mode == "valence3" else bag.label_9


def subset_bags(
    bags: Sequence[BagSample],
    exp_prefix: Optional[str] = None,
    subject_groups: Optional[Sequence[str]] = None,
    task_key: Optional[str] = None,
    task: Optional[str] = None,
) -> List[BagSample]:
    group_set = set(subject_groups) if subject_groups is not None else None
    out: List[BagSample] = []
    for bag in bags:
        if exp_prefix is not None and bag.exp_prefix != exp_prefix:
            continue
        if task_key is not None and bag.task_key != task_key:
            continue
        if task is not None and bag.task != task:
            continue
        if group_set is not None and bag.subject_group not in group_set:
            continue
        out.append(bag)
    return out
