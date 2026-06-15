"""Load trial-level self-report ratings and build bar summaries."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils.validation_selfreport.performance_dual_ratings import (
    PERFORMANCE_EXPERIENCE_PREFIX,
    PERFORMANCE_INTENTION_PREFIX,
)
from utils.validation_style import EMOTIONS_8, EMOTIONS_9
from utils.validation_vocab import TASK_DISPLAY, TASK_KEYS

N_SUBS_FULL = 53
N_SUBS_QUICK = 5

PERCEPTION_CLIP_THRESHOLD = 95.0

TASK_FILE_NUM_TO_KEY: dict[int, str] = {
    1: "perception",
    2: "production",
    3: "performance",
}
TASK_FILE_NUMBERS: tuple[int, ...] = (1, 2, 3)

TARGET_LEVELS: list[str] = EMOTIONS_9


def task_display_for_file_num(file_num: int) -> str:
    """Return manuscript task display name for a raw file number (exp1/2/3)."""
    return TASK_DISPLAY[TASK_FILE_NUM_TO_KEY[file_num]]


def _list_rating_csvs(sub_dir: Path, file_num: int) -> list[Path]:
    """Return rating CSV paths for a task file number. Production may have multiple sessions."""
    pat = re.compile(rf"^exp{file_num}_(\d{{14}})_rating\.csv$")
    matches: list[tuple[str, Path]] = []
    for f in sub_dir.iterdir():
        m = pat.match(f.name)
        if m:
            matches.append((m.group(1), f))
    if not matches:
        return []
    matches.sort(key=lambda x: x[0])
    if file_num == 2:
        return [p for _, p in matches]
    return [matches[-1][1]]


def _pick_rating_csv(sub_dir: Path, file_num: int) -> Path | None:
    """Pick the latest rating CSV for a given subject directory and task file number."""
    csvs = _list_rating_csvs(sub_dir, file_num)
    return csvs[-1] if csvs else None


def load_subject_info(data_raw: Path) -> pd.DataFrame:
    """Load subject metadata with a normalized ``subID`` column."""
    sub_info = pd.read_csv(data_raw / "subject_info.csv")
    if "id" in sub_info.columns and "subID" not in sub_info.columns:
        sub_info = sub_info.rename(columns={"id": "subID"})
    return sub_info


def load_all_ratings(data_raw: Path, *, n_subjects: int) -> pd.DataFrame:
    """Load discrete-emotion ratings for Perception, Production, and Performance."""
    frames: list[pd.DataFrame] = []
    for sub_no in range(1, n_subjects + 1):
        sub_dir = data_raw / str(sub_no)
        for file_num in TASK_FILE_NUMBERS:
            fps = _list_rating_csvs(sub_dir, file_num)
            if not fps:
                warnings.warn(f"Missing rating file sub={sub_no} exp={file_num}")
                continue
            task_key = TASK_FILE_NUM_TO_KEY[file_num]
            for fp in fps:
                df = pd.read_csv(fp)

                if file_num == 1:
                    df = df.rename(columns={"videoEmotion": "targetEmotion"})
                    df = df[df["percCompleted"] >= PERCEPTION_CLIP_THRESHOLD].copy()
                elif file_num == 2:
                    df = df[df["tellCompletedPerc"] > 1].copy()
                else:
                    feel_cols = [f"feel_{e}" for e in EMOTIONS_8]
                    missing = [c for c in feel_cols if c not in df.columns]
                    if missing:
                        raise ValueError(
                            f"Performance task file missing feel_* columns: {fp}"
                        )
                    df = df[df["actCompletedPerc"] > 1].copy()
                    for e in EMOTIONS_8:
                        df[e] = df[f"feel_{e}"]

                keep = ["targetEmotion"] + EMOTIONS_8
                keep = [c for c in keep if c in df.columns]
                df = df[keep].copy()
                df["subID"] = sub_no
                df["task_key"] = task_key
                frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_additional_selfreport_ratings_long(
    data_raw: Path, *, n_subjects: int
) -> pd.DataFrame:
    """Load additional self-report measures (valence, arousal, etc.) in long format."""
    additional_perception_production = [
        "valence", "arousal", "liking", "familiarity",
    ]
    additional_performance = [
        "self_valence", "self_arousal", "others_valence", "others_arousal",
        "innerDriven", "outerDriven", "familiarity", "liking",
        "actingCredibility", "scriptCredibility", "emotionCredibility", "roleConfidence",
    ]
    frames: list[pd.DataFrame] = []
    for sub_no in range(1, n_subjects + 1):
        sub_dir = data_raw / str(sub_no)
        for file_num in TASK_FILE_NUMBERS:
            task_key = TASK_FILE_NUM_TO_KEY[file_num]
            for fp in _list_rating_csvs(sub_dir, file_num):
                df = pd.read_csv(fp)
                if file_num == 1:
                    df = df.rename(columns={"videoEmotion": "targetEmotion"})
                    df = df[df["percCompleted"] >= PERCEPTION_CLIP_THRESHOLD].copy()
                    want = additional_perception_production
                elif file_num == 2:
                    df = df[df["tellCompletedPerc"] > 1].copy()
                    want = additional_perception_production
                else:
                    df = df[df["actCompletedPerc"] > 1].copy()
                    want = additional_performance

                present = [c for c in want if c in df.columns]
                if len(present) < len(want):
                    miss = set(want) - set(present)
                    warnings.warn(
                        f"Missing additional self-report columns sub={sub_no} "
                        f"task={task_key}: {miss}"
                    )

                cols_keep = ["targetEmotion"] + present
                chunk = df[cols_keep].copy()
                chunk["subID"] = sub_no
                chunk["task_key"] = task_key
                melted = chunk.melt(
                    id_vars=["subID", "task_key", "targetEmotion"],
                    var_name="dimension",
                    value_name="value",
                )
                melted = melted.dropna(subset=["value"])
                frames.append(melted)

    return pd.concat(frames, ignore_index=True)


def load_performance_dual_ratings(data_raw: Path, *, n_subjects: int) -> pd.DataFrame:
    """Load Performance-task emotional-experience and expressive-intention ratings."""
    frames: list[pd.DataFrame] = []
    for sub_no in range(1, n_subjects + 1):
        sub_dir = data_raw / str(sub_no)
        fp = _pick_rating_csv(sub_dir, 3)
        if fp is None:
            warnings.warn(f"Missing Performance task rating file sub={sub_no}")
            continue
        df = pd.read_csv(fp)
        feel_cols = [f"feel_{e}" for e in EMOTIONS_8]
        act_cols = [f"act_{e}" for e in EMOTIONS_8]
        missing = [c for c in feel_cols + act_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Performance task file missing feel_*/act_* columns: {fp}"
            )

        df = df[df["actCompletedPerc"] > 1].copy()
        out = pd.DataFrame({"subID": sub_no, "targetEmotion": df["targetEmotion"]})
        for e in EMOTIONS_8:
            out[f"{PERFORMANCE_EXPERIENCE_PREFIX}{e}"] = df[f"feel_{e}"].values
            out[f"{PERFORMANCE_INTENTION_PREFIX}{e}"] = df[f"act_{e}"].values
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def build_bar_summary(df: pd.DataFrame, task_key: str) -> pd.DataFrame:
    """Build per-target-category bar summary for one task."""
    df_task = df[df["task_key"] == task_key].copy()
    long = df_task[["subID", "targetEmotion"] + EMOTIONS_8].melt(
        id_vars=["subID", "targetEmotion"],
        value_vars=EMOTIONS_8,
        var_name="emotion",
        value_name="rating",
    ).dropna(subset=["rating"])

    long["type"] = np.where(
        long["emotion"] == long["targetEmotion"], "Matching", "Other"
    )

    agg = (
        long.groupby(["targetEmotion", "emotion", "type"])["rating"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["targetEmotion", "emotion", "type", "mean", "std_raw", "n"]
    agg["se"] = agg["std_raw"] / np.sqrt(agg["n"])
    agg = agg.drop(columns=["std_raw"])
    agg["task_key"] = task_key
    return agg


def build_bar_summary_all(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bar summaries across all three tasks."""
    bar_all = pd.concat(
        [build_bar_summary(df, tk) for tk in TASK_KEYS], ignore_index=True
    )
    bar_all["plot_fill"] = np.where(
        bar_all["type"] == "Matching", "Target dimension", "Non-target dimension"
    )
    bar_all["task"] = bar_all["task_key"].map(TASK_DISPLAY)
    return bar_all
