"""Performance-task experience/intention summaries and group-wise profiles."""
from __future__ import annotations

import pandas as pd

from utils.validation_style import EMOTIONS_8
from utils.validation_vocab import GROUP_KEYS

LAYER_EXPERIENCE = "Emotional experience"
LAYER_INTENTION = "Expressive intention"
LAYER_DIFFERENCE = "Difference"
PERFORMANCE_EXPERIENCE_PREFIX = "experience_"
PERFORMANCE_INTENTION_PREFIX = "intention_"

GROUP_ORDER: list[str] = list(GROUP_KEYS)
LAYER_ORDER: list[str] = [LAYER_EXPERIENCE, LAYER_INTENTION, LAYER_DIFFERENCE]


def _build_layer_summary(
    df: pd.DataFrame, prefix: str, layer_label: str
) -> pd.DataFrame:
    cols = [f"{prefix}{e}" for e in EMOTIONS_8]
    chunk = df[["subID", "targetEmotion"] + cols].copy()
    chunk.columns = ["subID", "targetEmotion"] + EMOTIONS_8
    long = chunk.melt(
        id_vars=["subID", "targetEmotion"],
        value_vars=EMOTIONS_8,
        var_name="emotion",
        value_name="rating",
    )
    agg = long.groupby(["targetEmotion", "emotion"])["rating"].mean().reset_index()
    agg.columns = ["targetEmotion", "emotion", "mean_rating"]
    agg["layer"] = layer_label
    return agg


def _build_layer_by_group(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}{e}" for e in EMOTIONS_8]
    long = df[["subID", "group", "targetEmotion"] + cols].melt(
        id_vars=["subID", "group", "targetEmotion"],
        value_vars=cols,
        var_name="emo_col",
        value_name="rating",
    )
    long["emotion"] = long["emo_col"].str.replace(f"^{prefix}", "", regex=True)
    return (
        long.groupby(["group", "targetEmotion", "emotion"])["rating"]
        .mean()
        .reset_index(name="mean_rating")
    )


def add_diff_columns(performance_dual: pd.DataFrame) -> pd.DataFrame:
    """Add experience-minus-intention columns to Performance dual ratings."""
    out = performance_dual.copy()
    for e in EMOTIONS_8:
        out[f"diff_{e}"] = (
            out[f"{PERFORMANCE_EXPERIENCE_PREFIX}{e}"]
            - out[f"{PERFORMANCE_INTENTION_PREFIX}{e}"]
        )
    return out


def build_performance_dual_summaries(
    performance_dual: pd.DataFrame,
    sub_info: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build Performance-task layer summaries and group-wise heatmap inputs."""
    experience_summary = _build_layer_summary(
        performance_dual, PERFORMANCE_EXPERIENCE_PREFIX, LAYER_EXPERIENCE
    )
    intention_summary = _build_layer_summary(
        performance_dual, PERFORMANCE_INTENTION_PREFIX, LAYER_INTENTION
    )
    dual_summary = pd.concat(
        [experience_summary, intention_summary], ignore_index=True
    )

    diff_summary = experience_summary[["targetEmotion", "emotion", "mean_rating"]].merge(
        intention_summary[["targetEmotion", "emotion", "mean_rating"]],
        on=["targetEmotion", "emotion"],
        suffixes=("_experience", "_intention"),
    )
    diff_summary["diff_mean"] = (
        diff_summary["mean_rating_experience"] - diff_summary["mean_rating_intention"]
    )

    performance_with_diff = add_diff_columns(performance_dual)
    performance_g = performance_with_diff.merge(
        sub_info[["subID", "group"]], on="subID", how="left"
    )

    return {
        "experience_summary": experience_summary,
        "intention_summary": intention_summary,
        "dual_summary": dual_summary,
        "diff_summary": diff_summary,
        "performance_with_diff": performance_with_diff,
        "experience_by_group": _build_layer_by_group(
            performance_g, PERFORMANCE_EXPERIENCE_PREFIX
        ),
        "intention_by_group": _build_layer_by_group(
            performance_g, PERFORMANCE_INTENTION_PREFIX
        ),
        "diff_by_group": _build_layer_by_group(performance_g, "diff_"),
    }
