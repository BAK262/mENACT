"""Cross-mode speech decoding evaluation (Performance -> Production transfer)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from utils.validation_speech.cross_speaker import (
    SVM_C,
    bootstrap_ci_by_trial,
    labels_for_mode,
    make_prediction_rows,
    predict_train_fixed_c_then_test,
)
from utils.validation_speech.trials import BagSample, subset_bags
from utils.validation_vocab import (
    ACTORS_GROUPS,
    CROSS_MODE_TEST,
    CROSS_MODE_TRAIN,
    GROUP_DISPLAY,
    LABEL_MODES,
    NON_ACTORS_GROUP,
    TASK_KEY_TO_EXP_PREFIX,
)

CROSS_MODE_CONDITION_ORDER: tuple[str, ...] = (
    "cross_mode_all",
    "cross_mode_actors",
    "cross_mode_non_actors",
)

CROSS_MODE_DISPLAY: Dict[str, str] = {
    "cross_mode_all": "All",
    "cross_mode_actors": "Actors",
    "cross_mode_non_actors": GROUP_DISPLAY[NON_ACTORS_GROUP],
}

TRAIN_EXP_PREFIX = TASK_KEY_TO_EXP_PREFIX[CROSS_MODE_TRAIN]
TEST_EXP_PREFIX = TASK_KEY_TO_EXP_PREFIX[CROSS_MODE_TEST]


def evaluate_cross_mode(
    bags: Sequence[BagSample],
    label_mode: str,
    random_state: int,
    model_name: str,
) -> pd.DataFrame:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    pred_rows: List[Dict[str, object]] = []
    train_bags = subset_bags(bags, exp_prefix=TRAIN_EXP_PREFIX)
    test_bags = subset_bags(bags, exp_prefix=TEST_EXP_PREFIX)
    if train_bags and test_bags:
        y_true, y_pred, best_c = predict_train_fixed_c_then_test(
            train_bags=train_bags,
            test_bags=test_bags,
            label_mode=label_mode,
            random_state=random_state + 2000,
            c_value=float(SVM_C),
        )
        pred_rows.extend(
            make_prediction_rows(
                test_bags=test_bags,
                y_true=y_true,
                y_pred=y_pred,
                label_mode=label_mode,
                test_id="cross_mode_all",
                train_desc=f"cross_mode_{CROSS_MODE_TRAIN}_to_{CROSS_MODE_TEST}",
                c_value=best_c,
                model_name=model_name,
                fold_id=1,
            )
        )

    all_pred_df = pd.DataFrame(pred_rows)
    if not all_pred_df.empty:
        c_all_rows = all_pred_df[all_pred_df["test_id"] == "cross_mode_all"].copy()
        gen_df = c_all_rows[c_all_rows["subject_group"] == NON_ACTORS_GROUP].copy()
        if not gen_df.empty:
            gen_df.loc[:, "test_id"] = "cross_mode_non_actors"
            gen_df.loc[:, "train_desc"] = "cross_mode_group_slice_non_actors"
            pred_rows.extend(gen_df.to_dict(orient="records"))
        act_df = c_all_rows[c_all_rows["subject_group"].isin(ACTORS_GROUPS)].copy()
        if not act_df.empty:
            act_df.loc[:, "test_id"] = "cross_mode_actors"
            act_df.loc[:, "train_desc"] = "cross_mode_group_slice_actors"
            pred_rows.extend(act_df.to_dict(orient="records"))

    return pd.DataFrame(pred_rows)


def summarize_cross_mode(
    pred_df: pd.DataFrame,
    label_mode: str,
    model_name: str,
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:
    labels_full = labels_for_mode(label_mode)
    empty_metrics = bootstrap_ci_by_trial(pd.DataFrame(), labels_full, n_bootstrap=n_bootstrap, seed=0)
    rows: List[Dict[str, object]] = []
    for idx, test_id in enumerate(CROSS_MODE_CONDITION_ORDER):
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
            seed=random_state + idx + 4000,
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


def combined_summary_for_figures(
    cross_speaker_summary: pd.DataFrame,
    cross_mode_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cross-speaker and cross-mode summaries for figure plotting."""
    return pd.concat([cross_speaker_summary, cross_mode_summary], ignore_index=True)


__all__ = [
    "CROSS_MODE_CONDITION_ORDER",
    "CROSS_MODE_DISPLAY",
    "evaluate_cross_mode",
    "summarize_cross_mode",
    "combined_summary_for_figures",
]
