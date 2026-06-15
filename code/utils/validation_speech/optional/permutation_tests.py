"""
Optional permutation tests for speech decoding validation (no retraining).

Reads saved prediction CSVs from ``results/validation_speech/data/cross_speaker/`` and
``data/cross_mode/`` and writes ``data/permutation_tests/speech_permutation_summary.csv``.

Usage (from project root):
  python code/utils/validation_speech/optional/permutation_tests.py
  python code/utils/validation_speech/optional/permutation_tests.py --n-perm 10000
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[3]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from utils.validation_paths import get_project_root, get_result_dir
from utils.validation_speech.cross_speaker import (
    CROSS_SPEAKER_CONDITION_ORDER,
    CROSS_SPEAKER_DISPLAY,
    labels_for_mode,
    pooled_metrics_pair,
)
from utils.validation_vocab import FEATURE_SETS, GROUP_DISPLAY, LABEL_MODES, NON_ACTORS_GROUP

LOGGER = logging.getLogger("speech_permutation_tests")

CHANCE = {"valence3": 0.333, "emotion9": 0.111}
DEFAULT_N_PERM = 10_000
DEFAULT_SEED = 42


def _actor_slice(subject_group: str) -> Optional[str]:
    g = str(subject_group).strip().lower()
    if g in {"professional", "amateur"}:
        return "actors"
    if g == NON_ACTORS_GROUP:
        return "non_actors"
    return None


def _pooled_f1_from_trials(trials: pd.DataFrame, label_mode: str) -> float:
    labels_full = labels_for_mode(label_mode)
    if trials.empty:
        return float("nan")
    mf1, _ = pooled_metrics_pair(
        trials["y_true"].to_numpy(dtype=object),
        trials["y_pred"].to_numpy(dtype=object),
        labels_full,
    )
    return float(mf1)


def permutation_p_vs_chance(
    trials: pd.DataFrame,
    label_mode: str,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    labels_full = labels_for_mode(label_mode)
    y_true = trials["y_true"].to_numpy(dtype=object)
    y_pred = trials["y_pred"].to_numpy(dtype=object)
    if y_true.size == 0:
        return float("nan"), float("nan")
    obs, _ = pooled_metrics_pair(y_true, y_pred, labels_full)
    count = 0
    for _ in range(int(n_perm)):
        perm_true = rng.permutation(y_true)
        null_f1, _ = pooled_metrics_pair(perm_true, y_pred, labels_full)
        if np.isfinite(null_f1) and null_f1 >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(obs), float(p)


def _subject_group_map(trials: pd.DataFrame) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for sid, grp in trials.groupby("subject_id")["subject_group"]:
        sl = _actor_slice(str(grp.iloc[0]))
        if sl is not None:
            out[int(sid)] = sl
    return out


def _trials_for_subjects(trials: pd.DataFrame, subject_ids: Sequence[int]) -> pd.DataFrame:
    sid_set = {int(s) for s in subject_ids}
    return trials[trials["subject_id"].isin(sid_set)].copy()


def permutation_p_actors_vs_non(
    trials: pd.DataFrame,
    label_mode: str,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, int, int, float]:
    group_map = _subject_group_map(trials)
    actor_ids = [sid for sid, g in group_map.items() if g == "actors"]
    non_ids = [sid for sid, g in group_map.items() if g == "non_actors"]
    if not actor_ids or not non_ids:
        return float("nan"), float("nan"), float("nan"), len(actor_ids), len(non_ids), float("nan")

    f1_a = _pooled_f1_from_trials(_trials_for_subjects(trials, actor_ids), label_mode)
    f1_n = _pooled_f1_from_trials(_trials_for_subjects(trials, non_ids), label_mode)
    obs_diff = float(f1_a - f1_n)

    all_ids = np.asarray(sorted(group_map.keys()), dtype=int)
    labels = np.asarray([group_map[int(s)] for s in all_ids], dtype=object)
    n_a = int(np.sum(labels == "actors"))

    count = 0
    for _ in range(int(n_perm)):
        perm_labels = rng.permutation(labels)
        perm_actor = all_ids[perm_labels == "actors"]
        perm_non = all_ids[perm_labels == "non_actors"]
        if perm_actor.size != n_a or perm_non.size == 0:
            continue
        pf1_a = _pooled_f1_from_trials(_trials_for_subjects(trials, perm_actor), label_mode)
        pf1_n = _pooled_f1_from_trials(_trials_for_subjects(trials, perm_non), label_mode)
        if not np.isfinite(pf1_a) or not np.isfinite(pf1_n):
            continue
        if abs(pf1_a - pf1_n) >= abs(obs_diff):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs_diff, float(f1_a), float(f1_n), len(actor_ids), len(non_ids), float(p)


def _slice_trials(trials: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    if slice_name == "All":
        return trials.copy()
    if slice_name == "Actors":
        return trials[trials["subject_group"].isin(["professional", "amateur"])].copy()
    if slice_name == "Non-actors":
        return trials[trials["subject_group"] == NON_ACTORS_GROUP].copy()
    raise ValueError(slice_name)


def _load_preds(pred_dir: Path, model: str, label_mode: str) -> pd.DataFrame:
    path = pred_dir / f"predictions_{model}_{label_mode}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")
    return pd.read_csv(path)


def run_permutation_tests(
    result_dir: Path,
    models: Sequence[str],
    label_modes: Sequence[str],
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    cs_dir = result_dir / "data" / "cross_speaker"
    cm_dir = result_dir / "data" / "cross_mode"
    out_dir = result_dir / "data" / "permutation_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, object]] = []

    for model in models:
        for label_mode in label_modes:
            chance = CHANCE[label_mode]
            pred_cs = _load_preds(cs_dir, model, label_mode)
            for test_id in CROSS_SPEAKER_CONDITION_ORDER:
                trials = pred_cs[pred_cs["test_id"] == test_id]
                obs, p = permutation_p_vs_chance(trials, label_mode, n_perm, rng)
                rows.append(
                    {
                        "family": "cross_speaker_vs_chance",
                        "evaluation": "cross-speaker",
                        "model": model,
                        "label_mode": label_mode,
                        "slice": CROSS_SPEAKER_DISPLAY[test_id],
                        "comparison": "vs_chance",
                        "method": "trial_label_permutation",
                        "n_perm": int(n_perm),
                        "n_trials": len(trials),
                        "n_subjects": trials["subject_id"].nunique(),
                        "pooled_f1": obs,
                        "chance": chance,
                        "p_value": p,
                    }
                )

    for model in models:
        for label_mode in label_modes:
            chance = CHANCE[label_mode]
            pred_cm = _load_preds(cm_dir, model, label_mode)
            trials_all = pred_cm[pred_cm["test_id"] == "cross_mode_all"]
            for slice_name in ["All", "Actors", "Non-actors"]:
                trials = _slice_trials(trials_all, slice_name)
                obs, p = permutation_p_vs_chance(trials, label_mode, n_perm, rng)
                rows.append(
                    {
                        "family": "cross_mode_vs_chance",
                        "evaluation": "cross-mode",
                        "model": model,
                        "label_mode": label_mode,
                        "slice": slice_name,
                        "comparison": "vs_chance",
                        "method": "trial_label_permutation",
                        "n_perm": int(n_perm),
                        "n_trials": len(trials),
                        "n_subjects": trials["subject_id"].nunique(),
                        "pooled_f1": obs,
                        "chance": chance,
                        "p_value": p,
                    }
                )

    for model in models:
        for label_mode in label_modes:
            trials = _load_preds(cm_dir, model, label_mode)
            trials = trials[trials["test_id"] == "cross_mode_all"]
            diff, f1_a, f1_n, n_a, n_b, p = permutation_p_actors_vs_non(trials, label_mode, n_perm, rng)
            rows.append(
                {
                    "family": "cross_mode_actors_vs_non",
                    "evaluation": "cross-mode",
                    "model": model,
                    "label_mode": label_mode,
                    "slice": "Actors_vs_Non-actors",
                    "comparison": "actors_vs_non_actors",
                    "method": "subject_cluster_permutation",
                    "n_perm": int(n_perm),
                    "n_trials": len(trials),
                    "n_subjects": int(n_a + n_b),
                    "n_subjects_actors": n_a,
                    "n_subjects_non_actors": n_b,
                    "pooled_f1": diff,
                    "pooled_f1_actors": f1_a,
                    "pooled_f1_non_actors": f1_n,
                    "chance": float("nan"),
                    "p_value": p,
                }
            )

    out_df = pd.DataFrame(rows)
    csv_path = out_dir / "speech_permutation_summary.csv"
    out_df.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s", csv_path)
    return out_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech decoding permutation tests (optional).")
    parser.add_argument("--models", nargs="+", default=list(FEATURE_SETS))
    parser.add_argument("--label-modes", nargs="+", default=list(LABEL_MODES))
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result_dir = get_result_dir("speech")
    run_permutation_tests(
        result_dir=result_dir,
        models=[m.lower() for m in args.models],
        label_modes=[m.lower() for m in args.label_modes],
        n_perm=int(args.n_perm),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
