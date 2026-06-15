"""Orchestrate cross-speaker and cross-mode speech validation benchmark."""
from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from utils.validation_paths import get_project_root
from utils.validation_speech.cross_mode import evaluate_cross_mode, summarize_cross_mode
from utils.validation_speech.cross_speaker import evaluate_cross_speaker, summarize_cross_speaker
from utils.validation_speech.features import check_and_align_trial_mean_manifests, load_trial_mean_manifest_csvs
from utils.validation_speech.figures import plot_speech_decoding_validation_figure
from utils.validation_speech.optional.permutation_tests import run_permutation_tests
from utils.validation_speech.tables import build_acc_f1_tex_tables
from utils.validation_speech.trials import load_subject_group_map, to_bags

LOGGER = logging.getLogger(__name__)


def run_speech_benchmark(
    *,
    result_dir: Path,
    feature_sets: Sequence[str],
    label_modes: Sequence[str],
    random_state: int,
    n_bootstrap: int,
    allowed_subject_ids: Sequence[int],
    trial_align: str,
    n_perm: int,
) -> Dict[str, object]:
    data_dir = result_dir / "data"
    cs_dir = data_dir / "cross_speaker"
    cm_dir = data_dir / "cross_mode"
    tables_dir = result_dir / "tables"
    align_report_dir = data_dir / "trial_align"
    for d in (cs_dir, cm_dir, tables_dir, align_report_dir):
        d.mkdir(parents=True, exist_ok=True)

    feature_dfs = load_trial_mean_manifest_csvs(result_dir, models=feature_sets)
    feature_dfs = check_and_align_trial_mean_manifests(
        feature_dfs,
        manifest_dir=result_dir,
        align_mode=str(trial_align),
        report_dir=align_report_dir,
    )
    subject_group_map = load_subject_group_map(get_project_root())
    allowed_set = set(int(s) for s in allowed_subject_ids)

    def _eval_model(model_name: str) -> None:
        bags = to_bags(feature_dfs[model_name], label_mode="emotion9", subject_group_map=subject_group_map)
        if allowed_set:
            bags = [b for b in bags if int(b.subject_id) in allowed_set]

        for label_mode in label_modes:
            rs = random_state + (0 if label_mode == "valence3" else 500)
            cs_pred = evaluate_cross_speaker(bags, label_mode, rs, model_name)
            cm_pred = evaluate_cross_mode(bags, label_mode, rs, model_name)
            cs_sum = summarize_cross_speaker(cs_pred, label_mode, model_name, n_bootstrap, random_state)
            cm_sum = summarize_cross_mode(cm_pred, label_mode, model_name, n_bootstrap, random_state)

            cs_pred.to_csv(cs_dir / f"predictions_{model_name}_{label_mode}.csv", index=False)
            cs_sum.to_csv(cs_dir / f"summary_{model_name}_{label_mode}.csv", index=False)
            cm_pred.to_csv(cm_dir / f"predictions_{model_name}_{label_mode}.csv", index=False)
            cm_sum.to_csv(cm_dir / f"summary_{model_name}_{label_mode}.csv", index=False)

    max_workers = max(1, min(2, len(feature_sets)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_eval_model, feature_sets))

    tex_outputs: Dict[str, Dict[str, str]] = {}
    for label_mode in label_modes:
        summary_by_model = {
            m: pd.concat(
                [
                    pd.read_csv(cs_dir / f"summary_{m}_{label_mode}.csv"),
                    pd.read_csv(cm_dir / f"summary_{m}_{label_mode}.csv"),
                ],
                ignore_index=True,
            )
            for m in feature_sets
        }
        written = build_acc_f1_tex_tables(
            summary_by_model,
            tables_dir,
            label_mode,
        )
        tex_outputs[label_mode] = {k: str(v) for k, v in written.items()}
        plot_speech_decoding_validation_figure(summary_by_model, label_mode, result_dir)

    perm_df = run_permutation_tests(
        result_dir=result_dir,
        models=list(feature_sets),
        label_modes=list(label_modes),
        n_perm=int(n_perm),
        seed=int(random_state),
    )

    root = get_project_root()

    def _rel(p: Path) -> str:
        return p.resolve().relative_to(root.resolve()).as_posix()

    LOGGER.info("Wrote tables and figures under %s", result_dir)
    return {
        "cross_speaker_data_dir": _rel(cs_dir),
        "cross_mode_data_dir": _rel(cm_dir),
        "permutation_summary_rows": int(len(perm_df)),
        "tex_outputs": tex_outputs,
    }
