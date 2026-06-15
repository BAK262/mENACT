"""Validation workflow: cross-speaker and cross-mode speech decoding (trial-mean features, RBF-SVM).

Outputs under ``results/validation_speech/``:
  - ``trial_mean_manifest_<feature_set>.csv``
  - ``data/cross_speaker/``, ``data/cross_mode/`` — predictions and summaries
  - ``tables/tab_speech_cross_speaker_*``, ``tables/tab_speech_cross_mode_*``
  - ``figures/fig_speech_decoding_{valence3,emotion9}_validation.{pdf,png}``

Usage (from project root):
  python code/validation_speech.py           # default: --quick
  python code/validation_speech.py --quick   # 5 subjects, eGeMAPS only
  python code/validation_speech.py --full    # all released speech trials, eGeMAPS + HuBERT
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.validation_paths import ensure_result_layout, get_project_root, get_result_dir
from utils.validation_speech.benchmark import run_speech_benchmark
from utils.validation_speech.features import (
    build_trial_mean_feature_cache,
    filter_available_feature_sets,
    validate_feature_sets,
)
from utils.validation_speech.trials import discover_trials
from utils.validation_style import apply_style
from utils.validation_vocab import FEATURE_SETS, LABEL_MODES

LOGGER = logging.getLogger(__name__)

N_SUBJECTS_QUICK = 5
BOOTSTRAP_QUICK = 50
BOOTSTRAP_FULL = 200
N_PERM_QUICK = 200
N_PERM_FULL = 10_000


def _setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"run_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    LOGGER.info("Log file: %s", log_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech decoding validation (Section IV).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help=f"smoke test ({N_SUBJECTS_QUICK} subjects, eGeMAPS only)")
    mode.add_argument("--full", action="store_true", help="full run (all released speech trials)")
    parser.add_argument("--feature-sets", nargs="+", default=None, help=f"subset of: {' '.join(FEATURE_SETS)}")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--trial-workers", type=int, default=8)
    parser.add_argument("--trial-align", default="strict", choices=["strict", "intersection", "none"])
    return parser.parse_args()


def _load_or_build_manifests(
    *,
    root: Path,
    result_dir: Path,
    trials,
    feature_sets,
    force_recompute: bool,
    trial_workers: int,
) -> Dict[str, object]:
    prefix = "trial_mean_manifest"
    feature_dfs: Dict[str, object] = {}
    if not force_recompute and all((result_dir / f"{prefix}_{m}.csv").is_file() for m in feature_sets):
        LOGGER.info("Reusing existing trial-mean manifests from %s", result_dir)
        return feature_dfs

    built = build_trial_mean_feature_cache(
        root=root,
        trials=trials,
        models=feature_sets,
        cache_dir=result_dir / "cache",
        force_recompute=force_recompute,
        trial_workers=trial_workers,
    )
    for model_name, df in built.items():
        df.to_csv(result_dir / f"{prefix}_{model_name}.csv", index=False)
    return feature_dfs


def main() -> None:
    args = _parse_args()
    use_full = args.full and not args.quick
    n_subjects = N_SUBJECTS_QUICK
    mode_label = "full" if use_full else "quick"
    n_bootstrap = args.bootstrap_iters or (BOOTSTRAP_FULL if use_full else BOOTSTRAP_QUICK)
    n_perm = N_PERM_FULL if use_full else N_PERM_QUICK

    apply_style()
    root = get_project_root()
    feature_sets = validate_feature_sets(args.feature_sets) if args.feature_sets else (
        list(FEATURE_SETS) if use_full else ["egemaps"]
    )
    feature_sets = filter_available_feature_sets(root, feature_sets)
    result_dir = get_result_dir("speech")
    ensure_result_layout(result_dir)
    (result_dir / "cache").mkdir(parents=True, exist_ok=True)
    _setup_logging(result_dir / "logs")

    LOGGER.info("validation_speech [%s]: features=%s", mode_label, feature_sets)
    trials = discover_trials(root, subject_limit=None if use_full else n_subjects)
    if not trials:
        raise RuntimeError("No trials discovered.")
    n_subjects_actual = len({t.subject_id for t in trials})
    LOGGER.info("trials=%d subjects=%d", len(trials), n_subjects_actual)

    _load_or_build_manifests(
        root=root,
        result_dir=result_dir,
        trials=trials,
        feature_sets=feature_sets,
        force_recompute=bool(args.force_recompute),
        trial_workers=int(args.trial_workers),
    )

    info = run_speech_benchmark(
        result_dir=result_dir,
        feature_sets=feature_sets,
        label_modes=list(LABEL_MODES),
        random_state=int(args.random_state),
        n_bootstrap=int(n_bootstrap),
        allowed_subject_ids=sorted({t.subject_id for t in trials}),
        trial_align=str(args.trial_align),
        n_perm=int(n_perm),
    )
    cfg = {
        "workflow": "validation_speech",
        "mode": mode_label,
        "n_subjects": n_subjects_actual,
        "feature_sets": feature_sets,
        "bootstrap_iters": int(n_bootstrap),
        "permutation_iters": int(n_perm),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **info,
    }
    (result_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Done. Outputs in %s", result_dir)


if __name__ == "__main__":
    main()
