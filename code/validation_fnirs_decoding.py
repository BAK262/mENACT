"""
validation_fnirs_decoding: trial-level attention MIL fNIRS emotion decoding.

Outputs under results/validation_fnirs_decoding/:
  tables/  — 4 manuscript LaTeX tables (valence3 + emotion9)
  figures/ — 2 grouped-bar figures (valence3 + emotion9)
  data/    — feature caches, trial predictions, CSV summaries
  tables/  — Friedman macro-F1 tests
  logs/    — run logs

CUDA required for MIL training.

Usage (from project root):
  python code/validation_fnirs_decoding.py --quick
  python code/validation_fnirs_decoding.py --full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Set

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from utils.validation_fnirs.common import get_fnirs_preproc_root, setup_logging
from utils.validation_fnirs.features import (
    FS,
    build_trial_bags_for_task,
    ensure_per_task_feature_caches,
    label_classes_and_strings,
    load_long_for_task,
)
from utils.validation_fnirs.model import resolve_device
from utils.validation_fnirs.report import write_root_aggregates
from utils.validation_fnirs.stats import DEFAULT_N_PERM_FULL, DEFAULT_N_PERM_QUICK
from utils.validation_fnirs.task_wise_loso import run_task_wise_loso
from utils.validation_paths import ensure_result_layout, get_result_dir
from utils.validation_style import TASKS, apply_style
from utils.validation_vocab import LABEL_MODES

LOGGER = logging.getLogger("validation_fnirs_decoding")

N_SUBJECTS_FULL = 53
N_SUBJECTS_QUICK = 5
BOOTSTRAP_N_MANUSCRIPT = 2_000
BOOTSTRAP_N_QUICK = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fNIRS trial-level attention MIL decoding (task-wise LOSO CV).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help=f"smoke test with {N_SUBJECTS_QUICK} subjects (default)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help=f"full analysis with all {N_SUBJECTS_FULL} subjects",
    )
    parser.add_argument("--preproc", type=str, default="AEPO_001filt02")
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--fixation-seconds", type=float, default=3.0)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--reuse-long",
        action="store_true",
        help="Reuse per-task feature caches under data/features/ if present.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--bootstrap-n-manuscript",
        type=int,
        default=int(BOOTSTRAP_N_MANUSCRIPT),
        help="Bootstrap resamples for manuscript tables/figures CIs (default: 2000).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--bag-max-windows", type=int, default=15)
    parser.add_argument(
        "--bag-shuffle-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle window order within each bag during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_full = args.full and not args.quick
    subject_limit = None if use_full else N_SUBJECTS_QUICK
    mode_label = "full" if use_full else "quick"
    if not use_full:
        args.bootstrap_n_quick = BOOTSTRAP_N_QUICK
        args.bootstrap_n_manuscript = min(int(args.bootstrap_n_manuscript), BOOTSTRAP_N_QUICK)
        args.max_epochs = min(int(args.max_epochs), 30)
        args.patience = min(int(args.patience), 10)

    apply_style()
    result_dir = get_result_dir("fnirs_decoding")
    ensure_result_layout(result_dir)
    setup_logging(result_dir / "logs", logger_name="validation_fnirs_decoding")

    preproc_root = get_fnirs_preproc_root(args.preproc)

    LOGGER.info(
        "validation_fnirs_decoding [%s] | preproc=%s | subjects=%s | cv=loso",
        mode_label,
        args.preproc,
        "all" if subject_limit is None else str(subject_limit),
    )

    tasks_to_cache: Set[str] = set(TASKS)
    ensure_per_task_feature_caches(
        result_dir=result_dir,
        preproc_root=preproc_root,
        tasks_to_write=list(TASKS),
        subject_limit=subject_limit,
        fs=float(args.fs),
        fixation_seconds=float(args.fixation_seconds),
        window_seconds=float(args.window_seconds),
        stride_seconds=float(args.stride_seconds),
        reuse_long=bool(args.reuse_long),
    )

    device = resolve_device()

    for task in TASKS:
        long_df = load_long_for_task(
            result_dir=result_dir,
            task=task,
            subject_limit=subject_limit,
            fs=float(args.fs),
            window_seconds=float(args.window_seconds),
        )
        for label_mode in LABEL_MODES:
            bags = build_trial_bags_for_task(long_df, task=task, label_mode=label_mode)
            if not bags:
                LOGGER.warning("No bags for task=%s label_mode=%s; skipping.", task, label_mode)
                continue
            classes, class_to_idx = label_classes_and_strings(label_mode)
            run_task_wise_loso(
                bags=bags,
                classes=classes,
                class_to_idx=class_to_idx,
                label_mode=label_mode,
                task=task,
                result_dir=result_dir,
                args=args,
                device=device,
            )

    write_root_aggregates(
        result_dir=result_dir,
        args=args,
        n_perm=DEFAULT_N_PERM_FULL if use_full else DEFAULT_N_PERM_QUICK,
    )
    LOGGER.info("validation_fnirs_decoding [%s]: outputs saved under %s", mode_label, result_dir)


if __name__ == "__main__":
    main()
