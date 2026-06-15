"""Validation workflow: trait questionnaires and ECS example heatmaps.

Outputs:
  (1) LaTeX table: questionnaire dimensions by experience group with one-way
      ANOVA, Benjamini--Hochberg FDR ($q$), and Tukey HSD post-hoc summaries.
  (2) Appendix figure: violin+jitter for 15 questionnaire dimensions.
  (3) Appendix figure: three ECS case heatmaps (high/low/heterogeneous).
  (4) CSV summaries under ``data/``.

Usage (from project root):
  python code/validation_traits.py           # default: --quick
  python code/validation_traits.py --quick   # 5 subjects smoke test
  python code/validation_traits.py --full    # all 53 subjects
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from utils.validation_paths import ensure_result_layout, get_project_root, get_result_dir
from utils.validation_style import apply_style
from utils.validation_traits.figures import plot_ecs_heatmaps, plot_trait_violins
from utils.validation_traits.load import load_data
from utils.validation_traits.tables import build_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

N_SUBJECTS_FULL = 53
N_SUBJECTS_QUICK = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trait questionnaire validation (Section IV).",
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    use_full = args.full and not args.quick
    n_subjects = N_SUBJECTS_FULL if use_full else N_SUBJECTS_QUICK
    mode_label = "full" if use_full else "quick"

    root = get_project_root()
    result_dir = get_result_dir("traits")
    ensure_result_layout(result_dir)
    tables_dir = result_dir / "tables"
    data_dir = result_dir / "data"

    apply_style()

    logger.info(
        "validation_traits [%s]: loading trait data for %d subjects …",
        mode_label, n_subjects,
    )
    traits, matrices = load_data(root, n_subjects=n_subjects)

    logger.info("Building LaTeX table …")
    build_table(traits, tables_dir, data_dir)

    logger.info("Plotting trait violin facets …")
    plot_trait_violins(traits, result_dir)

    logger.info("Plotting ECS heatmaps …")
    plot_ecs_heatmaps(traits, matrices, result_dir)

    ecs_cases = pd.DataFrame([{
        "high": int(traits.loc[traits["mean_pairwise"].idxmax(), "subID"]),
        "low": int(traits.loc[traits["mean_pairwise"].idxmin(), "subID"]),
        "mixed": int(traits.loc[traits["sd_pairwise"].idxmax(), "subID"]),
    }])
    ecs_cases.to_csv(data_dir / "ecs_case_subject_ids.csv", index=False)

    logger.info(
        "validation_traits [%s]: all outputs saved under %s",
        mode_label, result_dir,
    )


if __name__ == "__main__":
    main()
