"""Validation workflow – trial-level multidimensional self-reports vs target conditions.

Outputs
-------
(1) Main figure  : figures/fig_trial_selfreport_target_structure.pdf
(2) Appendix fig : figures/fig_appendix_performance_experience_intention_diff_grid.pdf
(3) Appendix tab : tables/tab_appendix_selfreport_aux_dims_by_task_target_tabular.tex
(4) CSV summaries: data/bar_summary, Performance experience/intention profiles, inferential tests

Performance-task discrete-intensity main analyses use emotional experience (feel_*) only.

Usage (from project root):
    python code/validation_selfreport.py --quick
    python code/validation_selfreport.py --full
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.validation_paths import ensure_result_layout, get_project_root, get_result_dir
from utils.validation_selfreport.figures import (
    plot_performance_experience_intention_grid,
    plot_target_structure_figure,
)
from utils.validation_selfreport.performance_dual_ratings import (
    build_performance_dual_summaries,
)
from utils.validation_selfreport.ratings import (
    N_SUBS_FULL,
    N_SUBS_QUICK,
    build_bar_summary_all,
    load_additional_selfreport_ratings_long,
    load_all_ratings,
    load_performance_dual_ratings,
    load_subject_info,
)
from utils.validation_selfreport.stats import write_selfreport_stats
from utils.validation_selfreport.tables import aggregate_aux_dims, write_aux_dims_table
from utils.validation_style import apply_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run(*, n_subjects: int, result_dir: Path) -> None:
    """Run self-report validation for *n_subjects* participants."""
    ensure_result_layout(result_dir)
    data_dir = result_dir / "data"
    tables_dir = result_dir / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    root = get_project_root()
    data_raw = root / "data" / "all_raw"

    logger.info("Loading all ratings (n_subjects=%d) ...", n_subjects)
    combined = load_all_ratings(data_raw, n_subjects=n_subjects)
    sub_info = load_subject_info(data_raw)

    bar_all = build_bar_summary_all(combined)
    bar_all.to_csv(data_dir / "bar_summary_all_tasks.csv", index=False)

    logger.info("Loading Performance experience/intention ratings ...")
    performance_dual = load_performance_dual_ratings(data_raw, n_subjects=n_subjects)
    performance_summaries = build_performance_dual_summaries(performance_dual, sub_info)

    performance_summaries["dual_summary"].to_csv(
        data_dir / "performance_experience_intention_mean_profiles.csv", index=False
    )
    performance_summaries["diff_summary"].to_csv(
        data_dir / "performance_experience_minus_intention.csv", index=False
    )

    logger.info("Loading additional self-report measures ...")
    additional_long = load_additional_selfreport_ratings_long(
        data_raw, n_subjects=n_subjects
    )
    aux_agg = aggregate_aux_dims(additional_long)

    logger.info("Plotting main self-report figure ...")
    plot_target_structure_figure(bar_all, result_dir)

    logger.info("Plotting appendix heatmap grid ...")
    plot_performance_experience_intention_grid(
        performance_summaries["experience_by_group"],
        performance_summaries["intention_by_group"],
        performance_summaries["diff_by_group"],
        result_dir,
    )

    logger.info("Running self-report inferential tests ...")
    write_selfreport_stats(
        ratings=combined,
        performance_dual=performance_dual,
        sub_info=sub_info,
        data_dir=data_dir,
    )

    logger.info("Generating appendix LaTeX table ...")
    write_aux_dims_table(aux_agg, tables_dir)

    logger.info("All outputs saved to %s", result_dir)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Trial-level self-report validation (Perception / Production / Performance)."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help=f"Smoke test on first {N_SUBS_QUICK} subjects (default).",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help=f"Full analysis on all {N_SUBS_FULL} subjects.",
    )
    args = parser.parse_args(argv)

    n_subjects = N_SUBS_FULL if args.full else N_SUBS_QUICK
    result_dir = get_result_dir("selfreport")
    apply_style()
    run(n_subjects=n_subjects, result_dir=result_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
