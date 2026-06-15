"""LaTeX tables for self-report validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.validation_selfreport.ratings import TARGET_LEVELS
from utils.validation_style import task_caption_label, write_tabular
from utils.validation_vocab import TASK_DISPLAY, TASK_KEYS


AUX_DIM_ORDER: dict[str, list[str]] = {
    "perception": ["valence", "arousal", "liking", "familiarity"],
    "production": ["valence", "arousal", "liking", "familiarity"],
    "performance": [
        "self_valence", "self_arousal", "others_valence", "others_arousal",
        "innerDriven", "outerDriven", "familiarity", "liking",
        "actingCredibility", "scriptCredibility", "emotionCredibility", "roleConfidence",
    ],
}


def aggregate_aux_dims(additional_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate additional self-report measures by task, dimension, and target."""
    aux_agg = (
        additional_long.groupby(["task_key", "dimension", "targetEmotion"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    aux_agg.columns = ["task_key", "dimension", "targetEmotion", "m", "s"]
    aux_agg["cell"] = aux_agg.apply(lambda r: f"{r['m']:.1f} ({r['s']:.1f})", axis=1)
    return aux_agg


def write_aux_dims_table(aux_agg: pd.DataFrame, tables_dir: Path) -> Path:
    """Write appendix LaTeX table for additional self-report measures."""
    ncol_tab = len(TARGET_LEVELS) + 1

    tex_lines: list[str] = [
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.05cm} "
        + "".join([r"p{0.076\textwidth}"] * len(TARGET_LEVELS))
        + r"@{}}",
        r"\toprule",
        "Released dimension & " + " & ".join(TARGET_LEVELS) + r" \\",
        r"\midrule",
    ]

    task_labels_tex = {
        task_key: task_caption_label(TASK_DISPLAY[task_key]) for task_key in TASK_KEYS
    }

    for task_idx, task_key in enumerate(TASK_KEYS):
        if task_idx > 0:
            tex_lines.append(r"\addlinespace[0.28ex]")
        tex_lines.append(
            rf"\multicolumn{{{ncol_tab}}}{{@{{}}l@{{}}}}{{\textit{{{task_labels_tex[task_key]}}}}} \\"
        )
        dims = AUX_DIM_ORDER[task_key]
        available = aux_agg[aux_agg["task_key"] == task_key]["dimension"].unique()
        dims = [d for d in dims if d in available]

        for dim in dims:
            dim_display = r"\texttt{" + dim.replace("_", r"\_") + "}"
            cells: list[str] = []
            for te in TARGET_LEVELS:
                hit = aux_agg[
                    (aux_agg["task_key"] == task_key)
                    & (aux_agg["dimension"] == dim)
                    & (aux_agg["targetEmotion"] == te)
                ]
                if len(hit) == 1:
                    cells.append(hit["cell"].values[0])
                else:
                    cells.append("---")
            tex_lines.append(rf"\quad {dim_display} & " + " & ".join(cells) + r" \\")

    tex_lines.extend([r"\bottomrule", r"\end{tabular}", "}"])

    out_path = tables_dir / "tab_appendix_selfreport_aux_dims_by_task_target_tabular.tex"
    write_tabular(tex_lines, out_path)
    return out_path
