"""LaTeX table and CSV summaries for trait group comparisons."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from utils.validation_style import write_tabular
from utils.validation_traits.load import (
    BFI_DOMAINS,
    BEQ_DIMS,
    GROUP_ORDER,
    QCAE_DIMS,
)

logger = logging.getLogger(__name__)

GROUP_ABBR: dict[str, str] = {
    "Professional": "Prof",
    "Amateur": "Am",
    "Non-actor": "Non",
}


def _pretty_dim(name: str) -> str:
    """Convert camelCase dimension name to Title Case with spaces."""
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name).title()


def _fmt_ms(values: pd.Series) -> str:
    """Format as 'mean (sd)' with one decimal place."""
    return f"{values.mean():.1f} ({values.std():.1f})"


def _fmt_f(f_val: float | None) -> str:
    if f_val is None or not np.isfinite(f_val):
        return "---"
    return f"{f_val:.2f}"


def _fmt_p(p_val: float | None) -> str:
    if p_val is None or not np.isfinite(p_val):
        return "---"
    if p_val < 0.001 or abs(p_val - 0.001) < 1e-12:
        return "<0.001"
    return f"{p_val:.3f}"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _group_anova(
    y: pd.Series, g: pd.Series,
) -> tuple[float | None, float | None]:
    """One-way ANOVA across experience groups. Returns (F, p)."""
    mask = y.notna() & g.notna()
    if mask.sum() < 6:
        return None, None
    groups_data = [
        y[mask & (g == grp)].values for grp in GROUP_ORDER if (mask & (g == grp)).any()
    ]
    if len(groups_data) < 2:
        return None, None
    f_stat, p_val = stats.f_oneway(*groups_data)
    return float(f_stat), float(p_val)


def _merge_posthoc_pairs(pairs: set[tuple[str, str]]) -> str:
    """Collapse Tukey pairs into compact ``Prof\\&Am$>$Non`` notation."""
    remaining = set(pairs)
    out: list[str] = []

    if ("Prof", "Non") in remaining and ("Am", "Non") in remaining:
        out.append(r"Prof\&Am$>$Non")
        remaining.discard(("Prof", "Non"))
        remaining.discard(("Am", "Non"))

    if ("Am", "Prof") in remaining and ("Am", "Non") in remaining:
        out.append(r"Am$>$Prof\&Non")
        remaining.discard(("Am", "Prof"))
        remaining.discard(("Am", "Non"))

    order = {"Prof": 0, "Am": 1, "Non": 2}
    for high, low in sorted(remaining, key=lambda t: (order[t[0]], order[t[1]])):
        out.append(f"{high}$>${low}")

    if not out:
        return "---"
    return "; ".join(out)


def _tukey_posthoc_summary(
    y: pd.Series,
    g: pd.Series,
    *,
    q_val: float | None,
) -> str:
    """Summarize significant Tukey HSD pairs when omnibus FDR q < 0.05."""
    if q_val is None or not np.isfinite(q_val) or q_val >= 0.05:
        return "---"
    sub = pd.DataFrame({"y": y, "g": g}).dropna()
    if sub["g"].nunique() < 2:
        return "---"
    res = pairwise_tukeyhsd(sub["y"], sub["g"], alpha=0.05)
    pairs: set[tuple[str, str]] = set()
    for row in res.summary().data[1:]:
        g1, g2, meandiff, _p_adj, _lower, _upper, reject = row
        if str(reject) != "True":
            continue
        diff = float(meandiff)
        if diff > 0:
            higher, lower = str(g2), str(g1)
        else:
            higher, lower = str(g1), str(g2)
        pairs.add((GROUP_ABBR[higher], GROUP_ABBR[lower]))
    return _merge_posthoc_pairs(pairs)


def build_table(
    traits: pd.DataFrame,
    tables_dir: Path,
    data_dir: Path,
) -> None:
    """Generate questionnaire group-tests LaTeX tabular and CSV summary."""
    n_total = len(traits)
    ns = {g: int((traits["group"] == g).sum()) for g in GROUP_ORDER}
    df1 = len(GROUP_ORDER) - 1
    df2 = n_total - len(GROUP_ORDER)

    sections: list[tuple[str, str, str]] = [("BDI-II", "Total", "depressScore")]
    for dim in BFI_DOMAINS:
        sections.append(("BFI-2", _pretty_dim(dim), dim))
    for dim in QCAE_DIMS:
        sections.append(("QCAE", _pretty_dim(dim), dim))
    for dim in BEQ_DIMS:
        sections.append(("BEQ", _pretty_dim(dim), dim))

    anova_ps: list[float | None] = []
    for _sec, _dim_display, col in sections:
        _f_val, p_val = _group_anova(traits[col], traits["group"])
        anova_ps.append(p_val)

    finite_ps = [p for p in anova_ps if p is not None and np.isfinite(p)]
    if finite_ps:
        _, q_vals, _, _ = multipletests(finite_ps, method="fdr_bh")
        q_by_idx = {
            i: q_vals[j]
            for j, i in enumerate(
                idx for idx, p in enumerate(anova_ps) if p is not None and np.isfinite(p)
            )
        }
    else:
        q_by_idx = {}

    csv_rows: list[dict[str, str]] = []
    table_data: list[tuple[str, str, dict[str, str]]] = []

    for idx, (sec, dim_display, col) in enumerate(sections):
        vals = traits[col]
        f_val, p_val = _group_anova(vals, traits["group"])
        q_val = q_by_idx.get(idx)
        posthoc = _tukey_posthoc_summary(vals, traits["group"], q_val=q_val)
        row: dict[str, str] = {
            "section": sec,
            "dimension": dim_display,
            "overall": _fmt_ms(vals),
            "F": _fmt_f(f_val),
            "p": _fmt_p(p_val),
            "q": _fmt_p(q_val),
            "posthoc": posthoc,
        }
        for g in GROUP_ORDER:
            row[g] = _fmt_ms(traits.loc[traits["group"] == g, col])
        csv_rows.append(row)
        table_data.append((sec, dim_display, row))

    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(
        data_dir / "trait_group_summary_table.csv", index=False,
    )
    logger.info("Saved trait summary CSV")

    n_cols = 8
    lines: list[str] = [
        r"\begin{tabular}{@{}"
        r" >{\raggedright\arraybackslash}p{3.6cm}"
        r" *{4}{c}"
        r" cc"
        r" c"
        r"@{}}",
        r"\toprule",
        (
            "Variable & Overall & Professional & Amateur & Non-actor"
            f" & $F({df1},{df2})$ & $q$ & Post-hoc \\\\"
        ),
        (
            f" & ($N$={n_total}) & ($n$={ns['Professional']})"
            f" & ($n$={ns['Amateur']}) & ($n$={ns['Non-actor']})"
            " & & & \\\\"
        ),
        r"\midrule",
    ]

    prev_sec: str | None = None
    for sec, dim_display, row in table_data:
        if sec != prev_sec:
            if prev_sec is not None:
                lines.append(r"\addlinespace[0.28ex]")
            lines.append(
                f"\\multicolumn{{{n_cols}}}{{@{{}}l@{{}}}}"
                f"{{\\textit{{{sec}}}}} \\\\"
            )
            prev_sec = sec
        lines.append(
            f"\\quad {_tex_escape(dim_display)}"
            f" & {row['overall']}"
            f" & {row['Professional']}"
            f" & {row['Amateur']}"
            f" & {row['Non-actor']}"
            f" & {row['F']}"
            f" & {row['q']}"
            f" & {row['posthoc']} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_tabular(
        lines,
        tables_dir / "tab_trait_demographics_scales_group_tests_tabular.tex",
    )
