"""Load trait questionnaires and ECS similarity matrices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.validation_style import EMOTIONS_9
from utils.validation_vocab import GROUP_DISPLAY, GROUP_KEYS

GROUP_ORDER: list[str] = [GROUP_DISPLAY[k] for k in GROUP_KEYS]

ECS_TAXONOMY: list[str] = EMOTIONS_9

BFI_DOMAINS: list[str] = [
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "negativeEmotionality",
    "openMindedness",
]
QCAE_DIMS: list[str] = [
    "perspectiveTaking",
    "onlineSimulation",
    "emotionContagion",
    "proximalResponsivity",
    "peripheralResponsivity",
]
BEQ_DIMS: list[str] = [
    "positiveExpressivity",
    "negativeExpressivity",
    "negativeInhibition",
    "positiveImpulseStrength",
    "negativeImpulseStrength",
]

INSTRUMENT_ORDER: list[str] = ["BFI-2", "QCAE", "BEQ"]
FAMILY_DIMS: dict[str, list[str]] = {
    "BFI-2": BFI_DOMAINS,
    "QCAE": QCAE_DIMS,
    "BEQ": BEQ_DIMS,
}
ALL_TRAIT_DIMS: list[str] = BFI_DOMAINS + QCAE_DIMS + BEQ_DIMS


def _build_ecs_matrix(fp: Path) -> np.ndarray:
    """Build a 9x9 symmetric similarity matrix from an ECS CSV."""
    dt = pd.read_csv(fp)
    mat = np.full((9, 9), np.nan)
    np.fill_diagonal(mat, 10.0)
    emo_idx = {e: i for i, e in enumerate(ECS_TAXONOMY)}
    for _, row in dt.iterrows():
        a, b = str(row["emotion1"]), str(row["emotion2"])
        if a in emo_idx and b in emo_idx:
            v = float(row["similarity"])
            mat[emo_idx[a], emo_idx[b]] = v
            mat[emo_idx[b], emo_idx[a]] = v
    return mat


def load_data(
    root: Path,
    *,
    n_subjects: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load trait data and ECS matrices for *n_subjects* participants.

    Returns
    -------
    traits : DataFrame with one row per subject (wide format).
    matrices : dict mapping subID (str) -> 9x9 ndarray.
    """
    sub_info = pd.read_csv(root / "data" / "all_raw" / "subject_info.csv")
    if "id" in sub_info.columns:
        sub_info = sub_info.rename(columns={"id": "subID"})
    sub_info["group"] = sub_info["group"].map(GROUP_DISPLAY)

    subject_ids = list(range(1, n_subjects + 1))
    trait_frames: list[pd.DataFrame] = []
    for s in subject_ids:
        p = root / "data" / "all_raw" / str(s)
        bfi = pd.read_csv(p / "trait_bfi2.csv").drop(columns=["id"], errors="ignore")
        qcae = pd.read_csv(p / "trait_qcae.csv").drop(columns=["id"], errors="ignore")
        beq = pd.read_csv(p / "trait_beq.csv").drop(columns=["id"], errors="ignore")
        for df in (bfi, qcae, beq):
            df["subID"] = s
        merged = bfi.merge(qcae, on="subID").merge(beq, on="subID")
        trait_frames.append(merged)

    traits = pd.concat(trait_frames, ignore_index=True)
    traits = traits.merge(sub_info, on="subID", how="left")

    matrices: dict[str, np.ndarray] = {}
    ecs_rows: list[dict[str, float | int]] = []
    for s in subject_ids:
        fp = root / "data" / "all_raw" / str(s) / "trait_ecs.csv"
        mat = _build_ecs_matrix(fp)
        matrices[str(s)] = mat
        upper = mat[np.triu_indices(9, k=1)]
        ecs_rows.append({
            "subID": s,
            "mean_pairwise": float(np.nanmean(upper)),
            "sd_pairwise": float(np.nanstd(upper, ddof=1)),
        })

    traits = traits.merge(pd.DataFrame(ecs_rows), on="subID", how="left")
    return traits, matrices
