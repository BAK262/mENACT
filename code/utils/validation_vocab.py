"""Manuscript-aligned vocabulary for validation workflows.

Single source of truth for task keys, display labels, group slices, label modes,
and feature-set identifiers. Internal implementation keys only; use TASK_DISPLAY
and GROUP_DISPLAY for reader-facing text.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Tasks (manuscript: Perception, Production, Performance)
# ---------------------------------------------------------------------------
TASK_KEYS: tuple[str, ...] = ("perception", "production", "performance")

TASK_DISPLAY: dict[str, str] = {
    "perception": "Perception",
    "production": "Production",
    "performance": "Performance",
}

# Raw filename prefixes (exp1/exp2/exp3) -> canonical task keys.
LEGACY_TASK_ALIASES: dict[str, str] = {
    "exp1": "perception",
    "exp2": "production",
    "exp3": "performance"
}

# Speech: expressive modes exclude passive perception.
EXPRESSIVE_MODES: tuple[str, ...] = ("production", "performance")

# Cross-mode evaluation direction (manuscript-fixed).
CROSS_MODE_TRAIN: str = "performance"
CROSS_MODE_TEST: str = "production"

# ---------------------------------------------------------------------------
# Classification label modes
# ---------------------------------------------------------------------------
LABEL_MODES: tuple[str, ...] = ("valence3", "emotion9")

# ---------------------------------------------------------------------------
# Acting-experience groups
# ---------------------------------------------------------------------------
GROUP_KEYS: tuple[str, ...] = ("professional", "amateur", "general")

GROUP_DISPLAY: dict[str, str] = {
    "professional": "Professional",
    "amateur": "Amateur",
    "general": "Non-actor",
}

# Subject slices used in cross-mode and similar analyses.
ACTORS_GROUPS: tuple[str, ...] = ("professional", "amateur")
NON_ACTORS_GROUP: str = "general"

# Canonical task key -> raw rating/audio filename prefix (exp1/exp2/exp3).
TASK_KEY_TO_EXP_PREFIX: dict[str, str] = {
    task_key: exp_prefix for exp_prefix, task_key in LEGACY_TASK_ALIASES.items()
}

# ---------------------------------------------------------------------------
# Speech feature backends
# ---------------------------------------------------------------------------
FEATURE_SETS: tuple[str, ...] = ("egemaps", "hubert")
