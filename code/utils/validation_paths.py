"""Project and validation result directory helpers."""
from __future__ import annotations

from pathlib import Path

_MARKER_FILES = ("VERSION",)
_MARKER_DIRS = ("data",)


def get_project_root() -> Path:
    """Return project root by locating VERSION file or data/ marker."""
    start = Path(__file__).resolve().parent
    for parent in (start, *start.parents):
        if any((parent / name).is_file() for name in _MARKER_FILES):
            return parent
        if any((parent / name).is_dir() for name in _MARKER_DIRS):
            return parent
    return Path(__file__).resolve().parents[2]


def get_result_dir(stem: str) -> Path:
    """Return ``results/validation_<stem>/`` under the project root."""
    return get_project_root() / "results" / f"validation_{stem}"


def ensure_result_layout(result_dir: Path) -> None:
    """Create standard validation result subdirectories."""
    for sub in ("tables", "figures", "data", "logs"):
        (result_dir / sub).mkdir(parents=True, exist_ok=True)
