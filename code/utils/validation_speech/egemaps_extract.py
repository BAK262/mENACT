"""
OpenSMILE eGeMAPS (v01b) extraction for validation_speech workflow only.

Expects bundled OpenSMILE under code/utils/opensmile-3.0-win-x64 on Windows.
"""

from __future__ import annotations

import io
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def python_opensmile_available() -> bool:
    try:
        import opensmile  # noqa: F401

        return True
    except Exception:
        return False


def opensmile_paths(code_dir: Path) -> Tuple[Path, Path]:
    """Return (SMILExtract.exe, config root)."""
    root = code_dir / "utils" / "opensmile-3.0-win-x64"
    exe = root / "bin" / "SMILExtract.exe"
    cfg = root / "config"
    return exe, cfg


def egemaps_config_file(config_root: Path) -> Path:
    return config_root / "egemaps" / "v01b" / "eGeMAPSv01b.conf"


def extract_egemaps_from_wav(
    wav_path: Path,
    smile_exe: Path,
    config_file: Path,
    instname: str,
    out_csv: Path,
) -> None:
    """Run SMILExtract on a single WAV; write features to out_csv."""
    if platform.system().lower() != "windows":
        raise RuntimeError("OpenSMILE eGeMAPS extraction is only supported on Windows in this workflow.")
    if not smile_exe.is_file():
        raise FileNotFoundError(f"OpenSMILE executable not found: {smile_exe}")
    if not config_file.is_file():
        raise FileNotFoundError(f"eGeMAPS config not found: {config_file}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    opts = [
        "-configfile",
        str(config_file),
        "-appendcsvlld",
        "0",
        "-timestampcsvlld",
        "1",
        "-headercsvlld",
        "1",
        "-inputfile",
        str(wav_path.resolve()),
        "-csvoutput",
        str(out_csv.resolve()),
        "-instname",
        instname,
    ]
    cmd = [str(smile_exe.resolve())] + opts
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(out_csv.parent),
        timeout=90,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"SMILExtract failed (code {proc.returncode}): {proc.stderr[:2000] or proc.stdout[:2000]}"
        )


def extract_egemaps_from_wav_python(wav_path: Path) -> np.ndarray:
    """Extract eGeMAPS v01b functionals via python-opensmile."""
    try:
        import opensmile
    except Exception as exc:
        raise RuntimeError(
            "python-opensmile not available. Install `opensmile` in ee_validation_speech environment."
        ) from exc

    if not wav_path.is_file():
        raise FileNotFoundError(str(wav_path))

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    df = smile.process_file(str(wav_path))
    if df is None or df.empty:
        raise RuntimeError(f"python-opensmile produced empty dataframe for {wav_path}")

    vec = df.to_numpy(dtype=np.float64).reshape(-1)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec.astype(np.float32)


def extract_egemaps_any_backend_from_wav(
    *,
    wav_path: Path,
    code_dir: Path,
    instname: str,
    out_csv: Optional[Path] = None,
) -> np.ndarray:
    """
    Backend selector:
    - Windows: prefer bundled OpenSMILE executable if present; fall back to python-opensmile.
    - Non-Windows: use python-opensmile (required).
    """
    sys = platform.system().lower()
    if sys != "windows":
        return extract_egemaps_from_wav_python(wav_path)

    smile_exe, cfg_root = opensmile_paths(code_dir)
    conf = egemaps_config_file(cfg_root)
    if smile_exe.is_file() and conf.is_file():
        if out_csv is None:
            raise ValueError("out_csv must be provided when using the OpenSMILE executable backend.")
        extract_egemaps_from_wav(wav_path, smile_exe, conf, instname=instname, out_csv=out_csv)
        return parse_smile_feature_csv(out_csv)

    if python_opensmile_available():
        return extract_egemaps_from_wav_python(wav_path)
    raise FileNotFoundError(f"Missing OpenSMILE executable/config, and python-opensmile not installed: {smile_exe}")


def parse_smile_feature_csv(csv_path: Path) -> np.ndarray:
    """Parse OpenSMILE CSV; return a 1-D float32 vector (mean over frames if multiple rows)."""
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))
    lines: list[str] = []
    for line in csv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("@"):
            continue
        lines.append(line)
    if not lines:
        raise ValueError(f"No data lines in {csv_path}")

    buf = "\n".join(lines)
    try:
        df = pd.read_csv(io.StringIO(buf), sep=";", engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(
            io.StringIO(buf),
            sep=";",
            engine="python",
            error_bad_lines=False,
            warn_bad_lines=False,
        )
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        try:
            df = pd.read_csv(io.StringIO(buf), sep=",", engine="python", on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(
                io.StringIO(buf),
                sep=",",
                engine="python",
                error_bad_lines=False,
                warn_bad_lines=False,
            )
        numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric columns parsed from {csv_path}")
    vec = numeric.to_numpy(dtype=np.float64)
    pooled = np.nanmean(vec, axis=0)
    pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
    return pooled.astype(np.float32)
