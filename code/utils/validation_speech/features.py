"""Trial-mean feature manifests for speech validation."""
from __future__ import annotations

import concurrent.futures
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.validation_speech.egemaps_extract import (
    extract_egemaps_any_backend_from_wav,
    parse_smile_feature_csv,
)
from utils.validation_speech.hubert_extract import (
    HUBERT_FRAME_SUFFIX,
    hubert_frame_cache_path,
)
from utils.validation_speech.trials import TrialRecord, emotion_to_valence3
from utils.validation_vocab import FEATURE_SETS, LEGACY_TASK_ALIASES, TASK_DISPLAY

LOGGER = logging.getLogger(__name__)

LEGACY_FEATURE_DIRS_REL = {
    "hubert": Path("data/speech_features/audio_only/hubert_base_encoder_l11_frame"),
}
DEFAULT_SPEECH_CACHE_REL = Path("results/validation_speech/cache")


def resolve_hubert_frame_npy(
    root: Path,
    trial_key: str,
    cache_dir: Path,
) -> Optional[Path]:
    """Cache-first HuBERT frame path; fall back to legacy restricted release dir."""
    cache_path = hubert_frame_cache_path(cache_dir, trial_key)
    if cache_path.is_file():
        return cache_path
    legacy = root / LEGACY_FEATURE_DIRS_REL["hubert"] / f"{trial_key}.npy"
    if legacy.is_file():
        return legacy
    return None


def build_frame_path_map(
    root: Path,
    cache_dir: Path,
    models: Sequence[str],
    trial_keys: Sequence[str],
) -> Dict[str, Dict[str, Path]]:
    """Resolve per-trial frame feature paths (HuBERT only; extensible)."""
    out: Dict[str, Dict[str, Path]] = {}
    if "hubert" in models:
        hubert_map: Dict[str, Path] = {}
        for trial_key in trial_keys:
            path = resolve_hubert_frame_npy(root, trial_key, cache_dir)
            if path is not None:
                hubert_map[trial_key] = path
        out["hubert"] = hubert_map
    return out


def hubert_frames_available(root: Path, cache_dir: Optional[Path] = None) -> bool:
    cache_root = cache_dir or (root / DEFAULT_SPEECH_CACHE_REL)
    legacy = root / LEGACY_FEATURE_DIRS_REL["hubert"]
    if legacy.is_dir() and any(legacy.glob("*.npy")):
        return True
    if cache_root.is_dir() and any(cache_root.glob(f"*/*_{HUBERT_FRAME_SUFFIX}.npy")):
        return True
    return False


def get_code_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_ffmpeg_decode_wav(mp4_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(mp4_path.resolve()),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(wav_path.resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {mp4_path}: {(proc.stderr or proc.stdout)[:1200]}")


def _load_feature_npy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = np.load(path, allow_pickle=True).item()
    emb = np.asarray(obj["embeddings"], dtype=np.float32)
    ts = np.asarray(obj["timestamps"], dtype=np.float32)
    if emb.ndim != 2 or ts.ndim != 2 or ts.shape[1] != 2 or ts.shape[0] != emb.shape[0]:
        raise ValueError(f"Invalid feature payload for {path}")
    return emb, ts


def pool_frames_full_trial(emb: np.ndarray, ts: np.ndarray) -> Optional[np.ndarray]:
    if emb.shape[0] == 0:
        return None
    return emb.mean(axis=0).astype(np.float32)


def load_trial_mean_manifest_csvs(result_dir: Path, models: Sequence[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for m in models:
        fp = result_dir / f"trial_mean_manifest_{m}.csv"
        if not fp.is_file():
            raise FileNotFoundError(f"Missing trial-mean manifest: {fp}")
        df = pd.read_csv(fp)
        if df.empty:
            raise RuntimeError(f"Empty trial-mean manifest: {fp}")
        out[m] = df
        LOGGER.info(
            "Loaded trial-mean manifest | model=%s | rows=%d trials=%d subjects=%d",
            m,
            len(df),
            df["trial_key"].nunique() if "trial_key" in df.columns else -1,
            df["subject_id"].nunique() if "subject_id" in df.columns else -1,
        )
    return out


def _trial_key_set(df: pd.DataFrame, *, model_name: str, source_path: Path) -> set[str]:
    if "trial_key" not in df.columns:
        raise RuntimeError(f"Bad manifest for model={model_name}: missing trial_key column ({source_path})")
    keys = df["trial_key"].astype(str)
    if keys.isna().any():
        raise RuntimeError(f"Bad manifest for model={model_name}: trial_key contains NaN ({source_path})")
    return set(keys.tolist())


def check_and_align_trial_mean_manifests(
    feature_dfs: Dict[str, pd.DataFrame],
    *,
    manifest_dir: Path,
    align_mode: str,
    report_dir: Path,
) -> Dict[str, pd.DataFrame]:
    mode = str(align_mode).strip().lower()
    if mode not in {"strict", "intersection", "none"}:
        raise ValueError(f"Unsupported trial align mode: {align_mode}")
    if mode == "none" or len(feature_dfs) <= 1:
        return feature_dfs

    report_dir.mkdir(parents=True, exist_ok=True)
    model_names = sorted(feature_dfs.keys())
    model_to_path = {m: (manifest_dir / f"trial_mean_manifest_{m}.csv") for m in model_names}

    key_sets: Dict[str, set[str]] = {
        m: _trial_key_set(feature_dfs[m], model_name=m, source_path=model_to_path[m]) for m in model_names
    }
    sizes = {m: len(s) for m, s in key_sets.items()}
    common = set.intersection(*[key_sets[m] for m in model_names]) if model_names else set()

    base = model_names[0]
    base_set = key_sets[base]
    rows: List[Dict[str, object]] = []
    for m in model_names:
        s = key_sets[m]
        rows.append(
            {
                "model": m,
                "n_trials": int(sizes[m]),
                "n_common": int(len(common)),
                "n_missing_vs_common": int(len(common - s)),
                "n_extra_vs_common": int(len(s - common)),
                "n_missing_vs_base": int(len(base_set - s)),
                "n_extra_vs_base": int(len(s - base_set)),
            }
        )
    pd.DataFrame(rows).to_csv(report_dir / "trial_align_report_summary.csv", index=False)

    for m in model_names:
        s = key_sets[m]
        missing_vs_common = sorted(common - s)
        extra_vs_common = sorted(s - common)
        if missing_vs_common:
            (report_dir / f"trial_align_missing_vs_common_{m}.txt").write_text(
                "\n".join(missing_vs_common) + "\n", encoding="utf-8"
            )
        if extra_vs_common:
            (report_dir / f"trial_align_extra_vs_common_{m}.txt").write_text(
                "\n".join(extra_vs_common) + "\n", encoding="utf-8"
            )

    all_equal = all(key_sets[m] == base_set for m in model_names[1:])
    if mode == "strict" and not all_equal:
        msg = (
            "Cross-model trial_key mismatch detected. This invalidates fair model comparison.\n"
            f"Models={model_names}\n"
            f"Counts={sizes}\n"
            f"Common={len(common)}\n"
            f"ReportDir={report_dir}\n"
            "Fix by recomputing manifests, or rerun with --trial-align intersection."
        )
        raise RuntimeError(msg)

    if mode == "intersection":
        if not common:
            raise RuntimeError("Trial intersection is empty; cannot proceed with aligned evaluation.")
        aligned: Dict[str, pd.DataFrame] = {}
        for m in model_names:
            df = feature_dfs[m]
            aligned[m] = df[df["trial_key"].astype(str).isin(common)].copy().reset_index(drop=True)
            LOGGER.info(
                "Aligned manifest to intersection | model=%s | before_trials=%d after_trials=%d",
                m,
                int(df["trial_key"].astype(str).nunique()) if "trial_key" in df.columns else -1,
                int(aligned[m]["trial_key"].astype(str).nunique()) if "trial_key" in aligned[m].columns else -1,
            )
        return aligned

    return feature_dfs


def build_trial_mean_feature_cache(
    root: Path,
    trials: List[TrialRecord],
    models: Sequence[str],
    cache_dir: Path,
    force_recompute: bool,
    trial_workers: int,
) -> Dict[str, pd.DataFrame]:
    """One row per trial: full-trial mean pooling (neural) or full-wav eGeMAPS."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    trial_keys = [t.trial_key for t in trials]
    frame_path_map = build_frame_path_map(root, cache_dir, models, trial_keys)
    needed_frame_models = [m for m in models if m in frame_path_map]
    if needed_frame_models:
        for model_name in needed_frame_models:
            LOGGER.info(
                "Frame cache | model=%s | resolved_trials=%d",
                model_name,
                len(frame_path_map.get(model_name, {})),
            )

    code_dir = get_code_dir()
    rows_by_model: Dict[str, List[Dict[str, object]]] = {m: [] for m in models}

    def _process_trial(t: TrialRecord) -> Dict[str, List[Dict[str, object]]]:
        local_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in models}
        trial_key = t.trial_key
        needed_frame_models = [m for m in models if m in frame_path_map]
        if any(trial_key not in frame_path_map.get(m, {}) for m in needed_frame_models):
            return local_rows
        trial_dir = cache_dir / trial_key
        trial_dir.mkdir(parents=True, exist_ok=True)
        wav_path = trial_dir / f"{trial_key}.wav"

        frame_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for m in needed_frame_models:
            frame_cache[m] = _load_feature_npy(frame_path_map[m][trial_key])

        base = {
            "trial_key": trial_key,
            "subject_id": t.subject_id,
            "exp_prefix": t.exp_prefix,
            "task_key": t.task_key,
            "task": t.task,
            "emotion_9": t.emotion,
            "emotion_3": emotion_to_valence3(t.emotion),
            "segment_id": 0,
            "segment_start_s": 0.0,
            "segment_end_s": 0.0,
        }
        for m in models:
            vec: Optional[np.ndarray]
            if m == "egemaps":
                if force_recompute or not wav_path.exists():
                    _run_ffmpeg_decode_wav(t.audio_path, wav_path)
                out_csv = trial_dir / f"{trial_key}_egemaps_full.csv"
                if force_recompute or not out_csv.exists():
                    try:
                        vec = extract_egemaps_any_backend_from_wav(
                            wav_path=wav_path,
                            code_dir=code_dir,
                            instname=trial_key,
                            out_csv=out_csv,
                        )
                    except Exception:
                        continue
                else:
                    try:
                        vec = parse_smile_feature_csv(out_csv)
                    except Exception:
                        continue
            else:
                emb, ts = frame_cache[m]
                vec = pool_frames_full_trial(emb, ts)
                if vec is None:
                    continue
            row = dict(base)
            for j, v in enumerate(vec.tolist()):
                row[f"f_{j}"] = float(v)
            local_rows[m].append(row)
        return local_rows

    tw = max(1, int(trial_workers))
    tasks_payload = [
        (t.subject_id, t.exp_prefix, t.timestamp, t.emotion, str(t.audio_path)) for t in trials
    ]
    frame_models = [m for m in models if m in frame_path_map]
    egemaps_models = [m for m in models if m == "egemaps"]

    if tw == 1:
        done = 0
        for t in trials:
            out_rows = _process_trial(t)
            for m in models:
                rows_by_model[m].extend(out_rows[m])
            done += 1
            if done % 20 == 0:
                LOGGER.info("Trial-mean feature cache progress: %d trials processed", done)
    else:
        done = 0
        if frame_models:
            frame_workers = max(1, min(8, tw))
            LOGGER.info("Frame-mean workers=%d (capped from trial_workers=%d)", frame_workers, tw)
            with concurrent.futures.ThreadPoolExecutor(max_workers=frame_workers) as ex:
                futures = [
                    ex.submit(_process_trial_frame_only, tp, frame_models, frame_path_map) for tp in tasks_payload
                ]
                for fut in concurrent.futures.as_completed(futures):
                    out_rows = fut.result()
                    for m in frame_models:
                        rows_by_model[m].extend(out_rows[m])
                    done += 1
                    if done % 50 == 0:
                        LOGGER.info("Trial-mean frame-cache progress: %d trials processed", done)

        if egemaps_models:
            import platform as _platform

            smile_root = code_dir / "utils" / "opensmile-3.0-win-x64"
            smile_exe = smile_root / "bin" / "SMILExtract.exe"
            conf = smile_root / "config" / "egemaps" / "v01b" / "eGeMAPSv01b.conf"
            use_process_pool = (
                _platform.system().lower() == "windows"
                and smile_exe.is_file()
                and conf.is_file()
            )
            if not use_process_pool:
                eg_workers = max(1, min(4, tw // 2))
                LOGGER.info(
                    "eGeMAPS extraction workers=%d (threaded; bundled OpenSMILE unavailable)",
                    eg_workers,
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=eg_workers) as ex:
                    futures = [ex.submit(_process_trial, t) for t in trials]
                    done_eg = 0
                    for fut in concurrent.futures.as_completed(futures):
                        out_rows = fut.result()
                        rows_by_model["egemaps"].extend(out_rows.get("egemaps", []))
                        done_eg += 1
                        if done_eg % 20 == 0:
                            LOGGER.info("Trial-mean egemaps progress: %d trials processed", done_eg)
                egemaps_models = []

        if egemaps_models:
            eg_workers = max(1, min(6, tw // 4))
            LOGGER.info("eGeMAPS extraction workers=%d (capped from trial_workers=%d)", eg_workers, tw)
            with concurrent.futures.ProcessPoolExecutor(max_workers=eg_workers) as ex:
                futures = [
                    ex.submit(
                        _trial_mean_egemaps_worker,
                        task_payload=tp,
                        cache_dir=str(cache_dir),
                        force_recompute=bool(force_recompute),
                        smile_exe=str(smile_exe),
                        conf=str(conf),
                    )
                    for tp in tasks_payload
                ]
                done_eg = 0
                for fut in concurrent.futures.as_completed(futures):
                    out_rows = fut.result()
                    rows_by_model["egemaps"].extend(out_rows["egemaps"])
                    done_eg += 1
                    if done_eg % 20 == 0:
                        LOGGER.info("Trial-mean egemaps progress: %d trials processed", done_eg)

    out: Dict[str, pd.DataFrame] = {}
    for m in models:
        df = pd.DataFrame(rows_by_model[m])
        if df.empty:
            raise RuntimeError(f"No trial-mean rows built for model={m}")
        out[m] = df
        LOGGER.info(
            "Trial-mean cache | model=%s | rows=%d trials=%d subjects=%d",
            m,
            len(df),
            df["trial_key"].nunique(),
            df["subject_id"].nunique(),
        )
    return out


def _process_trial_frame_only(
    task_payload: Tuple[int, str, str, str, str],
    models: Sequence[str],
    frame_path_map: Dict[str, Dict[str, Path]],
) -> Dict[str, List[Dict[str, object]]]:
    subject_id, exp_prefix, timestamp, emotion, _audio_path_s = task_payload
    trial_key = f"{subject_id}_{exp_prefix}_{timestamp}_{emotion}"
    local_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in models}
    for m in models:
        if trial_key not in frame_path_map.get(m, {}):
            return local_rows
    task_key = LEGACY_TASK_ALIASES[str(exp_prefix)]
    base = {
        "trial_key": trial_key,
        "subject_id": int(subject_id),
        "exp_prefix": str(exp_prefix),
        "task_key": task_key,
        "task": TASK_DISPLAY[task_key],
        "emotion_9": str(emotion),
        "emotion_3": emotion_to_valence3(str(emotion)),
        "segment_id": 0,
        "segment_start_s": 0.0,
        "segment_end_s": 0.0,
    }
    for m in models:
        emb, ts = _load_feature_npy(frame_path_map[m][trial_key])
        vec = pool_frames_full_trial(emb, ts)
        if vec is None:
            continue
        row = dict(base)
        for j, v in enumerate(vec.tolist()):
            row[f"f_{j}"] = float(v)
        local_rows[m].append(row)
    return local_rows


def _trial_mean_egemaps_worker(
    *,
    task_payload: Tuple[int, str, str, str, str],
    cache_dir: str,
    force_recompute: bool,
    smile_exe: str,
    conf: str,
) -> Dict[str, List[Dict[str, object]]]:
    import platform as _platform

    if _platform.system().lower() != "windows":
        return {"egemaps": []}
    subject_id, exp_prefix, timestamp, emotion, audio_path_s = task_payload
    audio_path = Path(audio_path_s)
    trial_key = f"{subject_id}_{exp_prefix}_{timestamp}_{emotion}"
    local_rows: Dict[str, List[Dict[str, object]]] = {"egemaps": []}

    trial_dir = Path(cache_dir) / trial_key
    trial_dir.mkdir(parents=True, exist_ok=True)
    wav_path = trial_dir / f"{trial_key}.wav"
    if force_recompute or not wav_path.exists():
        _run_ffmpeg_decode_wav(audio_path, wav_path)

    out_csv = trial_dir / f"{trial_key}_egemaps_full.csv"
    if force_recompute or not out_csv.exists():
        try:
            from utils.validation_speech.egemaps_extract import extract_egemaps_from_wav

            extract_egemaps_from_wav(wav_path, Path(smile_exe), Path(conf), instname=trial_key, out_csv=out_csv)
        except Exception:
            return local_rows
    try:
        vec = parse_smile_feature_csv(out_csv)
    except Exception:
        return local_rows

    task_key = LEGACY_TASK_ALIASES[str(exp_prefix)]
    base = {
        "trial_key": trial_key,
        "subject_id": int(subject_id),
        "exp_prefix": str(exp_prefix),
        "task_key": task_key,
        "task": TASK_DISPLAY[task_key],
        "emotion_9": str(emotion),
        "emotion_3": emotion_to_valence3(str(emotion)),
        "segment_id": 0,
        "segment_start_s": 0.0,
        "segment_end_s": 0.0,
    }
    row = dict(base)
    for j, v in enumerate(vec.tolist()):
        row[f"f_{j}"] = float(v)
    local_rows["egemaps"].append(row)
    return local_rows


def validate_feature_sets(models: Sequence[str]) -> List[str]:
    allowed = set(FEATURE_SETS)
    out = [m.lower() for m in models if m.lower() in allowed]
    if not out:
        return list(FEATURE_SETS)
    return out


def filter_available_feature_sets(
    root: Path,
    models: Sequence[str],
    cache_dir: Optional[Path] = None,
) -> List[str]:
    """Drop feature sets whose on-disk frame caches are absent (e.g. HuBERT in open release)."""
    validated = validate_feature_sets(models)
    speech_cache = cache_dir or (root / DEFAULT_SPEECH_CACHE_REL)
    available: List[str] = []
    for model_name in validated:
        if model_name == "hubert":
            if hubert_frames_available(root, speech_cache):
                available.append(model_name)
            else:
                manifest = root / "results/validation_speech/trial_mean_manifest_hubert.csv"
                if manifest.is_file():
                    available.append(model_name)
                    LOGGER.info(
                        "HuBERT frame cache absent; reusing precomputed manifest %s",
                        manifest,
                    )
                else:
                    LOGGER.warning(
                        "Skipping feature set hubert: no frame cache under %s or legacy data/speech_features",
                        speech_cache,
                    )
            continue
        available.append(model_name)
    if not available:
        raise FileNotFoundError(
            "No speech feature sets available. Requested: "
            + ", ".join(validated)
            + ". Provide restricted frame features or pass --feature-sets egemaps."
        )
    return available
