"""
HuBERT layer-11 frame extraction for validation_speech workflow.

Replicates author-side ``speech_feature_extraction`` HuBERT settings used by
``manuscript_analysis_speech``:
  - Model: TencentGameMate/chinese-hubert-base (``model_name=base``)
  - Layer: 11 (0-indexed encoder hidden state)
  - Audio: mono 16 kHz (ffmpeg decode from MP4, same as eGeMAPS cache)
  - Frames: ~50 Hz after CNN downsampling (320x); timestamps [start_s, end_s] per frame
  - Long audio: 30 s bidirectional windows (480000 samples), matching author context_utils

Output ``.npy`` dict (``allow_pickle=True``):
  - ``embeddings``: float32 array (n_frames, 768)
  - ``timestamps``: float array (n_frames, 2)
  - ``metadata``: optional dict (model_name, layer, etc.)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DOWNSAMPLING_FACTOR = 320
OUTPUT_FRAMES_PER_SECOND = SAMPLE_RATE / DOWNSAMPLING_FACTOR
DEFAULT_MAX_AUDIO_CONTEXT_SAMPLES = 30 * SAMPLE_RATE
HUBERT_MODEL_NAME = "base"
HUBERT_MODEL_ID = "TencentGameMate/chinese-hubert-base"
HUBERT_LAYER = 11
HUBERT_FRAME_SUFFIX = f"hubert_l{HUBERT_LAYER}_frame"


def hubert_frame_npy_name(trial_key: str) -> str:
    return f"{trial_key}_{HUBERT_FRAME_SUFFIX}.npy"


def hubert_frame_cache_path(cache_dir: Path, trial_key: str) -> Path:
    return cache_dir / trial_key / hubert_frame_npy_name(trial_key)


def split_with_bidirectional_window(
    sequence_length: int,
    max_window_length: int,
    chunk_size: int,
) -> List[Dict[str, int]]:
    """Match author ``context_utils.split_with_bidirectional_window`` (finite window)."""
    chunks: List[Dict[str, int]] = []
    max_window_length_int = int(max_window_length)
    chunk_size = int(chunk_size)

    for chunk_start in range((max_window_length_int - chunk_size) // 2, sequence_length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, sequence_length)
        chunk_length = chunk_end - chunk_start
        remaining_context = max_window_length_int - chunk_length

        prefix_size = remaining_context // 2
        suffix_size = remaining_context - prefix_size

        window_start = max(0, chunk_start - prefix_size)
        window_end = min(sequence_length, chunk_end + suffix_size)

        chunks.append(
            {
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "window_start": window_start,
                "window_end": window_end,
            }
        )

    if not chunks:
        chunks.append(
            {
                "chunk_start": 0,
                "chunk_end": sequence_length,
                "window_start": 0,
                "window_end": sequence_length,
            }
        )
    return chunks


def run_ffmpeg_decode_wav(mp4_path: Path, wav_path: Path) -> None:
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
            str(SAMPLE_RATE),
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


def load_wav_mono_16k(wav_path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(sr) != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=SAMPLE_RATE)
    return np.asarray(audio, dtype=np.float32)


def load_hubert_model(device: str):
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_ID)
    model = HubertModel.from_pretrained(HUBERT_MODEL_ID)
    model.to(device)
    model.eval()
    return model, processor, device


def extract_hubert_layer11_frames(
    audio: np.ndarray,
    *,
    model,
    processor,
    device: str,
    layer: int = HUBERT_LAYER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [T, D], timestamps [T, 2])."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    audio_tensor = inputs.input_values.to(device).squeeze(0)
    n_samples = int(audio_tensor.shape[0])

    chunks = split_with_bidirectional_window(
        sequence_length=n_samples,
        max_window_length=DEFAULT_MAX_AUDIO_CONTEXT_SAMPLES,
        chunk_size=DEFAULT_MAX_AUDIO_CONTEXT_SAMPLES,
    )

    emb_list: List[np.ndarray] = []
    ts_list: List[Tuple[float, float]] = []

    for chunk_info in chunks:
        chunk_results = _extract_single_window(
            chunk_info,
            audio_tensor=audio_tensor,
            n_samples=n_samples,
            model=model,
            processor=processor,
            device=device,
            layer=layer,
        )
        chunk_start_sample = chunk_info["chunk_start"]
        chunk_end_sample = chunk_info["chunk_end"]
        window_start_sample = chunk_info["window_start"]

        chunk_start_frame = int(chunk_start_sample / DOWNSAMPLING_FACTOR)
        chunk_end_frame = int(chunk_end_sample / DOWNSAMPLING_FACTOR)
        window_start_frame = int(window_start_sample / DOWNSAMPLING_FACTOR)

        chunk_frame_start = chunk_start_frame - window_start_frame
        chunk_frame_end = chunk_end_frame - window_start_frame

        emb = chunk_results["embeddings"][chunk_frame_start:chunk_frame_end]
        ts = chunk_results["timestamps"][chunk_frame_start:chunk_frame_end]
        if emb.shape[0]:
            emb_list.append(emb)
            ts_list.extend(ts)

    if not emb_list:
        raise RuntimeError("HuBERT extraction produced zero frames")

    embeddings = np.concatenate(emb_list, axis=0).astype(np.float32)
    timestamps = np.asarray(ts_list, dtype=np.float64)
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError(f"Invalid timestamps shape: {timestamps.shape}")
    return embeddings, timestamps


def _extract_single_window(
    chunk_info: Dict[str, int],
    *,
    audio_tensor,
    n_samples: int,
    model,
    processor,
    device: str,
    layer: int,
) -> Dict[str, object]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    window_start_sample = max(0, min(chunk_info["window_start"], n_samples))
    window_end_sample = max(0, min(chunk_info["window_end"], n_samples))
    audio_window = audio_tensor[window_start_sample:window_end_sample]
    actual_length = int(audio_window.shape[0])

    batch_audio = pad_sequence([audio_window], batch_first=True, padding_value=0.0)
    window_start_time_sec = window_start_sample / SAMPLE_RATE

    with torch.no_grad():
        extractor_output = model.feature_extractor(batch_audio)
        extractor_output = extractor_output.transpose(1, 2)
        projection_output = model.feature_projection(extractor_output)
        hidden_states = projection_output[0] if isinstance(projection_output, tuple) else projection_output

        actual_output_length = hidden_states.shape[1]
        downsampling_ratio = actual_length / actual_output_length if actual_output_length > 0 else DOWNSAMPLING_FACTOR
        output_len = min(int(actual_length / downsampling_ratio), actual_output_length)

        attention_mask = torch.zeros((1, actual_output_length), dtype=torch.float32, device=device)
        attention_mask[0, :output_len] = 1.0

        x = hidden_states
        remaining_layers = {layer}
        collected = None
        for layer_idx, enc_layer in enumerate(model.encoder.layers):
            x = enc_layer(x, attention_mask=attention_mask)[0]
            if layer_idx in remaining_layers:
                collected = x
                remaining_layers.remove(layer_idx)
            if not remaining_layers:
                break

        if collected is None:
            raise ValueError(f"Layer {layer} not found in HuBERT encoder")

        layer_out = collected[0, :output_len, :].detach().cpu().numpy().astype(np.float32)
        embeddings_list = [layer_out[i] for i in range(layer_out.shape[0])]
        timestamps: List[Tuple[float, float]] = []
        for j in range(output_len):
            start_sec = window_start_time_sec + (j / OUTPUT_FRAMES_PER_SECOND)
            end_sec = window_start_time_sec + ((j + 1) / OUTPUT_FRAMES_PER_SECOND)
            timestamps.append((start_sec, end_sec))

    return {"embeddings": np.stack(embeddings_list), "timestamps": timestamps}


def save_hubert_frame_npy(
    output_path: Path,
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    *,
    trial_key: Optional[str] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "metadata": {
            "model_name": HUBERT_MODEL_NAME,
            "model_id": HUBERT_MODEL_ID,
            "layer": HUBERT_LAYER,
            "time_granularity": "frame",
            "trial_key": trial_key,
        },
    }
    np.save(output_path, payload, allow_pickle=True)


def extract_hubert_frames_from_wav(
    wav_path: Path,
    output_path: Path,
    *,
    device: str = "auto",
    skip_existing: bool = True,
) -> Path:
    if skip_existing and output_path.is_file():
        return output_path

    resolved_device = _resolve_device(device)
    model, processor, resolved_device = load_hubert_model(resolved_device)
    audio = load_wav_mono_16k(wav_path)
    embeddings, timestamps = extract_hubert_layer11_frames(
        audio, model=model, processor=processor, device=resolved_device
    )
    save_hubert_frame_npy(output_path, embeddings, timestamps, trial_key=output_path.stem.split("_hubert")[0])
    return output_path


def extract_hubert_frames_from_mp4(
    mp4_path: Path,
    cache_dir: Path,
    trial_key: str,
    *,
    device: str = "auto",
    skip_existing: bool = True,
    reuse_wav: bool = True,
) -> Path:
    out_npy = hubert_frame_cache_path(cache_dir, trial_key)
    if skip_existing and out_npy.is_file():
        return out_npy

    trial_dir = cache_dir / trial_key
    wav_path = trial_dir / f"{trial_key}.wav"
    if not reuse_wav or not wav_path.is_file():
        run_ffmpeg_decode_wav(mp4_path, wav_path)
    return extract_hubert_frames_from_wav(wav_path, out_npy, device=device, skip_existing=skip_existing)


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HuBERT L11 frame features for validation_speech.")
    parser.add_argument("--mp4", type=Path, help="Single MP4 input")
    parser.add_argument("--wav", type=Path, help="Single WAV input (16 kHz mono preferred)")
    parser.add_argument("--trial-key", type=str, default=None, help="Trial key for cache naming")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/validation_speech/cache"),
        help="Per-trial cache root (same parent as eGeMAPS)",
    )
    parser.add_argument("--device", default="auto", help="cuda:0, cpu, or auto")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .npy")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()
    skip_existing = not bool(args.force)

    if args.wav is not None:
        trial_key = args.trial_key or args.wav.stem
        out_path = hubert_frame_cache_path(args.cache_dir, trial_key)
        extract_hubert_frames_from_wav(args.wav, out_path, device=args.device, skip_existing=skip_existing)
        LOGGER.info("Wrote %s", out_path)
        return

    if args.mp4 is not None:
        trial_key = args.trial_key or args.mp4.stem
        out_path = extract_hubert_frames_from_mp4(
            args.mp4,
            args.cache_dir,
            trial_key,
            device=args.device,
            skip_existing=skip_existing,
        )
        LOGGER.info("Wrote %s", out_path)
        return

    raise SystemExit("Provide --mp4 or --wav")


if __name__ == "__main__":
    main()
