"""Speech preprocessing utilities for TESS emotion recognition.

Design Principles:
1. Never assume a fixed native sampling rate (TESS contains 2,799 files at 24,414 Hz
   and 1 file, OAF_food_fear.wav, at 96,000 Hz).
2. Convert to mono float32 and dynamically resample to 16,000 Hz.
3. Trim leading and trailing silence using librosa.effects.trim with automatic
   fallback if an utterance becomes shorter than min_duration_sec.
4. Preserve expressive loudness differences across utterances:
   - Default normalization ('dc_center_headroom') removes DC offset (zero-centering)
     and applies non-destructive headroom protection (scaling down only if peak > 0.999
     due to resampling overshoot). This aligns with microsoft/wavlm-base-plus
     (preprocessor_config.json has do_normalize=False) and retains vocal intensity cues.
   - Optional 'zscore', 'peak', and 'none' modes are also supported for ablations.
5. Preserve exact valid sequence lengths; pad or crop only when explicitly requested.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeechPreprocessConfig:
    """Configuration for speech waveform preprocessing."""

    target_sr: int = TARGET_SAMPLE_RATE
    trim_silence: bool = True
    top_db: float = 30.0
    relaxed_top_db: float = 45.0
    trim_frame_length: int = 512
    trim_hop_length: int = 128
    min_duration_sec: float = 0.35
    norm_mode: str = "dc_center_headroom"  # 'dc_center_headroom', 'zscore', 'peak', 'none'
    headroom_peak: float = 0.999
    max_duration_sec: Optional[float] = None
    pad_to_max: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessedSpeech:
    """Container for a preprocessed speech waveform and its diagnostic metadata."""

    waveform: np.ndarray
    sample_rate: int
    native_sample_rate: int
    num_samples: int
    valid_samples: int
    duration_sec: float
    raw_duration_sec: float
    rms_energy: float
    peak_amplitude: float
    trim_fallback_used: bool
    norm_mode: str


def normalize_waveform(
    waveform: np.ndarray,
    norm_mode: str = "dc_center_headroom",
    headroom_peak: float = 0.999,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize a 1D float32 waveform while preserving expressive dynamics by default.

    Modes:
    - 'dc_center_headroom' (default): Removes DC offset (mean = 0) and preserves the
      true relative amplitude scale across recordings, scaling down only if peak
      amplitude exceeds `headroom_peak` (e.g., from resampling filter overshoot).
    - 'zscore': Per-utterance zero-mean unit-variance normalization.
    - 'peak': Removes DC offset and scales peak amplitude to `headroom_peak`.
    - 'none': Returns waveform unchanged as float32.
    """
    y = np.asarray(waveform, dtype=np.float32)
    if y.size == 0:
        return y

    if norm_mode == "none":
        return y

    # Always remove DC offset using float64 accumulator for numerical precision
    dc_offset = float(np.mean(y, dtype=np.float64))
    y = (y - dc_offset).astype(np.float32)

    if norm_mode == "dc_center_headroom":
        peak = float(np.max(np.abs(y)))
        if peak > headroom_peak:
            y = (y * (headroom_peak / (peak + eps))).astype(np.float32)
        return y

    if norm_mode == "peak":
        peak = float(np.max(np.abs(y)))
        if peak > eps:
            y = (y * (headroom_peak / peak)).astype(np.float32)
        return y

    if norm_mode == "zscore":
        std = float(np.std(y, dtype=np.float64))
        if std > eps:
            y = (y / std).astype(np.float32)
        return y

    raise ValueError(
        f"Unsupported norm_mode='{norm_mode}'. Expected one of "
        "['dc_center_headroom', 'zscore', 'peak', 'none']."
    )


def preprocess_Raw_waveform(
    raw_waveform: np.ndarray,
    native_sr: int,
    config: Optional[SpeechPreprocessConfig] = None,
) -> PreprocessedSpeech:
    """Preprocess an in-memory waveform sampled at `native_sr` according to `config`."""
    cfg = config or SpeechPreprocessConfig()
    if native_sr <= 0:
        raise ValueError(f"native_sr must be positive, got {native_sr}")

    y = np.asarray(raw_waveform, dtype=np.float32)
    if y.ndim == 2:
        # Convert multi-channel audio (frames, channels) or (channels, frames) to mono
        if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
            y = np.mean(y, axis=0, dtype=np.float32)
        else:
            y = np.mean(y, axis=-1, dtype=np.float32)
    elif y.ndim != 1:
        raise ValueError(f"Expected 1D or 2D audio array, got shape {y.shape}")

    if y.size == 0:
        raise ValueError("Loaded waveform is empty (0 samples).")

    raw_duration_sec = float(y.shape[0] / float(native_sr))

    # Dynamic resampling to target_sr (16,000 Hz)
    if int(native_sr) != int(cfg.target_sr):
        y = librosa.resample(
            y,
            orig_sr=int(native_sr),
            target_sr=int(cfg.target_sr),
            res_type="soxr_hq",
        ).astype(np.float32)

    min_samples = max(int(round(cfg.min_duration_sec * cfg.target_sr)), 400)
    trim_fallback_used = False

    if cfg.trim_silence:
        trimmed, _ = librosa.effects.trim(
            y,
            top_db=cfg.top_db,
            frame_length=cfg.trim_frame_length,
            hop_length=cfg.trim_hop_length,
        )
        if trimmed.shape[0] < min_samples:
            # Try a gentler silence threshold first
            relaxed, _ = librosa.effects.trim(
                y,
                top_db=cfg.relaxed_top_db,
                frame_length=cfg.trim_frame_length,
                hop_length=cfg.trim_hop_length,
            )
            if relaxed.shape[0] >= min_samples:
                y = relaxed.astype(np.float32)
                trim_fallback_used = True
            else:
                # Retain untrimmed resampled signal
                trim_fallback_used = True
        else:
            y = trimmed.astype(np.float32)

    # Guard against degenerate ultra-short synthetic clips so CNN extractors never fail
    if y.shape[0] < min_samples:
        trim_fallback_used = True
        pad_len = min_samples - y.shape[0]
        y = np.pad(y, (0, pad_len), mode="constant", constant_values=0.0).astype(np.float32)

    # Record pre-normalization amplitude diagnostics
    rms_energy = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    peak_amplitude = float(np.max(np.abs(y)))

    # Apply documented amplitude normalization
    y = normalize_waveform(y, norm_mode=cfg.norm_mode, headroom_peak=cfg.headroom_peak)

    valid_samples = int(y.shape[0])

    # Optional model-specific cropping / fixed-length padding (disabled by default)
    if cfg.max_duration_sec is not None:
        max_samples = int(round(cfg.max_duration_sec * cfg.target_sr))
        if y.shape[0] > max_samples:
            y = y[:max_samples].astype(np.float32)
            valid_samples = int(max_samples)
        elif cfg.pad_to_max and y.shape[0] < max_samples:
            pad_width = max_samples - y.shape[0]
            y = np.pad(y, (0, pad_width), mode="constant", constant_values=0.0).astype(np.float32)

    if not np.isfinite(y).all():
        raise ValueError("Preprocessed waveform contains NaN or Inf values.")

    num_samples = int(y.shape[0])
    duration_sec = float(valid_samples / float(cfg.target_sr))

    return PreprocessedSpeech(
        waveform=y,
        sample_rate=int(cfg.target_sr),
        native_sample_rate=int(native_sr),
        num_samples=num_samples,
        valid_samples=valid_samples,
        duration_sec=round(duration_sec, 6),
        raw_duration_sec=round(raw_duration_sec, 6),
        rms_energy=round(rms_energy, 6),
        peak_amplitude=round(peak_amplitude, 6),
        trim_fallback_used=trim_fallback_used,
        norm_mode=cfg.norm_mode,
    )


def load_and_preprocess_speech(
    filepath: Union[str, Path],
    config: Optional[SpeechPreprocessConfig] = None,
) -> PreprocessedSpeech:
    """Load an audio file at its actual native sampling rate and preprocess it."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    raw_waveform, native_sr = sf.read(str(path), dtype="float32", always_2d=False)
    return preprocess_Raw_waveform(raw_waveform, native_sr=int(native_sr), config=config)
