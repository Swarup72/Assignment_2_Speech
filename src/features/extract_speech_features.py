"""Extract and cache WavLM Base Plus and MFCC+Delta+DeltaDelta speech features.

Features are cached per utterance (`{sample_id}.pt`) under:
- `data/features/wavlm/`  -> shape `(T_wavlm, 768)`, float32
- `data/features/mfcc/`   -> shape `(T_mfcc, 120)`, float32

Design Highlights:
- Frozen `microsoft/wavlm-base-plus` in `eval()` + `torch.inference_mode()`.
- MPS hardware acceleration with automatic CPU fallback if any MPS operator fails.
- Small-batch processing (`batch_size=16`) with exact valid frame length computation
  via `model._get_feat_extract_output_lengths` so zero padding is never saved.
- Checkpointable extraction: skips already-valid cached `.pt` files unless `overwrite=True`.
- Memory-safe: streams mini-batches directly to disk without holding all 2,800 tensors in RAM.
- Configurable CMVN for MFCC features (`apply_cmvn: bool = False` by default so utterance-level
  log-energy c0 and spectral tilt differences are preserved unless CMVN is requested).
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

from src.features.speech_preprocessing import (
    PreprocessedSpeech,
    SpeechPreprocessConfig,
    TARGET_SAMPLE_RATE,
    load_and_preprocess_speech,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "tess_manifest.csv"
DEFAULT_WAVLM_CACHE_DIR = PROJECT_ROOT / "data" / "features" / "wavlm"
DEFAULT_MFCC_CACHE_DIR = PROJECT_ROOT / "data" / "features" / "mfcc"
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"


@dataclass(frozen=True)
class MFCCConfig:
    """Configuration for 120-dim MFCC + Delta + Delta-Delta extraction."""

    sample_rate: int = TARGET_SAMPLE_RATE
    n_mfcc: int = 40
    win_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 80
    fmin: float = 20.0
    fmax: float = 8000.0
    delta_width: int = 9
    apply_cmvn: bool = False
    cmvn_norm_vars: bool = True

    @property
    def win_length(self) -> int:
        return int(round(self.sample_rate * (self.win_length_ms / 1000.0)))  # 400 @ 16kHz

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * (self.hop_length_ms / 1000.0)))  # 160 @ 16kHz

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["win_length_samples"] = self.win_length
        d["hop_length_samples"] = self.hop_length
        d["feature_dim"] = self.n_mfcc * 3
        return d


def apply_utterance_cmvn(
    features: np.ndarray,
    norm_vars: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply per-utterance Cepstral Mean (and optional Variance) Normalization over time axis 0."""
    x = np.asarray(features, dtype=np.float32)
    mean = np.mean(x, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    x = x - mean
    if norm_vars:
        std = np.std(x, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        x = x / np.maximum(std, eps)
    return x.astype(np.float32)


def compute_mfcc_features(
    waveform: np.ndarray,
    config: Optional[MFCCConfig] = None,
) -> np.ndarray:
    """Compute (T, 120) float32 MFCC + Delta + Delta-Delta matrix from a 16 kHz waveform."""
    cfg = config or MFCCConfig()
    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim != 1 or y.size == 0:
        raise ValueError(f"Expected non-empty 1D float32 waveform, got shape {y.shape}")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        center=True,
    ).astype(np.float32)  # (40, T)

    num_frames = mfcc.shape[1]
    width = min(cfg.delta_width, num_frames if num_frames % 2 == 1 else num_frames - 1)
    width = max(width, 3)

    if num_frames >= 3:
        delta1 = librosa.feature.delta(mfcc, width=width, order=1, mode="interp").astype(np.float32)
        delta2 = librosa.feature.delta(mfcc, width=width, order=2, mode="interp").astype(np.float32)
    else:
        delta1 = np.zeros_like(mfcc, dtype=np.float32)
        delta2 = np.zeros_like(mfcc, dtype=np.float32)

    stacked = np.concatenate([mfcc, delta1, delta2], axis=0).T.astype(np.float32)  # (T, 120)

    if cfg.apply_cmvn:
        stacked = apply_utterance_cmvn(stacked, norm_vars=cfg.cmvn_norm_vars)

    if not np.isfinite(stacked).all():
        raise ValueError("Computed MFCC features contain NaN or Inf values.")

    return stacked


def is_valid_cached_feature(pt_path: Path, expected_dim: int) -> bool:
    """Check whether a cached .pt feature file exists and has valid shape and length."""
    if not pt_path.exists() or pt_path.stat().st_size == 0:
        return False
    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        feats = payload.get("features")
        num_frames = payload.get("num_frames")
        if not isinstance(feats, torch.Tensor) or feats.ndim != 2:
            return False
        if feats.shape[1] != expected_dim or feats.shape[0] <= 0:
            return False
        if num_frames != int(feats.shape[0]):
            return False
        return bool(torch.isfinite(feats).all().item())
    except Exception:
        return False


def compute_directory_size_bytes(directory: Path) -> int:
    """Compute total disk size in bytes of all files inside `directory`."""
    if not directory.exists():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Select PyTorch execution device (prefers MPS on Apple Silicon)."""
    if preferred is not None:
        if preferred == "mps" and not torch.backends.mps.is_available():
            return torch.device("cpu")
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_hf_model_revision(model: WavLMModel, model_name: str = WAVLM_MODEL_NAME) -> str:
    """Retrieve the cached commit hash / revision of the HuggingFace model."""
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash:
        return str(commit_hash)
    return "local_cache_verified"


def extract_and_cache_mfcc_dataset(
    manifest_df: pd.DataFrame,
    cache_dir: Union[str, Path] = DEFAULT_MFCC_CACHE_DIR,
    preprocess_config: Optional[SpeechPreprocessConfig] = None,
    mfcc_config: Optional[MFCCConfig] = None,
    project_root: Path = PROJECT_ROOT,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Extract and cache (T, 120) MFCC+Delta+DeltaDelta features for all rows in `manifest_df`."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    pre_cfg = preprocess_config or SpeechPreprocessConfig()
    m_cfg = mfcc_config or MFCCConfig()

    t0 = time.perf_counter()
    num_extracted = 0
    num_skipped = 0
    frame_lengths: List[int] = []

    iterator = manifest_df.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(manifest_df), desc="Extracting MFCC (120-d)")

    for row in iterator:
        sample_id = str(row.sample_id)
        out_file = cache_path / f"{sample_id}.pt"

        if not overwrite and is_valid_cached_feature(out_file, expected_dim=120):
            payload = torch.load(out_file, map_location="cpu", weights_only=False)
            frame_lengths.append(int(payload["num_frames"]))
            num_skipped += 1
            continue

        audio_path = project_root / str(row.filepath)
        pre: PreprocessedSpeech = load_and_preprocess_speech(audio_path, config=pre_cfg)
        mfcc_mat = compute_mfcc_features(pre.waveform, config=m_cfg)  # (T, 120)
        feat_tensor = torch.from_numpy(mfcc_mat).to(dtype=torch.float32)

        num_frames = int(feat_tensor.shape[0])
        frame_lengths.append(num_frames)

        payload = {
            "sample_id": sample_id,
            "features": feat_tensor,
            "num_frames": num_frames,
            "feature_dim": 120,
            "num_samples_16k": pre.valid_samples,
            "native_sample_rate": pre.native_sample_rate,
            "duration_sec": pre.duration_sec,
            "raw_duration_sec": pre.raw_duration_sec,
            "rms_energy": pre.rms_energy,
            "peak_amplitude": pre.peak_amplitude,
            "trim_fallback_used": pre.trim_fallback_used,
            "cmvn_applied": m_cfg.apply_cmvn,
            "label_id": int(row.label_id),
            "emotion": str(row.emotion),
            "speaker": str(row.speaker),
            "word": str(row.word),
        }
        torch.save(payload, out_file)
        num_extracted += 1

    elapsed_sec = time.perf_counter() - t0
    total_cached = len(list(cache_path.glob("*.pt")))
    cache_size_bytes = compute_directory_size_bytes(cache_path)

    metadata = {
        "feature_type": "mfcc_delta_deltadelta",
        "feature_dim": 120,
        "preprocess_config": pre_cfg.to_dict(),
        "mfcc_config": m_cfg.to_dict(),
        "hardware_used": "cpu",
        "cpu_fallback_used": False,
        "num_manifest_samples": len(manifest_df),
        "num_newly_extracted": num_extracted,
        "num_skipped_cached": num_skipped,
        "num_cached_samples": total_cached,
        "min_frames": int(min(frame_lengths)) if frame_lengths else 0,
        "max_frames": int(max(frame_lengths)) if frame_lengths else 0,
        "mean_frames": round(float(np.mean(frame_lengths)), 2) if frame_lengths else 0.0,
        "extraction_time_sec": round(elapsed_sec, 4),
        "cache_size_bytes": cache_size_bytes,
        "cache_size_mb": round(cache_size_bytes / (1024 * 1024), 3),
    }

    with open(cache_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _extract_wavlm_batch(
    model: WavLMModel,
    feature_extractor: Wav2Vec2FeatureExtractor,
    waveforms: List[np.ndarray],
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.device, bool]:
    """Run WavLM Base Plus forward pass on a batch of 16 kHz waveforms with MPS->CPU fallback."""
    inputs = feature_extractor(
        waveforms,
        sampling_rate=TARGET_SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_values = inputs["input_values"]
    attention_mask = inputs["attention_mask"]
    input_lengths = attention_mask.sum(dim=-1)
    output_lengths = model._get_feat_extract_output_lengths(input_lengths).tolist()

    fallback_triggered = False
    active_device = device

    try:
        with torch.inference_mode():
            outputs = model(
                input_values=input_values.to(active_device),
                attention_mask=attention_mask.to(active_device),
            )
            hidden = outputs.last_hidden_state.detach().to("cpu", dtype=torch.float32)
    except RuntimeError as exc:
        if active_device.type == "mps":
            fallback_triggered = True
            active_device = torch.device("cpu")
            model.to(active_device)
            with torch.inference_mode():
                outputs = model(
                    input_values=input_values.to(active_device),
                    attention_mask=attention_mask.to(active_device),
                )
                hidden = outputs.last_hidden_state.detach().to("cpu", dtype=torch.float32)
        else:
            raise exc

    unpadded_tensors: List[torch.Tensor] = []
    for idx, valid_len in enumerate(output_lengths):
        vlen = max(1, min(int(valid_len), int(hidden.shape[1])))
        seq_tensor = hidden[idx, :vlen, :].clone()
        if not torch.isfinite(seq_tensor).all():
            raise ValueError(f"Non-finite WavLM features produced at batch index {idx}")
        unpadded_tensors.append(seq_tensor)

    return unpadded_tensors, active_device, fallback_triggered


def extract_and_cache_wavlm_dataset(
    manifest_df: pd.DataFrame,
    cache_dir: Union[str, Path] = DEFAULT_WAVLM_CACHE_DIR,
    preprocess_config: Optional[SpeechPreprocessConfig] = None,
    model_name: str = WAVLM_MODEL_NAME,
    batch_size: int = 16,
    device: Optional[str] = None,
    project_root: Path = PROJECT_ROOT,
    overwrite: bool = False,
    local_files_only: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Extract and cache (T, 768) WavLM Base Plus features for all rows in `manifest_df`."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    pre_cfg = preprocess_config or SpeechPreprocessConfig()

    exec_device = select_device(device)
    initial_device_str = str(exec_device)

    t0 = time.perf_counter()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = WavLMModel.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model.eval()
    model.requires_grad_(False)
    model.to(exec_device)
    model_revision = get_hf_model_revision(model, model_name=model_name)

    num_extracted = 0
    num_skipped = 0
    cpu_fallback_count = 0
    frame_lengths: List[int] = []

    pending_rows: List[Any] = []
    for row in manifest_df.itertuples(index=False):
        sample_id = str(row.sample_id)
        out_file = cache_path / f"{sample_id}.pt"
        if not overwrite and is_valid_cached_feature(out_file, expected_dim=768):
            payload = torch.load(out_file, map_location="cpu", weights_only=False)
            frame_lengths.append(int(payload["num_frames"]))
            num_skipped += 1
        else:
            pending_rows.append(row)

    num_batches = (len(pending_rows) + batch_size - 1) // batch_size
    batch_iter: Sequence[int] = range(0, len(pending_rows), batch_size)
    if show_progress and pending_rows:
        batch_iter = tqdm(batch_iter, total=num_batches, desc=f"Extracting WavLM ({exec_device})")

    for start_idx in batch_iter:
        batch_rows = pending_rows[start_idx : start_idx + batch_size]
        preprocessed_batch: List[PreprocessedSpeech] = [
            load_and_preprocess_speech(project_root / str(r.filepath), config=pre_cfg)
            for r in batch_rows
        ]
        waveforms = [p.waveform for p in preprocessed_batch]

        batch_tensors, exec_device, fallback_used = _extract_wavlm_batch(
            model=model,
            feature_extractor=feature_extractor,
            waveforms=waveforms,
            device=exec_device,
        )
        if fallback_used:
            cpu_fallback_count += 1

        for row, pre, feat_tensor in zip(batch_rows, preprocessed_batch, batch_tensors):
            sample_id = str(row.sample_id)
            out_file = cache_path / f"{sample_id}.pt"
            num_frames = int(feat_tensor.shape[0])
            frame_lengths.append(num_frames)

            payload = {
                "sample_id": sample_id,
                "features": feat_tensor,
                "num_frames": num_frames,
                "feature_dim": 768,
                "num_samples_16k": pre.valid_samples,
                "native_sample_rate": pre.native_sample_rate,
                "duration_sec": pre.duration_sec,
                "raw_duration_sec": pre.raw_duration_sec,
                "rms_energy": pre.rms_energy,
                "peak_amplitude": pre.peak_amplitude,
                "trim_fallback_used": pre.trim_fallback_used,
                "label_id": int(row.label_id),
                "emotion": str(row.emotion),
                "speaker": str(row.speaker),
                "word": str(row.word),
            }
            torch.save(payload, out_file)
            num_extracted += 1

    if exec_device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    elapsed_sec = time.perf_counter() - t0
    total_cached = len(list(cache_path.glob("*.pt")))
    cache_size_bytes = compute_directory_size_bytes(cache_path)

    metadata = {
        "feature_type": "wavlm_base_plus_last_hidden_state",
        "model_name": model_name,
        "model_revision": model_revision,
        "feature_dim": 768,
        "preprocess_config": pre_cfg.to_dict(),
        "batch_size": batch_size,
        "hardware_requested": initial_device_str,
        "hardware_used": str(exec_device),
        "cpu_fallback_used": cpu_fallback_count > 0,
        "cpu_fallback_count": cpu_fallback_count,
        "num_manifest_samples": len(manifest_df),
        "num_newly_extracted": num_extracted,
        "num_skipped_cached": num_skipped,
        "num_cached_samples": total_cached,
        "min_frames": int(min(frame_lengths)) if frame_lengths else 0,
        "max_frames": int(max(frame_lengths)) if frame_lengths else 0,
        "mean_frames": round(float(np.mean(frame_lengths)), 2) if frame_lengths else 0.0,
        "extraction_time_sec": round(elapsed_sec, 4),
        "cache_size_bytes": cache_size_bytes,
        "cache_size_mb": round(cache_size_bytes / (1024 * 1024), 3),
    }

    with open(cache_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache WavLM and MFCC speech features.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--wavlm-dir", type=Path, default=DEFAULT_WAVLM_CACHE_DIR)
    parser.add_argument("--mfcc-dir", type=Path, default=DEFAULT_MFCC_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--apply-cmvn", action="store_true", help="Apply CMVN during MFCC caching")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--modality", choices=["all", "wavlm", "mfcc"], default="all")
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)
    pre_cfg = SpeechPreprocessConfig()

    if args.modality in ("all", "mfcc"):
        mfcc_cfg = MFCCConfig(apply_cmvn=args.apply_cmvn)
        mfcc_meta = extract_and_cache_mfcc_dataset(
            manifest_df=manifest_df,
            cache_dir=args.mfcc_dir,
            preprocess_config=pre_cfg,
            mfcc_config=mfcc_cfg,
            overwrite=args.overwrite,
        )
        print("[extract_speech_features] MFCC summary:", json.dumps(mfcc_meta, indent=2))

    if args.modality in ("all", "wavlm"):
        wavlm_meta = extract_and_cache_wavlm_dataset(
            manifest_df=manifest_df,
            cache_dir=args.wavlm_dir,
            preprocess_config=pre_cfg,
            batch_size=args.batch_size,
            device=args.device,
            overwrite=args.overwrite,
        )
        print("[extract_speech_features] WavLM summary:", json.dumps(wavlm_meta, indent=2))


if __name__ == "__main__":
    main()
