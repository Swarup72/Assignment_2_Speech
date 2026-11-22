"""Automated smoke and full-cache verification tests for Step 03 preprocessing & features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.feature_dataset import (
    TESSFeatureDataset,
    create_feature_dataloader,
)
from src.features.extract_speech_features import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MFCC_CACHE_DIR,
    DEFAULT_WAVLM_CACHE_DIR,
    MFCCConfig,
    compute_mfcc_features,
    extract_and_cache_mfcc_dataset,
    extract_and_cache_wavlm_dataset,
    is_valid_cached_feature,
)
from src.features.extract_text_features import (
    DEFAULT_DISTILBERT_CACHE_DIR,
    extract_and_cache_distilbert_dataset,
    is_valid_cached_transcript,
)
from src.features.speech_preprocessing import (
    SpeechPreprocessConfig,
    TARGET_SAMPLE_RATE,
    load_and_preprocess_speech,
    preprocess_Raw_waveform,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest_df() -> pd.DataFrame:
    assert DEFAULT_MANIFEST_PATH.exists(), f"Manifest not found at {DEFAULT_MANIFEST_PATH}"
    return pd.read_csv(DEFAULT_MANIFEST_PATH)


@pytest.fixture(scope="module")
def representative_subset_df(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Select 16 recordings covering both speakers, all 7 emotions, repeated transcripts,
    standard 24,414 Hz files, and the exceptional 96,000 Hz OAF_food_fear recording.
    """
    # All 7 emotions for OAF & YAF on 'back' (14 files: repeated transcript 'Say the word back', 24414 Hz)
    # Plus OAF_food_fear (96000 Hz) and YAF_food_fear (24414 Hz, repeated transcript 'Say the word food')
    subset_ids = [
        "OAF_back_angry",
        "OAF_back_disgust",
        "OAF_back_fear",
        "OAF_back_happy",
        "OAF_back_neutral",
        "OAF_back_pleasant_surprise",
        "OAF_back_sad",
        "YAF_back_angry",
        "YAF_back_disgust",
        "YAF_back_fear",
        "YAF_back_happy",
        "YAF_back_neutral",
        "YAF_back_pleasant_surprise",
        "YAF_back_sad",
        "OAF_food_fear",
        "YAF_food_fear",
    ]
    sub = manifest_df[manifest_df["sample_id"].isin(subset_ids)].copy().reset_index(drop=True)
    assert len(sub) == 16
    assert set(sub["speaker"].unique()) == {"OAF", "YAF"}
    assert len(sub["emotion"].unique()) == 7
    assert set(sub["sample_rate"].unique()) == {24414, 96000}
    return sub


# ============================================================================
# PART 1: SMOKE TESTS ON REPRESENTATIVE SUBSET & EDGE CASES
# ============================================================================


def test_smoke_speech_preprocessing_24414hz_and_96000hz(manifest_df: pd.DataFrame):
    """Verify dynamic resampling to 16 kHz for both 24,414 Hz and 96,000 Hz recordings."""
    row_24k = manifest_df[manifest_df["sample_id"] == "OAF_back_angry"].iloc[0]
    row_96k = manifest_df[manifest_df["sample_id"] == "OAF_food_fear"].iloc[0]

    pre_24k = load_and_preprocess_speech(PROJECT_ROOT / row_24k["filepath"])
    pre_96k = load_and_preprocess_speech(PROJECT_ROOT / row_96k["filepath"])

    assert pre_24k.native_sample_rate == 24414
    assert pre_96k.native_sample_rate == 96000
    assert pre_24k.sample_rate == TARGET_SAMPLE_RATE == 16000
    assert pre_96k.sample_rate == TARGET_SAMPLE_RATE == 16000

    for pre in (pre_24k, pre_96k):
        assert pre.waveform.dtype == np.float32
        assert pre.waveform.ndim == 1
        assert pre.num_samples == pre.valid_samples == len(pre.waveform)
        assert pre.num_samples >= int(0.35 * 16000)
        assert np.isfinite(pre.waveform).all()
        assert abs(float(np.mean(pre.waveform))) < 1e-4
        assert float(np.max(np.abs(pre.waveform))) <= 1.0


def test_smoke_short_recording_fallback_and_loudness_preservation():
    """Verify short-clip silence-trimming fallback and preservation of relative loudness dynamics."""
    sr = 24414
    # Construct a mostly-silent clip with a tiny 50ms burst so aggressive trimming makes it < 0.35s
    short_raw = np.zeros(int(1.0 * sr), dtype=np.float32)
    short_raw[1000:2200] = 0.25 * np.sin(np.linspace(0, 40 * np.pi, 1200, dtype=np.float32))
    pre_short = preprocess_Raw_waveform(short_raw, native_sr=sr)
    assert pre_short.trim_fallback_used is True
    assert pre_short.num_samples >= int(0.35 * TARGET_SAMPLE_RATE)
    assert np.isfinite(pre_short.waveform).all()

    # Verify default normalization ('dc_center_headroom') preserves expressive loudness differences
    t = np.linspace(0, 1.0, TARGET_SAMPLE_RATE, dtype=np.float32)
    quiet_sig = 0.10 * np.sin(2 * np.pi * 220 * t)
    loud_sig = 0.60 * np.sin(2 * np.pi * 220 * t)
    cfg = SpeechPreprocessConfig(trim_silence=False, norm_mode="dc_center_headroom")
    pre_quiet = preprocess_Raw_waveform(quiet_sig, native_sr=TARGET_SAMPLE_RATE, config=cfg)
    pre_loud = preprocess_Raw_waveform(loud_sig, native_sr=TARGET_SAMPLE_RATE, config=cfg)

    rms_quiet = float(np.sqrt(np.mean(np.square(pre_quiet.waveform))))
    rms_loud = float(np.sqrt(np.mean(np.square(pre_loud.waveform))))
    assert rms_loud > 5.0 * rms_quiet, "Expressive loudness difference was destroyed by normalization!"


def test_smoke_mfcc_extraction_and_configurable_cmvn(manifest_df: pd.DataFrame):
    """Verify (T, 120) MFCC+Delta+DeltaDelta extraction and configurable CMVN option."""
    row = manifest_df[manifest_df["sample_id"] == "OAF_food_fear"].iloc[0]
    pre = load_and_preprocess_speech(PROJECT_ROOT / row["filepath"])

    mfcc_raw = compute_mfcc_features(pre.waveform, config=MFCCConfig(apply_cmvn=False))
    mfcc_cmvn = compute_mfcc_features(pre.waveform, config=MFCCConfig(apply_cmvn=True))

    assert mfcc_raw.ndim == 2 and mfcc_raw.shape[1] == 120
    assert mfcc_cmvn.shape == mfcc_raw.shape
    assert np.isfinite(mfcc_raw).all() and np.isfinite(mfcc_cmvn).all()

    # With CMVN enabled, each of the 120 feature channels has zero mean and unit variance over time
    channel_means = np.abs(np.mean(mfcc_cmvn, axis=0))
    channel_stds = np.std(mfcc_cmvn, axis=0)
    assert float(np.max(channel_means)) < 1e-4
    assert np.allclose(channel_stds, 1.0, atol=1e-2)
    # Without CMVN, c0 mean reflects true log-energy (non-zero)
    assert abs(float(np.mean(mfcc_raw[:, 0]))) > 1.0


def test_smoke_subset_caching_and_dataloader_collation(
    representative_subset_df: pd.DataFrame,
    tmp_path: Path,
):
    """Run end-to-end extraction, caching, checkpoint skip, and DataLoader padding on 16-sample subset."""
    wavlm_dir = tmp_path / "wavlm"
    mfcc_dir = tmp_path / "mfcc"
    db_dir = tmp_path / "distilbert"
    sub_manifest_path = tmp_path / "subset_manifest.csv"
    representative_subset_df.to_csv(sub_manifest_path, index=False)

    # 1. Extract MFCC on subset
    mfcc_meta1 = extract_and_cache_mfcc_dataset(
        manifest_df=representative_subset_df,
        cache_dir=mfcc_dir,
        show_progress=False,
    )
    assert mfcc_meta1["num_newly_extracted"] == 16
    assert mfcc_meta1["num_cached_samples"] == 16

    # Checkpoint test: second call must skip all 16 cached files
    mfcc_meta2 = extract_and_cache_mfcc_dataset(
        manifest_df=representative_subset_df,
        cache_dir=mfcc_dir,
        show_progress=False,
    )
    assert mfcc_meta2["num_newly_extracted"] == 0
    assert mfcc_meta2["num_skipped_cached"] == 16

    # 2. Extract WavLM on subset (batch_size=8)
    wavlm_meta1 = extract_and_cache_wavlm_dataset(
        manifest_df=representative_subset_df,
        cache_dir=wavlm_dir,
        batch_size=8,
        show_progress=False,
    )
    assert wavlm_meta1["num_newly_extracted"] == 16
    assert wavlm_meta1["num_cached_samples"] == 16
    assert wavlm_meta1["feature_dim"] == 768

    # Checkpoint test: second call must skip all 16 cached files
    wavlm_meta2 = extract_and_cache_wavlm_dataset(
        manifest_df=representative_subset_df,
        cache_dir=wavlm_dir,
        batch_size=8,
        show_progress=False,
    )
    assert wavlm_meta2["num_newly_extracted"] == 0
    assert wavlm_meta2["num_skipped_cached"] == 16

    # 3. Extract DistilBERT on subset (16 recordings share only 2 unique transcripts!)
    db_meta1 = extract_and_cache_distilbert_dataset(
        manifest_df=representative_subset_df,
        cache_dir=db_dir,
        batch_size=8,
    )
    assert db_meta1["num_manifest_samples"] == 16
    assert db_meta1["num_mapped_samples"] == 16
    assert db_meta1["num_unique_transcripts"] == 2
    assert db_meta1["num_cached_transcript_files"] == 2

    # 4. Verify TESSFeatureDataset & dynamic padding DataLoader
    dataset = TESSFeatureDataset(
        manifest_path=sub_manifest_path,
        split_type="word_indep_split",
        split=None,
        modalities=("wavlm", "mfcc", "distilbert"),
        wavlm_cache_dir=wavlm_dir,
        mfcc_cache_dir=mfcc_dir,
        distilbert_cache_dir=db_dir,
    )
    assert len(dataset) == 16

    loader = create_feature_dataloader(dataset, batch_size=6, shuffle=False)
    batch = next(iter(loader))

    assert len(batch["sample_ids"]) == 6
    assert batch["labels"].shape == (6,)
    assert batch["wavlm_features"].ndim == 3 and batch["wavlm_features"].shape[2] == 768
    assert batch["mfcc_features"].ndim == 3 and batch["mfcc_features"].shape[2] == 120
    assert batch["text_features"].ndim == 3 and batch["text_features"].shape[2] == 768

    # Verify masks and padding zeroes
    for i in range(6):
        w_len = int(batch["wavlm_lengths"][i].item())
        w_max = batch["wavlm_features"].shape[1]
        assert int(batch["wavlm_mask"][i].sum().item()) == w_len
        if w_len < w_max:
            assert torch.all(batch["wavlm_features"][i, w_len:, :] == 0.0)

        m_len = int(batch["mfcc_lengths"][i].item())
        m_max = batch["mfcc_features"].shape[1]
        assert int(batch["mfcc_mask"][i].sum().item()) == m_len
        if m_len < m_max:
            assert torch.all(batch["mfcc_features"][i, m_len:, :] == 0.0)

        t_len = int(batch["text_lengths"][i].item())
        assert int(batch["text_mask"][i].sum().item()) == t_len


# ============================================================================
# PART 2: FULL 2,800-UTTERANCE CACHE INTEGRITY TESTS
# ============================================================================


@pytest.mark.skipif(
    not (DEFAULT_WAVLM_CACHE_DIR / "metadata.json").exists(),
    reason="Full feature cache not yet generated",
)
def test_full_cache_all_2800_entries_and_splits(manifest_df: pd.DataFrame):
    """Verify every single cached entry across all 2,800 utterances for WavLM, MFCC, and DistilBERT."""
    # 1. Verify metadata files
    with open(DEFAULT_WAVLM_CACHE_DIR / "metadata.json", "r", encoding="utf-8") as f:
        wavlm_meta = json.load(f)
    with open(DEFAULT_MFCC_CACHE_DIR / "metadata.json", "r", encoding="utf-8") as f:
        mfcc_meta = json.load(f)
    with open(DEFAULT_DISTILBERT_CACHE_DIR / "metadata.json", "r", encoding="utf-8") as f:
        db_meta = json.load(f)
    with open(DEFAULT_DISTILBERT_CACHE_DIR / "sample_to_transcript.json", "r", encoding="utf-8") as f:
        sample_to_transcript = json.load(f)

    assert wavlm_meta["num_cached_samples"] == 2800
    assert mfcc_meta["num_cached_samples"] == 2800
    assert db_meta["num_unique_transcripts"] == 200
    assert db_meta["num_cached_transcript_files"] == 200
    assert db_meta["num_mapped_samples"] == 2800
    assert len(sample_to_transcript) == 2800

    # 2. Verify all 200 unique transcript embeddings
    transcript_files = sorted((DEFAULT_DISTILBERT_CACHE_DIR / "transcripts").glob("*.pt"))
    assert len(transcript_files) == 200
    for pt_file in transcript_files:
        assert is_valid_cached_transcript(pt_file, expected_dim=768), f"Invalid transcript cache: {pt_file}"

    # 3. Verify all 2,800 WavLM and MFCC feature files and label consistency
    for row in manifest_df.itertuples(index=False):
        sid = str(row.sample_id)
        wavlm_pt = DEFAULT_WAVLM_CACHE_DIR / f"{sid}.pt"
        mfcc_pt = DEFAULT_MFCC_CACHE_DIR / f"{sid}.pt"

        assert is_valid_cached_feature(wavlm_pt, expected_dim=768), f"Invalid WavLM cache for {sid}"
        assert is_valid_cached_feature(mfcc_pt, expected_dim=120), f"Invalid MFCC cache for {sid}"
        assert sid in sample_to_transcript
        assert sample_to_transcript[sid]["label_id"] == int(row.label_id)
        assert sample_to_transcript[sid]["transcript"] == str(row.transcript)

    # 4. Verify TESSFeatureDataset across both split protocols and all partitions
    expected_counts = {
        "word_indep_split": {"train": 1960, "val": 420, "test": 420},
        "cross_speaker_split": {"train": 1190, "val": 210, "test": 1400},
    }
    for split_type, part_map in expected_counts.items():
        for part, exp_len in part_map.items():
            ds = TESSFeatureDataset(
                split_type=split_type,
                split=part,
                modalities=("wavlm", "mfcc", "distilbert"),
                apply_mfcc_cmvn=True,
            )
            assert len(ds) == exp_len
            loader = create_feature_dataloader(ds, batch_size=16, shuffle=False)
            batch = next(iter(loader))
            assert batch["wavlm_features"].shape[0] == 16
            assert batch["wavlm_features"].shape[2] == 768
            assert batch["mfcc_features"].shape[0] == 16
            assert batch["mfcc_features"].shape[2] == 120
            assert batch["text_features"].shape[0] == 16
            assert batch["text_features"].shape[2] == 768
            assert torch.isfinite(batch["wavlm_features"]).all()
            assert torch.isfinite(batch["mfcc_features"]).all()
            assert torch.isfinite(batch["text_features"]).all()
