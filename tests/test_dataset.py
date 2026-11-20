"""Automated verification tests for the TESS dataset and data/tess_manifest.csv."""

from pathlib import Path

import pandas as pd
import pytest
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "tess_manifest.csv"

EXPECTED_EMOTIONS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "pleasant_surprise": 5,
    "sad": 6,
}

REQUIRED_COLUMNS = [
    "sample_id",
    "filepath",
    "speaker",
    "word",
    "transcript",
    "emotion",
    "label_id",
    "sample_rate",
    "duration_sec",
    "word_indep_split",
    "cross_speaker_split",
]


@pytest.fixture(scope="module")
def manifest_df() -> pd.DataFrame:
    assert MANIFEST_PATH.exists(), f"Manifest file not found at {MANIFEST_PATH}"
    return pd.read_csv(MANIFEST_PATH)


def test_manifest_columns_and_total_count(manifest_df: pd.DataFrame):
    """Verify required columns and exactly 2,800 unique recordings."""
    assert list(manifest_df.columns) == REQUIRED_COLUMNS
    assert len(manifest_df) == 2800, f"Expected 2800 rows, got {len(manifest_df)}"


def test_no_duplicate_canonical_sample_ids_or_filepaths(manifest_df: pd.DataFrame):
    """Verify no duplicate canonical sample_ids or (speaker, word, emotion) triples."""
    assert manifest_df["sample_id"].nunique() == 2800
    assert manifest_df["filepath"].nunique() == 2800

    triples = manifest_df[["speaker", "word", "emotion"]].drop_duplicates()
    assert len(triples) == 2800


def test_speaker_distribution(manifest_df: pd.DataFrame):
    """Verify 1,400 OAF recordings and 1,400 YAF recordings."""
    counts = manifest_df["speaker"].value_counts().to_dict()
    assert set(counts.keys()) == {"OAF", "YAF"}
    assert counts["OAF"] == 1400
    assert counts["YAF"] == 1400


def test_emotion_distribution_and_label_mapping(manifest_df: pd.DataFrame):
    """Verify 7 canonical emotion categories, 400 recordings per emotion, and consistent label_id."""
    emotion_counts = manifest_df["emotion"].value_counts().to_dict()
    assert set(emotion_counts.keys()) == set(EXPECTED_EMOTIONS.keys())
    for emo, count in emotion_counts.items():
        assert count == 400, f"Expected 400 recordings for {emo}, got {count}"

    for emo, expected_id in EXPECTED_EMOTIONS.items():
        observed_ids = manifest_df.loc[manifest_df["emotion"] == emo, "label_id"].unique()
        assert list(observed_ids) == [expected_id], (
            f"Emotion {emo} mapped to {observed_ids}, expected [{expected_id}]"
        )


def test_unique_words_and_complete_grid(manifest_df: pd.DataFrame):
    """Verify 200 unique target words, each appearing exactly 14 times (2 speakers x 7 emotions)."""
    assert manifest_df["word"].nunique() == 200
    word_counts = manifest_df["word"].value_counts()
    assert (word_counts == 14).all(), (
        f"Every word must have 14 recordings; found min={word_counts.min()}, max={word_counts.max()}"
    )


def test_transcript_construction(manifest_df: pd.DataFrame):
    """Verify every transcript matches 'Say the word {word}'."""
    expected_transcripts = "Say the word " + manifest_df["word"].astype(str)
    assert (manifest_df["transcript"] == expected_transcripts).all()


def test_all_audio_files_readable_and_metadata_valid(manifest_df: pd.DataFrame):
    """Verify all 2,800 audio files exist, are readable by soundfile, and have valid duration and sample rate."""
    assert (manifest_df["sample_rate"] > 0).all()
    assert (manifest_df["duration_sec"] > 0.3).all()
    assert (manifest_df["duration_sec"] < 10.0).all()

    for row in manifest_df.itertuples(index=False):
        audio_path = PROJECT_ROOT / row.filepath
        assert audio_path.exists(), f"Missing audio file: {audio_path}"
        info = sf.info(str(audio_path))
        assert info.samplerate == row.sample_rate
        assert info.frames > 0
        assert info.channels >= 1
