"""Automated verification tests for Word-Independent and Cross-Speaker splits."""

import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "tess_manifest.csv"
SPLITS_JSON_PATH = PROJECT_ROOT / "data" / "splits" / "word_splits.json"


@pytest.fixture(scope="module")
def manifest_df() -> pd.DataFrame:
    assert MANIFEST_PATH.exists(), f"Manifest file not found at {MANIFEST_PATH}"
    return pd.read_csv(MANIFEST_PATH)


def test_word_splits_metadata_file_exists_and_matches_manifest(manifest_df: pd.DataFrame):
    """Verify saved seed and exact word lists in data/splits/word_splits.json match the manifest."""
    assert SPLITS_JSON_PATH.exists(), f"Split metadata file not found at {SPLITS_JSON_PATH}"
    with open(SPLITS_JSON_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "seed" in meta and isinstance(meta["seed"], int)
    train_words = set(meta["word_indep_split"]["train_words"])
    val_words = set(meta["word_indep_split"]["val_words"])
    test_words = set(meta["word_indep_split"]["test_words"])

    assert len(train_words) == 140
    assert len(val_words) == 30
    assert len(test_words) == 30
    assert train_words.isdisjoint(val_words)
    assert train_words.isdisjoint(test_words)
    assert val_words.isdisjoint(test_words)

    assert set(manifest_df.loc[manifest_df["word_indep_split"] == "train", "word"]) == train_words
    assert set(manifest_df.loc[manifest_df["word_indep_split"] == "val", "word"]) == val_words
    assert set(manifest_df.loc[manifest_df["word_indep_split"] == "test", "word"]) == test_words


def test_word_independent_split_zero_word_leakage_and_counts(manifest_df: pd.DataFrame):
    """Verify 140/30/30 word split has zero word overlap and exact 1960/420/420 sample counts."""
    assert set(manifest_df["word_indep_split"].unique()) == {"train", "val", "test"}

    train_df = manifest_df[manifest_df["word_indep_split"] == "train"]
    val_df = manifest_df[manifest_df["word_indep_split"] == "val"]
    test_df = manifest_df[manifest_df["word_indep_split"] == "test"]

    assert len(train_df) == 140 * 14  # 1,960
    assert len(val_df) == 30 * 14     # 420
    assert len(test_df) == 30 * 14    # 420

    train_words = set(train_df["word"])
    val_words = set(val_df["word"])
    test_words = set(test_df["word"])

    assert len(train_words) == 140
    assert len(val_words) == 30
    assert len(test_words) == 30

    assert len(train_words & val_words) == 0, "Word leakage detected between train and val!"
    assert len(train_words & test_words) == 0, "Word leakage detected between train and test!"
    assert len(val_words & test_words) == 0, "Word leakage detected between val and test!"


def test_word_independent_split_speaker_and_emotion_balance(manifest_df: pd.DataFrame):
    """Verify both speakers and all 7 emotions are equally represented in every partition."""
    expected_per_emotion = {"train": 140 * 2, "val": 30 * 2, "test": 30 * 2}
    expected_per_speaker = {"train": 140 * 7, "val": 30 * 7, "test": 30 * 7}

    for split_name in ["train", "val", "test"]:
        sub = manifest_df[manifest_df["word_indep_split"] == split_name]
        assert set(sub["speaker"].unique()) == {"OAF", "YAF"}
        assert set(sub["emotion"].unique()) == {
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "pleasant_surprise",
            "sad",
        }
        for spk, cnt in sub["speaker"].value_counts().items():
            assert cnt == expected_per_speaker[split_name], (
                f"Split {split_name} speaker {spk} count={cnt}, expected {expected_per_speaker[split_name]}"
            )
        for emo, cnt in sub["emotion"].value_counts().items():
            assert cnt == expected_per_emotion[split_name], (
                f"Split {split_name} emotion {emo} count={cnt}, expected {expected_per_emotion[split_name]}"
            )


def test_cross_speaker_split_zero_speaker_leakage(manifest_df: pd.DataFrame):
    """Verify OAF -> YAF cross-speaker split has zero speaker leakage and disjoint OAF train/val words."""
    assert set(manifest_df["cross_speaker_split"].unique()) == {"train", "val", "test"}

    cs_train = manifest_df[manifest_df["cross_speaker_split"] == "train"]
    cs_val = manifest_df[manifest_df["cross_speaker_split"] == "val"]
    cs_test = manifest_df[manifest_df["cross_speaker_split"] == "test"]

    # Zero speaker leakage: train and val are exclusively OAF; test is exclusively YAF
    assert set(cs_train["speaker"].unique()) == {"OAF"}
    assert set(cs_val["speaker"].unique()) == {"OAF"}
    assert set(cs_test["speaker"].unique()) == {"YAF"}

    # Total counts: OAF train (170 words * 7 = 1190), OAF val (30 words * 7 = 210), YAF test (200 words * 7 = 1400)
    assert len(cs_train) + len(cs_val) == 1400
    assert len(cs_test) == 1400
    assert len(cs_val) == 30 * 7
    assert len(cs_train) == 170 * 7

    # Validation words within OAF are disjoint from training words within OAF
    oaf_train_words = set(cs_train["word"])
    oaf_val_words = set(cs_val["word"])
    assert oaf_train_words.isdisjoint(oaf_val_words)
    assert len(oaf_val_words) == 30
    assert len(oaf_train_words) == 170

    # All 7 emotions are equally represented in OAF train, OAF val, and YAF test
    assert (cs_train["emotion"].value_counts() == 170).all()
    assert (cs_val["emotion"].value_counts() == 30).all()
    assert (cs_test["emotion"].value_counts() == 200).all()
