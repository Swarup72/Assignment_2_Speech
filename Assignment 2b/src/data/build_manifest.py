"""Build deduplicated TESS dataset manifest and leakage-free data splits."""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_TESS_DIR = PROJECT_ROOT / "data" / "tess" / "raw"
MANIFEST_CSV_PATH = PROJECT_ROOT / "data" / "tess_manifest.csv"
SPLITS_JSON_PATH = PROJECT_ROOT / "data" / "splits" / "word_splits.json"

SPLIT_SEED = 42

# Consistent 7-class canonical emotion mapping (alphabetical 0..6)
EMOTION_TO_ID: Dict[str, int] = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "pleasant_surprise": 5,
    "sad": 6,
}

# Normalization lookup for raw folder emotion tokens and filename suffixes
RAW_EMOTION_MAP: Dict[str, str] = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "ps": "pleasant_surprise",
    "pleasant_surprise": "pleasant_surprise",
    "pleasant_surprised": "pleasant_surprise",
    "sad": "sad",
}

# Normalization lookup for speaker prefixes (handles OA_bite_neutral.wav edge case)
RAW_SPEAKER_MAP: Dict[str, str] = {
    "oaf": "OAF",
    "oa": "OAF",
    "yaf": "YAF",
}


def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    """Parse speaker and canonical emotion from a TESS folder name (e.g., OAF_Pleasant_surprise)."""
    parts = folder_name.strip().split("_", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Unrecognized TESS folder name format: {folder_name}")
    raw_spk, raw_emo = parts[0].lower(), parts[1].lower()
    if raw_spk not in RAW_SPEAKER_MAP:
        raise ValueError(f"Unrecognized speaker in folder {folder_name}: {raw_spk}")
    if raw_emo not in RAW_EMOTION_MAP:
        raise ValueError(f"Unrecognized emotion in folder {folder_name}: {raw_emo}")
    return RAW_SPEAKER_MAP[raw_spk], RAW_EMOTION_MAP[raw_emo]


def parse_filename(wav_path: Path) -> Tuple[str, str, str]:
    """Parse canonical (speaker, word, emotion) from a TESS .wav filename.

    Handles edge cases such as:
    - 'OA_bite_neutral.wav' (speaker prefix 'OA' instead of 'OAF')
    - '<Speaker>_<word>_ps.wav' ('ps' suffix for 'pleasant_surprise')
    """
    stem = wav_path.stem.strip()
    parts = [p.strip() for p in stem.split("_") if p.strip()]
    if len(parts) < 3:
        raise ValueError(f"Unexpected TESS filename structure: {wav_path.name}")

    raw_spk = parts[0].lower()
    raw_emo = parts[-1].lower()
    word = "_".join(parts[1:-1]).lower()

    if raw_spk not in RAW_SPEAKER_MAP:
        raise ValueError(f"Unknown speaker prefix '{raw_spk}' in file {wav_path}")
    if raw_emo not in RAW_EMOTION_MAP:
        raise ValueError(f"Unknown emotion suffix '{raw_emo}' in file {wav_path}")

    speaker = RAW_SPEAKER_MAP[raw_spk]
    emotion = RAW_EMOTION_MAP[raw_emo]
    return speaker, word, emotion


def discover_and_deduplicate_tess(raw_dir: Path = RAW_TESS_DIR) -> Tuple[List[Dict], Dict]:
    """Discover all TESS .wav files and deduplicate by canonical sample_id without deleting files."""
    all_wavs = sorted(
        raw_dir.rglob("*.wav"),
        key=lambda p: (len(p.relative_to(raw_dir).parts), str(p.relative_to(raw_dir))),
    )
    if not all_wavs:
        raise FileNotFoundError(f"No .wav files found in {raw_dir}")

    retained_by_id: Dict[str, Dict] = {}
    duplicate_count = 0
    speaker_prefix_anomalies: List[str] = []
    non_standard_sr_files: List[str] = []

    for wav_path in all_wavs:
        folder_spk, folder_emo = parse_folder_name(wav_path.parent.name)
        file_spk, word, file_emo = parse_filename(wav_path)

        if wav_path.stem.split("_")[0] != file_spk:
            speaker_prefix_anomalies.append(str(wav_path.relative_to(PROJECT_ROOT)))

        if folder_spk != file_spk or folder_emo != file_emo:
            raise ValueError(
                f"Folder/filename mismatch for {wav_path}: "
                f"folder=({folder_spk}, {folder_emo}) vs file=({file_spk}, {file_emo})"
            )

        sample_id = f"{file_spk}_{word}_{file_emo}"
        rel_path = str(wav_path.relative_to(PROJECT_ROOT))
        file_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()

        if sample_id in retained_by_id:
            duplicate_count += 1
            existing_hash = retained_by_id[sample_id]["_sha256"]
            if existing_hash != file_hash:
                raise ValueError(
                    f"Hash mismatch for duplicate sample_id {sample_id}: "
                    f"{retained_by_id[sample_id]['filepath']} vs {rel_path}"
                )
            continue

        info = sf.info(str(wav_path))
        if info.samplerate != 24414:
            non_standard_sr_files.append(f"{rel_path} ({info.samplerate} Hz)")

        retained_by_id[sample_id] = {
            "sample_id": sample_id,
            "filepath": rel_path,
            "speaker": file_spk,
            "word": word,
            "transcript": f"Say the word {word}",
            "emotion": file_emo,
            "label_id": EMOTION_TO_ID[file_emo],
            "sample_rate": int(info.samplerate),
            "duration_sec": round(float(info.duration), 6),
            "_sha256": file_hash,
        }

    records = sorted(
        retained_by_id.values(),
        key=lambda r: (r["speaker"], r["word"], r["label_id"]),
    )
    for r in records:
        r.pop("_sha256", None)

    stats = {
        "total_raw_wav_files_scanned": len(all_wavs),
        "unique_canonical_recordings_retained": len(records),
        "duplicate_files_skipped": duplicate_count,
        "speaker_prefix_anomalies_normalized": sorted(set(speaker_prefix_anomalies)),
        "non_24414hz_files_detected": non_standard_sr_files,
    }
    return records, stats


def assign_splits(records: List[Dict], seed: int = SPLIT_SEED) -> Tuple[pd.DataFrame, Dict]:
    """Assign deterministic Word-Independent (140/30/30 words) and Cross-Speaker (OAF->YAF) splits."""
    unique_words = sorted({r["word"] for r in records})
    if len(unique_words) != 200:
        raise ValueError(f"Expected exactly 200 unique target words, found {len(unique_words)}")

    rng = random.Random(seed)
    shuffled_words = list(unique_words)
    rng.shuffle(shuffled_words)

    train_words = sorted(shuffled_words[:140])
    val_words = sorted(shuffled_words[140:170])
    test_words = sorted(shuffled_words[170:200])

    word_to_primary_split: Dict[str, str] = {}
    for w in train_words:
        word_to_primary_split[w] = "train"
    for w in val_words:
        word_to_primary_split[w] = "val"
    for w in test_words:
        word_to_primary_split[w] = "test"

    # Cross-speaker OAF -> YAF split:
    # - OAF validation uses the 30 validation words (210 recordings)
    # - OAF training uses the remaining 170 words (1,190 recordings)
    # - YAF testing uses all 200 YAF words (1,400 recordings)
    oaf_val_word_set = set(val_words)
    oaf_train_words = sorted(w for w in unique_words if w not in oaf_val_word_set)

    for r in records:
        w = r["word"]
        spk = r["speaker"]
        r["word_indep_split"] = word_to_primary_split[w]
        if spk == "OAF":
            r["cross_speaker_split"] = "val" if w in oaf_val_word_set else "train"
        elif spk == "YAF":
            r["cross_speaker_split"] = "test"
        else:
            raise ValueError(f"Unexpected speaker: {spk}")

    df = pd.DataFrame(records)[
        [
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
    ]

    split_metadata = {
        "seed": seed,
        "emotion_to_id": EMOTION_TO_ID,
        "word_indep_split": {
            "description": (
                "Word-independent split: 140 train words (1960 samples), "
                "30 val words (420 samples), 30 test words (420 samples). "
                "Zero target word overlap across partitions; both speakers and all 7 emotions in each partition."
            ),
            "counts": {
                "train_words": len(train_words),
                "val_words": len(val_words),
                "test_words": len(test_words),
                "train_samples": int((df["word_indep_split"] == "train").sum()),
                "val_samples": int((df["word_indep_split"] == "val").sum()),
                "test_samples": int((df["word_indep_split"] == "test").sum()),
            },
            "train_words": train_words,
            "val_words": val_words,
            "test_words": test_words,
        },
        "cross_speaker_split": {
            "description": (
                "Cross-speaker generalization split (OAF -> YAF): "
                "OAF is partitioned into 170 train words (1190 samples) and 30 val words (210 samples); "
                "YAF (all 200 words, 1400 samples) is held out exclusively for testing. "
                "Zero speaker overlap between train/val and test; word overlap between OAF and YAF is "
                "intentionally allowed because the objective is cross-speaker acoustic generalization."
            ),
            "counts": {
                "oaf_train_words": len(oaf_train_words),
                "oaf_val_words": len(val_words),
                "yaf_test_words": len(unique_words),
                "train_samples": int((df["cross_speaker_split"] == "train").sum()),
                "val_samples": int((df["cross_speaker_split"] == "val").sum()),
                "test_samples": int((df["cross_speaker_split"] == "test").sum()),
            },
            "oaf_train_words": oaf_train_words,
            "oaf_val_words": val_words,
            "yaf_test_words": unique_words,
        },
    }
    return df, split_metadata


def build_and_save_manifest() -> Tuple[pd.DataFrame, Dict]:
    """Run full discovery, deduplication, split assignment, and save manifest + split metadata."""
    records, dedup_stats = discover_and_deduplicate_tess(RAW_TESS_DIR)
    df, split_meta = assign_splits(records, seed=SPLIT_SEED)
    split_meta["deduplication_stats"] = dedup_stats

    MANIFEST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(MANIFEST_CSV_PATH, index=False)
    with open(SPLITS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print(f"[build_manifest] Saved {len(df)} rows to {MANIFEST_CSV_PATH}")
    print(f"[build_manifest] Saved split metadata to {SPLITS_JSON_PATH}")
    print(f"[build_manifest] Deduplication summary: {json.dumps(dedup_stats, indent=2)}")
    return df, split_meta


if __name__ == "__main__":
    build_and_save_manifest()
