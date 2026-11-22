"""Unified PyTorch Dataset and dynamic padding collate_fn for cached TESS features.

Supports:
- Split protocols: `word_indep_split` and `cross_speaker_split`
- Partition filtering: `'train'`, `'val'`, `'test'`, or `None` (all rows)
- Configurable modalities: any subset of `('wavlm', 'mfcc', 'distilbert')`
- Optional on-the-fly CMVN for MFCC baseline features (`apply_mfcc_cmvn: bool = False`)
- Lazy disk loading per sample (`__getitem__`) so all 2,800 feature tensors are never
  loaded into RAM simultaneously.
- Dynamic mini-batch padding (`tess_feature_collate_fn`) producing padded feature tensors,
  exact sequence length tensors `(B,)`, boolean/integer attention masks `(B, T_max)`,
  emotion labels `(B,)`, and rich sample metadata (`sample_ids`, `speakers`, `words`,
  `emotions`, `transcripts`).
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.features.extract_speech_features import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MFCC_CACHE_DIR,
    DEFAULT_WAVLM_CACHE_DIR,
    apply_utterance_cmvn,
)
from src.features.extract_text_features import (
    DEFAULT_DISTILBERT_CACHE_DIR,
    transcript_to_cache_id,
)


VALID_SPLIT_COLUMNS = ("word_indep_split", "cross_speaker_split")
VALID_PARTITIONS = ("train", "val", "test")
VALID_MODALITIES = ("wavlm", "mfcc", "distilbert")


def build_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Construct a (B, T_max) boolean validity mask where True = valid token/frame, False = padding."""
    if lengths.ndim != 1:
        raise ValueError(f"Expected 1D lengths tensor, got shape {tuple(lengths.shape)}")
    t_max = int(max_len if max_len is not None else lengths.max().item())
    arange = torch.arange(t_max, device=lengths.device).unsqueeze(0)  # (1, T_max)
    return arange < lengths.unsqueeze(1)  # (B, T_max) bool


class TESSFeatureDataset(Dataset):
    """Unified lazy-loading PyTorch Dataset for cached TESS speech and text features."""

    def __init__(
        self,
        manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
        split_type: str = "word_indep_split",
        split: Optional[str] = "train",
        modalities: Sequence[str] = ("wavlm", "mfcc", "distilbert"),
        wavlm_cache_dir: Union[str, Path] = DEFAULT_WAVLM_CACHE_DIR,
        mfcc_cache_dir: Union[str, Path] = DEFAULT_MFCC_CACHE_DIR,
        distilbert_cache_dir: Union[str, Path] = DEFAULT_DISTILBERT_CACHE_DIR,
        apply_mfcc_cmvn: bool = False,
        manifest_df: Optional[pd.DataFrame] = None,
    ) -> None:
        super().__init__()
        if split_type not in VALID_SPLIT_COLUMNS:
            raise ValueError(
                f"Invalid split_type='{split_type}'. Must be one of {VALID_SPLIT_COLUMNS}."
            )
        if split is not None and split not in VALID_PARTITIONS:
            raise ValueError(
                f"Invalid split='{split}'. Must be one of {VALID_PARTITIONS} or None."
            )
        if not modalities:
            raise ValueError("At least one modality must be requested.")
        for mod in modalities:
            if mod not in VALID_MODALITIES:
                raise ValueError(
                    f"Unsupported modality='{mod}'. Expected subset of {VALID_MODALITIES}."
                )

        self.manifest_path = Path(manifest_path)
        self.split_type = split_type
        self.split = split
        self.modalities = tuple(dict.fromkeys(modalities))
        self.wavlm_cache_dir = Path(wavlm_cache_dir)
        self.mfcc_cache_dir = Path(mfcc_cache_dir)
        self.distilbert_cache_dir = Path(distilbert_cache_dir)
        self.apply_mfcc_cmvn = apply_mfcc_cmvn

        df = manifest_df.copy() if manifest_df is not None else pd.read_csv(self.manifest_path)
        if split is not None:
            df = df[df[self.split_type] == split].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No samples found for split_type='{split_type}', split='{split}'."
            )

        self.df = df

        # Load lightweight sample_id -> transcript mapping JSON if distilbert is enabled
        self.sample_to_transcript: Dict[str, Dict[str, Any]] = {}
        if "distilbert" in self.modalities:
            mapping_file = self.distilbert_cache_dir / "sample_to_transcript.json"
            if mapping_file.exists():
                with open(mapping_file, "r", encoding="utf-8") as f:
                    self.sample_to_transcript = json.load(f)

    def __len__(self) -> int:
        return len(self.df)

    def _load_wavlm(self, sample_id: str) -> Tuple[torch.Tensor, int]:
        pt_path = self.wavlm_cache_dir / f"{sample_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Cached WavLM feature file missing: {pt_path}")
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        feats: torch.Tensor = payload["features"].to(dtype=torch.float32)
        num_frames = int(payload["num_frames"])
        return feats, num_frames

    def _load_mfcc(self, sample_id: str) -> Tuple[torch.Tensor, int]:
        pt_path = self.mfcc_cache_dir / f"{sample_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Cached MFCC feature file missing: {pt_path}")
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        feats: torch.Tensor = payload["features"].to(dtype=torch.float32)
        if self.apply_mfcc_cmvn and not payload.get("cmvn_applied", False):
            cmvn_arr = apply_utterance_cmvn(feats.numpy(), norm_vars=True)
            feats = torch.from_numpy(cmvn_arr).to(dtype=torch.float32)
        num_frames = int(payload["num_frames"])
        return feats, num_frames

    def _load_distilbert(self, sample_id: str, transcript: str) -> Dict[str, Any]:
        if sample_id in self.sample_to_transcript:
            rel_file = self.sample_to_transcript[sample_id]["feature_file"]
            pt_path = self.distilbert_cache_dir / rel_file
        else:
            t_id = transcript_to_cache_id(transcript)
            pt_path = self.distilbert_cache_dir / "transcripts" / f"{t_id}.pt"

        if not pt_path.exists():
            raise FileNotFoundError(f"Cached DistilBERT feature file missing: {pt_path}")
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        return {
            "features": payload["features"].to(dtype=torch.float32),
            "attention_mask": payload["attention_mask"].to(dtype=torch.long),
            "input_ids": payload["input_ids"].to(dtype=torch.long),
            "num_tokens": int(payload["num_tokens"]),
            "transcript_id": str(payload["transcript_id"]),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        sample_id = str(row["sample_id"])
        transcript = str(row["transcript"])

        item: Dict[str, Any] = {
            "sample_id": sample_id,
            "label_id": int(row["label_id"]),
            "emotion": str(row["emotion"]),
            "speaker": str(row["speaker"]),
            "word": str(row["word"]),
            "transcript": transcript,
            "word_indep_split": str(row["word_indep_split"]),
            "cross_speaker_split": str(row["cross_speaker_split"]),
        }

        if "wavlm" in self.modalities:
            wavlm_feats, wavlm_len = self._load_wavlm(sample_id)
            item["wavlm_features"] = wavlm_feats
            item["wavlm_length"] = wavlm_len

        if "mfcc" in self.modalities:
            mfcc_feats, mfcc_len = self._load_mfcc(sample_id)
            item["mfcc_features"] = mfcc_feats
            item["mfcc_length"] = mfcc_len

        if "distilbert" in self.modalities:
            db_data = self._load_distilbert(sample_id, transcript)
            item["text_features"] = db_data["features"]
            item["text_attention_mask"] = db_data["attention_mask"]
            item["text_input_ids"] = db_data["input_ids"]
            item["text_length"] = db_data["num_tokens"]
            item["transcript_id"] = db_data["transcript_id"]

        return item


def tess_feature_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dynamically pad variable-length speech and text features within a mini-batch."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    collated: Dict[str, Any] = {
        "sample_ids": [b["sample_id"] for b in batch],
        "labels": torch.tensor([b["label_id"] for b in batch], dtype=torch.long),
        "emotions": [b["emotion"] for b in batch],
        "speakers": [b["speaker"] for b in batch],
        "words": [b["word"] for b in batch],
        "transcripts": [b["transcript"] for b in batch],
        "word_indep_splits": [b["word_indep_split"] for b in batch],
        "cross_speaker_splits": [b["cross_speaker_split"] for b in batch],
    }

    if "wavlm_features" in batch[0]:
        wavlm_seqs = [b["wavlm_features"] for b in batch]
        wavlm_lengths = torch.tensor([b["wavlm_length"] for b in batch], dtype=torch.long)
        wavlm_padded = pad_sequence(wavlm_seqs, batch_first=True, padding_value=0.0)
        wavlm_mask = build_padding_mask(wavlm_lengths, max_len=wavlm_padded.shape[1])
        collated["wavlm_features"] = wavlm_padded
        collated["wavlm_lengths"] = wavlm_lengths
        collated["wavlm_mask"] = wavlm_mask
        collated["wavlm_attention_mask"] = wavlm_mask.to(dtype=torch.long)

    if "mfcc_features" in batch[0]:
        mfcc_seqs = [b["mfcc_features"] for b in batch]
        mfcc_lengths = torch.tensor([b["mfcc_length"] for b in batch], dtype=torch.long)
        mfcc_padded = pad_sequence(mfcc_seqs, batch_first=True, padding_value=0.0)
        mfcc_mask = build_padding_mask(mfcc_lengths, max_len=mfcc_padded.shape[1])
        collated["mfcc_features"] = mfcc_padded
        collated["mfcc_lengths"] = mfcc_lengths
        collated["mfcc_mask"] = mfcc_mask
        collated["mfcc_attention_mask"] = mfcc_mask.to(dtype=torch.long)

    if "text_features" in batch[0]:
        text_seqs = [b["text_features"] for b in batch]
        text_lengths = torch.tensor([b["text_length"] for b in batch], dtype=torch.long)
        text_padded = pad_sequence(text_seqs, batch_first=True, padding_value=0.0)
        text_mask = build_padding_mask(text_lengths, max_len=text_padded.shape[1])
        text_ids_padded = pad_sequence(
            [b["text_input_ids"] for b in batch], batch_first=True, padding_value=0
        )
        collated["text_features"] = text_padded
        collated["text_lengths"] = text_lengths
        collated["text_mask"] = text_mask
        collated["text_attention_mask"] = text_mask.to(dtype=torch.long)
        collated["text_input_ids"] = text_ids_padded
        collated["transcript_ids"] = [b["transcript_id"] for b in batch]

    return collated


def create_feature_dataloader(
    dataset: TESSFeatureDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader bound to `tess_feature_collate_fn`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tess_feature_collate_fn,
    )
