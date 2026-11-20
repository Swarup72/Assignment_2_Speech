"""Extract and cache DistilBERT token embeddings by unique transcript.

Because TESS consists of 200 unique target words spoken across 2 speakers and
7 emotions (2,800 recordings total), there are exactly 200 unique transcripts
of the form `"Say the word {word}"`.

Design Highlights:
- Uses official `DistilBertTokenizerFast` and frozen `DistilBertModel`
  (`distilbert-base-uncased`) in `eval()` mode under `torch.inference_mode()`.
- Extracts final hidden-state token embeddings of shape `(L, 768)` along with
  `input_ids`, `attention_mask` `(L,)`, and `num_tokens = L` for each unique
  transcript.
- Caches one `.pt` file per unique transcript under
  `data/features/distilbert/transcripts/{transcript_id}.pt` (200 files for the
  full TESS corpus, eliminating 14x redundant storage).
- Maintains a complete mapping from every `sample_id` (2,800 entries) to its
  corresponding transcript feature file in
  `data/features/distilbert/sample_to_transcript.json`.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

from src.features.extract_speech_features import (
    compute_directory_size_bytes,
    select_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "tess_manifest.csv"
DEFAULT_DISTILBERT_CACHE_DIR = PROJECT_ROOT / "data" / "features" / "distilbert"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"


def clean_transcript_text(text: str) -> str:
    """Clean and normalize transcript text while preserving lexical content."""
    if not isinstance(text, str):
        raise TypeError(f"Expected transcript string, got {type(text)}")
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        raise ValueError("Transcript text is empty after cleaning.")
    return cleaned


def transcript_to_cache_id(transcript: str) -> str:
    """Create a deterministic, filesystem-safe identifier for a unique transcript."""
    cleaned = clean_transcript_text(transcript).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if len(slug) > 48:
        digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
        slug = f"{slug[:36]}_{digest}"
    return slug


def is_valid_cached_transcript(pt_path: Path, expected_dim: int = 768) -> bool:
    """Verify a cached transcript .pt file exists and contains valid (L, 768) embeddings and masks."""
    if not pt_path.exists() or pt_path.stat().st_size == 0:
        return False
    try:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        feats = payload.get("features")
        mask = payload.get("attention_mask")
        input_ids = payload.get("input_ids")
        num_tokens = payload.get("num_tokens")
        if not isinstance(feats, torch.Tensor) or feats.ndim != 2:
            return False
        if feats.shape[1] != expected_dim or feats.shape[0] <= 0:
            return False
        if not isinstance(mask, torch.Tensor) or mask.ndim != 1 or mask.shape[0] != feats.shape[0]:
            return False
        if not isinstance(input_ids, torch.Tensor) or input_ids.shape[0] != feats.shape[0]:
            return False
        if num_tokens != int(feats.shape[0]):
            return False
        if not (mask == 1).all().item():
            return False
        return bool(torch.isfinite(feats).all().item())
    except Exception:
        return False


def _extract_distilbert_batch(
    model: DistilBertModel,
    tokenizer: DistilBertTokenizerFast,
    transcripts: List[str],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], torch.device, bool]:
    """Tokenize and extract unpadded (L, 768) DistilBERT embeddings for a batch of transcripts."""
    cleaned_texts = [clean_transcript_text(t) for t in transcripts]
    encoded = tokenizer(
        cleaned_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    fallback_triggered = False
    active_device = device

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids.to(active_device),
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
                    input_ids=input_ids.to(active_device),
                    attention_mask=attention_mask.to(active_device),
                )
                hidden = outputs.last_hidden_state.detach().to("cpu", dtype=torch.float32)
        else:
            raise exc

    results: List[Dict[str, Any]] = []
    for idx, raw_t in enumerate(transcripts):
        valid_len = int(attention_mask[idx].sum().item())
        seq_ids = input_ids[idx, :valid_len].clone().to(dtype=torch.long)
        seq_mask = attention_mask[idx, :valid_len].clone().to(dtype=torch.long)
        seq_feats = hidden[idx, :valid_len, :].clone()

        if not torch.isfinite(seq_feats).all():
            raise ValueError(f"Non-finite DistilBERT features for transcript '{raw_t}'")

        results.append(
            {
                "transcript_id": transcript_to_cache_id(raw_t),
                "transcript": raw_t,
                "cleaned_text": cleaned_texts[idx],
                "input_ids": seq_ids,
                "attention_mask": seq_mask,
                "tokens": tokenizer.convert_ids_to_tokens(seq_ids.tolist()),
                "features": seq_feats,
                "num_tokens": valid_len,
                "feature_dim": 768,
            }
        )

    return results, active_device, fallback_triggered


def extract_and_cache_distilbert_dataset(
    manifest_df: pd.DataFrame,
    cache_dir: Union[str, Path] = DEFAULT_DISTILBERT_CACHE_DIR,
    model_name: str = DISTILBERT_MODEL_NAME,
    batch_size: int = 32,
    device: Optional[str] = None,
    overwrite: bool = False,
    local_files_only: bool = True,
) -> Dict[str, Any]:
    """Extract and cache unique transcript DistilBERT embeddings and sample_id mapping."""
    cache_path = Path(cache_dir)
    transcripts_dir = cache_path / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    exec_device = select_device(device)
    initial_device_str = str(exec_device)
    t0 = time.perf_counter()

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model = DistilBertModel.from_pretrained(
        model_name, local_files_only=local_files_only
    )
    model.eval()
    model.requires_grad_(False)
    model.to(exec_device)
    model_revision = getattr(model.config, "_commit_hash", None) or "local_cache_verified"

    unique_transcripts = sorted(manifest_df["transcript"].astype(str).unique().tolist())

    pending_transcripts: List[str] = []
    num_skipped = 0
    token_lengths: List[int] = []

    for transcript in unique_transcripts:
        t_id = transcript_to_cache_id(transcript)
        pt_file = transcripts_dir / f"{t_id}.pt"
        if not overwrite and is_valid_cached_transcript(pt_file, expected_dim=768):
            payload = torch.load(pt_file, map_location="cpu", weights_only=False)
            token_lengths.append(int(payload["num_tokens"]))
            num_skipped += 1
        else:
            pending_transcripts.append(transcript)

    num_extracted = 0
    cpu_fallback_count = 0

    for start_idx in range(0, len(pending_transcripts), batch_size):
        batch_texts = pending_transcripts[start_idx : start_idx + batch_size]
        batch_payloads, exec_device, fallback_used = _extract_distilbert_batch(
            model=model,
            tokenizer=tokenizer,
            transcripts=batch_texts,
            device=exec_device,
        )
        if fallback_used:
            cpu_fallback_count += 1

        for item in batch_payloads:
            t_id = item["transcript_id"]
            pt_file = transcripts_dir / f"{t_id}.pt"
            torch.save(item, pt_file)
            token_lengths.append(int(item["num_tokens"]))
            num_extracted += 1

    # Build complete sample_id -> transcript feature mapping for all manifest rows
    sample_to_transcript: Dict[str, Dict[str, Any]] = {}
    for row in manifest_df.itertuples(index=False):
        sample_id = str(row.sample_id)
        transcript = str(row.transcript)
        t_id = transcript_to_cache_id(transcript)
        rel_file = f"transcripts/{t_id}.pt"
        sample_to_transcript[sample_id] = {
            "sample_id": sample_id,
            "transcript": transcript,
            "transcript_id": t_id,
            "feature_file": rel_file,
            "word": str(row.word),
            "speaker": str(row.speaker),
            "emotion": str(row.emotion),
            "label_id": int(row.label_id),
        }

    mapping_path = cache_path / "sample_to_transcript.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(sample_to_transcript, f, indent=2)

    if exec_device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    elapsed_sec = time.perf_counter() - t0
    total_cached_transcripts = len(list(transcripts_dir.glob("*.pt")))
    cache_size_bytes = compute_directory_size_bytes(cache_path)

    metadata = {
        "feature_type": "distilbert_base_uncased_last_hidden_state",
        "model_name": model_name,
        "model_revision": str(model_revision),
        "feature_dim": 768,
        "batch_size": batch_size,
        "hardware_requested": initial_device_str,
        "hardware_used": str(exec_device),
        "cpu_fallback_used": cpu_fallback_count > 0,
        "cpu_fallback_count": cpu_fallback_count,
        "num_manifest_samples": len(manifest_df),
        "num_mapped_samples": len(sample_to_transcript),
        "num_unique_transcripts": len(unique_transcripts),
        "num_newly_extracted_transcripts": num_extracted,
        "num_skipped_cached_transcripts": num_skipped,
        "num_cached_transcript_files": total_cached_transcripts,
        "min_tokens": int(min(token_lengths)) if token_lengths else 0,
        "max_tokens": int(max(token_lengths)) if token_lengths else 0,
        "mean_tokens": round(float(sum(token_lengths) / len(token_lengths)), 2) if token_lengths else 0.0,
        "extraction_time_sec": round(elapsed_sec, 4),
        "cache_size_bytes": cache_size_bytes,
        "cache_size_mb": round(cache_size_bytes / (1024 * 1024), 3),
    }

    with open(cache_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache DistilBERT transcript features.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_DISTILBERT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)
    meta = extract_and_cache_distilbert_dataset(
        manifest_df=manifest_df,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    print("[extract_text_features] DistilBERT summary:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
