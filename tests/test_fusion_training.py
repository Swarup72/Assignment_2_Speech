"""Unit and integration tests for Stage 08B Multimodal Fusion training & evaluation.

Uses purely synthetic tensors, lightweight mock base encoders, and temporary
directories (`tmp_path`). Does not read the TESS dataset, feature caches, or
any of the 18 frozen experiment checkpoints.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.fusion_pipeline.model import ID_TO_EMOTION, NUM_EMOTION_CLASSES
from models.fusion_pipeline.test import (
    EMOTION_NAMES,
    aggregate_fusion_results,
    compute_full_evaluation_metrics,
    evaluate_fusion_run,
)
from models.fusion_pipeline.train import (
    DEFAULT_LAMBDA_GRID,
    FusionTrainConfig,
    _validate_resume_compatibility,
    assert_encoder_frozen,
    calibrate_late_fusion_from_logits,
    extract_frozen_multimodal_features,
    freeze_encoder,
    get_fusion_run_output_dir,
    load_frozen_speech_encoder,
    load_frozen_text_encoder,
    normalize_split_column,
    resolve_speech_checkpoint_path,
    resolve_text_checkpoint_path,
    train_single_fusion_run,
    verify_and_align_modalities,
)


# =============================================================================
# Lightweight Mock Encoders & Synthetic Paired Dataset Helpers
# =============================================================================


class _MockFrozenEncoder(nn.Module):
    """Lightweight deterministic encoder mirroring the speech/text forward contract."""

    def __init__(
        self,
        input_dim: int = 16,
        repr_dim: int = 256,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, repr_dim)
        self.head = nn.Linear(repr_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled = features.mean(dim=1)
        emb = torch.tanh(self.proj(self.drop(pooled)))
        logits = self.head(emb)
        return {"logits": logits, "embedding": emb}


def _mock_speech_builder(
    architecture: str = "wavlm_bigru_attn",
    dropout: float = 0.3,
    num_classes: int = NUM_EMOTION_CLASSES,
) -> nn.Module:
    return _MockFrozenEncoder(input_dim=16, repr_dim=256, num_classes=num_classes, dropout=dropout)


def _mock_text_builder(
    architecture: str = "distilbert_bilstm_attn",
    dropout: float = 0.3,
    num_classes: int = NUM_EMOTION_CLASSES,
) -> nn.Module:
    return _MockFrozenEncoder(input_dim=16, repr_dim=256, num_classes=num_classes, dropout=dropout)


def _save_mock_checkpoint(
    path: Path,
    architecture: str,
    split_type: str,
    seed: int,
    builder: Any,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = builder(architecture=architecture, dropout=0.3, num_classes=NUM_EMOTION_CLASSES)
    payload = {
        "architecture": architecture,
        "split_type": split_type,
        "seed": seed,
        "best_epoch": 5,
        "best_val_macro_f1": 0.9,
        "model_config": {
            "architecture": architecture,
            "dropout": 0.3,
            "num_classes": NUM_EMOTION_CLASSES,
        },
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


class _SyntheticMultimodalDataset(Dataset):
    """Synthetic paired speech+text dataset producing batches matching tess_feature_collate_fn."""

    def __init__(self, num_samples: int = 21, seed: int = 42, prefix: str = "sample") -> None:
        g = torch.Generator().manual_seed(seed)
        self.num_samples = num_samples
        self.labels = [i % NUM_EMOTION_CLASSES for i in range(num_samples)]
        # Make speech features linearly separable by label while text features are noisy
        self.speech_feats = []
        self.text_feats = []
        for i in range(num_samples):
            y = self.labels[i]
            base_s = torch.randn(4, 16, generator=g) * 0.2
            base_s[:, y] += 2.5
            self.speech_feats.append(base_s)
            self.text_feats.append(torch.randn(3, 16, generator=g) * 0.5)
        self.sample_ids = [f"{prefix}_{i:03d}" for i in range(num_samples)]
        self.speakers = ["OAF" if i % 2 == 0 else "YAF" for i in range(num_samples)]
        self.words = [f"word{i % 5}" for i in range(num_samples)]
        self.emotions = [ID_TO_EMOTION[y] for y in self.labels]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_ids[idx],
            "label": self.labels[idx],
            "speaker": self.speakers[idx],
            "word": self.words[idx],
            "emotion": self.emotions[idx],
            "wavlm_features": self.speech_feats[idx],
            "wavlm_length": 4,
            "text_features": self.text_feats[idx],
            "text_length": 3,
        }


def _synthetic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    b_size = len(batch)
    return {
        "sample_ids": [b["sample_id"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "speakers": [b["speaker"] for b in batch],
        "words": [b["word"] for b in batch],
        "emotions": [b["emotion"] for b in batch],
        "wavlm_features": torch.stack([b["wavlm_features"] for b in batch], dim=0),
        "wavlm_lengths": torch.tensor([b["wavlm_length"] for b in batch], dtype=torch.long),
        "wavlm_mask": torch.ones((b_size, 4), dtype=torch.bool),
        "text_features": torch.stack([b["text_features"] for b in batch], dim=0),
        "text_lengths": torch.tensor([b["text_length"] for b in batch], dtype=torch.long),
        "text_mask": torch.ones((b_size, 3), dtype=torch.bool),
    }


def _make_loader(num_samples: int, seed: int, prefix: str, batch_size: int = 8) -> DataLoader:
    return DataLoader(
        _SyntheticMultimodalDataset(num_samples=num_samples, seed=seed, prefix=prefix),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_synthetic_collate_fn,
    )


# =============================================================================
# 1. Checkpoint Integration & Validation Tests
# =============================================================================


def test_split_alias_resolution_and_checkpoint_paths(tmp_path: Path):
    """Verify canonical split normalization and speech/text alias path resolution."""
    assert normalize_split_column("word_independent") == "word_indep_split"
    assert normalize_split_column("word_indep_split") == "word_indep_split"
    assert normalize_split_column("cross_speaker") == "cross_speaker_split"
    assert normalize_split_column("cross_speaker_split") == "cross_speaker_split"

    with pytest.raises(ValueError, match="Unsupported split"):
        normalize_split_column("invalid_split")

    sp_path = resolve_speech_checkpoint_path(tmp_path / "speech", "word_independent", 42)
    tx_path = resolve_text_checkpoint_path(tmp_path / "text", "word_indep_split", 42)
    assert sp_path == tmp_path / "speech" / "word_indep_split" / "wavlm_bigru_attn" / "seed_42" / "best_model.pt"
    assert tx_path == tmp_path / "text" / "distilbert_bilstm_attn" / "word_independent" / "seed_42" / "best_model.pt"


def test_checkpoint_validation_rejects_seed_split_arch_and_missing(tmp_path: Path):
    """Verify seed mismatch, split mismatch, architecture mismatch, and missing files raise errors."""
    ckpt = _save_mock_checkpoint(
        tmp_path / "speech.pt",
        architecture="wavlm_bigru_attn",
        split_type="word_indep_split",
        seed=42,
        builder=_mock_speech_builder,
    )

    # Matching metadata loads cleanly and freezes encoder
    model, meta = load_frozen_speech_encoder(
        checkpoint_path=ckpt,
        expected_split_type="word_indep_split",
        expected_seed=42,
        expected_architecture="wavlm_bigru_attn",
        model_builder=_mock_speech_builder,
    )
    assert_encoder_frozen(model, name="speech_encoder")
    assert meta["seed"] == 42
    assert meta["split_type"] == "word_indep_split"

    # Seed mismatch rejected
    with pytest.raises(ValueError, match="seed mismatch"):
        load_frozen_speech_encoder(
            ckpt, "word_indep_split", 123, "wavlm_bigru_attn", model_builder=_mock_speech_builder
        )

    # Split mismatch rejected
    with pytest.raises(ValueError, match="split mismatch"):
        load_frozen_speech_encoder(
            ckpt, "cross_speaker_split", 42, "wavlm_bigru_attn", model_builder=_mock_speech_builder
        )

    # Architecture mismatch rejected
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_frozen_speech_encoder(
            ckpt, "word_indep_split", 42, "mfcc_cnn_bilstm", model_builder=_mock_speech_builder
        )

    # Missing checkpoint rejected
    with pytest.raises(FileNotFoundError, match="not found"):
        load_frozen_text_encoder(
            tmp_path / "missing.pt",
            "word_indep_split",
            42,
            "distilbert_bilstm_attn",
            model_builder=_mock_text_builder,
        )


# =============================================================================
# 2. Encoder Freezing & Gradient Isolation Tests
# =============================================================================


def test_encoders_remain_frozen_and_receive_no_gradients(tmp_path: Path):
    """Verify frozen speech/text encoders stay in eval() mode with zero gradients during training."""
    sp_ckpt = _save_mock_checkpoint(
        resolve_speech_checkpoint_path(tmp_path / "speech", "word_indep_split", 42),
        "wavlm_bigru_attn",
        "word_indep_split",
        42,
        _mock_speech_builder,
    )
    tx_ckpt = _save_mock_checkpoint(
        resolve_text_checkpoint_path(tmp_path / "text", "word_indep_split", 42),
        "distilbert_bilstm_attn",
        "word_indep_split",
        42,
        _mock_text_builder,
    )

    sp_enc, _ = load_frozen_speech_encoder(
        sp_ckpt, "word_indep_split", 42, model_builder=_mock_speech_builder
    )
    tx_enc, _ = load_frozen_text_encoder(
        tx_ckpt, "word_indep_split", 42, model_builder=_mock_text_builder
    )

    sp_weights_before = {k: v.clone() for k, v in sp_enc.state_dict().items()}
    tx_weights_before = {k: v.clone() for k, v in tx_enc.state_dict().items()}

    loader = _make_loader(num_samples=14, seed=42, prefix="tr", batch_size=7)
    bundle, _, _ = extract_frozen_multimodal_features(loader, sp_enc, tx_enc, torch.device("cpu"))

    assert not bundle["speech_embeddings"].requires_grad
    assert not bundle["text_embeddings"].requires_grad
    assert_encoder_frozen(sp_enc, "speech")
    assert_encoder_frozen(tx_enc, "text")

    for k, v in sp_enc.state_dict().items():
        assert torch.equal(v, sp_weights_before[k])
    for k, v in tx_enc.state_dict().items():
        assert torch.equal(v, tx_weights_before[k])
    for p in list(sp_enc.parameters()) + list(tx_enc.parameters()):
        assert p.grad is None


# =============================================================================
# 3. Dataset & Sample Alignment Tests
# =============================================================================


def test_sample_alignment_reorders_by_id_and_rejects_defects():
    """Verify explicit sample_id alignment and rejection of mismatches, duplicates, and label errors."""
    n = 4
    sp_bundle = {
        "sample_ids": ["s0", "s1", "s2", "s3"],
        "labels": torch.tensor([0, 1, 2, 3]),
        "embeddings": torch.randn(n, 256),
        "logits": torch.randn(n, 7),
    }
    # Reversed order in text bundle -> must align explicitly by sample_id
    tx_emb = torch.randn(n, 256)
    tx_logits = torch.randn(n, 7)
    tx_bundle_reordered = {
        "sample_ids": ["s3", "s2", "s1", "s0"],
        "labels": torch.tensor([3, 2, 1, 0]),
        "embeddings": tx_emb,
        "logits": tx_logits,
    }
    aligned = verify_and_align_modalities(sp_bundle, tx_bundle_reordered)
    assert aligned["sample_ids"] == ["s0", "s1", "s2", "s3"]
    assert torch.equal(aligned["text_embeddings"][0], tx_emb[3])
    assert torch.equal(aligned["text_logits"][0], tx_logits[3])

    # Mismatched sample IDs rejected
    bad_ids = dict(tx_bundle_reordered, sample_ids=["s3", "s2", "s1", "s_MISSING"])
    with pytest.raises(ValueError, match="Sample ID mismatch"):
        verify_and_align_modalities(sp_bundle, bad_ids)

    # Duplicate sample IDs rejected
    dup_ids = dict(tx_bundle_reordered, sample_ids=["s0", "s0", "s1", "s2"])
    with pytest.raises(ValueError, match="Duplicate sample_ids"):
        verify_and_align_modalities(sp_bundle, dup_ids)

    # Mismatched labels rejected
    bad_labels = dict(tx_bundle_reordered, labels=torch.tensor([3, 2, 1, 6]))
    with pytest.raises(ValueError, match="Label mismatch"):
        verify_and_align_modalities(sp_bundle, bad_labels)


# =============================================================================
# 4. Training, Early Stopping, Resume & Completed-Run Protection Tests
# =============================================================================


@pytest.mark.parametrize("arch", ["gated_fusion", "concat_fusion"])
def test_trainable_fusion_train_evaluate_and_completed_run_protection(tmp_path: Path, arch: str):
    """Verify end-to-end synthetic training, checkpointing, completed-run skip, and evaluation."""
    speech_root = tmp_path / "results" / "speech"
    text_root = tmp_path / "results" / "text"
    fusion_root = tmp_path / "results" / "fusion"

    _save_mock_checkpoint(
        resolve_speech_checkpoint_path(speech_root, "word_indep_split", 42),
        "wavlm_bigru_attn",
        "word_indep_split",
        42,
        _mock_speech_builder,
    )
    _save_mock_checkpoint(
        resolve_text_checkpoint_path(text_root, "word_indep_split", 42),
        "distilbert_bilstm_attn",
        "word_indep_split",
        42,
        _mock_text_builder,
    )

    train_loader = _make_loader(num_samples=14, seed=42, prefix="tr", batch_size=7)
    val_loader = _make_loader(num_samples=7, seed=100, prefix="va", batch_size=7)
    test_loader = _make_loader(num_samples=7, seed=200, prefix="te", batch_size=1)  # B=1 test!

    cfg = FusionTrainConfig(
        architecture=arch,
        split_type="word_indep_split",
        seed=42,
        batch_size=7,
        max_epochs=3,
        patience=2,
        device="cpu",
        speech_results_root=str(speech_root),
        text_results_root=str(text_root),
        output_root=str(fusion_root),
    )

    summary = train_single_fusion_run(
        cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        speech_model_builder=_mock_speech_builder,
        text_model_builder=_mock_text_builder,
    )
    run_dir = get_fusion_run_output_dir(fusion_root, arch, "word_indep_split", 42)
    assert (run_dir / "best_model.pt").exists()
    assert (run_dir / "last_checkpoint.pt").exists()
    assert (run_dir / "run_summary.json").exists()
    assert (run_dir / "config.json").exists()
    assert summary["epochs_trained"] >= 1
    assert summary["num_trainable_params"] > 0

    # Completed-run protection: calling again without overwrite skips retraining
    mtime_before = (run_dir / "best_model.pt").stat().st_mtime_ns
    summary_repeat = train_single_fusion_run(
        cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        speech_model_builder=_mock_speech_builder,
        text_model_builder=_mock_text_builder,
    )
    assert (run_dir / "best_model.pt").stat().st_mtime_ns == mtime_before
    assert summary_repeat["best_epoch"] == summary["best_epoch"]

    # Evaluate on batch size 1 test loader
    eval_metrics = evaluate_fusion_run(
        architecture=arch,
        split_type="word_indep_split",
        seed=42,
        output_root=fusion_root,
        speech_results_root=speech_root,
        text_results_root=text_root,
        device="cpu",
        test_loader=test_loader,
        speech_model_builder=_mock_speech_builder,
        text_model_builder=_mock_text_builder,
    )

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "test_predictions.csv").exists()
    assert (run_dir / "confusion_matrix.csv").exists()
    assert (run_dir / "classification_report.json").exists()
    assert (run_dir / "test_embeddings.npz").exists()
    if arch == "gated_fusion":
        assert (run_dir / "gate_activations.npz").exists()
        assert (run_dir / "gate_summary.json").exists()

    # Validate prediction CSV schema and 7 canonical probability columns
    pred_df = pd.read_csv(run_dir / "test_predictions.csv")
    assert len(pred_df) == 7
    expected_prob_cols = [f"prob_{emo}" for emo in EMOTION_NAMES]
    for col in ["sample_id", "speaker", "word", "true_emotion", "pred_emotion", "confidence"] + expected_prob_cols:
        assert col in pred_df.columns
    prob_sums = pred_df[expected_prob_cols].sum(axis=1).to_numpy()
    assert np.allclose(prob_sums, 1.0, atol=1e-4)

    # Validate confusion matrix shape (7, 7) and finite metrics
    assert np.array(eval_metrics["confusion_matrix"]).shape == (7, 7)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "log_loss"):
        assert np.isfinite(eval_metrics[key])

    # Overwrite protection on evaluation
    with pytest.raises(FileExistsError, match="already exist"):
        evaluate_fusion_run(
            architecture=arch,
            split_type="word_indep_split",
            seed=42,
            output_root=fusion_root,
            speech_results_root=speech_root,
            text_results_root=text_root,
            device="cpu",
            test_loader=test_loader,
            speech_model_builder=_mock_speech_builder,
            text_model_builder=_mock_text_builder,
        )


def test_resume_from_interrupted_checkpoint_and_incompatibility_guard(tmp_path: Path):
    """Verify resuming continues from last_epoch+1 and rejects incompatible hyperparameters."""
    speech_root = tmp_path / "speech"
    text_root = tmp_path / "text"
    fusion_root = tmp_path / "fusion"

    _save_mock_checkpoint(
        resolve_speech_checkpoint_path(speech_root, "word_indep_split", 42),
        "wavlm_bigru_attn",
        "word_indep_split",
        42,
        _mock_speech_builder,
    )
    _save_mock_checkpoint(
        resolve_text_checkpoint_path(text_root, "word_indep_split", 42),
        "distilbert_bilstm_attn",
        "word_indep_split",
        42,
        _mock_text_builder,
    )

    train_loader = _make_loader(num_samples=14, seed=42, prefix="tr", batch_size=7)
    val_loader = _make_loader(num_samples=7, seed=100, prefix="va", batch_size=7)

    cfg = FusionTrainConfig(
        architecture="gated_fusion",
        split_type="word_indep_split",
        seed=42,
        batch_size=7,
        max_epochs=2,
        patience=5,
        device="cpu",
        speech_results_root=str(speech_root),
        text_results_root=str(text_root),
        output_root=str(fusion_root),
    )
    train_single_fusion_run(
        cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        speech_model_builder=_mock_speech_builder,
        text_model_builder=_mock_text_builder,
    )

    run_dir = get_fusion_run_output_dir(fusion_root, "gated_fusion", "word_indep_split", 42)
    resume_payload = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert resume_payload["last_epoch"] == 2

    # Incompatibility check: different seed or lr must raise ValueError
    bad_cfg = FusionTrainConfig(
        architecture="gated_fusion",
        split_type="word_indep_split",
        seed=123,
        batch_size=7,
        max_epochs=2,
    )
    with pytest.raises(ValueError, match="Incompatible fusion resume"):
        _validate_resume_compatibility(resume_payload, bad_cfg)

    bad_lr_cfg = FusionTrainConfig(
        architecture="gated_fusion",
        split_type="word_indep_split",
        seed=42,
        batch_size=7,
        max_epochs=2,
        lr=5e-4,
    )
    with pytest.raises(ValueError, match="Incompatible fusion resume hyperparameter"):
        _validate_resume_compatibility(resume_payload, bad_lr_cfg)


# =============================================================================
# 5. Late-Logit Fusion Calibration & Evaluation Tests
# =============================================================================


def test_late_fusion_calibration_grid_tiebreak_and_lambda_one():
    """Verify lambda grid includes 0 & 1, uses val data only, breaks ties by log-loss, accepts lambda=1.0."""
    assert 0.0 in DEFAULT_LAMBDA_GRID
    assert 1.0 in DEFAULT_LAMBDA_GRID
    assert len(DEFAULT_LAMBDA_GRID) == 21

    # Construct validation logits where speech is 100% accurate and text is uniform noise
    val_labels = torch.arange(NUM_EMOTION_CLASSES, dtype=torch.long).repeat(2)  # 14 samples
    speech_logits = torch.full((14, NUM_EMOTION_CLASSES), -3.0)
    for i, y in enumerate(val_labels.tolist()):
        speech_logits[i, y] = 5.0
    text_logits = torch.zeros((14, NUM_EMOTION_CLASSES))  # uniform 1/7 probabilities

    best_lam, best_rec, search_rows = calibrate_late_fusion_from_logits(
        val_speech_logits=speech_logits,
        val_text_logits=text_logits,
        val_labels=val_labels,
    )
    assert len(search_rows) == 21
    assert best_rec["val_macro_f1"] == pytest.approx(1.0)
    # All lambda > 0 give F1=1.0, so log-loss tie-breaker MUST select lambda=1.0
    assert best_lam == pytest.approx(1.0)


def test_late_fusion_end_to_end_calibration_evaluation_and_aggregation(tmp_path: Path):
    """Verify late fusion creates zero trainable params, saves calibration artifacts, and aggregates."""
    speech_root = tmp_path / "speech"
    text_root = tmp_path / "text"
    fusion_root = tmp_path / "fusion"

    _save_mock_checkpoint(
        resolve_speech_checkpoint_path(speech_root, "cross_speaker_split", 999),
        "wavlm_bigru_attn",
        "cross_speaker_split",
        999,
        _mock_speech_builder,
    )
    _save_mock_checkpoint(
        resolve_text_checkpoint_path(text_root, "cross_speaker_split", 999),
        "distilbert_bilstm_attn",
        "cross_speaker_split",
        999,
        _mock_text_builder,
    )

    val_loader = _make_loader(num_samples=14, seed=10, prefix="va", batch_size=7)
    test_loader = _make_loader(num_samples=7, seed=20, prefix="te", batch_size=7)

    cfg = FusionTrainConfig(
        architecture="late_fusion",
        split_type="cross_speaker",  # alias -> cross_speaker_split
        seed=999,
        device="cpu",
        speech_results_root=str(speech_root),
        text_results_root=str(text_root),
        output_root=str(fusion_root),
    )

    calib_summary = train_single_fusion_run(
        cfg,
        val_loader=val_loader,
        speech_model_builder=_mock_speech_builder,
        text_model_builder=_mock_text_builder,
    )
    assert calib_summary["num_trainable_params"] == 0
    assert 0.0 <= calib_summary["selected_lambda"] <= 1.0

    run_dir = get_fusion_run_output_dir(fusion_root, "late_fusion", "cross_speaker_split", 999)
    assert (run_dir / "selected_lambda.json").exists()
    assert (run_dir / "lambda_search.csv").exists()
    assert not (run_dir / "best_model.pt").exists()

    # Evaluate on untouched test partition
    eval_metrics = evaluate_fusion_run(
        architecture="late_fusion",
        split_type="cross_speaker_split",
        seed=999,
        output_root=fusion_root,
        speech_results_root=speech_root,
        text_results_root=text_root,
        device="cpu",
        test_loader=test_loader,
        speech_model_builder=_mock_speech_builder,
        text_model_builder=_mock_text_builder,
    )
    assert eval_metrics["selected_lambda"] == calib_summary["selected_lambda"]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "test_predictions.csv").exists()

    # Aggregate results
    agg = aggregate_fusion_results(
        output_root=fusion_root,
        splits=("cross_speaker_split",),
        architectures=("late_fusion",),
        seeds=(999,),
    )
    assert len(agg["runs"]) == 1
    assert len(agg["aggregated"]) == 1
    assert (fusion_root / "summaries" / "fusion_all_runs.csv").exists()
    assert (fusion_root / "summaries" / "fusion_summary_mean_std.json").exists()
