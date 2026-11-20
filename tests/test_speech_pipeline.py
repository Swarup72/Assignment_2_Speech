"""Automated verification tests for the Speech-Only Emotion Recognition Pipeline (Step 04)."""

from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

import pandas as pd
import pytest
import torch

import models.speech_pipeline.train as speech_train_mod
from models.speech_pipeline.model import (
    SPEECH_ARCHITECTURES,
    build_speech_model,
)
from models.speech_pipeline.test import (
    evaluate_and_aggregate,
    evaluate_checkpoint,
)
from models.speech_pipeline.train import (
    SpeechTrainConfig,
    set_reproducible_seed,
    train_single_run,
)
from src.data.build_manifest import EMOTION_TO_ID
from src.data.feature_dataset import build_padding_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _create_synthetic_speech_fixture(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create a lightweight synthetic manifest and WavLM/MFCC feature caches in `tmp_path`."""
    wavlm_dir = tmp_path / "features" / "wavlm"
    mfcc_dir = tmp_path / "features" / "mfcc"
    wavlm_dir.mkdir(parents=True, exist_ok=True)
    mfcc_dir.mkdir(parents=True, exist_ok=True)

    set_reproducible_seed(77)
    rows: List[Dict[str, object]] = []
    words_by_split = {
        "train": ["alpha", "bravo"],
        "val": ["charlie"],
        "test": ["delta"],
    }

    for split_name, words in words_by_split.items():
        for word in words:
            for spk in ("OAF", "YAF"):
                for emo, label_id in EMOTION_TO_ID.items():
                    sid = f"{spk}_{word}_{emo}"
                    t_wavlm = 18 + (label_id * 2)
                    t_mfcc = 30 + (label_id * 2)
                    # Add class-correlated signal so synthetic metrics are well-defined
                    class_bias_w = torch.full((t_wavlm, 768), float(label_id) * 0.25)
                    w_feat = torch.randn(t_wavlm, 768) * 0.2 + class_bias_w
                    torch.save(
                        {
                            "sample_id": sid,
                            "features": w_feat.to(torch.float32),
                            "num_frames": t_wavlm,
                            "feature_dim": 768,
                            "label_id": label_id,
                        },
                        wavlm_dir / f"{sid}.pt",
                    )

                    class_bias_m = torch.full((t_mfcc, 120), float(label_id) * 0.25)
                    m_feat = torch.randn(t_mfcc, 120) * 0.2 + class_bias_m
                    torch.save(
                        {
                            "sample_id": sid,
                            "features": m_feat.to(torch.float32),
                            "num_frames": t_mfcc,
                            "feature_dim": 120,
                            "cmvn_applied": False,
                            "label_id": label_id,
                        },
                        mfcc_dir / f"{sid}.pt",
                    )

                    rows.append(
                        {
                            "sample_id": sid,
                            "filepath": f"synthetic/{sid}.wav",
                            "speaker": spk,
                            "word": word,
                            "transcript": f"Say the word {word}",
                            "emotion": emo,
                            "label_id": label_id,
                            "sample_rate": 16000,
                            "duration_sec": 1.5,
                            "word_indep_split": split_name,
                            "cross_speaker_split": (
                                "test"
                                if spk == "YAF"
                                else ("val" if split_name == "val" else "train")
                            ),
                        }
                    )

    manifest_path = tmp_path / "synthetic_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path, wavlm_dir, mfcc_dir


@pytest.mark.parametrize(
    "architecture,input_dim",
    [
        ("wavlm_bigru_attn", 768),
        ("mfcc_cnn_bilstm", 120),
    ],
)
def test_forward_output_shapes_and_attention_properties(architecture: str, input_dim: int):
    """Verify forward pass output shapes, 7-class logits, 256-d h_s, and attention weight invariants."""
    set_reproducible_seed(42)
    model = build_speech_model(architecture=architecture, dropout=0.2).eval()

    batch_size, max_len = 4, 40
    lengths = torch.tensor([40, 31, 19, 8], dtype=torch.long)
    mask = build_padding_mask(lengths, max_len=max_len)
    features = torch.randn(batch_size, max_len, input_dim)

    with torch.no_grad():
        out = model(features, lengths=lengths, mask=mask)

    assert set(out.keys()) >= {"logits", "embedding", "h_s", "attention_weights"}
    assert out["logits"].shape == (batch_size, 7)
    assert out["embedding"].shape == (batch_size, 256)
    assert torch.equal(out["embedding"], out["h_s"])
    assert out["attention_weights"].shape == (batch_size, max_len)

    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["embedding"]).all()
    assert torch.isfinite(out["attention_weights"]).all()

    # Verify attention weights sum to 1 over valid positions and are strictly 0 at padded positions
    attn = out["attention_weights"]
    for i, valid_len in enumerate(lengths.tolist()):
        valid_sum = attn[i, :valid_len].sum().item()
        assert abs(valid_sum - 1.0) < 1e-5, f"Valid attention sum={valid_sum}, expected 1.0"
        if valid_len < max_len:
            padded_weights = attn[i, valid_len:]
            assert torch.all(padded_weights == 0.0), (
                f"Non-zero attention weight on padded positions: {padded_weights}"
            )


@pytest.mark.parametrize(
    "architecture,input_dim",
    [
        ("wavlm_bigru_attn", 768),
        ("mfcc_cnn_bilstm", 120),
    ],
)
def test_backward_pass_and_gradient_flow(architecture: str, input_dim: int):
    """Verify backward pass succeeds and all trainable parameters receive finite non-zero gradients."""
    set_reproducible_seed(123)
    model = build_speech_model(architecture=architecture, dropout=0.1).train()

    lengths = torch.tensor([25, 18, 12], dtype=torch.long)
    features = torch.randn(3, 25, input_dim, requires_grad=True)
    labels = torch.tensor([0, 3, 6], dtype=torch.long)

    out = model(features, lengths=lengths)
    loss = torch.nn.functional.cross_entropy(out["logits"], labels)
    loss.backward()

    assert features.grad is not None and torch.isfinite(features.grad).all()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter {name}"
            assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for parameter {name}"


@pytest.mark.parametrize(
    "architecture,input_dim",
    [
        ("wavlm_bigru_attn", 768),
        ("mfcc_cnn_bilstm", 120),
    ],
)
def test_padding_mask_invariance(architecture: str, input_dim: int):
    """Verify adding extra padded frames (even filled with non-zero noise) does not alter h_s or logits."""
    set_reproducible_seed(999)
    model = build_speech_model(architecture=architecture, dropout=0.0).eval()

    valid_len = 22
    x_unpadded = torch.randn(2, valid_len, input_dim)
    lengths = torch.tensor([valid_len, 15], dtype=torch.long)
    mask_short = build_padding_mask(lengths, max_len=valid_len)

    # Create an extended tensor with 30 extra trailing frames filled with arbitrary noise
    extra_pad = 30
    noise_tail = torch.randn(2, extra_pad, input_dim) * 5.0
    x_extended = torch.cat([x_unpadded, noise_tail], dim=1)
    mask_extended = build_padding_mask(lengths, max_len=valid_len + extra_pad)

    with torch.no_grad():
        out_short = model(x_unpadded, lengths=lengths, mask=mask_short)
        out_ext = model(x_extended, lengths=lengths, mask=mask_extended)

    assert torch.allclose(out_short["embedding"], out_ext["embedding"], atol=1e-5, rtol=1e-5), (
        f"Padding invariance violated for {architecture} embedding: "
        f"max diff={(out_short['embedding'] - out_ext['embedding']).abs().max().item()}"
    )
    assert torch.allclose(out_short["logits"], out_ext["logits"], atol=1e-5, rtol=1e-5), (
        f"Padding invariance violated for {architecture} logits: "
        f"max diff={(out_short['logits'] - out_ext['logits']).abs().max().item()}"
    )
    assert torch.allclose(
        out_short["attention_weights"],
        out_ext["attention_weights"][:, :valid_len],
        atol=1e-5,
    )
    assert torch.all(out_ext["attention_weights"][:, valid_len:] == 0.0)


@pytest.mark.parametrize(
    "architecture,input_dim",
    [
        ("wavlm_bigru_attn", 768),
        ("mfcc_cnn_bilstm", 120),
    ],
)
def test_cpu_and_mps_execution_and_seed_reproducibility(architecture: str, input_dim: int):
    """Verify deterministic initialization with fixed seeds and execution on CPU and MPS/CUDA."""
    set_reproducible_seed(42)
    m1 = build_speech_model(architecture=architecture, dropout=0.0).eval()
    set_reproducible_seed(42)
    m2 = build_speech_model(architecture=architecture, dropout=0.0).eval()

    x = torch.randn(3, 20, input_dim)
    lengths = torch.tensor([20, 14, 9], dtype=torch.long)

    with torch.no_grad():
        o1 = m1(x, lengths=lengths)
        o2 = m2(x, lengths=lengths)

    assert torch.equal(o1["logits"], o2["logits"])
    assert torch.equal(o1["embedding"], o2["embedding"])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        m_mps = m1.to(device)
        with torch.no_grad():
            o_mps = m_mps(x.to(device), lengths=lengths.to(device))
        assert o_mps["logits"].device.type == "mps"
        assert torch.allclose(o1["logits"], o_mps["logits"].cpu(), atol=1e-4, rtol=1e-4)
        assert torch.allclose(o1["embedding"], o_mps["embedding"].cpu(), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("architecture", SPEECH_ARCHITECTURES)
def test_checkpoint_save_load_and_embedding_export(architecture: str, tmp_path: Path):
    """Verify end-to-end training checkpoint save, test evaluation load, and (N_test, 256) embedding export."""
    manifest_path, wavlm_dir, mfcc_dir = _create_synthetic_speech_fixture(tmp_path)
    out_root = tmp_path / "results_speech"

    cfg = SpeechTrainConfig(
        architecture=architecture,
        split_type="word_indep_split",
        seed=42,
        batch_size=14,
        max_epochs=2,
        patience=2,
        device="cpu",
        manifest_path=str(manifest_path),
        wavlm_cache_dir=str(wavlm_dir),
        mfcc_cache_dir=str(mfcc_dir),
        output_root=str(out_root),
    )

    train_summary = train_single_run(cfg)
    ckpt_path = Path(train_summary["checkpoint_path"])
    assert ckpt_path.exists()

    eval_summary = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        manifest_path=manifest_path,
        wavlm_cache_dir=wavlm_dir,
        mfcc_cache_dir=mfcc_dir,
        batch_size=14,
        device="cpu",
    )
    n_test = 14
    assert eval_summary["num_test_samples"] == n_test
    assert eval_summary["embedding_shape"] == [n_test, 256]

    run_dir = ckpt_path.parent
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "per_class_metrics.csv").exists()
    assert (run_dir / "confusion_matrix.csv").exists()
    assert (run_dir / "test_predictions.csv").exists()
    assert (run_dir / "test_embeddings.pt").exists()
    assert (run_dir / "test_embeddings.npz").exists()

    emb_payload = torch.load(run_dir / "test_embeddings.pt", map_location="cpu", weights_only=False)
    assert emb_payload["embeddings"].shape == (n_test, 256)
    assert torch.isfinite(emb_payload["embeddings"]).all()


def test_early_stopping_does_not_reset_patience_on_loss_only_tiebreak(tmp_path: Path):
    """Verify loss tie-breaks update best_model.pt on tied Macro-F1 but do NOT reset patience."""
    manifest_path, wavlm_dir, mfcc_dir = _create_synthetic_speech_fixture(tmp_path)
    out_root = tmp_path / "early_stop_test"

    # Simulate 10 max epochs with patience=3 where val Macro-F1 stays flat at 0.8500
    # while val_loss decreases every epoch (0.90 -> 0.85 -> 0.80 -> 0.75 ...).
    # Epoch 1 sets best_val_macro_f1=0.8500 (patience=0).
    # Epochs 2, 3, 4 tie on Macro-F1 (0.8500) with improving val_loss:
    #   - loss_tiebreak is True, so best_epoch updates to 2, 3, 4
    #   - f1_improved is False, so patience increments to 1, 2, 3 -> stops at Epoch 4!
    val_call_count = {"n": 0}

    def fake_run_epoch(model, dataloader, criterion, architecture, device, optimizer=None, grad_clip_norm=1.0):
        if optimizer is not None:
            return (
                {"loss": 0.5, "accuracy": 0.85, "macro_precision": 0.85, "macro_recall": 0.85, "macro_f1": 0.85},
                device,
                False,
            )
        val_call_count["n"] += 1
        ep = val_call_count["n"]
        return (
            {
                "loss": 0.95 - (ep * 0.05),
                "accuracy": 0.85,
                "macro_precision": 0.85,
                "macro_recall": 0.85,
                "macro_f1": 0.85,
            },
            device,
            False,
        )

    cfg = SpeechTrainConfig(
        architecture="mfcc_cnn_bilstm",
        split_type="word_indep_split",
        seed=42,
        batch_size=14,
        max_epochs=10,
        patience=3,
        device="cpu",
        manifest_path=str(manifest_path),
        wavlm_cache_dir=str(wavlm_dir),
        mfcc_cache_dir=str(mfcc_dir),
        output_root=str(out_root),
    )

    with patch.object(speech_train_mod, "run_epoch", side_effect=fake_run_epoch):
        summary = train_single_run(cfg)

    # Must stop after 1 (initial) + 3 (patience) = 4 epochs, NOT run all 10 epochs!
    assert summary["epochs_trained"] == 4
    # Best checkpoint must be updated via loss_tiebreak to epoch 4 (lowest val_loss = 0.75)
    assert summary["best_epoch"] == 4
    assert abs(summary["best_val_macro_f1"] - 0.85) < 1e-6
    assert abs(summary["best_val_loss"] - 0.75) < 1e-6

    history = summary["history"]
    assert [h["epoch"] for h in history] == [1, 2, 3, 4]
    assert [h["f1_improved"] for h in history] == [True, False, False, False]
    assert [h["loss_tiebreak"] for h in history] == [False, True, True, True]
    assert [h["patience_counter"] for h in history] == [0, 1, 2, 3]


def test_checkpoint_resume_mid_run_and_skip_completed_run(tmp_path: Path):
    """Verify epoch-level resume from last_checkpoint.pt and completed-run protection."""
    manifest_path, wavlm_dir, mfcc_dir = _create_synthetic_speech_fixture(tmp_path)
    out_root = tmp_path / "resume_test"
    run_dir = out_root / "word_indep_split" / "mfcc_cnn_bilstm" / "seed_42"

    real_run_epoch = speech_train_mod.run_epoch
    epoch_tracker = {"val_calls": 0}

    def interrupting_run_epoch(*args, **kwargs):
        is_train = kwargs.get("optimizer") is not None
        if not is_train:
            epoch_tracker["val_calls"] += 1
            # Simulate Colab disconnect at the start of Epoch 3 validation
            # (after Epochs 1 and 2 completed and wrote last_checkpoint.pt)
            if epoch_tracker["val_calls"] == 3:
                raise KeyboardInterrupt("Simulated Colab disconnect during epoch 3")
        return real_run_epoch(*args, **kwargs)

    cfg = SpeechTrainConfig(
        architecture="mfcc_cnn_bilstm",
        split_type="word_indep_split",
        seed=42,
        batch_size=14,
        max_epochs=4,
        patience=5,
        device="cpu",
        manifest_path=str(manifest_path),
        wavlm_cache_dir=str(wavlm_dir),
        mfcc_cache_dir=str(mfcc_dir),
        output_root=str(out_root),
    )

    with patch.object(speech_train_mod, "run_epoch", side_effect=interrupting_run_epoch):
        with pytest.raises(KeyboardInterrupt, match="Simulated Colab disconnect"):
            train_single_run(cfg)

    # Verify last_checkpoint.pt exists after epoch 2 and final training_history.json does not exist yet
    last_ckpt_path = run_dir / "last_checkpoint.pt"
    assert last_ckpt_path.exists()
    assert not (run_dir / "training_history.json").exists()

    saved_resume = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
    assert saved_resume["last_epoch"] == 2
    assert len(saved_resume["history"]) == 2
    assert "model_state_dict" in saved_resume
    assert "optimizer_state_dict" in saved_resume
    assert "scheduler_state_dict" in saved_resume
    assert "train_generator_state" in saved_resume

    # Verify incompatible config is rejected on resume
    bad_cfg = SpeechTrainConfig(
        architecture="mfcc_cnn_bilstm",
        split_type="word_indep_split",
        seed=42,
        batch_size=14,
        max_epochs=4,
        patience=5,
        lr=5e-2,  # Mismatched learning rate
        device="cpu",
        manifest_path=str(manifest_path),
        wavlm_cache_dir=str(wavlm_dir),
        mfcc_cache_dir=str(mfcc_dir),
        output_root=str(out_root),
    )
    with pytest.raises(ValueError, match="Incompatible resume checkpoint"):
        train_single_run(bad_cfg)

    # Now resume with the matching config; confirm only epochs 3 and 4 execute
    resumed_val_calls = {"n": 0}

    def counting_run_epoch(*args, **kwargs):
        if kwargs.get("optimizer") is None:
            resumed_val_calls["n"] += 1
        return real_run_epoch(*args, **kwargs)

    with patch.object(speech_train_mod, "run_epoch", side_effect=counting_run_epoch):
        summary = train_single_run(cfg)

    # Only epochs 3 and 4 should have run during the resumed invocation
    assert resumed_val_calls["n"] == 2
    assert summary["epochs_trained"] == 4
    assert [h["epoch"] for h in summary["history"]] == [1, 2, 3, 4]
    # After successful completion, last_checkpoint.pt is removed and final artifacts exist
    assert not last_ckpt_path.exists()
    assert (run_dir / "training_history.json").exists()
    assert (run_dir / "best_model.pt").exists()

    # Completed-run protection: invoking train_single_run again without overwrite must skip training
    ckpt_mtime_before = (run_dir / "best_model.pt").stat().st_mtime_ns
    skip_calls = {"n": 0}

    def fail_if_called(*args, **kwargs):
        skip_calls["n"] += 1
        raise AssertionError("run_epoch should not be called for an already-completed run!")

    with patch.object(speech_train_mod, "run_epoch", side_effect=fail_if_called):
        skipped_summary = train_single_run(cfg)

    assert skip_calls["n"] == 0
    assert skipped_summary["epochs_trained"] == 4
    assert (run_dir / "best_model.pt").stat().st_mtime_ns == ckpt_mtime_before


def test_evaluate_main_skips_missing_mac_checkpoints_gracefully(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
):
    """Verify test evaluation warns and skips absent checkpoints while evaluating available ones."""
    manifest_path, wavlm_dir, mfcc_dir = _create_synthetic_speech_fixture(tmp_path)
    out_root = tmp_path / "partial_results"

    # Train ONLY 1 combination (mfcc_cnn_bilstm on word_indep_split, seed=42),
    # leaving wavlm_bigru_attn absent (simulating Colab when Mac runs are stored locally).
    cfg = SpeechTrainConfig(
        architecture="mfcc_cnn_bilstm",
        split_type="word_indep_split",
        seed=42,
        batch_size=14,
        max_epochs=2,
        patience=2,
        device="cpu",
        manifest_path=str(manifest_path),
        wavlm_cache_dir=str(wavlm_dir),
        mfcc_cache_dir=str(mfcc_dir),
        output_root=str(out_root),
    )
    train_single_run(cfg)

    # Request evaluation across BOTH architectures on word_indep_split (wavlm_bigru_attn is missing)
    summary = evaluate_and_aggregate(
        output_root=out_root,
        splits=("word_indep_split",),
        architectures=("wavlm_bigru_attn", "mfcc_cnn_bilstm"),
        seeds=(42,),
        manifest_path=manifest_path,
        wavlm_cache_dir=wavlm_dir,
        mfcc_cache_dir=mfcc_dir,
        batch_size=14,
        device="cpu",
    )

    captured = capsys.readouterr().out
    assert "WARNING: Missing checkpoint for split=word_indep_split, architecture=wavlm_bigru_attn, seed=42" in captured
    assert summary["num_evaluated"] == 1
    assert len(summary["skipped_combinations"]) == 1
    assert summary["skipped_combinations"][0]["architecture"] == "wavlm_bigru_attn"
    assert len(summary["runs"]) == 1
    assert summary["runs"][0]["architecture"] == "mfcc_cnn_bilstm"
    assert len(summary["aggregated"]) == 1
    assert (out_root / "summaries" / "speech_all_runs.json").exists()
    assert (out_root / "summaries" / "speech_summary_mean_std.json").exists()
