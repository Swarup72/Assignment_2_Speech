"""Comprehensive unit & dry-run smoke tests for the Text-Only Training & Evaluation Pipeline (Stage 07B).

Uses strictly synthetic DistilBERT feature caches and manifests inside temporary directories.
Never accesses real TESS caches, never modifies `results/speech/`, and never launches real experiments.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import models.text_pipeline.train as text_train_mod
from models.text_pipeline.model import build_text_model
from models.text_pipeline.test import (
    evaluate_and_aggregate_text,
    evaluate_text_checkpoint,
)
from models.text_pipeline.train import (
    TextTrainConfig,
    compute_classification_metrics,
    get_text_run_output_dir,
    run_text_epoch,
    set_reproducible_seed,
    train_single_text_run,
)
from src.data.build_manifest import EMOTION_TO_ID
from src.data.feature_dataset import (
    TESSFeatureDataset,
    tess_feature_collate_fn,
)


def _create_synthetic_text_fixture(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a small synthetic DistilBERT transcript cache and manifest in `tmp_path`."""
    db_dir = tmp_path / "features" / "distilbert"
    transcripts_dir = db_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    set_reproducible_seed(101)
    words_by_split = {
        "train": ["alpha", "bravo"],
        "val": ["charlie"],
        "test": ["delta"],
    }

    # Cache one unique embedding tensor per unique transcript (matching TESS architecture)
    all_words = [w for ws in words_by_split.values() for w in ws]
    for w_idx, word in enumerate(all_words):
        transcript = f"Say the word {word}"
        t_id = f"say_the_word_{word}"
        num_tokens = 6 + (w_idx % 2)  # Mix of 6 and 7 tokens to test dynamic padding
        feats = torch.randn(num_tokens, 768, dtype=torch.float32) * 0.25
        input_ids = torch.randint(100, 1000, (num_tokens,), dtype=torch.long)
        attn_mask = torch.ones((num_tokens,), dtype=torch.long)
        torch.save(
            {
                "transcript_id": t_id,
                "transcript": transcript,
                "cleaned_text": transcript,
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "features": feats,
                "num_tokens": num_tokens,
                "feature_dim": 768,
            },
            transcripts_dir / f"{t_id}.pt",
        )

    rows: List[Dict[str, object]] = []
    sample_to_transcript: Dict[str, Dict[str, object]] = {}

    for split_name, words in words_by_split.items():
        for word in words:
            transcript = f"Say the word {word}"
            t_id = f"say_the_word_{word}"
            for spk in ("OAF", "YAF"):
                for emo, label_id in EMOTION_TO_ID.items():
                    sid = f"{spk}_{word}_{emo}"
                    rows.append(
                        {
                            "sample_id": sid,
                            "filepath": f"synthetic/{sid}.wav",
                            "speaker": spk,
                            "word": word,
                            "transcript": transcript,
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
                    sample_to_transcript[sid] = {
                        "sample_id": sid,
                        "transcript": transcript,
                        "transcript_id": t_id,
                        "feature_file": f"transcripts/{t_id}.pt",
                        "word": word,
                        "speaker": spk,
                        "emotion": emo,
                        "label_id": label_id,
                    }

    manifest_path = tmp_path / "synthetic_text_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    (db_dir / "sample_to_transcript.json").write_text(
        json.dumps(sample_to_transcript, indent=2), encoding="utf-8"
    )
    return manifest_path, db_dir


def test_dataset_loading_train_and_val(tmp_path: Path):
    """Items 1 & 2: Verify training and validation TESSFeatureDataset loading and collation."""
    manifest_path, db_dir = _create_synthetic_text_fixture(tmp_path)

    train_ds = TESSFeatureDataset(
        manifest_path=manifest_path,
        split_type="word_indep_split",
        split="train",
        modalities=("distilbert",),
        distilbert_cache_dir=db_dir,
    )
    val_ds = TESSFeatureDataset(
        manifest_path=manifest_path,
        split_type="word_indep_split",
        split="val",
        modalities=("distilbert",),
        distilbert_cache_dir=db_dir,
    )

    assert len(train_ds) == 28  # 2 words * 2 speakers * 7 emotions
    assert len(val_ds) == 14    # 1 word * 2 speakers * 7 emotions

    loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=tess_feature_collate_fn)
    batch = next(iter(loader))
    assert set(batch.keys()) >= {"text_features", "text_lengths", "text_mask", "text_input_ids", "labels"}
    assert batch["text_features"].shape[0] == 8
    assert batch["text_features"].shape[2] == 768
    assert batch["text_mask"].dtype == torch.bool


def test_training_step_optimizer_update_and_metrics(tmp_path: Path):
    """Items 3, 4, 5, 6: Verify forward/backward step, optimizer parameter update, and Macro-F1 math."""
    manifest_path, db_dir = _create_synthetic_text_fixture(tmp_path)
    set_reproducible_seed(42)

    train_ds = TESSFeatureDataset(
        manifest_path=manifest_path,
        split_type="word_indep_split",
        split="train",
        modalities=("distilbert",),
        distilbert_cache_dir=db_dir,
    )
    val_ds = TESSFeatureDataset(
        manifest_path=manifest_path,
        split_type="word_indep_split",
        split="val",
        modalities=("distilbert",),
        distilbert_cache_dir=db_dir,
    )
    train_loader = DataLoader(train_ds, batch_size=14, shuffle=False, collate_fn=tess_feature_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=14, shuffle=False, collate_fn=tess_feature_collate_fn)

    model = build_text_model("distilbert_bilstm_attn", dropout=0.1)
    before_weights = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    train_metrics, _, fb = run_text_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        device=torch.device("cpu"),
        optimizer=optimizer,
    )
    assert fb is False
    assert np.isfinite(train_metrics["loss"])

    # Confirm optimizer actually updated model parameters
    any_changed = any(
        not torch.equal(before_weights[k], v.detach()) for k, v in model.named_parameters()
    )
    assert any_changed, "Optimizer step did not update model parameters!"

    val_metrics, _, _ = run_text_epoch(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=torch.device("cpu"),
        optimizer=None,
    )
    for key in ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert key in val_metrics and np.isfinite(val_metrics[key])

    # Exact Macro-F1 sanity check against known predictions
    known = compute_classification_metrics([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 2])
    assert abs(known["accuracy"] - (5 / 6)) < 1e-6
    assert 0.0 < known["macro_f1"] < 1.0


def test_early_stopping_f1_resets_patience_and_loss_tiebreak_does_not(tmp_path: Path):
    """Items 7, 8, 9: Verify Macro-F1 improvement resets patience, while loss-only tie-break does not."""
    manifest_path, db_dir = _create_synthetic_text_fixture(tmp_path)
    out_root = tmp_path / "early_stop_text"

    # Validation schedule:
    # Ep 1: F1=0.20, loss=1.90 -> f1_improved=True  (patience=0, best_ep=1)
    # Ep 2: F1=0.20, loss=1.85 -> loss_tiebreak=True (patience=1, best_ep=2)
    # Ep 3: F1=0.35, loss=1.75 -> f1_improved=True  (patience=0, best_ep=3)  <-- resets patience!
    # Ep 4: F1=0.35, loss=1.70 -> loss_tiebreak=True (patience=1, best_ep=4)
    # Ep 5: F1=0.35, loss=1.65 -> loss_tiebreak=True (patience=2, best_ep=5)  <-- stops (patience=2)!
    val_schedule = [
        (0.20, 1.90),
        (0.20, 1.85),
        (0.35, 1.75),
        (0.35, 1.70),
        (0.35, 1.65),
        (0.35, 1.60),
    ]
    call_idx = {"val": 0}

    def fake_run_text_epoch(model, dataloader, criterion, device, optimizer=None, grad_clip_norm=1.0):
        if optimizer is not None:
            return (
                {"loss": 1.5, "accuracy": 0.3, "macro_precision": 0.3, "macro_recall": 0.3, "macro_f1": 0.3},
                device,
                False,
            )
        f1_v, loss_v = val_schedule[call_idx["val"]]
        call_idx["val"] += 1
        return (
            {"loss": loss_v, "accuracy": f1_v, "macro_precision": f1_v, "macro_recall": f1_v, "macro_f1": f1_v},
            device,
            False,
        )

    cfg = TextTrainConfig(
        architecture="distilbert_bilstm_attn",
        split_type="word_independent",
        seed=42,
        batch_size=14,
        max_epochs=10,
        patience=2,
        device="cpu",
        manifest_path=str(manifest_path),
        distilbert_cache_dir=str(db_dir),
        output_root=str(out_root),
    )

    with patch.object(text_train_mod, "run_text_epoch", side_effect=fake_run_text_epoch):
        summary = train_single_text_run(cfg)

    assert summary["epochs_trained"] == 5
    assert summary["best_epoch"] == 5
    assert abs(summary["best_val_macro_f1"] - 0.35) < 1e-6
    assert abs(summary["best_val_loss"] - 1.65) < 1e-6

    hist = summary["history"]
    assert [h["f1_improved"] for h in hist] == [True, False, True, False, False]
    assert [h["loss_tiebreak"] for h in hist] == [False, True, False, True, True]
    assert [h["patience_counter"] for h in hist] == [0, 1, 0, 1, 2]


def test_checkpoint_save_load_resume_and_completed_skip(tmp_path: Path):
    """Items 10, 11, 12, 13: Verify checkpoint artifacts, mid-run resume, and completed-run skip."""
    manifest_path, db_dir = _create_synthetic_text_fixture(tmp_path)
    out_root = tmp_path / "resume_text"
    run_dir = get_text_run_output_dir(out_root, "distilbert_bilstm_attn", "word_independent", 42)

    real_run_epoch = text_train_mod.run_text_epoch
    val_counter = {"n": 0}

    def interrupting_epoch(*args, **kwargs):
        if kwargs.get("optimizer") is None:
            val_counter["n"] += 1
            if val_counter["n"] == 3:
                raise KeyboardInterrupt("Simulated disconnect at epoch 3")
        return real_run_epoch(*args, **kwargs)

    cfg = TextTrainConfig(
        architecture="distilbert_bilstm_attn",
        split_type="word_independent",
        seed=42,
        batch_size=14,
        max_epochs=4,
        patience=5,
        device="cpu",
        manifest_path=str(manifest_path),
        distilbert_cache_dir=str(db_dir),
        output_root=str(out_root),
    )

    with patch.object(text_train_mod, "run_text_epoch", side_effect=interrupting_epoch):
        with pytest.raises(KeyboardInterrupt, match="Simulated disconnect"):
            train_single_text_run(cfg)

    last_ckpt = run_dir / "last_checkpoint.pt"
    assert last_ckpt.exists()
    assert (run_dir / "best_model.pt").exists()
    assert not (run_dir / "training_history.json").exists()

    resume_data = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    assert resume_data["last_epoch"] == 2
    assert "optimizer_state_dict" in resume_data
    assert "scheduler_state_dict" in resume_data
    assert "train_generator_state" in resume_data

    # Reject incompatible resume config
    bad_cfg = TextTrainConfig(
        architecture="distilbert_bilstm_attn",
        split_type="word_independent",
        seed=42,
        batch_size=14,
        max_epochs=4,
        patience=5,
        lr=9e-2,
        device="cpu",
        manifest_path=str(manifest_path),
        distilbert_cache_dir=str(db_dir),
        output_root=str(out_root),
    )
    with pytest.raises(ValueError, match="Incompatible resume checkpoint"):
        train_single_text_run(bad_cfg)

    # Resume and verify only epochs 3 and 4 execute
    resumed_epochs = {"val_calls": 0}

    def counting_epoch(*args, **kwargs):
        if kwargs.get("optimizer") is None:
            resumed_epochs["val_calls"] += 1
        return real_run_epoch(*args, **kwargs)

    with patch.object(text_train_mod, "run_text_epoch", side_effect=counting_epoch):
        summary = train_single_text_run(cfg)

    assert resumed_epochs["val_calls"] == 2
    assert summary["epochs_trained"] == 4
    assert [h["epoch"] for h in summary["history"]] == [1, 2, 3, 4]

    for fname in (
        "best_model.pt",
        "last_checkpoint.pt",
        "train_config.json",
        "training_history.csv",
        "training_history.json",
    ):
        assert (run_dir / fname).exists(), f"Missing training artifact: {fname}"

    # Completed-run skip protection
    mtime_before = (run_dir / "best_model.pt").stat().st_mtime_ns
    with patch.object(text_train_mod, "run_text_epoch", side_effect=AssertionError("Should skip!")):
        skipped = train_single_text_run(cfg)
    assert skipped["epochs_trained"] == 4
    assert (run_dir / "best_model.pt").stat().st_mtime_ns == mtime_before


def test_reproducibility_with_same_seed(tmp_path: Path):
    """Item 15: Verify identical seed produces identical training loss and best_val_macro_f1."""
    manifest_path, db_dir = _create_synthetic_text_fixture(tmp_path)

    cfg1 = TextTrainConfig(
        architecture="distilbert_bilstm_attn",
        split_type="word_independent",
        seed=123,
        batch_size=14,
        max_epochs=2,
        patience=5,
        device="cpu",
        manifest_path=str(manifest_path),
        distilbert_cache_dir=str(db_dir),
        output_root=str(tmp_path / "run_a"),
    )
    cfg2 = TextTrainConfig(
        architecture="distilbert_bilstm_attn",
        split_type="word_independent",
        seed=123,
        batch_size=14,
        max_epochs=2,
        patience=5,
        device="cpu",
        manifest_path=str(manifest_path),
        distilbert_cache_dir=str(db_dir),
        output_root=str(tmp_path / "run_b"),
    )

    s1 = train_single_text_run(cfg1)
    s2 = train_single_text_run(cfg2)

    assert [h["train_loss"] for h in s1["history"]] == [h["train_loss"] for h in s2["history"]]
    assert [h["val_loss"] for h in s1["history"]] == [h["val_loss"] for h in s2["history"]]
    assert s1["best_val_macro_f1"] == s2["best_val_macro_f1"]


def test_evaluation_pipeline_smoke_no_grad_embeddings_overwrite_and_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
):
    """Items 14, 16, 17, 18, 19, 20: Smoke-test 1-epoch train -> eval, no-grad, embeddings, overwrite & missing."""
    manifest_path, db_dir = _create_synthetic_text_fixture(tmp_path)
    out_root = tmp_path / "eval_smoke_text"

    # Controlled 1-epoch synthetic training smoke test
    cfg = TextTrainConfig(
        architecture="distilbert_bilstm_attn",
        split_type="word_independent",
        seed=42,
        batch_size=14,
        max_epochs=1,
        patience=2,
        device="cpu",
        manifest_path=str(manifest_path),
        distilbert_cache_dir=str(db_dir),
        output_root=str(out_root),
    )
    train_summary = train_single_text_run(cfg)
    ckpt_path = Path(train_summary["checkpoint_path"])
    run_dir = ckpt_path.parent

    # Verify gradient computation is strictly disabled during evaluation
    grad_flags: List[bool] = []
    orig_build = build_text_model

    def spy_build_text_model(*args, **kwargs):
        m = orig_build(*args, **kwargs)
        orig_fwd = m.forward

        def wrapped_fwd(*f_args, **f_kwargs):
            grad_flags.append(torch.is_grad_enabled())
            return orig_fwd(*f_args, **f_kwargs)

        m.forward = wrapped_fwd  # type: ignore[method-assign]
        return m

    with patch("models.text_pipeline.test.build_text_model", side_effect=spy_build_text_model):
        eval_summary = evaluate_text_checkpoint(
            checkpoint_path=ckpt_path,
            output_dir=run_dir,
            manifest_path=manifest_path,
            distilbert_cache_dir=db_dir,
            batch_size=14,
            device="cpu",
            overwrite=False,
        )

    assert len(grad_flags) > 0 and all(flag is False for flag in grad_flags)
    assert eval_summary["num_test_samples"] == 14
    assert eval_summary["embedding_shape"] == [14, 256]

    for fname in (
        "test_metrics.json",
        "per_class_metrics.csv",
        "confusion_matrix.csv",
        "test_predictions.csv",
        "test_embeddings.pt",
        "test_embeddings.npz",
    ):
        assert (run_dir / fname).exists(), f"Missing eval artifact: {fname}"

    # Verify confusion matrix & predictions CSV integrity
    cm_df = pd.read_csv(run_dir / "confusion_matrix.csv", index_col=0)
    assert cm_df.shape == (7, 7)
    assert int(cm_df.values.sum()) == 14

    preds_df = pd.read_csv(run_dir / "test_predictions.csv")
    assert len(preds_df) == 14
    assert set(preds_df.columns) >= {"sample_id", "true_emotion", "pred_emotion", "true_label_id", "pred_label_id"}

    # Verify embeddings shape (N_test, 256) and attention weights saved
    emb_pt = torch.load(run_dir / "test_embeddings.pt", map_location="cpu", weights_only=False)
    emb_npz = np.load(run_dir / "test_embeddings.npz")
    assert tuple(emb_pt["embeddings"].shape) == (14, 256)
    assert emb_npz["embeddings"].shape == (14, 256)
    assert np.allclose(emb_pt["embeddings"].numpy(), emb_npz["embeddings"])
    assert emb_pt["attention_weights"].shape[0] == 14

    # Item 20: Overwrite protection on evaluation
    with pytest.raises(FileExistsError, match="Evaluation outputs already exist"):
        evaluate_text_checkpoint(
            checkpoint_path=ckpt_path,
            output_dir=run_dir,
            manifest_path=manifest_path,
            distilbert_cache_dir=db_dir,
            batch_size=14,
            device="cpu",
            overwrite=False,
        )

    # Item 14: Missing checkpoint handling in single and batch evaluation
    with pytest.raises(FileNotFoundError):
        evaluate_text_checkpoint(checkpoint_path=tmp_path / "does_not_exist.pt")

    batch_summary = evaluate_and_aggregate_text(
        output_root=out_root,
        splits=("word_independent", "cross_speaker"),
        architectures=("distilbert_bilstm_attn",),
        seeds=(42, 123),
        manifest_path=manifest_path,
        distilbert_cache_dir=db_dir,
        batch_size=14,
        device="cpu",
        overwrite=False,
    )
    out_text = capsys.readouterr().out
    assert "WARNING: Missing checkpoint" in out_text
    assert len(batch_summary["skipped_missing"]) == 3
    assert len(batch_summary["skipped_existing"]) == 1
    assert len(batch_summary["runs"]) == 1
    assert (out_root / "summaries" / "text_all_runs.json").exists()
    assert (out_root / "summaries" / "text_summary_mean_std.json").exists()
