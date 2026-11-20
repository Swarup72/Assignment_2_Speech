"""Evaluation pipeline for Text-Only Emotion Recognition (Stage 07B).

Loads a trained text checkpoint (`best_model.pt`), runs gradient-free inference on
the corresponding test partition, computes Accuracy, Macro-Precision, Macro-Recall,
Macro-F1, per-class Precision/Recall/F1, and confusion matrix, exports contextual
256-dimensional embeddings `h_t` and token attention weights, and aggregates
multi-seed summaries under `results/text/summaries/`.

Includes overwrite protection: never overwrites existing evaluation outputs unless
`overwrite=True` / `--overwrite` is explicitly requested.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from models.text_pipeline.model import (
    ID_TO_EMOTION,
    NUM_EMOTION_CLASSES,
    TEXT_ARCHITECTURES,
    build_text_model,
)
from models.text_pipeline.train import (
    DEFAULT_TEXT_RESULTS_DIR,
    SUPPORTED_SPLIT_CHOICES,
    _extract_text_batch_inputs,
    get_text_run_output_dir,
    normalize_split_column,
    release_device_memory,
    split_column_to_dir_name,
)
from src.data.feature_dataset import (
    TESSFeatureDataset,
    tess_feature_collate_fn,
)
from src.features.extract_speech_features import (
    DEFAULT_MANIFEST_PATH,
    select_device,
)
from src.features.extract_text_features import DEFAULT_DISTILBERT_CACHE_DIR


EMOTION_NAMES = [ID_TO_EMOTION[i] for i in range(NUM_EMOTION_CLASSES)]

EVAL_OUTPUT_FILES = (
    "test_metrics.json",
    "per_class_metrics.csv",
    "confusion_matrix.csv",
    "test_predictions.csv",
    "test_embeddings.pt",
    "test_embeddings.npz",
)


def _pad_attention_list(attn_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Pad variable-width `(B_i, T_i)` attention tensors to `(N_total, T_max)` with 0.0."""
    max_len = max(t.shape[1] for t in attn_tensors)
    padded_chunks: List[torch.Tensor] = []
    for t in attn_tensors:
        if t.shape[1] < max_len:
            pad_cols = max_len - t.shape[1]
            t = torch.nn.functional.pad(t, (0, pad_cols), mode="constant", value=0.0)
        padded_chunks.append(t)
    return torch.cat(padded_chunks, dim=0)


def evaluate_text_checkpoint(
    checkpoint_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
    distilbert_cache_dir: Union[str, Path] = DEFAULT_DISTILBERT_CACHE_DIR,
    batch_size: int = 32,
    device: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single trained text checkpoint on its test split and export embeddings."""
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(output_dir).expanduser() if output_dir is not None else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overwrite protection: refuse to overwrite existing evaluation results unless explicitly approved
    existing_eval_files = [f for f in EVAL_OUTPUT_FILES if (out_dir / f).exists()]
    if existing_eval_files and not overwrite:
        raise FileExistsError(
            f"Evaluation outputs already exist in {out_dir} ({existing_eval_files}). "
            f"Pass overwrite=True / --overwrite explicitly to replace them."
        )

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    architecture = str(payload["architecture"])
    split_type = normalize_split_column(str(payload["split_type"]))
    split_name = split_column_to_dir_name(split_type)
    seed = int(payload["seed"])
    model_cfg = payload.get("model_config", {})
    train_cfg = payload.get("train_config", {})

    test_dataset = TESSFeatureDataset(
        manifest_path=manifest_path,
        split_type=split_type,
        split="test",
        modalities=("distilbert",),
        distilbert_cache_dir=distilbert_cache_dir,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=tess_feature_collate_fn,
    )

    exec_device = select_device(device)
    model = build_text_model(
        architecture=architecture,
        dropout=float(model_cfg.get("dropout", 0.3)),
        num_classes=int(model_cfg.get("num_classes", NUM_EMOTION_CLASSES)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(exec_device)
    model.eval()

    all_sample_ids: List[str] = []
    all_speakers: List[str] = []
    all_words: List[str] = []
    all_transcripts: List[str] = []
    all_true_emotions: List[str] = []
    all_true_labels: List[int] = []
    all_pred_labels: List[int] = []
    all_probs: List[np.ndarray] = []
    all_embeddings: List[torch.Tensor] = []
    all_attn_weights: List[torch.Tensor] = []
    all_text_lengths: List[int] = []

    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch in test_loader:
            features, lengths, mask, labels = _extract_text_batch_inputs(batch, exec_device)
            outputs = model(features, lengths=lengths, mask=mask)
            logits = outputs["logits"].detach().cpu()
            probs = torch.softmax(logits, dim=-1).numpy()
            preds = logits.argmax(dim=-1).tolist()
            h_t = outputs["embedding"].detach().cpu().to(dtype=torch.float32)
            attn = outputs["attention_weights"].detach().cpu().to(dtype=torch.float32)

            all_sample_ids.extend(batch["sample_ids"])
            all_speakers.extend(batch["speakers"])
            all_words.extend(batch["words"])
            all_transcripts.extend(batch["transcripts"])
            all_true_emotions.extend(batch["emotions"])
            all_true_labels.extend(labels.detach().cpu().tolist())
            all_pred_labels.extend(preds)
            all_probs.append(probs)
            all_embeddings.append(h_t)
            all_attn_weights.append(attn)
            all_text_lengths.extend(lengths.detach().cpu().tolist())

    release_device_memory(exec_device)
    eval_time_sec = time.perf_counter() - t0

    probs_matrix = np.concatenate(all_probs, axis=0)  # (N_test, 7)
    embeddings_tensor = torch.cat(all_embeddings, dim=0)  # (N_test, 256)
    attn_tensor = _pad_attention_list(all_attn_weights)  # (N_test, T_max)
    lengths_tensor = torch.tensor(all_text_lengths, dtype=torch.long)

    acc = float(accuracy_score(all_true_labels, all_pred_labels))
    macro_prec = float(precision_score(all_true_labels, all_pred_labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(all_true_labels, all_pred_labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_true_labels, all_pred_labels, average="macro", zero_division=0))

    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        all_true_labels,
        all_pred_labels,
        labels=list(range(NUM_EMOTION_CLASSES)),
        zero_division=0,
    )
    per_class_f1: Dict[str, float] = {}
    per_class_rows: List[Dict[str, Any]] = []
    for idx, emo_name in enumerate(EMOTION_NAMES):
        per_class_f1[emo_name] = round(float(f1_cls[idx]), 6)
        per_class_rows.append(
            {
                "label_id": idx,
                "emotion": emo_name,
                "precision": round(float(p_cls[idx]), 6),
                "recall": round(float(r_cls[idx]), 6),
                "f1": round(float(f1_cls[idx]), 6),
                "support": int(sup_cls[idx]),
            }
        )

    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=list(range(NUM_EMOTION_CLASSES)))
    cm_df = pd.DataFrame(cm, index=EMOTION_NAMES, columns=EMOTION_NAMES)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    pred_rows: List[Dict[str, Any]] = []
    for i, sid in enumerate(all_sample_ids):
        true_id = all_true_labels[i]
        pred_id = all_pred_labels[i]
        row_dict: Dict[str, Any] = {
            "sample_id": sid,
            "speaker": all_speakers[i],
            "word": all_words[i],
            "transcript": all_transcripts[i],
            "true_emotion": all_true_emotions[i],
            "true_label_id": true_id,
            "pred_emotion": ID_TO_EMOTION[pred_id],
            "pred_label_id": pred_id,
            "correct": bool(true_id == pred_id),
            "confidence": round(float(probs_matrix[i, pred_id]), 6),
        }
        for c_idx, emo_name in enumerate(EMOTION_NAMES):
            row_dict[f"prob_{emo_name}"] = round(float(probs_matrix[i, c_idx]), 6)
        pred_rows.append(row_dict)

    pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False)

    # Export learned 256-d Contextual Modelling representations h_t + attention weights
    embedding_payload = {
        "architecture": architecture,
        "split_type": split_type,
        "split_name": split_name,
        "seed": seed,
        "block": "contextual_modelling_block",
        "embeddings": embeddings_tensor,  # (N_test, 256)
        "attention_weights": attn_tensor,  # (N_test, T_max)
        "text_lengths": lengths_tensor,
        "labels": torch.tensor(all_true_labels, dtype=torch.long),
        "predictions": torch.tensor(all_pred_labels, dtype=torch.long),
        "sample_ids": all_sample_ids,
        "speakers": all_speakers,
        "words": all_words,
        "transcripts": all_transcripts,
        "emotions": all_true_emotions,
    }
    torch.save(embedding_payload, out_dir / "test_embeddings.pt")
    np.savez_compressed(
        out_dir / "test_embeddings.npz",
        embeddings=embeddings_tensor.numpy(),
        attention_weights=attn_tensor.numpy(),
        text_lengths=lengths_tensor.numpy(),
        labels=np.array(all_true_labels, dtype=np.int64),
        predictions=np.array(all_pred_labels, dtype=np.int64),
        sample_ids=np.array(all_sample_ids),
        speakers=np.array(all_speakers),
        words=np.array(all_words),
        emotions=np.array(all_true_emotions),
    )

    test_summary = {
        "architecture": architecture,
        "split_type": split_type,
        "split_name": split_name,
        "seed": seed,
        "checkpoint_path": str(ckpt_path),
        "best_epoch": int(payload.get("best_epoch", 0)),
        "best_val_macro_f1": round(float(payload.get("best_val_macro_f1", 0.0)), 6),
        "num_test_samples": len(test_dataset),
        "embedding_shape": list(embeddings_tensor.shape),
        "attention_shape": list(attn_tensor.shape),
        "hardware_used": str(exec_device),
        "eval_time_sec": round(eval_time_sec, 4),
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "emotion_labels": EMOTION_NAMES,
        "model_config": model_cfg,
        "train_config": train_cfg,
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    print(
        f"[text_test] {architecture} | {split_name} | seed={seed} | "
        f"Acc={acc*100:.2f}% | Macro-P={macro_prec*100:.2f}% | "
        f"Macro-R={macro_rec*100:.2f}% | Macro-F1={macro_f1*100:.2f}%",
        flush=True,
    )
    return test_summary


def aggregate_text_results(
    output_root: Union[str, Path] = DEFAULT_TEXT_RESULTS_DIR,
    splits: Sequence[str] = ("word_independent", "cross_speaker"),
    architectures: Sequence[str] = TEXT_ARCHITECTURES,
    seeds: Sequence[int] = (42, 123, 999),
) -> Dict[str, Any]:
    """Aggregate individual text test metrics across seeds and write summary CSV/JSON tables."""
    root = Path(output_root).expanduser()
    summary_dir = root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    normalized_splits = list(dict.fromkeys(normalize_split_column(s) for s in splits))
    all_runs: List[Dict[str, Any]] = []

    for split_col in normalized_splits:
        for arch in architectures:
            for seed in seeds:
                run_dir = get_text_run_output_dir(root, arch, split_col, seed)
                metrics_file = run_dir / "test_metrics.json"
                if not metrics_file.exists():
                    continue
                with open(metrics_file, "r", encoding="utf-8") as f:
                    m = json.load(f)

                row: Dict[str, Any] = {
                    "architecture": arch,
                    "split_type": split_col,
                    "split_name": split_column_to_dir_name(split_col),
                    "seed": seed,
                    "best_epoch": m["best_epoch"],
                    "val_macro_f1": m["best_val_macro_f1"],
                    "test_accuracy": m["accuracy"],
                    "test_macro_precision": m["macro_precision"],
                    "test_macro_recall": m["macro_recall"],
                    "test_macro_f1": m["macro_f1"],
                    "num_test_samples": m["num_test_samples"],
                    "checkpoint_path": m["checkpoint_path"],
                }
                for emo_name in EMOTION_NAMES:
                    row[f"f1_{emo_name}"] = m["per_class_f1"][emo_name]
                all_runs.append(row)

    runs_df = pd.DataFrame(all_runs)
    if len(runs_df) > 0:
        runs_df.to_csv(summary_dir / "text_all_runs.csv", index=False)
    with open(summary_dir / "text_all_runs.json", "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    agg_rows: List[Dict[str, Any]] = []
    for split_col in normalized_splits:
        for arch in architectures:
            sub = [r for r in all_runs if r["split_type"] == split_col and r["architecture"] == arch]
            if not sub:
                continue

            accs = np.array([r["test_accuracy"] for r in sub], dtype=np.float64)
            precs = np.array([r["test_macro_precision"] for r in sub], dtype=np.float64)
            recs = np.array([r["test_macro_recall"] for r in sub], dtype=np.float64)
            f1s = np.array([r["test_macro_f1"] for r in sub], dtype=np.float64)
            ddof = 1 if len(sub) > 1 else 0

            agg_item: Dict[str, Any] = {
                "architecture": arch,
                "split_type": split_col,
                "split_name": split_column_to_dir_name(split_col),
                "num_seeds": len(sub),
                "seeds": [r["seed"] for r in sub],
                "accuracy_mean": round(float(np.mean(accs)), 6),
                "accuracy_std": round(float(np.std(accs, ddof=ddof)), 6),
                "macro_precision_mean": round(float(np.mean(precs)), 6),
                "macro_precision_std": round(float(np.std(precs, ddof=ddof)), 6),
                "macro_recall_mean": round(float(np.mean(recs)), 6),
                "macro_recall_std": round(float(np.std(recs, ddof=ddof)), 6),
                "macro_f1_mean": round(float(np.mean(f1s)), 6),
                "macro_f1_std": round(float(np.std(f1s, ddof=ddof)), 6),
                "accuracy_formatted": f"{np.mean(accs)*100:.2f} ± {np.std(accs, ddof=ddof)*100:.2f}%",
                "macro_precision_formatted": f"{np.mean(precs)*100:.2f} ± {np.std(precs, ddof=ddof)*100:.2f}%",
                "macro_recall_formatted": f"{np.mean(recs)*100:.2f} ± {np.std(recs, ddof=ddof)*100:.2f}%",
                "macro_f1_formatted": f"{np.mean(f1s)*100:.2f} ± {np.std(f1s, ddof=ddof)*100:.2f}%",
            }
            for emo_name in EMOTION_NAMES:
                emo_vals = np.array([r[f"f1_{emo_name}"] for r in sub], dtype=np.float64)
                agg_item[f"f1_{emo_name}_mean"] = round(float(np.mean(emo_vals)), 6)
                agg_item[f"f1_{emo_name}_std"] = round(float(np.std(emo_vals, ddof=ddof)), 6)
            agg_rows.append(agg_item)

    agg_df = pd.DataFrame(agg_rows)
    if len(agg_df) > 0:
        agg_df.to_csv(summary_dir / "text_summary_mean_std.csv", index=False)
    with open(summary_dir / "text_summary_mean_std.json", "w", encoding="utf-8") as f:
        json.dump(agg_rows, f, indent=2)

    return {"runs": all_runs, "aggregated": agg_rows}


def evaluate_and_aggregate_text(
    output_root: Union[str, Path] = DEFAULT_TEXT_RESULTS_DIR,
    splits: Sequence[str] = ("word_independent", "cross_speaker"),
    architectures: Sequence[str] = TEXT_ARCHITECTURES,
    seeds: Sequence[int] = (42, 123, 999),
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
    distilbert_cache_dir: Union[str, Path] = DEFAULT_DISTILBERT_CACHE_DIR,
    batch_size: int = 32,
    device: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Evaluate available text checkpoints, warn & skip missing ones, and aggregate summaries."""
    evaluated_runs: List[Dict[str, Any]] = []
    skipped_missing: List[Dict[str, Any]] = []
    skipped_existing: List[Dict[str, Any]] = []

    normalized_splits = list(dict.fromkeys(normalize_split_column(s) for s in splits))

    for split_col in normalized_splits:
        for arch in architectures:
            for seed in seeds:
                run_dir = get_text_run_output_dir(output_root, arch, split_col, seed)
                ckpt_path = run_dir / "best_model.pt"
                if not ckpt_path.exists():
                    print(
                        f"[text_test] WARNING: Missing checkpoint for "
                        f"architecture={arch}, split={split_column_to_dir_name(split_col)}, "
                        f"seed={seed} at {ckpt_path}; skipping combination.",
                        flush=True,
                    )
                    skipped_missing.append(
                        {
                            "architecture": arch,
                            "split_type": split_col,
                            "seed": seed,
                            "expected_checkpoint": str(ckpt_path),
                        }
                    )
                    continue

                if (run_dir / "test_metrics.json").exists() and not overwrite:
                    print(
                        f"[text_test] SKIPPING already-evaluated run: "
                        f"{arch} | {split_column_to_dir_name(split_col)} | seed={seed} "
                        f"(pass --overwrite to re-evaluate).",
                        flush=True,
                    )
                    skipped_existing.append(
                        {
                            "architecture": arch,
                            "split_type": split_col,
                            "seed": seed,
                            "run_dir": str(run_dir),
                        }
                    )
                    continue

                res = evaluate_text_checkpoint(
                    checkpoint_path=ckpt_path,
                    output_dir=run_dir,
                    manifest_path=manifest_path,
                    distilbert_cache_dir=distilbert_cache_dir,
                    batch_size=batch_size,
                    device=device,
                    overwrite=overwrite,
                )
                evaluated_runs.append(res)

    summary = aggregate_text_results(
        output_root=output_root,
        splits=normalized_splits,
        architectures=architectures,
        seeds=seeds,
    )
    summary["skipped_missing"] = skipped_missing
    summary["skipped_existing"] = skipped_existing
    summary["num_evaluated"] = len(evaluated_runs)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained text-only emotion recognition checkpoints.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a single best_model.pt checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory when evaluating a single --checkpoint.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(TEXT_ARCHITECTURES),
        choices=list(TEXT_ARCHITECTURES),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["word_independent", "cross_speaker"],
        choices=list(SUPPORTED_SPLIT_CHOICES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TEXT_RESULTS_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow overwriting existing evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None:
        evaluate_text_checkpoint(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            overwrite=args.overwrite,
        )
        return

    summary = evaluate_and_aggregate_text(
        output_root=args.output_root,
        splits=args.splits,
        architectures=args.architectures,
        seeds=args.seeds,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    print("[text_test] Aggregated summary:", flush=True)
    for item in summary["aggregated"]:
        print(
            f"  {item['architecture']} | {item['split_name']} ({item['num_seeds']} seeds): "
            f"Acc={item['accuracy_formatted']} | "
            f"Macro-P={item['macro_precision_formatted']} | "
            f"Macro-R={item['macro_recall_formatted']} | "
            f"Macro-F1={item['macro_f1_formatted']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
