"""Evaluation and aggregation pipeline for Multimodal Fusion (Stage 08B).

Evaluates finalized fusion runs (`gated_fusion`, `concat_fusion`, `late_fusion`)
on the untouched test partition using the matching frozen speech and text encoders,
computes canonical emotion classification metrics, exports predictions, fused
embeddings, gate activations (for gated fusion), confusion matrices, classification
reports, and aggregates multi-seed summaries under `results/fusion/summaries/`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from models.fusion_pipeline.model import (
    FUSION_ARCHITECTURES,
    ID_TO_EMOTION,
    LateFusionEnsemble,
    NUM_EMOTION_CLASSES,
    build_fusion_model,
)
from models.fusion_pipeline.train import (
    DEFAULT_FUSION_RESULTS_DIR,
    DEFAULT_SPEECH_ARCHITECTURE,
    DEFAULT_SPEECH_RESULTS_DIR,
    DEFAULT_TEXT_ARCHITECTURE,
    DEFAULT_TEXT_RESULTS_DIR,
    SUPPORTED_SPLIT_CHOICES,
    atomic_json_dump,
    extract_frozen_multimodal_features,
    get_fusion_run_output_dir,
    load_frozen_speech_encoder,
    load_frozen_text_encoder,
    normalize_split_column,
    release_device_memory,
    resolve_speech_checkpoint_path,
    resolve_text_checkpoint_path,
)
from models.speech_pipeline.model import build_speech_model
from models.text_pipeline.model import build_text_model
from src.data.feature_dataset import TESSFeatureDataset, tess_feature_collate_fn
from src.features.extract_speech_features import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_WAVLM_CACHE_DIR,
    select_device,
)
from src.features.extract_text_features import DEFAULT_DISTILBERT_CACHE_DIR


EMOTION_NAMES: List[str] = [ID_TO_EMOTION[i] for i in range(NUM_EMOTION_CLASSES)]

EVAL_OUTPUT_FILES = (
    "metrics.json",
    "test_predictions.csv",
    "confusion_matrix.csv",
    "per_class_metrics.csv",
    "classification_report.json",
)


def compute_full_evaluation_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Compute canonical evaluation metrics, per-class scores, confusion matrix, and report."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    probs_clipped = np.clip(np.asarray(probs_matrix, dtype=np.float64), 1e-12, 1.0)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=-1, keepdims=True)
    labels_range = list(range(NUM_EMOTION_CLASSES))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    ll = float(log_loss(y_true_arr, probs_clipped, labels=labels_range))

    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels_range,
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

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_range)
    cls_report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels_range,
        target_names=EMOTION_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "log_loss": round(ll, 6),
        "per_class_f1": per_class_f1,
        "per_class_rows": per_class_rows,
        "confusion_matrix": cm,
        "classification_report": cls_report,
    }


def _export_evaluation_artifacts(
    out_dir: Path,
    architecture: str,
    split_type: str,
    seed: int,
    test_bundle: Dict[str, Any],
    probs_matrix: np.ndarray,
    pred_labels: List[int],
    eval_time_sec: float,
    hardware_used: str,
    extra_metadata: Dict[str, Any],
    fused_embeddings: Optional[np.ndarray] = None,
    gate_activations: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Persist all standardized fusion test artifacts and return the metrics dictionary."""
    sample_ids = test_bundle["sample_ids"]
    true_labels = test_bundle["labels"].detach().cpu().tolist()
    speakers = test_bundle.get("speakers", [""] * len(sample_ids))
    words = test_bundle.get("words", [""] * len(sample_ids))
    true_emotions = test_bundle.get(
        "emotions", [ID_TO_EMOTION[int(y)] for y in true_labels]
    )

    eval_res = compute_full_evaluation_metrics(
        y_true=true_labels,
        y_pred=pred_labels,
        probs_matrix=probs_matrix,
    )

    # Per-class CSV and Confusion Matrix CSV
    pd.DataFrame(eval_res["per_class_rows"]).to_csv(
        out_dir / "per_class_metrics.csv", index=False
    )
    cm_df = pd.DataFrame(
        eval_res["confusion_matrix"], index=EMOTION_NAMES, columns=EMOTION_NAMES
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    atomic_json_dump(eval_res["classification_report"], out_dir / "classification_report.json")

    # Test predictions CSV with canonical 7 probability columns
    pred_rows: List[Dict[str, Any]] = []
    for i, sid in enumerate(sample_ids):
        true_id = int(true_labels[i])
        pred_id = int(pred_labels[i])
        row: Dict[str, Any] = {
            "sample_id": sid,
            "speaker": speakers[i],
            "word": words[i],
            "true_emotion": true_emotions[i],
            "true_label_id": true_id,
            "pred_emotion": ID_TO_EMOTION[pred_id],
            "pred_label_id": pred_id,
            "correct": bool(true_id == pred_id),
            "confidence": round(float(probs_matrix[i, pred_id]), 6),
        }
        for c_idx, emo_name in enumerate(EMOTION_NAMES):
            row[f"prob_{emo_name}"] = round(float(probs_matrix[i, c_idx]), 6)
        pred_rows.append(row)

    pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False)

    # Fused embeddings (for gated_fusion and concat_fusion)
    if fused_embeddings is not None:
        npz_kwargs: Dict[str, Any] = {
            "embeddings": fused_embeddings.astype(np.float32),
            "labels": np.array(true_labels, dtype=np.int64),
            "predictions": np.array(pred_labels, dtype=np.int64),
            "sample_ids": np.array(sample_ids),
            "speakers": np.array(speakers),
            "words": np.array(words),
            "emotions": np.array(true_emotions),
        }
        if gate_activations is not None:
            npz_kwargs["gates"] = gate_activations.astype(np.float32)
        np.savez_compressed(out_dir / "test_embeddings.npz", **npz_kwargs)

    # Dedicated gate activation artifacts for GatedMultimodalFusion analysis
    gate_stats: Optional[Dict[str, Any]] = None
    if gate_activations is not None:
        np.savez_compressed(
            out_dir / "gate_activations.npz",
            gates=gate_activations.astype(np.float32),
            sample_ids=np.array(sample_ids),
            labels=np.array(true_labels, dtype=np.int64),
            predictions=np.array(pred_labels, dtype=np.int64),
            emotions=np.array(true_emotions),
        )
        per_emotion_mean_gate: Dict[str, float] = {}
        for c_idx, emo_name in enumerate(EMOTION_NAMES):
            mask = np.array(true_labels, dtype=np.int64) == c_idx
            if np.any(mask):
                per_emotion_mean_gate[emo_name] = round(float(gate_activations[mask].mean()), 6)
        gate_stats = {
            "mean_gate": round(float(gate_activations.mean()), 6),
            "std_gate": round(float(gate_activations.std()), 6),
            "min_gate": round(float(gate_activations.min()), 6),
            "max_gate": round(float(gate_activations.max()), 6),
            "per_emotion_mean_gate": per_emotion_mean_gate,
        }
        atomic_json_dump(gate_stats, out_dir / "gate_summary.json")

    metrics_payload: Dict[str, Any] = {
        "architecture": architecture,
        "split_type": split_type,
        "seed": int(seed),
        "num_test_samples": len(sample_ids),
        "hardware_used": hardware_used,
        "eval_time_sec": round(eval_time_sec, 4),
        "accuracy": eval_res["accuracy"],
        "macro_precision": eval_res["macro_precision"],
        "macro_recall": eval_res["macro_recall"],
        "macro_f1": eval_res["macro_f1"],
        "weighted_f1": eval_res["weighted_f1"],
        "log_loss": eval_res["log_loss"],
        "per_class_f1": eval_res["per_class_f1"],
        "confusion_matrix": eval_res["confusion_matrix"].tolist(),
        "emotion_labels": EMOTION_NAMES,
    }
    if fused_embeddings is not None:
        metrics_payload["embedding_shape"] = list(fused_embeddings.shape)
    if gate_stats is not None:
        metrics_payload["gate_stats"] = gate_stats
    metrics_payload.update(extra_metadata)

    atomic_json_dump(metrics_payload, out_dir / "metrics.json")
    return metrics_payload


def evaluate_fusion_run(
    architecture: str,
    split_type: str,
    seed: int,
    output_root: Union[str, Path] = DEFAULT_FUSION_RESULTS_DIR,
    speech_results_root: Union[str, Path] = DEFAULT_SPEECH_RESULTS_DIR,
    text_results_root: Union[str, Path] = DEFAULT_TEXT_RESULTS_DIR,
    speech_architecture: str = DEFAULT_SPEECH_ARCHITECTURE,
    text_architecture: str = DEFAULT_TEXT_ARCHITECTURE,
    speech_checkpoint_path: Optional[Union[str, Path]] = None,
    text_checkpoint_path: Optional[Union[str, Path]] = None,
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
    wavlm_cache_dir: Union[str, Path] = DEFAULT_WAVLM_CACHE_DIR,
    distilbert_cache_dir: Union[str, Path] = DEFAULT_DISTILBERT_CACHE_DIR,
    batch_size: int = 32,
    device: Optional[str] = None,
    overwrite: bool = False,
    test_loader: Optional[DataLoader] = None,
    speech_model_builder: Callable[..., nn.Module] = build_speech_model,
    text_model_builder: Callable[..., nn.Module] = build_text_model,
) -> Dict[str, Any]:
    """Evaluate a single trained (`gated_fusion`/`concat_fusion`) or calibrated (`late_fusion`) run."""
    split_col = normalize_split_column(split_type)
    run_dir = get_fusion_run_output_dir(output_root, architecture, split_col, seed)
    if not run_dir.exists():
        raise FileNotFoundError(f"Fusion run directory not found: {run_dir}")

    existing_eval = [f for f in EVAL_OUTPUT_FILES if (run_dir / f).exists()]
    if existing_eval and not overwrite:
        raise FileExistsError(
            f"Evaluation outputs already exist in {run_dir} ({existing_eval}). "
            f"Pass overwrite=True / --overwrite explicitly to replace them."
        )

    speech_ckpt = (
        Path(speech_checkpoint_path)
        if speech_checkpoint_path is not None
        else resolve_speech_checkpoint_path(
            speech_results_root, split_col, seed, speech_architecture
        )
    )
    text_ckpt = (
        Path(text_checkpoint_path)
        if text_checkpoint_path is not None
        else resolve_text_checkpoint_path(
            text_results_root, split_col, seed, text_architecture
        )
    )

    exec_device = select_device(device)
    speech_encoder, speech_meta = load_frozen_speech_encoder(
        checkpoint_path=speech_ckpt,
        expected_split_type=split_col,
        expected_seed=seed,
        expected_architecture=speech_architecture,
        device=exec_device,
        model_builder=speech_model_builder,
    )
    text_encoder, text_meta = load_frozen_text_encoder(
        checkpoint_path=text_ckpt,
        expected_split_type=split_col,
        expected_seed=seed,
        expected_architecture=text_architecture,
        device=exec_device,
        model_builder=text_model_builder,
    )

    if test_loader is None:
        test_dataset = TESSFeatureDataset(
            manifest_path=manifest_path,
            split_type=split_col,
            split="test",
            modalities=("wavlm", "distilbert"),
            wavlm_cache_dir=wavlm_cache_dir,
            distilbert_cache_dir=distilbert_cache_dir,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=tess_feature_collate_fn,
        )

    t0 = time.perf_counter()
    test_bundle, exec_device, _ = extract_frozen_multimodal_features(
        dataloader=test_loader,
        speech_encoder=speech_encoder,
        text_encoder=text_encoder,
        device=exec_device,
    )

    if architecture == "late_fusion":
        sel_path = run_dir / "selected_lambda.json"
        if not sel_path.exists():
            raise FileNotFoundError(
                f"Late fusion calibration artifact not found: {sel_path}"
            )
        with open(sel_path, "r", encoding="utf-8") as f:
            sel_payload = json.load(f)
        selected_lambda = float(sel_payload["selected_lambda"])
        ensemble = LateFusionEnsemble(
            lambda_weight=selected_lambda, num_classes=NUM_EMOTION_CLASSES
        ).eval()

        with torch.inference_mode():
            out = ensemble(
                test_bundle["speech_logits"],
                test_bundle["text_logits"],
                lambda_weight=selected_lambda,
            )
        probs_matrix = out["probabilities"].detach().cpu().numpy()
        pred_labels = out["predictions"].detach().cpu().tolist()
        eval_time_sec = time.perf_counter() - t0
        release_device_memory(exec_device)

        return _export_evaluation_artifacts(
            out_dir=run_dir,
            architecture=architecture,
            split_type=split_col,
            seed=seed,
            test_bundle=test_bundle,
            probs_matrix=probs_matrix,
            pred_labels=pred_labels,
            eval_time_sec=eval_time_sec,
            hardware_used=str(exec_device),
            extra_metadata={
                "selected_lambda": selected_lambda,
                "calibration_path": str(sel_path),
                "speech_encoder_meta": speech_meta,
                "text_encoder_meta": text_meta,
            },
        )

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != architecture:
        raise ValueError(
            f"Fusion checkpoint architecture mismatch: expected '{architecture}', "
            f"got '{payload.get('architecture')}'."
        )
    if normalize_split_column(str(payload.get("split_type"))) != split_col:
        raise ValueError(
            f"Fusion checkpoint split mismatch: expected '{split_col}', "
            f"got '{payload.get('split_type')}'."
        )
    if int(payload.get("seed")) != int(seed):
        raise ValueError(
            f"Fusion checkpoint seed mismatch: expected {seed}, got {payload.get('seed')}."
        )

    model_cfg = payload.get("model_config", {})
    fusion_model = build_fusion_model(
        architecture=architecture,
        speech_dim=int(model_cfg.get("speech_dim", 256)),
        text_dim=int(model_cfg.get("text_dim", 256)),
        fusion_dim=int(model_cfg.get("fusion_dim", 256)),
        mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 128)),
        num_classes=int(model_cfg.get("num_classes", NUM_EMOTION_CLASSES)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )
    fusion_model.load_state_dict(payload["state_dict"])
    fusion_model.to(exec_device)
    fusion_model.eval()

    with torch.inference_mode():
        h_s = test_bundle["speech_embeddings"].to(exec_device)
        h_t = test_bundle["text_embeddings"].to(exec_device)
        outputs = fusion_model(h_s, h_t)
        logits = outputs["logits"].detach().cpu()
        probs_matrix = torch.softmax(logits, dim=-1).numpy()
        pred_labels = logits.argmax(dim=-1).tolist()
        fused_emb = outputs["embedding"].detach().cpu().numpy()
        gate_arr = (
            outputs["gate"].detach().cpu().numpy() if "gate" in outputs else None
        )

    eval_time_sec = time.perf_counter() - t0
    release_device_memory(exec_device)

    return _export_evaluation_artifacts(
        out_dir=run_dir,
        architecture=architecture,
        split_type=split_col,
        seed=seed,
        test_bundle=test_bundle,
        probs_matrix=probs_matrix,
        pred_labels=pred_labels,
        eval_time_sec=eval_time_sec,
        hardware_used=str(exec_device),
        extra_metadata={
            "checkpoint_path": str(ckpt_path),
            "best_epoch": int(payload.get("best_epoch", 0)),
            "best_val_macro_f1": round(float(payload.get("best_val_macro_f1", 0.0)), 6),
            "speech_encoder_meta": speech_meta,
            "text_encoder_meta": text_meta,
            "model_config": model_cfg,
        },
        fused_embeddings=fused_emb,
        gate_activations=gate_arr,
    )


def aggregate_fusion_results(
    output_root: Union[str, Path] = DEFAULT_FUSION_RESULTS_DIR,
    splits: Sequence[str] = ("word_indep_split", "cross_speaker_split"),
    architectures: Sequence[str] = FUSION_ARCHITECTURES,
    seeds: Sequence[int] = (42, 123, 999),
) -> Dict[str, Any]:
    """Aggregate completed fusion evaluation metrics across seeds into `results/fusion/summaries/`."""
    root = Path(output_root).expanduser()
    summary_dir = root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    normalized_splits = list(dict.fromkeys(normalize_split_column(s) for s in splits))
    all_runs: List[Dict[str, Any]] = []

    for split_col in normalized_splits:
        for arch in architectures:
            for seed in seeds:
                run_dir = get_fusion_run_output_dir(root, arch, split_col, seed)
                metrics_file = run_dir / "metrics.json"
                if not metrics_file.exists():
                    continue
                with open(metrics_file, "r", encoding="utf-8") as f:
                    m = json.load(f)

                row: Dict[str, Any] = {
                    "architecture": arch,
                    "split_type": split_col,
                    "seed": int(seed),
                    "test_accuracy": m["accuracy"],
                    "test_macro_precision": m["macro_precision"],
                    "test_macro_recall": m["macro_recall"],
                    "test_macro_f1": m["macro_f1"],
                    "test_weighted_f1": m["weighted_f1"],
                    "test_log_loss": m["log_loss"],
                    "num_test_samples": m["num_test_samples"],
                }
                if "selected_lambda" in m:
                    row["selected_lambda"] = m["selected_lambda"]
                for emo_name in EMOTION_NAMES:
                    row[f"f1_{emo_name}"] = m["per_class_f1"][emo_name]
                all_runs.append(row)

    if all_runs:
        pd.DataFrame(all_runs).to_csv(summary_dir / "fusion_all_runs.csv", index=False)
    atomic_json_dump(all_runs, summary_dir / "fusion_all_runs.json")

    agg_rows: List[Dict[str, Any]] = []
    for split_col in normalized_splits:
        for arch in architectures:
            sub = [
                r for r in all_runs if r["split_type"] == split_col and r["architecture"] == arch
            ]
            if not sub:
                continue

            accs = np.array([r["test_accuracy"] for r in sub], dtype=np.float64)
            precs = np.array([r["test_macro_precision"] for r in sub], dtype=np.float64)
            recs = np.array([r["test_macro_recall"] for r in sub], dtype=np.float64)
            f1s = np.array([r["test_macro_f1"] for r in sub], dtype=np.float64)
            wf1s = np.array([r["test_weighted_f1"] for r in sub], dtype=np.float64)
            ddof = 1 if len(sub) > 1 else 0

            agg_item: Dict[str, Any] = {
                "architecture": arch,
                "split_type": split_col,
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
                "weighted_f1_mean": round(float(np.mean(wf1s)), 6),
                "weighted_f1_std": round(float(np.std(wf1s, ddof=ddof)), 6),
                "accuracy_formatted": f"{np.mean(accs)*100:.2f} ± {np.std(accs, ddof=ddof)*100:.2f}%",
                "macro_f1_formatted": f"{np.mean(f1s)*100:.2f} ± {np.std(f1s, ddof=ddof)*100:.2f}%",
            }
            for emo_name in EMOTION_NAMES:
                vals = np.array([r[f"f1_{emo_name}"] for r in sub], dtype=np.float64)
                agg_item[f"f1_{emo_name}_mean"] = round(float(np.mean(vals)), 6)
                agg_item[f"f1_{emo_name}_std"] = round(float(np.std(vals, ddof=ddof)), 6)
            agg_rows.append(agg_item)

    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(summary_dir / "fusion_summary_mean_std.csv", index=False)
    atomic_json_dump(agg_rows, summary_dir / "fusion_summary_mean_std.json")

    return {"runs": all_runs, "aggregated": agg_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate finalized multimodal fusion models and aggregate multi-seed summaries."
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(FUSION_ARCHITECTURES),
        choices=list(FUSION_ARCHITECTURES),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["word_indep_split", "cross_speaker_split"],
        choices=list(SUPPORTED_SPLIT_CHOICES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--speech-results-root", type=Path, default=DEFAULT_SPEECH_RESULTS_DIR)
    parser.add_argument("--text-results-root", type=Path, default=DEFAULT_TEXT_RESULTS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FUSION_RESULTS_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow overwriting existing evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized_splits = list(dict.fromkeys(normalize_split_column(s) for s in args.splits))
    for split_col in normalized_splits:
        for arch in args.architectures:
            for seed in args.seeds:
                run_dir = get_fusion_run_output_dir(args.output_root, arch, split_col, seed)
                expected_artifact = (
                    run_dir / "selected_lambda.json"
                    if arch == "late_fusion"
                    else run_dir / "best_model.pt"
                )
                if not expected_artifact.exists():
                    print(
                        f"[fusion_test] WARNING: Missing artifact for {arch} | {split_col} | "
                        f"seed={seed} at {expected_artifact}; skipping.",
                        flush=True,
                    )
                    continue
                if (run_dir / "metrics.json").exists() and not args.overwrite:
                    continue
                evaluate_fusion_run(
                    architecture=arch,
                    split_type=split_col,
                    seed=seed,
                    output_root=args.output_root,
                    speech_results_root=args.speech_results_root,
                    text_results_root=args.text_results_root,
                    batch_size=args.batch_size,
                    device=args.device,
                    overwrite=args.overwrite,
                )

    aggregate_fusion_results(
        output_root=args.output_root,
        splits=normalized_splits,
        architectures=args.architectures,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
