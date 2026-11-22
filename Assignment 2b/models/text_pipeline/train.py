"""Training pipeline for Text-Only Emotion Recognition (Stage 07B).

Uses `DistilBERTBiLSTMAttentionClassifier` on cached DistilBERT token embeddings
loaded via `TESSFeatureDataset` and `tess_feature_collate_fn`.

Key Features:
- Supports `word_independent` (`word_indep_split`) and `cross_speaker` (`cross_speaker_split`)
  evaluation protocols across configurable random seeds (`42`, `123`, `999`).
- AdamW (`lr=1e-3`, `weight_decay=1e-2`), CosineAnnealingLR (`eta_min=1e-5`),
  CrossEntropyLoss (`label_smoothing=0.1`).
- Decoupled early stopping:
  - `f1_improved`: resets `epochs_without_improvement = 0` and updates `best_model.pt`.
  - `loss_tiebreak`: updates `best_model.pt` on tied Macro-F1 with lower validation loss,
    without resetting `epochs_without_improvement`.
- Atomic epoch-level resume checkpoint (`last_checkpoint.pt`) preserving latest model weights,
  optimizer, scheduler, history, patience counter, and Python/NumPy/Torch/CUDA/DataLoader RNG states.
- Completed-run protection: skips finished runs (`best_model.pt` + `training_history.json` exist)
  unless `--overwrite` is explicitly requested.
- Outputs isolated under `results/text/{architecture}/{split_dir}/seed_{seed}/` (or `--output-root`).
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from models.text_pipeline.model import (
    ID_TO_EMOTION,
    NUM_EMOTION_CLASSES,
    TEXT_ARCHITECTURES,
    build_text_model,
)
from src.data.feature_dataset import (
    TESSFeatureDataset,
    VALID_SPLIT_COLUMNS,
    tess_feature_collate_fn,
)
from src.features.extract_speech_features import (
    DEFAULT_MANIFEST_PATH,
    select_device,
)
from src.features.extract_text_features import DEFAULT_DISTILBERT_CACHE_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_RESULTS_DIR = PROJECT_ROOT / "Results" / "text"

F1_IMPROVEMENT_TOL = 1e-6
LOSS_TIEBREAK_TOL = 1e-6

# Map friendly CLI/directory split names and manifest column names cleanly
SPLIT_ALIAS_TO_COLUMN: Dict[str, str] = {
    "word_independent": "word_indep_split",
    "word_indep_split": "word_indep_split",
    "cross_speaker": "cross_speaker_split",
    "cross_speaker_split": "cross_speaker_split",
}
COLUMN_TO_SPLIT_DIR: Dict[str, str] = {
    "word_indep_split": "word_independent",
    "cross_speaker_split": "cross_speaker",
}
SUPPORTED_SPLIT_CHOICES = (
    "word_independent",
    "cross_speaker",
    "word_indep_split",
    "cross_speaker_split",
)


def normalize_split_column(split_name: str) -> str:
    """Resolve a CLI split name (`word_independent` or `word_indep_split`) to manifest column name."""
    if split_name not in SPLIT_ALIAS_TO_COLUMN:
        raise ValueError(
            f"Unsupported split '{split_name}'. Expected one of {SUPPORTED_SPLIT_CHOICES}."
        )
    col = SPLIT_ALIAS_TO_COLUMN[split_name]
    assert col in VALID_SPLIT_COLUMNS
    return col


def split_column_to_dir_name(split_column: str) -> str:
    """Convert manifest split column (`word_indep_split`) to output directory name (`word_independent`)."""
    col = normalize_split_column(split_column)
    return COLUMN_TO_SPLIT_DIR[col]


@dataclass
class TextTrainConfig:
    """Configuration for a single text-only emotion recognition training run."""

    architecture: str = "distilbert_bilstm_attn"
    split_type: str = "word_indep_split"
    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 12
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    dropout: float = 0.3
    grad_clip_norm: float = 1.0
    device: Optional[str] = None
    manifest_path: str = str(DEFAULT_MANIFEST_PATH)
    distilbert_cache_dir: str = str(DEFAULT_DISTILBERT_CACHE_DIR)
    output_root: str = str(DEFAULT_TEXT_RESULTS_DIR)
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.split_type = normalize_split_column(self.split_type)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and cuDNN deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def release_device_memory(device: torch.device) -> None:
    """Synchronize and release cached accelerator memory on MPS or CUDA."""
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def get_text_run_output_dir(
    output_root: Union[str, Path],
    architecture: str,
    split_type: str,
    seed: int,
) -> Path:
    """Return canonical directory path `{output_root}/{architecture}/{split_dir}/seed_{seed}`."""
    split_dir = split_column_to_dir_name(split_type)
    return Path(output_root).expanduser() / architecture / split_dir / f"seed_{seed}"


def is_text_run_completed(run_dir: Path) -> bool:
    """Check whether a text training run has completed and saved valid final artifacts."""
    best_ckpt = run_dir / "best_model.pt"
    hist_json = run_dir / "training_history.json"
    return (
        best_ckpt.exists()
        and best_ckpt.stat().st_size > 0
        and hist_json.exists()
        and hist_json.stat().st_size > 0
    )


def _validate_resume_compatibility(
    resume_payload: Dict[str, Any],
    config: TextTrainConfig,
) -> None:
    """Verify that `last_checkpoint.pt` matches the requested run configuration and seed."""
    saved_arch = resume_payload.get("architecture")
    saved_split = resume_payload.get("split_type")
    saved_seed = resume_payload.get("seed")

    if (
        saved_arch != config.architecture
        or saved_split != config.split_type
        or int(saved_seed) != int(config.seed)
    ):
        raise ValueError(
            f"Incompatible resume checkpoint identity: checkpoint has "
            f"(architecture={saved_arch}, split_type={saved_split}, seed={saved_seed}) "
            f"but current config requested "
            f"(architecture={config.architecture}, split_type={config.split_type}, seed={config.seed})."
        )

    saved_cfg = resume_payload.get("train_config", {})
    critical_keys = (
        "batch_size",
        "max_epochs",
        "lr",
        "min_lr",
        "weight_decay",
        "label_smoothing",
        "dropout",
    )
    current_cfg = config.to_dict()
    for key in critical_keys:
        if key in saved_cfg and saved_cfg[key] != current_cfg[key]:
            raise ValueError(
                f"Incompatible resume checkpoint hyperparameter '{key}': "
                f"checkpoint has {saved_cfg[key]!r} vs requested {current_cfg[key]!r}."
            )


def _extract_text_batch_inputs(
    batch: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract `(text_features, text_lengths, text_mask, labels)` onto `device`."""
    features = batch["text_features"].to(device)
    lengths = batch["text_lengths"].to(device)
    mask = batch["text_mask"].to(device)
    labels = batch["labels"].to(device)
    return features, lengths, mask, labels


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    """Compute Accuracy, Macro-Precision, Macro-Recall, and Macro-F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_text_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = 1.0,
) -> Tuple[Dict[str, float], torch.device, bool]:
    """Run one training or validation epoch for the text model with MPS->CPU fallback."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    fallback_triggered = False
    active_device = device

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for batch in dataloader:
            try:
                features, lengths, mask, labels = _extract_text_batch_inputs(batch, active_device)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                out = model(features, lengths=lengths, mask=mask)
                logits = out["logits"]
                loss = criterion(logits, labels)
                if is_train:
                    loss.backward()
                    if grad_clip_norm is not None and grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
            except RuntimeError as exc:
                if active_device.type == "mps":
                    fallback_triggered = True
                    active_device = torch.device("cpu")
                    model.to(active_device)
                    features, lengths, mask, labels = _extract_text_batch_inputs(batch, active_device)
                    if is_train:
                        optimizer.zero_grad(set_to_none=True)
                    out = model(features, lengths=lengths, mask=mask)
                    logits = out["logits"]
                    loss = criterion(logits, labels)
                    if is_train:
                        loss.backward()
                        if grad_clip_norm is not None and grad_clip_norm > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()
                else:
                    raise exc

            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_samples += batch_size

            all_preds.extend(logits.detach().argmax(dim=-1).cpu().tolist())
            all_targets.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss
    return metrics, active_device, fallback_triggered


def train_single_text_run(config: TextTrainConfig) -> Dict[str, Any]:
    """Train a single text-only run with Macro-F1 early stopping, resume, and skip protection."""
    if config.architecture not in TEXT_ARCHITECTURES:
        raise ValueError(f"Invalid text architecture: {config.architecture}")
    split_col = normalize_split_column(config.split_type)
    config.split_type = split_col

    run_dir = get_text_run_output_dir(
        output_root=config.output_root,
        architecture=config.architecture,
        split_type=config.split_type,
        seed=config.seed,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Completed-run protection: never overwrite completed runs unless overwrite=True
    if not config.overwrite and is_text_run_completed(run_dir):
        with open(run_dir / "training_history.json", "r", encoding="utf-8") as f:
            existing_summary = json.load(f)
        print(
            f"[text_train] SKIPPING completed run: {config.architecture} | "
            f"{split_column_to_dir_name(config.split_type)} | seed={config.seed} "
            f"(best_epoch={existing_summary.get('best_epoch')}, "
            f"best_val_macro_f1={existing_summary.get('best_val_macro_f1')})",
            flush=True,
        )
        return existing_summary

    set_reproducible_seed(config.seed)

    train_dataset = TESSFeatureDataset(
        manifest_path=config.manifest_path,
        split_type=config.split_type,
        split="train",
        modalities=("distilbert",),
        distilbert_cache_dir=config.distilbert_cache_dir,
    )
    val_dataset = TESSFeatureDataset(
        manifest_path=config.manifest_path,
        split_type=config.split_type,
        split="val",
        modalities=("distilbert",),
        distilbert_cache_dir=config.distilbert_cache_dir,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=tess_feature_collate_fn,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=tess_feature_collate_fn,
    )

    exec_device = select_device(config.device)
    initial_device_str = str(exec_device)

    model = build_text_model(
        architecture=config.architecture,
        dropout=config.dropout,
        num_classes=NUM_EMOTION_CLASSES,
    ).to(exec_device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.max_epochs, 1),
        eta_min=config.min_lr,
    )

    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    cpu_fallback_used = False
    elapsed_prior_sec = 0.0
    history: List[Dict[str, Any]] = []
    checkpoint_path = run_dir / "best_model.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"

    # Resume from last_checkpoint.pt if an interrupted run exists
    if not config.overwrite and last_checkpoint_path.exists():
        resume_payload = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        _validate_resume_compatibility(resume_payload, config)

        model.load_state_dict(resume_payload["model_state_dict"])
        model.to(exec_device)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])

        start_epoch = int(resume_payload["last_epoch"]) + 1
        best_val_macro_f1 = float(resume_payload["best_val_macro_f1"])
        best_val_loss = float(resume_payload["best_val_loss"])
        best_epoch = int(resume_payload["best_epoch"])
        epochs_without_improvement = int(resume_payload["epochs_without_improvement"])
        history = list(resume_payload.get("history", []))
        elapsed_prior_sec = float(resume_payload.get("elapsed_sec", 0.0))
        cpu_fallback_used = bool(resume_payload.get("cpu_fallback_used", False))

        if resume_payload.get("train_generator_state") is not None:
            train_generator.set_state(resume_payload["train_generator_state"].cpu())
        if resume_payload.get("python_rng_state") is not None:
            random.setstate(resume_payload["python_rng_state"])
        if resume_payload.get("numpy_rng_state") is not None:
            np.random.set_state(resume_payload["numpy_rng_state"])
        if resume_payload.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_state_all") is not None:
            try:
                cuda_states = [s.cpu() for s in resume_payload["cuda_rng_state_all"]]
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

        print(
            f"[text_train] RESUMING {config.architecture} | "
            f"{split_column_to_dir_name(config.split_type)} | seed={config.seed} "
            f"from epoch {start_epoch}/{config.max_epochs} "
            f"(best_val_macro_f1={best_val_macro_f1:.4f} @ ep {best_epoch}, "
            f"patience={epochs_without_improvement}/{config.patience})",
            flush=True,
        )

    t0 = time.perf_counter()

    for epoch in range(start_epoch, config.max_epochs + 1):
        if epochs_without_improvement >= config.patience:
            break

        epoch_t0 = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_metrics, exec_device, train_fb = run_text_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=exec_device,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_metrics, exec_device, val_fb = run_text_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=exec_device,
            optimizer=None,
        )
        scheduler.step()

        if train_fb or val_fb:
            cpu_fallback_used = True

        val_f1 = val_metrics["macro_f1"]
        val_loss = val_metrics["loss"]

        # Decoupled early stopping & checkpoint tie-breaker
        f1_improved = val_f1 > (best_val_macro_f1 + F1_IMPROVEMENT_TOL)
        loss_tiebreak = (
            (not f1_improved)
            and (abs(val_f1 - best_val_macro_f1) <= F1_IMPROVEMENT_TOL)
            and (val_loss < best_val_loss - LOSS_TIEBREAK_TOL)
        )
        should_save_ckpt = f1_improved or loss_tiebreak

        if f1_improved:
            best_val_macro_f1 = val_f1
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if loss_tiebreak:
                best_val_loss = val_loss

        if should_save_ckpt:
            best_epoch = epoch
            state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "architecture": config.architecture,
                    "split_type": config.split_type,
                    "split_name": split_column_to_dir_name(config.split_type),
                    "seed": config.seed,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_accuracy": val_metrics["accuracy"],
                    "best_val_loss": best_val_loss,
                    "model_config": {
                        "architecture": config.architecture,
                        "dropout": config.dropout,
                        "num_classes": NUM_EMOTION_CLASSES,
                    },
                    "train_config": config.to_dict(),
                    "state_dict": state_dict_cpu,
                },
                checkpoint_path,
            )

        epoch_time_sec = round(time.perf_counter() - epoch_t0, 4)
        epoch_record = {
            "epoch": epoch,
            "lr": round(current_lr, 8),
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "train_macro_f1": round(train_metrics["macro_f1"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_macro_precision": round(val_metrics["macro_precision"], 6),
            "val_macro_recall": round(val_metrics["macro_recall"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "f1_improved": f1_improved,
            "loss_tiebreak": loss_tiebreak,
            "is_best": should_save_ckpt,
            "patience_counter": epochs_without_improvement,
            "epoch_time_sec": epoch_time_sec,
        }
        history.append(epoch_record)

        print(
            f"[text_train] {config.architecture} | {split_column_to_dir_name(config.split_type)} | "
            f"seed={config.seed} | ep={epoch:02d}/{config.max_epochs:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f} (best={best_val_macro_f1:.4f} @ ep {best_epoch}) | "
            f"f1_imp={f1_improved} | tiebreak={loss_tiebreak} | "
            f"patience={epochs_without_improvement}/{config.patience} | {epoch_time_sec:.2f}s",
            flush=True,
        )

        # Save atomic epoch-level resume checkpoint
        latest_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        resume_state = {
            "architecture": config.architecture,
            "split_type": config.split_type,
            "split_name": split_column_to_dir_name(config.split_type),
            "seed": config.seed,
            "train_config": config.to_dict(),
            "last_epoch": epoch,
            "model_state_dict": latest_state_cpu,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "elapsed_sec": elapsed_prior_sec + (time.perf_counter() - t0),
            "cpu_fallback_used": cpu_fallback_used,
            "train_generator_state": train_generator.get_state().cpu(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state().cpu(),
            "cuda_rng_state_all": (
                [s.cpu() for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        }
        tmp_last_ckpt = run_dir / "last_checkpoint.pt.tmp"
        torch.save(resume_state, tmp_last_ckpt)
        tmp_last_ckpt.replace(last_checkpoint_path)

        if epochs_without_improvement >= config.patience:
            print(
                f"[text_train] Early stopping triggered at epoch {epoch} "
                f"(no Macro-F1 improvement for {config.patience} epochs).",
                flush=True,
            )
            break

    release_device_memory(exec_device)
    total_train_time_sec = elapsed_prior_sec + (time.perf_counter() - t0)

    history_df = pd.DataFrame(history)
    history_csv_path = run_dir / "training_history.csv"
    history_json_path = run_dir / "training_history.json"
    config_json_path = run_dir / "train_config.json"

    history_df.to_csv(history_csv_path, index=False)

    train_summary = {
        "architecture": config.architecture,
        "split_type": config.split_type,
        "split_name": split_column_to_dir_name(config.split_type),
        "seed": config.seed,
        "num_trainable_params": num_params,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 6),
        "best_val_loss": round(best_val_loss, 6),
        "hardware_requested": initial_device_str,
        "hardware_used": str(exec_device),
        "cpu_fallback_used": cpu_fallback_used,
        "training_time_sec": round(total_train_time_sec, 4),
        "checkpoint_path": str(checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "config": config.to_dict(),
        "history": history,
    }

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    print(
        f"[text_train] COMPLETED {config.architecture} | {split_column_to_dir_name(config.split_type)} | "
        f"seed={config.seed} | best_epoch={best_epoch}/{len(history)} | "
        f"best_val_macro_f1={best_val_macro_f1:.4f} | time={total_train_time_sec:.2f}s ({exec_device})",
        flush=True,
    )
    return train_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text-only emotion recognition models.")
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TEXT_RESULTS_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly overwrite completed runs instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split_arg in args.splits:
        split_col = normalize_split_column(split_arg)
        for arch in args.architectures:
            for seed in args.seeds:
                cfg = TextTrainConfig(
                    architecture=arch,
                    split_type=split_col,
                    seed=seed,
                    batch_size=args.batch_size,
                    max_epochs=args.epochs,
                    patience=args.patience,
                    lr=args.lr,
                    min_lr=args.min_lr,
                    weight_decay=args.weight_decay,
                    label_smoothing=args.label_smoothing,
                    dropout=args.dropout,
                    device=args.device,
                    output_root=str(args.output_root),
                    overwrite=args.overwrite,
                )
                train_single_text_run(cfg)


if __name__ == "__main__":
    main()
