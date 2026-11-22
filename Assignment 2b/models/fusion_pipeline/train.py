"""Training and calibration pipeline for Multimodal Fusion (Stage 08B).

Integrates the frozen unimodal speech (`WavLMBiGRUAttentionClassifier`) and text
(`DistilBERTBiLSTMAttentionClassifier`) checkpoints with the Stage 08A fusion
architectures:
- `gated_fusion` (`GatedMultimodalFusion`): Trainable gated multimodal fusion head.
- `concat_fusion` (`ConcatFusion`): Trainable feature concatenation fusion head.
- `late_fusion` (`LateFusionEnsemble`): Zero-parameter validation-calibrated
  decision-level probability ensemble.

Key Safeguards:
- Strict same-split, same-seed, same-architecture checkpoint validation.
- Frozen encoders strictly in `eval()` mode with `requires_grad=False`, excluded
  from the optimizer, and run exclusively under `torch.no_grad()`.
- Explicit `sample_id` and label alignment verification with duplicate/missing checks.
- Zero test-set access during training, early stopping, or lambda calibration.
- Decoupled Macro-F1 early stopping with validation-loss checkpoint tie-breaking.
- Atomic checkpoint saving (`best_model.pt`, `last_checkpoint.pt`), full RNG/optimizer
  state resume, and completed-run protection.
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from models.fusion_pipeline.model import (
    FUSION_ARCHITECTURES,
    ID_TO_EMOTION,
    LateFusionEnsemble,
    NUM_EMOTION_CLASSES,
    build_fusion_model,
)
from models.speech_pipeline.model import build_speech_model
from models.text_pipeline.model import build_text_model
from src.data.feature_dataset import (
    TESSFeatureDataset,
    VALID_SPLIT_COLUMNS,
    tess_feature_collate_fn,
)
from src.features.extract_speech_features import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_WAVLM_CACHE_DIR,
    select_device,
)
from src.features.extract_text_features import DEFAULT_DISTILBERT_CACHE_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEECH_RESULTS_DIR = PROJECT_ROOT / "Results" / "speech"
DEFAULT_TEXT_RESULTS_DIR = PROJECT_ROOT / "Results" / "text"
DEFAULT_FUSION_RESULTS_DIR = PROJECT_ROOT / "Results" / "fusion"

DEFAULT_SPEECH_ARCHITECTURE = "wavlm_bigru_attn"
DEFAULT_TEXT_ARCHITECTURE = "distilbert_bilstm_attn"
TRAINABLE_FUSION_ARCHITECTURES = ("gated_fusion", "concat_fusion")
DEFAULT_LAMBDA_GRID: Tuple[float, ...] = tuple(round(i * 0.05, 2) for i in range(21))

F1_IMPROVEMENT_TOL = 1e-6
LOSS_TIEBREAK_TOL = 1e-6

# Standardize fusion split names to manifest columns while accepting text aliases
SPLIT_ALIAS_TO_COLUMN: Dict[str, str] = {
    "word_indep_split": "word_indep_split",
    "word_independent": "word_indep_split",
    "cross_speaker_split": "cross_speaker_split",
    "cross_speaker": "cross_speaker_split",
}
COLUMN_TO_TEXT_SPLIT_DIR: Dict[str, str] = {
    "word_indep_split": "word_independent",
    "cross_speaker_split": "cross_speaker",
}
SUPPORTED_SPLIT_CHOICES = (
    "word_indep_split",
    "cross_speaker_split",
    "word_independent",
    "cross_speaker",
)


def normalize_split_column(split_name: str) -> str:
    """Resolve split aliases (`word_independent`/`word_indep_split`) to canonical column names."""
    if split_name not in SPLIT_ALIAS_TO_COLUMN:
        raise ValueError(
            f"Unsupported split '{split_name}'. Expected one of {SUPPORTED_SPLIT_CHOICES}."
        )
    col = SPLIT_ALIAS_TO_COLUMN[split_name]
    assert col in VALID_SPLIT_COLUMNS
    return col


def resolve_speech_checkpoint_path(
    speech_results_root: Union[str, Path],
    split_type: str,
    seed: int,
    speech_architecture: str = DEFAULT_SPEECH_ARCHITECTURE,
) -> Path:
    """Return canonical speech checkpoint path `{root}/{split_col}/{arch}/seed_{seed}/best_model.pt`."""
    split_col = normalize_split_column(split_type)
    return (
        Path(speech_results_root).expanduser()
        / split_col
        / speech_architecture
        / f"seed_{int(seed)}"
        / "best_model.pt"
    )


def resolve_text_checkpoint_path(
    text_results_root: Union[str, Path],
    split_type: str,
    seed: int,
    text_architecture: str = DEFAULT_TEXT_ARCHITECTURE,
) -> Path:
    """Return canonical text checkpoint path `{root}/{arch}/{text_split_dir}/seed_{seed}/best_model.pt`."""
    split_col = normalize_split_column(split_type)
    text_split_dir = COLUMN_TO_TEXT_SPLIT_DIR[split_col]
    return (
        Path(text_results_root).expanduser()
        / text_architecture
        / text_split_dir
        / f"seed_{int(seed)}"
        / "best_model.pt"
    )


def get_fusion_run_output_dir(
    output_root: Union[str, Path],
    architecture: str,
    split_type: str,
    seed: int,
) -> Path:
    """Return canonical fusion output path `{output_root}/{architecture}/{split_col}/seed_{seed}`."""
    if architecture not in FUSION_ARCHITECTURES:
        raise ValueError(
            f"Unknown fusion architecture '{architecture}'. Expected one of {FUSION_ARCHITECTURES}."
        )
    split_col = normalize_split_column(split_type)
    return Path(output_root).expanduser() / architecture / split_col / f"seed_{int(seed)}"


def atomic_torch_save(payload: Any, target_path: Union[str, Path]) -> None:
    """Atomically write a PyTorch artifact via a temporary sibling file."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(target)


def atomic_json_dump(payload: Any, target_path: Union[str, Path]) -> None:
    """Atomically write a UTF-8 JSON file via a temporary sibling file."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(target)


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, PyTorch, and cuDNN deterministic flags."""
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


def freeze_encoder(model: nn.Module) -> nn.Module:
    """Place an encoder strictly in eval() mode and disable gradients on all parameters."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def assert_encoder_frozen(model: nn.Module, name: str = "encoder") -> None:
    """Verify that an encoder is in eval() mode and has zero trainable parameters."""
    if model.training:
        raise RuntimeError(f"Frozen {name} must be in eval() mode, but model.training is True.")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(
            f"Frozen {name} has trainable parameters that must have requires_grad=False: "
            f"{trainable[:5]}"
        )


def validate_base_checkpoint_metadata(
    payload: Dict[str, Any],
    expected_architecture: str,
    expected_split_type: str,
    expected_seed: int,
    modality_label: str,
) -> Dict[str, Any]:
    """Validate frozen base checkpoint identity (architecture, split, seed, and state_dict)."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid {modality_label} checkpoint format: expected dict, got {type(payload).__name__}."
        )
    if "state_dict" not in payload:
        raise ValueError(f"{modality_label} checkpoint is missing required 'state_dict' key.")

    saved_arch = payload.get("architecture")
    if saved_arch != expected_architecture:
        raise ValueError(
            f"{modality_label} checkpoint architecture mismatch: expected '{expected_architecture}', "
            f"got '{saved_arch}'."
        )

    raw_split = payload.get("split_type")
    if raw_split is None:
        raise ValueError(f"{modality_label} checkpoint is missing required 'split_type' metadata.")
    saved_split = normalize_split_column(str(raw_split))
    expected_split = normalize_split_column(expected_split_type)
    if saved_split != expected_split:
        raise ValueError(
            f"{modality_label} checkpoint split mismatch: expected '{expected_split}', "
            f"got '{saved_split}'."
        )

    if "seed" not in payload or payload["seed"] is None:
        raise ValueError(f"{modality_label} checkpoint is missing required 'seed' metadata.")
    saved_seed = int(payload["seed"])
    if saved_seed != int(expected_seed):
        raise ValueError(
            f"{modality_label} checkpoint seed mismatch: expected {int(expected_seed)}, "
            f"got {saved_seed}."
        )

    model_cfg = payload.get("model_config", {})
    if not isinstance(model_cfg, dict):
        raise ValueError(f"{modality_label} checkpoint 'model_config' must be a dict.")
    cfg_arch = model_cfg.get("architecture")
    if cfg_arch is not None and cfg_arch != expected_architecture:
        raise ValueError(
            f"{modality_label} checkpoint model_config architecture mismatch: "
            f"expected '{expected_architecture}', got '{cfg_arch}'."
        )

    return {
        "architecture": saved_arch,
        "split_type": saved_split,
        "seed": saved_seed,
        "best_epoch": int(payload.get("best_epoch", 0)),
        "best_val_macro_f1": float(payload.get("best_val_macro_f1", 0.0)),
        "model_config": dict(model_cfg),
    }


def load_frozen_speech_encoder(
    checkpoint_path: Union[str, Path],
    expected_split_type: str,
    expected_seed: int,
    expected_architecture: str = DEFAULT_SPEECH_ARCHITECTURE,
    device: Optional[torch.device] = None,
    model_builder: Callable[..., nn.Module] = build_speech_model,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load, validate, and freeze a trained speech checkpoint."""
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Frozen speech checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = validate_base_checkpoint_metadata(
        payload=payload,
        expected_architecture=expected_architecture,
        expected_split_type=expected_split_type,
        expected_seed=expected_seed,
        modality_label="speech",
    )
    model_cfg = meta["model_config"]
    model = model_builder(
        architecture=expected_architecture,
        dropout=float(model_cfg.get("dropout", 0.3)),
        num_classes=int(model_cfg.get("num_classes", NUM_EMOTION_CLASSES)),
    )
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model.to(device)
    freeze_encoder(model)
    assert_encoder_frozen(model, name="speech_encoder")
    meta["checkpoint_path"] = str(ckpt_path)
    return model, meta


def load_frozen_text_encoder(
    checkpoint_path: Union[str, Path],
    expected_split_type: str,
    expected_seed: int,
    expected_architecture: str = DEFAULT_TEXT_ARCHITECTURE,
    device: Optional[torch.device] = None,
    model_builder: Callable[..., nn.Module] = build_text_model,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load, validate, and freeze a trained text checkpoint."""
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Frozen text checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = validate_base_checkpoint_metadata(
        payload=payload,
        expected_architecture=expected_architecture,
        expected_split_type=expected_split_type,
        expected_seed=expected_seed,
        modality_label="text",
    )
    model_cfg = meta["model_config"]
    model = model_builder(
        architecture=expected_architecture,
        dropout=float(model_cfg.get("dropout", 0.3)),
        num_classes=int(model_cfg.get("num_classes", NUM_EMOTION_CLASSES)),
    )
    model.load_state_dict(payload["state_dict"])
    if device is not None:
        model.to(device)
    freeze_encoder(model)
    assert_encoder_frozen(model, name="text_encoder")
    meta["checkpoint_path"] = str(ckpt_path)
    return model, meta


def verify_and_align_modalities(
    speech_bundle: Dict[str, Any],
    text_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Explicitly align speech and text modality bundles by `sample_id` and validate labels.

    Checks:
    - No empty bundles.
    - No duplicate `sample_ids` in either modality.
    - Exact set equality of `sample_ids` (no missing or extra samples).
    - Reorders text entries by `sample_id` to match speech order (never assumes row order).
    - Exact label equality for every aligned `sample_id`.
    """
    speech_ids: List[str] = [str(s) for s in speech_bundle["sample_ids"]]
    text_ids: List[str] = [str(s) for s in text_bundle["sample_ids"]]

    if not speech_ids or not text_ids:
        raise ValueError("Cannot align empty speech or text modality bundle.")

    if len(set(speech_ids)) != len(speech_ids):
        raise ValueError("Duplicate sample_ids detected in speech modality bundle.")
    if len(set(text_ids)) != len(text_ids):
        raise ValueError("Duplicate sample_ids detected in text modality bundle.")

    speech_set = set(speech_ids)
    text_set = set(text_ids)
    if speech_set != text_set:
        missing_in_text = sorted(speech_set - text_set)[:5]
        missing_in_speech = sorted(text_set - speech_set)[:5]
        raise ValueError(
            f"Sample ID mismatch between speech and text modalities: "
            f"missing_in_text={missing_in_text}, missing_in_speech={missing_in_speech}."
        )

    text_index_map = {sid: idx for idx, sid in enumerate(text_ids)}
    reorder_indices = [text_index_map[sid] for sid in speech_ids]

    speech_labels = torch.as_tensor(speech_bundle["labels"], dtype=torch.long)
    text_labels = torch.as_tensor(text_bundle["labels"], dtype=torch.long)[reorder_indices]

    if not torch.equal(speech_labels, text_labels):
        mismatch_pos = (speech_labels != text_labels).nonzero(as_tuple=False).flatten()[:5].tolist()
        examples = [
            (speech_ids[i], int(speech_labels[i].item()), int(text_labels[i].item()))
            for i in mismatch_pos
        ]
        raise ValueError(
            f"Label mismatch between aligned speech and text samples (sample_id, speech_y, text_y): "
            f"{examples}"
        )

    aligned: Dict[str, Any] = {
        "sample_ids": speech_ids,
        "labels": speech_labels,
        "speech_embeddings": torch.as_tensor(
            speech_bundle["embeddings"], dtype=torch.float32
        ),
        "text_embeddings": torch.as_tensor(
            text_bundle["embeddings"], dtype=torch.float32
        )[reorder_indices],
        "speech_logits": torch.as_tensor(
            speech_bundle["logits"], dtype=torch.float32
        ),
        "text_logits": torch.as_tensor(
            text_bundle["logits"], dtype=torch.float32
        )[reorder_indices],
    }

    for meta_key in ("speakers", "words", "emotions", "transcripts"):
        if meta_key in speech_bundle:
            aligned[meta_key] = list(speech_bundle[meta_key])
        elif meta_key in text_bundle:
            src_list = list(text_bundle[meta_key])
            aligned[meta_key] = [src_list[i] for i in reorder_indices]

    return aligned


def extract_frozen_multimodal_features(
    dataloader: DataLoader,
    speech_encoder: nn.Module,
    text_encoder: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, Any], torch.device, bool]:
    """Run frozen speech and text encoders under `torch.no_grad()` on a paired multimodal loader.

    Returns:
        Tuple `(aligned_bundle, active_device, cpu_fallback_used)`.
    """
    assert_encoder_frozen(speech_encoder, name="speech_encoder")
    assert_encoder_frozen(text_encoder, name="text_encoder")

    active_device = device
    speech_encoder.to(active_device)
    text_encoder.to(active_device)
    cpu_fallback_used = False

    speech_embs: List[torch.Tensor] = []
    text_embs: List[torch.Tensor] = []
    speech_logits_list: List[torch.Tensor] = []
    text_logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    sample_ids: List[str] = []
    speakers: List[str] = []
    words: List[str] = []
    emotions: List[str] = []
    transcripts: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            try:
                wavlm_x = batch["wavlm_features"].to(active_device)
                wavlm_len = batch["wavlm_lengths"].to(active_device)
                wavlm_mask = batch["wavlm_mask"].to(active_device)
                text_x = batch["text_features"].to(active_device)
                text_len = batch["text_lengths"].to(active_device)
                text_mask = batch["text_mask"].to(active_device)

                out_s = speech_encoder(wavlm_x, lengths=wavlm_len, mask=wavlm_mask)
                out_t = text_encoder(text_x, lengths=text_len, mask=text_mask)
            except RuntimeError as exc:
                if active_device.type == "mps":
                    cpu_fallback_used = True
                    active_device = torch.device("cpu")
                    speech_encoder.to(active_device)
                    text_encoder.to(active_device)
                    wavlm_x = batch["wavlm_features"].to(active_device)
                    wavlm_len = batch["wavlm_lengths"].to(active_device)
                    wavlm_mask = batch["wavlm_mask"].to(active_device)
                    text_x = batch["text_features"].to(active_device)
                    text_len = batch["text_lengths"].to(active_device)
                    text_mask = batch["text_mask"].to(active_device)

                    out_s = speech_encoder(wavlm_x, lengths=wavlm_len, mask=wavlm_mask)
                    out_t = text_encoder(text_x, lengths=text_len, mask=text_mask)
                else:
                    raise exc

            speech_embs.append(out_s["embedding"].detach().cpu().to(dtype=torch.float32))
            text_embs.append(out_t["embedding"].detach().cpu().to(dtype=torch.float32))
            speech_logits_list.append(out_s["logits"].detach().cpu().to(dtype=torch.float32))
            text_logits_list.append(out_t["logits"].detach().cpu().to(dtype=torch.float32))
            labels_list.append(batch["labels"].detach().cpu().to(dtype=torch.long))

            batch_ids = [str(s) for s in batch["sample_ids"]]
            sample_ids.extend(batch_ids)
            if "speakers" in batch:
                speakers.extend([str(s) for s in batch["speakers"]])
            if "words" in batch:
                words.extend([str(w) for w in batch["words"]])
            if "emotions" in batch:
                emotions.extend([str(e) for e in batch["emotions"]])
            if "transcripts" in batch:
                transcripts.extend([str(t) for t in batch["transcripts"]])

    if not sample_ids:
        raise ValueError("Multimodal dataloader yielded zero samples.")

    speech_bundle: Dict[str, Any] = {
        "sample_ids": sample_ids,
        "labels": torch.cat(labels_list, dim=0),
        "embeddings": torch.cat(speech_embs, dim=0),
        "logits": torch.cat(speech_logits_list, dim=0),
    }
    text_bundle: Dict[str, Any] = {
        "sample_ids": sample_ids,
        "labels": torch.cat(labels_list, dim=0),
        "embeddings": torch.cat(text_embs, dim=0),
        "logits": torch.cat(text_logits_list, dim=0),
    }
    if speakers:
        speech_bundle["speakers"] = speakers
    if words:
        speech_bundle["words"] = words
    if emotions:
        speech_bundle["emotions"] = emotions
    if transcripts:
        speech_bundle["transcripts"] = transcripts

    aligned = verify_and_align_modalities(speech_bundle, text_bundle)
    return aligned, active_device, cpu_fallback_used


@dataclass
class FusionTrainConfig:
    """Configuration for a single multimodal fusion training or calibration run."""

    architecture: str = "gated_fusion"
    split_type: str = "word_indep_split"
    seed: int = 42
    speech_architecture: str = DEFAULT_SPEECH_ARCHITECTURE
    text_architecture: str = DEFAULT_TEXT_ARCHITECTURE
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
    wavlm_cache_dir: str = str(DEFAULT_WAVLM_CACHE_DIR)
    distilbert_cache_dir: str = str(DEFAULT_DISTILBERT_CACHE_DIR)
    speech_results_root: str = str(DEFAULT_SPEECH_RESULTS_DIR)
    text_results_root: str = str(DEFAULT_TEXT_RESULTS_DIR)
    output_root: str = str(DEFAULT_FUSION_RESULTS_DIR)
    speech_checkpoint_path: Optional[str] = None
    text_checkpoint_path: Optional[str] = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in FUSION_ARCHITECTURES:
            raise ValueError(
                f"Invalid fusion architecture '{self.architecture}'. "
                f"Expected one of {FUSION_ARCHITECTURES}."
            )
        self.split_type = normalize_split_column(self.split_type)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_fusion_run_completed(run_dir: Path, architecture: str) -> bool:
    """Check whether a fusion training or calibration run has finalized its artifacts."""
    if architecture == "late_fusion":
        sel_json = run_dir / "selected_lambda.json"
        search_csv = run_dir / "lambda_search.csv"
        return (
            sel_json.exists()
            and sel_json.stat().st_size > 0
            and search_csv.exists()
            and search_csv.stat().st_size > 0
        )
    best_ckpt = run_dir / "best_model.pt"
    run_summary = run_dir / "run_summary.json"
    return (
        best_ckpt.exists()
        and best_ckpt.stat().st_size > 0
        and run_summary.exists()
        and run_summary.stat().st_size > 0
    )


def _validate_resume_compatibility(
    resume_payload: Dict[str, Any],
    config: FusionTrainConfig,
) -> None:
    """Verify that `last_checkpoint.pt` matches the requested fusion run identity and hyperparameters."""
    saved_arch = resume_payload.get("architecture")
    saved_split = resume_payload.get("split_type")
    saved_seed = resume_payload.get("seed")

    if (
        saved_arch != config.architecture
        or saved_split != config.split_type
        or int(saved_seed) != int(config.seed)
    ):
        raise ValueError(
            f"Incompatible fusion resume checkpoint identity: checkpoint has "
            f"(architecture={saved_arch}, split_type={saved_split}, seed={saved_seed}) "
            f"vs requested (architecture={config.architecture}, split_type={config.split_type}, "
            f"seed={config.seed})."
        )

    saved_cfg = resume_payload.get("train_config", {})
    critical_keys = (
        "speech_architecture",
        "text_architecture",
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
                f"Incompatible fusion resume hyperparameter '{key}': "
                f"checkpoint has {saved_cfg[key]!r} vs requested {current_cfg[key]!r}."
            )


def compute_basic_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    """Compute Accuracy, Macro-Precision, Macro-Recall, Macro-F1, and Weighted-F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def run_fusion_epoch(
    fusion_model: nn.Module,
    embedding_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """Run one training or validation epoch on pre-extracted frozen `(h_s, h_t, y)` tensors."""
    is_train = optimizer is not None
    fusion_model.train(mode=is_train)

    total_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for h_s, h_t, labels in embedding_loader:
            h_s = h_s.to(device)
            h_t = h_t.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            out = fusion_model(h_s, h_t)
            logits = out["logits"]
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(fusion_model.parameters(), grad_clip_norm)
                optimizer.step()

            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_samples += batch_size

            all_preds.extend(logits.detach().argmax(dim=-1).cpu().tolist())
            all_targets.extend(labels.detach().cpu().tolist())

    metrics = compute_basic_metrics(all_targets, all_preds)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def train_trainable_fusion_from_embeddings(
    config: FusionTrainConfig,
    train_bundle: Dict[str, Any],
    val_bundle: Dict[str, Any],
    speech_meta: Dict[str, Any],
    text_meta: Dict[str, Any],
    run_dir: Path,
    exec_device: torch.device,
) -> Dict[str, Any]:
    """Train `GatedMultimodalFusion` or `ConcatFusion` on aligned train/val embeddings."""
    if config.architecture not in TRAINABLE_FUSION_ARCHITECTURES:
        raise ValueError(
            f"Expected trainable fusion architecture in {TRAINABLE_FUSION_ARCHITECTURES}, "
            f"got '{config.architecture}'."
        )

    set_reproducible_seed(config.seed)

    train_ds = TensorDataset(
        train_bundle["speech_embeddings"],
        train_bundle["text_embeddings"],
        train_bundle["labels"],
    )
    val_ds = TensorDataset(
        val_bundle["speech_embeddings"],
        val_bundle["text_embeddings"],
        val_bundle["labels"],
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    speech_dim = int(train_bundle["speech_embeddings"].shape[1])
    text_dim = int(train_bundle["text_embeddings"].shape[1])
    fusion_model = build_fusion_model(
        architecture=config.architecture,
        speech_dim=speech_dim,
        text_dim=text_dim,
        fusion_dim=256,
        mlp_hidden_dim=128,
        num_classes=NUM_EMOTION_CLASSES,
        dropout=config.dropout,
    ).to(exec_device)

    trainable_params = [p for p in fusion_model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        trainable_params,
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
    elapsed_prior_sec = 0.0
    history: List[Dict[str, Any]] = []

    checkpoint_path = run_dir / "best_model.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"

    if not config.overwrite and last_checkpoint_path.exists():
        resume_payload = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        _validate_resume_compatibility(resume_payload, config)

        fusion_model.load_state_dict(resume_payload["model_state_dict"])
        fusion_model.to(exec_device)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])

        start_epoch = int(resume_payload["last_epoch"]) + 1
        best_val_macro_f1 = float(resume_payload["best_val_macro_f1"])
        best_val_loss = float(resume_payload["best_val_loss"])
        best_epoch = int(resume_payload["best_epoch"])
        epochs_without_improvement = int(resume_payload["epochs_without_improvement"])
        history = list(resume_payload.get("history", []))
        elapsed_prior_sec = float(resume_payload.get("elapsed_sec", 0.0))

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

    model_config = {
        "architecture": config.architecture,
        "speech_dim": speech_dim,
        "text_dim": text_dim,
        "fusion_dim": 256,
        "mlp_hidden_dim": 128,
        "num_classes": NUM_EMOTION_CLASSES,
        "dropout": config.dropout,
    }

    t0 = time.perf_counter()
    for epoch in range(start_epoch, config.max_epochs + 1):
        if epochs_without_improvement >= config.patience:
            break

        epoch_t0 = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_metrics = run_fusion_epoch(
            fusion_model=fusion_model,
            embedding_loader=train_loader,
            criterion=criterion,
            device=exec_device,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_metrics = run_fusion_epoch(
            fusion_model=fusion_model,
            embedding_loader=val_loader,
            criterion=criterion,
            device=exec_device,
            optimizer=None,
        )
        scheduler.step()

        val_f1 = val_metrics["macro_f1"]
        val_loss = val_metrics["loss"]

        # Decoupled early stopping: only Macro-F1 resets patience; loss breaks ties
        f1_improved = val_f1 > (best_val_macro_f1 + F1_IMPROVEMENT_TOL)
        loss_tiebreak = (
            (not f1_improved)
            and (abs(val_f1 - best_val_macro_f1) <= F1_IMPROVEMENT_TOL)
            and (val_loss < best_val_loss - LOSS_TIEBREAK_TOL)
        )
        should_save_best = f1_improved or loss_tiebreak

        if f1_improved:
            best_val_macro_f1 = val_f1
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if loss_tiebreak:
                best_val_loss = val_loss

        if should_save_best:
            best_epoch = epoch
            state_dict_cpu = {k: v.detach().cpu() for k, v in fusion_model.state_dict().items()}
            atomic_torch_save(
                {
                    "architecture": config.architecture,
                    "split_type": config.split_type,
                    "seed": config.seed,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_accuracy": val_metrics["accuracy"],
                    "best_val_loss": best_val_loss,
                    "model_config": model_config,
                    "train_config": config.to_dict(),
                    "speech_encoder_meta": speech_meta,
                    "text_encoder_meta": text_meta,
                    "state_dict": state_dict_cpu,
                },
                checkpoint_path,
            )

        epoch_time_sec = round(time.perf_counter() - epoch_t0, 4)
        history.append(
            {
                "epoch": epoch,
                "lr": round(current_lr, 8),
                "train_loss": round(train_metrics["loss"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "train_macro_f1": round(train_metrics["macro_f1"], 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_macro_precision": round(val_metrics["macro_precision"], 6),
                "val_macro_recall": round(val_metrics["macro_recall"], 6),
                "val_macro_f1": round(val_f1, 6),
                "val_weighted_f1": round(val_metrics["weighted_f1"], 6),
                "f1_improved": f1_improved,
                "loss_tiebreak": loss_tiebreak,
                "is_best": should_save_best,
                "patience_counter": epochs_without_improvement,
                "epoch_time_sec": epoch_time_sec,
            }
        )

        # Atomic epoch-level resume checkpoint
        latest_state_cpu = {k: v.detach().cpu() for k, v in fusion_model.state_dict().items()}
        resume_state = {
            "architecture": config.architecture,
            "split_type": config.split_type,
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
        atomic_torch_save(resume_state, last_checkpoint_path)

        if epochs_without_improvement >= config.patience:
            break

    total_train_time_sec = elapsed_prior_sec + (time.perf_counter() - t0)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    run_summary = {
        "architecture": config.architecture,
        "split_type": config.split_type,
        "seed": config.seed,
        "num_trainable_params": num_params,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 6),
        "best_val_loss": round(best_val_loss, 6),
        "hardware_used": str(exec_device),
        "training_time_sec": round(total_train_time_sec, 4),
        "checkpoint_path": str(checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "speech_encoder_meta": speech_meta,
        "text_encoder_meta": text_meta,
        "model_config": model_config,
        "config": config.to_dict(),
        "history": history,
    }
    atomic_json_dump(config.to_dict(), run_dir / "config.json")
    atomic_json_dump(run_summary, run_dir / "run_summary.json")
    return run_summary


def calibrate_late_fusion_from_logits(
    val_speech_logits: torch.Tensor,
    val_text_logits: torch.Tensor,
    val_labels: torch.Tensor,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """Calibrate `LateFusionEnsemble` strictly on validation logits.

    Primary objective: maximize validation Macro-F1.
    Tie-breaker: minimize validation log-loss.
    `lambda_weight = 1.0` (speech-only) and `0.0` (text-only) are valid outcomes.
    """
    if not lambda_grid:
        raise ValueError("`lambda_grid` must contain at least one candidate lambda value.")

    ensemble = LateFusionEnsemble(lambda_weight=0.5, num_classes=NUM_EMOTION_CLASSES)
    y_true = val_labels.detach().cpu().numpy().astype(np.int64)
    all_class_labels = list(range(NUM_EMOTION_CLASSES))

    search_rows: List[Dict[str, Any]] = []
    best_lambda: Optional[float] = None
    best_f1 = -1.0
    best_loss = float("inf")
    best_record: Dict[str, Any] = {}

    for raw_lam in lambda_grid:
        lam = round(float(raw_lam), 6)
        out = ensemble(val_speech_logits, val_text_logits, lambda_weight=lam)
        probs = np.clip(out["probabilities"].detach().cpu().numpy(), 1e-12, 1.0)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        preds = out["predictions"].detach().cpu().numpy().astype(np.int64)

        metrics = compute_basic_metrics(y_true, preds)
        val_ll = float(log_loss(y_true, probs, labels=all_class_labels))

        record = {
            "lambda_weight": lam,
            "val_accuracy": round(metrics["accuracy"], 6),
            "val_macro_precision": round(metrics["macro_precision"], 6),
            "val_macro_recall": round(metrics["macro_recall"], 6),
            "val_macro_f1": round(metrics["macro_f1"], 6),
            "val_weighted_f1": round(metrics["weighted_f1"], 6),
            "val_log_loss": round(val_ll, 6),
        }
        search_rows.append(record)

        f1_better = metrics["macro_f1"] > (best_f1 + F1_IMPROVEMENT_TOL)
        loss_tiebreak = (
            (not f1_better)
            and (abs(metrics["macro_f1"] - best_f1) <= F1_IMPROVEMENT_TOL)
            and (val_ll < best_loss - LOSS_TIEBREAK_TOL)
        )
        if f1_better or loss_tiebreak or best_lambda is None:
            best_lambda = lam
            best_f1 = metrics["macro_f1"]
            best_loss = val_ll
            best_record = dict(record)

    assert best_lambda is not None
    return best_lambda, best_record, search_rows


def run_late_fusion_calibration(
    config: FusionTrainConfig,
    val_bundle: Dict[str, Any],
    speech_meta: Dict[str, Any],
    text_meta: Dict[str, Any],
    run_dir: Path,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> Dict[str, Any]:
    """Calibrate `LateFusionEnsemble` on validation logits and persist calibration artifacts."""
    t0 = time.perf_counter()
    best_lambda, best_record, search_rows = calibrate_late_fusion_from_logits(
        val_speech_logits=val_bundle["speech_logits"],
        val_text_logits=val_bundle["text_logits"],
        val_labels=val_bundle["labels"],
        lambda_grid=lambda_grid,
    )
    calib_time_sec = round(time.perf_counter() - t0, 4)

    pd.DataFrame(search_rows).to_csv(run_dir / "lambda_search.csv", index=False)

    selected_payload = {
        "architecture": "late_fusion",
        "split_type": config.split_type,
        "seed": config.seed,
        "selected_lambda": best_lambda,
        "num_trainable_params": 0,
        "val_samples": int(len(val_bundle["sample_ids"])),
        "best_val_metrics": best_record,
        "calibration_time_sec": calib_time_sec,
        "speech_encoder_meta": speech_meta,
        "text_encoder_meta": text_meta,
        "config": config.to_dict(),
    }
    atomic_json_dump(selected_payload, run_dir / "selected_lambda.json")
    atomic_json_dump(config.to_dict(), run_dir / "config.json")
    atomic_json_dump(selected_payload, run_dir / "run_summary.json")
    return selected_payload


def train_single_fusion_run(
    config: FusionTrainConfig,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    speech_model_builder: Callable[..., nn.Module] = build_speech_model,
    text_model_builder: Callable[..., nn.Module] = build_text_model,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
) -> Dict[str, Any]:
    """Execute one fusion training (`gated_fusion`/`concat_fusion`) or calibration (`late_fusion`) run."""
    run_dir = get_fusion_run_output_dir(
        output_root=config.output_root,
        architecture=config.architecture,
        split_type=config.split_type,
        seed=config.seed,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Completed-run protection
    if not config.overwrite and is_fusion_run_completed(run_dir, config.architecture):
        summary_file = (
            run_dir / "selected_lambda.json"
            if config.architecture == "late_fusion"
            else run_dir / "run_summary.json"
        )
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)

    speech_ckpt = (
        Path(config.speech_checkpoint_path)
        if config.speech_checkpoint_path is not None
        else resolve_speech_checkpoint_path(
            speech_results_root=config.speech_results_root,
            split_type=config.split_type,
            seed=config.seed,
            speech_architecture=config.speech_architecture,
        )
    )
    text_ckpt = (
        Path(config.text_checkpoint_path)
        if config.text_checkpoint_path is not None
        else resolve_text_checkpoint_path(
            text_results_root=config.text_results_root,
            split_type=config.split_type,
            seed=config.seed,
            text_architecture=config.text_architecture,
        )
    )

    exec_device = select_device(config.device)

    speech_encoder, speech_meta = load_frozen_speech_encoder(
        checkpoint_path=speech_ckpt,
        expected_split_type=config.split_type,
        expected_seed=config.seed,
        expected_architecture=config.speech_architecture,
        device=exec_device,
        model_builder=speech_model_builder,
    )
    text_encoder, text_meta = load_frozen_text_encoder(
        checkpoint_path=text_ckpt,
        expected_split_type=config.split_type,
        expected_seed=config.seed,
        expected_architecture=config.text_architecture,
        device=exec_device,
        model_builder=text_model_builder,
    )

    if val_loader is None:
        val_dataset = TESSFeatureDataset(
            manifest_path=config.manifest_path,
            split_type=config.split_type,
            split="val",
            modalities=("wavlm", "distilbert"),
            wavlm_cache_dir=config.wavlm_cache_dir,
            distilbert_cache_dir=config.distilbert_cache_dir,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=tess_feature_collate_fn,
        )

    val_bundle, exec_device, _ = extract_frozen_multimodal_features(
        dataloader=val_loader,
        speech_encoder=speech_encoder,
        text_encoder=text_encoder,
        device=exec_device,
    )

    if config.architecture == "late_fusion":
        summary = run_late_fusion_calibration(
            config=config,
            val_bundle=val_bundle,
            speech_meta=speech_meta,
            text_meta=text_meta,
            run_dir=run_dir,
            lambda_grid=lambda_grid,
        )
        release_device_memory(exec_device)
        return summary

    if train_loader is None:
        train_dataset = TESSFeatureDataset(
            manifest_path=config.manifest_path,
            split_type=config.split_type,
            split="train",
            modalities=("wavlm", "distilbert"),
            wavlm_cache_dir=config.wavlm_cache_dir,
            distilbert_cache_dir=config.distilbert_cache_dir,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=tess_feature_collate_fn,
        )

    train_bundle, exec_device, _ = extract_frozen_multimodal_features(
        dataloader=train_loader,
        speech_encoder=speech_encoder,
        text_encoder=text_encoder,
        device=exec_device,
    )

    summary = train_trainable_fusion_from_embeddings(
        config=config,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        speech_meta=speech_meta,
        text_meta=text_meta,
        run_dir=run_dir,
        exec_device=exec_device,
    )
    release_device_memory(exec_device)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or calibrate multimodal fusion models on frozen speech & text encoders."
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--speech-results-root", type=Path, default=DEFAULT_SPEECH_RESULTS_DIR)
    parser.add_argument("--text-results-root", type=Path, default=DEFAULT_TEXT_RESULTS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FUSION_RESULTS_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly overwrite completed fusion runs instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split_arg in args.splits:
        split_col = normalize_split_column(split_arg)
        for arch in args.architectures:
            for seed in args.seeds:
                cfg = FusionTrainConfig(
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
                    speech_results_root=str(args.speech_results_root),
                    text_results_root=str(args.text_results_root),
                    output_root=str(args.output_root),
                    overwrite=args.overwrite,
                )
                train_single_fusion_run(cfg)


if __name__ == "__main__":
    main()
