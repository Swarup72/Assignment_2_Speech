"""Multimodal fusion architectures for Assignment 2b (Stage 08A).

Implements three independent fusion heads designed to operate on representations
and logits produced by the frozen unimodal speech (`WavLMBiGRUAttentionClassifier`,
`MFCCCNNBiLSTMAttentionClassifier`) and text (`DistilBERTBiLSTMAttentionClassifier`)
encoders:

1. `GatedMultimodalFusion` (`gated_fusion`):
   Element-wise modality gating over 256-d speech and text embeddings with a 2-layer
   MLP classifier (`256 -> 128 -> 7`).
2. `ConcatFusion` (`concat_fusion`):
   Feature concatenation (`512 -> 256` projection with LayerNorm, GELU, Dropout)
   followed by a 2-layer MLP classifier (`256 -> 128 -> 7`).
3. `LateFusionEnsemble` (`late_fusion`):
   Zero-parameter decision-level probability ensemble combining softmax distributions
   from frozen speech and text logits via `lambda_weight in [0, 1]`.

Uses the canonical 7-class emotion mapping from `src.data.build_manifest.EMOTION_TO_ID`.
"""

import math
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

from src.data.build_manifest import EMOTION_TO_ID


FUSION_ARCHITECTURES = ("gated_fusion", "concat_fusion", "late_fusion")
ID_TO_EMOTION: Dict[int, str] = {v: k for k, v in EMOTION_TO_ID.items()}
NUM_EMOTION_CLASSES: int = len(EMOTION_TO_ID)  # 7


def _validate_positive_int(name: str, value: int) -> None:
    """Validate that a dimension or class count is a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer, got {value!r}.")


def _validate_dropout(dropout: float) -> None:
    """Validate that dropout probability lies in `[0.0, 1.0)`."""
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError(f"`dropout` must be a float in [0.0, 1.0), got {dropout!r}.")
    if not math.isfinite(dropout) or not (0.0 <= float(dropout) < 1.0):
        raise ValueError(f"`dropout` must be in [0.0, 1.0), got {dropout}.")


def _validate_lambda_weight(lambda_weight: float) -> float:
    """Validate that a late-fusion weight is a finite float within `[0.0, 1.0]`."""
    if isinstance(lambda_weight, bool) or not isinstance(lambda_weight, (int, float)):
        raise ValueError(
            f"`lambda_weight` must be a finite number in [0.0, 1.0], got {lambda_weight!r}."
        )
    lam = float(lambda_weight)
    if not math.isfinite(lam) or not (0.0 <= lam <= 1.0):
        raise ValueError(f"`lambda_weight` must be within [0.0, 1.0], got {lambda_weight}.")
    return lam


def _validate_2d_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    first_name: str,
    second_name: str,
    expected_first_dim: int,
    expected_second_dim: int,
) -> None:
    """Validate two 2D tensors `(B, D1)` and `(B, D2)` for rank, shape, batch size, and device."""
    if not isinstance(first, torch.Tensor):
        raise ValueError(f"Expected `{first_name}` to be a torch.Tensor, got {type(first).__name__}.")
    if not isinstance(second, torch.Tensor):
        raise ValueError(f"Expected `{second_name}` to be a torch.Tensor, got {type(second).__name__}.")

    if first.ndim != 2:
        raise ValueError(
            f"Expected 2D `{first_name}` of shape (B, {expected_first_dim}), "
            f"got {first.ndim}D tensor with shape {tuple(first.shape)}."
        )
    if second.ndim != 2:
        raise ValueError(
            f"Expected 2D `{second_name}` of shape (B, {expected_second_dim}), "
            f"got {second.ndim}D tensor with shape {tuple(second.shape)}."
        )

    if first.shape[0] <= 0 or second.shape[0] <= 0:
        raise ValueError(
            f"Batch size must be >= 1, got `{first_name}` shape {tuple(first.shape)} "
            f"and `{second_name}` shape {tuple(second.shape)}."
        )

    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f"Batch size mismatch between `{first_name}` (B={first.shape[0]}) "
            f"and `{second_name}` (B={second.shape[0]})."
        )

    if first.shape[1] != expected_first_dim:
        raise ValueError(
            f"Expected `{first_name}` dimension {expected_first_dim}, got {first.shape[1]}."
        )
    if second.shape[1] != expected_second_dim:
        raise ValueError(
            f"Expected `{second_name}` dimension {expected_second_dim}, got {second.shape[1]}."
        )

    if first.device != second.device:
        raise ValueError(
            f"Device mismatch between `{first_name}` ({first.device}) "
            f"and `{second_name}` ({second.device})."
        )


class FusionEmotionMLPClassifier(nn.Module):
    """Two-layer MLP classification head for multimodal embeddings.

    Architecture:
        `Linear(input_dim, hidden_dim) -> LayerNorm(hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, num_classes)`
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        _validate_positive_int("input_dim", input_dim)
        _validate_positive_int("hidden_dim", hidden_dim)
        _validate_positive_int("num_classes", num_classes)
        _validate_dropout(dropout)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = float(dropout)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project `(B, input_dim)` fused embedding to `(B, num_classes)` logits."""
        return self.net(h)


class GatedMultimodalFusion(nn.Module):
    """Gated Multimodal Fusion head operating on 256-d speech and text embeddings.

    Computes:
        projected_speech = tanh(W_s * speech_embedding + b_s)
        projected_text   = tanh(W_t * text_embedding + b_t)
        gate             = sigmoid(W_g * [speech_embedding || text_embedding] + b_g)
        h_fused          = gate * projected_speech + (1 - gate) * projected_text
        logits           = classifier(h_fused)

    Designed to consume detached embeddings produced by frozen speech and text
    encoders in `eval()` mode under `torch.no_grad()`.
    """

    def __init__(
        self,
        speech_dim: int = 256,
        text_dim: int = 256,
        fusion_dim: int = 256,
        mlp_hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        _validate_positive_int("speech_dim", speech_dim)
        _validate_positive_int("text_dim", text_dim)
        _validate_positive_int("fusion_dim", fusion_dim)
        _validate_positive_int("mlp_hidden_dim", mlp_hidden_dim)
        _validate_positive_int("num_classes", num_classes)
        _validate_dropout(dropout)

        self.architecture_name = "gated_fusion"
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = float(dropout)

        self.speech_proj = nn.Linear(speech_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.gate_proj = nn.Linear(speech_dim + text_dim, fusion_dim)
        self.classifier = FusionEmotionMLPClassifier(
            input_dim=fusion_dim,
            hidden_dim=mlp_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        speech_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Fuse `(B, speech_dim)` speech and `(B, text_dim)` text embeddings via gating.

        Args:
            speech_embedding: Speech representation of shape `(B, speech_dim)`.
            text_embedding: Text representation of shape `(B, text_dim)`.

        Returns:
            Dictionary containing:
            - `logits`: `(B, num_classes)` class logits
            - `embedding`: `(B, fusion_dim)` gated multimodal representation
            - `gate`: `(B, fusion_dim)` element-wise modality gate in `[0, 1]`
        """
        _validate_2d_pair(
            first=speech_embedding,
            second=text_embedding,
            first_name="speech_embedding",
            second_name="text_embedding",
            expected_first_dim=self.speech_dim,
            expected_second_dim=self.text_dim,
        )

        projected_speech = torch.tanh(self.speech_proj(speech_embedding))
        projected_text = torch.tanh(self.text_proj(text_embedding))

        concat_inputs = torch.cat([speech_embedding, text_embedding], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat_inputs))

        h_fused = gate * projected_speech + (1.0 - gate) * projected_text
        logits = self.classifier(h_fused)

        return {
            "logits": logits,
            "embedding": h_fused,
            "gate": gate,
        }


class ConcatFusion(nn.Module):
    """Feature Concatenation Fusion head operating on 256-d speech and text embeddings.

    Computes:
        concat_inputs = [speech_embedding || text_embedding]           # (B, 512)
        h_fused       = Dropout(GELU(LayerNorm(Linear(concat_inputs)))) # (B, 256)
        logits        = classifier(h_fused)                            # (B, 7)
    """

    def __init__(
        self,
        speech_dim: int = 256,
        text_dim: int = 256,
        fusion_dim: int = 256,
        mlp_hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        _validate_positive_int("speech_dim", speech_dim)
        _validate_positive_int("text_dim", text_dim)
        _validate_positive_int("fusion_dim", fusion_dim)
        _validate_positive_int("mlp_hidden_dim", mlp_hidden_dim)
        _validate_positive_int("num_classes", num_classes)
        _validate_dropout(dropout)

        self.architecture_name = "concat_fusion"
        self.speech_dim = speech_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = float(dropout)

        self.fusion_proj = nn.Sequential(
            nn.Linear(speech_dim + text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = FusionEmotionMLPClassifier(
            input_dim=fusion_dim,
            hidden_dim=mlp_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        speech_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Fuse `(B, speech_dim)` speech and `(B, text_dim)` text embeddings via concatenation.

        Args:
            speech_embedding: Speech representation of shape `(B, speech_dim)`.
            text_embedding: Text representation of shape `(B, text_dim)`.

        Returns:
            Dictionary containing:
            - `logits`: `(B, num_classes)` class logits
            - `embedding`: `(B, fusion_dim)` projected multimodal representation
        """
        _validate_2d_pair(
            first=speech_embedding,
            second=text_embedding,
            first_name="speech_embedding",
            second_name="text_embedding",
            expected_first_dim=self.speech_dim,
            expected_second_dim=self.text_dim,
        )

        concat_inputs = torch.cat([speech_embedding, text_embedding], dim=-1)
        h_fused = self.fusion_proj(concat_inputs)
        logits = self.classifier(h_fused)

        return {
            "logits": logits,
            "embedding": h_fused,
        }


class LateFusionEnsemble(nn.Module):
    """Zero-parameter Late-Logit (decision-level) fusion ensemble.

    Combines class probability distributions from frozen speech and text logits:
        p_s     = softmax(speech_logits, dim=-1)
        p_t     = softmax(text_logits, dim=-1)
        p_fused = lambda_weight * p_s + (1 - lambda_weight) * p_t

    Contains no trainable parameters. Supports overriding `lambda_weight` per forward
    call or sweeping a sequence of candidate weights via `evaluate_lambdas()`.
    """

    def __init__(
        self,
        lambda_weight: float = 0.5,
        num_classes: int = NUM_EMOTION_CLASSES,
    ) -> None:
        super().__init__()
        _validate_positive_int("num_classes", num_classes)
        self.architecture_name = "late_fusion"
        self.lambda_weight: float = _validate_lambda_weight(lambda_weight)
        self.num_classes: int = num_classes

    def forward(
        self,
        speech_logits: torch.Tensor,
        text_logits: torch.Tensor,
        lambda_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Combine `(B, num_classes)` speech and text logits via weighted probability averaging.

        Args:
            speech_logits: Logits from the frozen speech model of shape `(B, num_classes)`.
            text_logits: Logits from the frozen text model of shape `(B, num_classes)`.
            lambda_weight: Optional override weight in `[0.0, 1.0]` for the speech modality.
                If omitted, uses `self.lambda_weight`.

        Returns:
            Dictionary containing:
            - `probabilities`: `(B, num_classes)` fused class probability distribution
            - `predictions`: `(B,)` predicted class indices (`torch.long`)
            - `lambda_weight`: `float` weight used for speech modality
        """
        lam = self.lambda_weight if lambda_weight is None else _validate_lambda_weight(lambda_weight)

        _validate_2d_pair(
            first=speech_logits,
            second=text_logits,
            first_name="speech_logits",
            second_name="text_logits",
            expected_first_dim=self.num_classes,
            expected_second_dim=self.num_classes,
        )

        p_s = torch.softmax(speech_logits, dim=-1)
        p_t = torch.softmax(text_logits, dim=-1)
        p_fused = lam * p_s + (1.0 - lam) * p_t
        predictions = torch.argmax(p_fused, dim=-1)

        return {
            "probabilities": p_fused,
            "predictions": predictions,
            "lambda_weight": lam,
        }

    def evaluate_lambdas(
        self,
        speech_logits: torch.Tensor,
        text_logits: torch.Tensor,
        lambda_weights: Iterable[float],
    ) -> Dict[float, Dict[str, Any]]:
        """Evaluate the ensemble over multiple candidate `lambda_weight` values."""
        results: Dict[float, Dict[str, Any]] = {}
        for weight in lambda_weights:
            lam = _validate_lambda_weight(weight)
            results[lam] = self.forward(
                speech_logits=speech_logits,
                text_logits=text_logits,
                lambda_weight=lam,
            )
        return results


def build_fusion_model(
    architecture: str = "gated_fusion",
    **kwargs: Any,
) -> nn.Module:
    """Factory function constructing a multimodal fusion model by architecture name."""
    if architecture == "gated_fusion":
        return GatedMultimodalFusion(**kwargs)
    if architecture == "concat_fusion":
        return ConcatFusion(**kwargs)
    if architecture == "late_fusion":
        return LateFusionEnsemble(**kwargs)
    raise ValueError(
        f"Unknown fusion architecture '{architecture}'. Expected one of {FUSION_ARCHITECTURES}."
    )
