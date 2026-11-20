"""Text-only emotion recognition architecture for Assignment 2b (Stage 07A).

Implements `DistilBERTBiLSTMAttentionClassifier` (`distilbert_bilstm_attn`):
- Input: Cached DistilBERT token embeddings `(B, T, 768)` + valid lengths and/or boolean mask.
- Projection: `Linear(768 -> 256)` + `LayerNorm(256)` + `GELU` + `Dropout`.
- Contextual Modelling: 2-layer Bidirectional LSTM (`hidden_size=128` per direction, output `256`)
  using `pack_padded_sequence` / `pad_packed_sequence` so padded tokens never affect either
  the forward or backward LSTM recurrence.
- Masked Additive Attention Pooling: Pools `(B, T, 256)` into a 256-d contextual representation
  `h_t` with normalized token attention weights `alpha_t` (`0.0` at padded positions, sum `1.0`
  over valid tokens).
- Classification Head: `Linear(256 -> 128)` + `GELU` + `Dropout` + `Linear(128 -> 7)`.

Uses the canonical 7-class emotion mapping from `src.data.build_manifest.EMOTION_TO_ID`.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.build_manifest import EMOTION_TO_ID
from src.data.feature_dataset import build_padding_mask


TEXT_ARCHITECTURES = ("distilbert_bilstm_attn",)
ID_TO_EMOTION: Dict[int, str] = {v: k for k, v in EMOTION_TO_ID.items()}
NUM_EMOTION_CLASSES: int = len(EMOTION_TO_ID)  # 7


def resolve_text_mask_and_lengths(
    features: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    expected_input_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate text feature tensor, sequence lengths, and padding mask consistency.

    Args:
        features: Input token embeddings of shape `(B, T, D)`.
        lengths: Optional 1D integer tensor of shape `(B,)` with valid token counts in `[1, T]`.
        mask: Optional 2D boolean/binary tensor of shape `(B, T)` where True/1 marks valid tokens.
        expected_input_dim: If provided, asserts `features.shape[2] == expected_input_dim`.

    Returns:
        Tuple `(lengths_resolved, mask_resolved)` where `lengths_resolved` has shape `(B,)`
        (`torch.long`) and `mask_resolved` has shape `(B, T)` (`torch.bool`) on `features.device`.

    Raises:
        ValueError: If tensor rank/shape, sequence lengths, or mask-length consistency are invalid.
    """
    if not isinstance(features, torch.Tensor):
        raise ValueError(f"Expected `features` to be a torch.Tensor, got {type(features).__name__}.")
    if features.ndim != 3:
        raise ValueError(
            f"Expected 3D text features of shape (B, T, D), got {features.ndim}D tensor "
            f"with shape {tuple(features.shape)}."
        )

    batch_size, max_len, feat_dim = features.shape
    if batch_size <= 0 or max_len <= 0:
        raise ValueError(
            f"Batch size and sequence length must be >= 1, got shape {tuple(features.shape)}."
        )
    if expected_input_dim is not None and feat_dim != expected_input_dim:
        raise ValueError(
            f"Expected text feature dimension {expected_input_dim}, got {feat_dim}."
        )

    lengths_resolved: Optional[torch.Tensor] = None
    mask_resolved: Optional[torch.Tensor] = None

    if lengths is not None:
        if not isinstance(lengths, torch.Tensor):
            raise ValueError(f"Expected `lengths` to be a torch.Tensor, got {type(lengths).__name__}.")
        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError(
                f"Expected 1D `lengths` of shape ({batch_size},), got {tuple(lengths.shape)}."
            )
        if lengths.is_floating_point() or lengths.dtype == torch.bool:
            raise ValueError(
                f"Expected integer dtype for `lengths`, got {lengths.dtype}."
            )
        lengths_resolved = lengths.to(device=features.device, dtype=torch.long)
        if (lengths_resolved <= 0).any() or (lengths_resolved > max_len).any():
            raise ValueError(
                f"All sequence lengths must satisfy 1 <= length <= {max_len}, "
                f"got {lengths_resolved.tolist()}."
            )

    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"Expected `mask` to be a torch.Tensor, got {type(mask).__name__}.")
        if mask.ndim != 2 or mask.shape != (batch_size, max_len):
            raise ValueError(
                f"Expected 2D `mask` of shape ({batch_size}, {max_len}), got {tuple(mask.shape)}."
            )
        mask_resolved = mask.to(device=features.device, dtype=torch.bool)
        mask_lengths = mask_resolved.sum(dim=1).to(dtype=torch.long)
        if (mask_lengths <= 0).any():
            raise ValueError(
                f"Every sequence in `mask` must have at least 1 valid token, "
                f"got valid counts {mask_lengths.tolist()}."
            )
        expected_contiguous = build_padding_mask(mask_lengths, max_len=max_len)
        if not torch.equal(mask_resolved, expected_contiguous):
            raise ValueError(
                "Provided `mask` is non-contiguous; valid tokens must form a contiguous "
                "prefix [0, length) with right-padding."
            )

    if lengths_resolved is None and mask_resolved is None:
        lengths_resolved = torch.full((batch_size,), max_len, dtype=torch.long, device=features.device)
        mask_resolved = torch.ones((batch_size, max_len), dtype=torch.bool, device=features.device)
    elif lengths_resolved is not None and mask_resolved is None:
        mask_resolved = build_padding_mask(lengths_resolved, max_len=max_len)
    elif mask_resolved is not None and lengths_resolved is None:
        lengths_resolved = mask_resolved.sum(dim=1).to(dtype=torch.long)
    else:
        expected_from_lengths = build_padding_mask(lengths_resolved, max_len=max_len)
        if not torch.equal(mask_resolved, expected_from_lengths):
            raise ValueError(
                f"Inconsistent `lengths` and `mask`: lengths={lengths_resolved.tolist()} "
                f"vs mask valid counts={mask_resolved.sum(dim=1).tolist()}."
            )

    return lengths_resolved, mask_resolved


class TextMaskedAdditiveAttentionPool(nn.Module):
    """Masked additive (Bahdanau-style) self-attention pooling over variable-length token sequences."""

    def __init__(self, input_dim: int = 256, attn_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, attn_dim, bias=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        sequence_output: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool `(B, T, D)` sequence into `(B, D)` representation using boolean `mask` `(B, T)`."""
        mask_bool = mask.to(device=sequence_output.device, dtype=torch.bool)
        energy = self.score(self.dropout(self.tanh(self.proj(sequence_output)))).squeeze(-1)  # (B, T)

        # Mask padded positions with -inf prior to softmax so exp(-inf) == 0.0
        masked_energy = energy.masked_fill(~mask_bool, float("-inf"))
        attn_weights = torch.softmax(masked_energy, dim=-1)

        # Explicitly zero padded positions and renormalize to guarantee exact 0.0 on padding
        mask_float = mask_bool.to(dtype=sequence_output.dtype)
        attn_weights = attn_weights * mask_float
        denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attn_weights = attn_weights / denom

        pooled = torch.bmm(attn_weights.unsqueeze(1), sequence_output).squeeze(1)  # (B, D)
        return pooled, attn_weights


class TextEmotionMLPClassifier(nn.Module):
    """Two-layer MLP classification head: `Linear(256 -> 128) -> GELU -> Dropout -> Linear(128 -> 7)`."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(h))))


class DistilBERTBiLSTMAttentionClassifier(nn.Module):
    """Text-only emotion classifier: Linear(768->256) + 2-Layer BiLSTM(128) + Masked Attention + MLP."""

    def __init__(
        self,
        input_dim: int = 768,
        proj_dim: int = 256,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        attn_dim: int = 128,
        mlp_hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")
        if num_classes != NUM_EMOTION_CLASSES:
            raise ValueError(
                f"Expected num_classes={NUM_EMOTION_CLASSES} matching canonical emotion mapping, "
                f"got {num_classes}."
            )

        self.architecture_name = "distilbert_bilstm_attn"
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.repr_dim = lstm_hidden_dim * 2  # 256
        self.num_classes = num_classes
        self.dropout_rate = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.post_rnn_norm = nn.LayerNorm(self.repr_dim)

        self.attention_pool = TextMaskedAdditiveAttentionPool(
            input_dim=self.repr_dim,
            attn_dim=attn_dim,
            dropout=min(dropout, 0.2),
        )

        self.classifier = TextEmotionMLPClassifier(
            input_dim=self.repr_dim,
            hidden_dim=mlp_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass on cached DistilBERT token embeddings `(B, T, 768)`.

        Args:
            features: Cached DistilBERT token embeddings of shape `(B, T, 768)`.
            lengths: Optional valid token lengths of shape `(B,)`.
            mask: Optional boolean validity mask of shape `(B, T)`.

        Returns:
            Dictionary with keys:
            - `logits`: `(B, 7)` class logits
            - `embedding`: `(B, 256)` attended contextual representation
            - `h_t`: `(B, 256)` alias for `embedding`
            - `attention_weights`: `(B, T)` normalized token attention weights
        """
        lengths_resolved, mask_resolved = resolve_text_mask_and_lengths(
            features=features,
            lengths=lengths,
            mask=mask,
            expected_input_dim=self.input_dim,
        )
        max_len = features.shape[1]
        mask_float = mask_resolved.unsqueeze(-1).to(dtype=features.dtype)

        # Project valid tokens and zero out padded positions
        proj = self.input_proj(features) * mask_float  # (B, T, 256)

        # Pack padded sequence so padded tokens never affect forward or backward BiLSTM passes
        packed = pack_padded_sequence(
            proj,
            lengths_resolved.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.bilstm(packed)
        rnn_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=max_len,
        )
        rnn_out = self.post_rnn_norm(rnn_out) * mask_float  # (B, T, 256)

        h_t, attn_weights = self.attention_pool(rnn_out, mask_resolved)  # (B, 256), (B, T)
        logits = self.classifier(h_t)  # (B, 7)

        return {
            "logits": logits,
            "embedding": h_t,
            "h_t": h_t,
            "attention_weights": attn_weights,
        }


def build_text_model(
    architecture: str = "distilbert_bilstm_attn",
    dropout: float = 0.3,
    num_classes: int = NUM_EMOTION_CLASSES,
    **kwargs: Any,
) -> nn.Module:
    """Factory function constructing a text emotion recognition model by architecture name."""
    if architecture == "distilbert_bilstm_attn":
        return DistilBERTBiLSTMAttentionClassifier(
            dropout=dropout,
            num_classes=num_classes,
            **kwargs,
        )
    raise ValueError(
        f"Unknown text architecture '{architecture}'. Expected one of {TEXT_ARCHITECTURES}."
    )
