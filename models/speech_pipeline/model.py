"""Speech-only emotion recognition architectures for Assignment 2b.

Implements:
1. `WavLMBiGRUAttentionClassifier` (`wavlm_bigru_attn`):
   - Input: Cached WavLM Base Plus frame sequences `(B, T, 768)` + valid lengths/mask.
   - Linear projection `768 -> 256` + LayerNorm + GELU + Dropout.
   - 2-layer Bidirectional GRU (`hidden_size=128`, output `256`) using `pack_padded_sequence`.
   - Masked additive self-attention pooling yielding 256-d temporal representation `h_s`
     and normalized frame attention weights `alpha_s` (`0` on padded frames, sum `1` on valid frames).
   - 2-layer MLP classifier: `256 -> 128 -> 7`.

2. `MFCCCNNBiLSTMAttentionClassifier` (`mfcc_cnn_bilstm`):
   - Input: Cached MFCC + Delta + Delta-Delta sequences `(B, T, 120)` + valid lengths/mask.
   - 2-block stride-1 length-preserving 1D convolutional front-end (`120 -> 192 -> 256`)
     with per-frame LayerNorm, GELU, Dropout, and explicit inter-layer zero-masking so
     convolutional padding never contaminates valid frame representations.
   - 2-layer Bidirectional LSTM (`hidden_size=128`, output `256`) using `pack_padded_sequence`.
   - Masked additive self-attention pooling yielding 256-d temporal representation `h_s`
     and normalized frame attention weights `alpha_s`.
   - 2-layer MLP classifier: `256 -> 128 -> 7`.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.feature_dataset import build_padding_mask


SPEECH_ARCHITECTURES = ("wavlm_bigru_attn", "mfcc_cnn_bilstm")
NUM_EMOTION_CLASSES = 7


def resolve_mask_and_lengths(
    x: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensure consistent 1D `lengths` (B,) and 2D boolean `mask` (B, T) for sequence tensor `x`."""
    batch_size, max_len, _ = x.shape
    if lengths is None and mask is None:
        lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=x.device)
        mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=x.device)
    elif lengths is not None and mask is None:
        lengths = lengths.to(device=x.device, dtype=torch.long)
        mask = build_padding_mask(lengths, max_len=max_len)
    elif mask is not None and lengths is None:
        mask = mask.to(device=x.device, dtype=torch.bool)
        lengths = mask.sum(dim=1).to(dtype=torch.long)
    else:
        lengths = lengths.to(device=x.device, dtype=torch.long)
        mask = build_padding_mask(lengths, max_len=max_len)

    if (lengths <= 0).any():
        raise ValueError(f"All sequence lengths must be >= 1, got lengths={lengths.tolist()}")
    return lengths, mask


class MaskedAdditiveAttentionPool(nn.Module):
    """Masked additive (Bahdanau-style) self-attention pooling over variable-length sequences."""

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
        """Pool `(B, T, D)` sequence into `(B, D)` representation using boolean validity `mask` `(B, T)`."""
        mask_bool = mask.to(device=sequence_output.device, dtype=torch.bool)
        energy = self.score(self.dropout(self.tanh(self.proj(sequence_output)))).squeeze(-1)  # (B, T)

        # Fill padded positions with -inf prior to softmax so exp(-inf) == 0.0 exactly
        masked_energy = energy.masked_fill(~mask_bool, float("-inf"))
        attn_weights = torch.softmax(masked_energy, dim=-1)

        # Multiply explicitly by float mask and renormalize to guarantee exact 0.0 on padding
        mask_float = mask_bool.to(dtype=sequence_output.dtype)
        attn_weights = attn_weights * mask_float
        denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attn_weights = attn_weights / denom

        pooled = torch.bmm(attn_weights.unsqueeze(1), sequence_output).squeeze(1)  # (B, D)
        return pooled, attn_weights


class EmotionMLPClassifier(nn.Module):
    """Two-layer MLP classification head: `256 -> 128 -> 7`."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.norm1(self.fc1(h))))
        return self.fc2(x)


class WavLMBiGRUAttentionClassifier(nn.Module):
    """Primary speech model: Linear(768->256) + 2-Layer BiGRU(128) + Masked Attention + MLP(256->128->7)."""

    def __init__(
        self,
        input_dim: int = 768,
        proj_dim: int = 256,
        gru_hidden_dim: int = 128,
        gru_layers: int = 2,
        attn_dim: int = 128,
        mlp_hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture_name = "wavlm_bigru_attn"
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers
        self.repr_dim = gru_hidden_dim * 2  # 256

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.bigru = nn.GRU(
            input_size=proj_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.post_rnn_norm = nn.LayerNorm(self.repr_dim)
        self.attention_pool = MaskedAdditiveAttentionPool(
            input_dim=self.repr_dim,
            attn_dim=attn_dim,
            dropout=min(dropout, 0.2),
        )
        self.classifier = EmotionMLPClassifier(
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
        lengths_resolved, mask_resolved = resolve_mask_and_lengths(features, lengths=lengths, mask=mask)
        max_len = features.shape[1]
        mask_float = mask_resolved.unsqueeze(-1).to(dtype=features.dtype)

        # Project valid frames and zero out padded positions
        proj = self.input_proj(features) * mask_float  # (B, T, 256)

        # Pack padded sequence so padded frames never influence BiGRU recurrence in either direction
        packed = pack_padded_sequence(
            proj,
            lengths_resolved.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.bigru(packed)
        rnn_out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=max_len,
        )
        rnn_out = self.post_rnn_norm(rnn_out) * mask_float  # (B, T, 256)

        h_s, attn_weights = self.attention_pool(rnn_out, mask_resolved)  # (B, 256), (B, T)
        logits = self.classifier(h_s)  # (B, 7)

        return {
            "logits": logits,
            "embedding": h_s,
            "h_s": h_s,
            "attention_weights": attn_weights,
        }


class MaskedConv1DBlock(nn.Module):
    """Length-preserving stride-1 Conv1D block with per-frame LayerNorm and strict zero-masking.

    Zeroing out padded positions before and after the kernel_size=3, padding=1 convolution ensures
    that the right-boundary neighbor at t = L_i is always 0.0 regardless of batch padding length T_max,
    preserving strict padding-mask invariance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd for length-preserving padding, got {kernel_size}")
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask_float: torch.Tensor) -> torch.Tensor:
        """Apply masked 1D convolution to `x` of shape `(B, T, C_in)` with `mask_float` `(B, T, 1)`."""
        x_masked = x * mask_float
        conv_out = self.conv(x_masked.transpose(1, 2)).transpose(1, 2)  # (B, T, C_out)
        out = self.drop(self.act(self.norm(conv_out)))
        return out * mask_float


class MFCCCNNBiLSTMAttentionClassifier(nn.Module):
    """Baseline speech model: 2-Layer Masked Conv1D + 2-Layer BiLSTM(128) + Masked Attention + MLP."""

    def __init__(
        self,
        input_dim: int = 120,
        conv_channels: Tuple[int, int] = (192, 256),
        kernel_size: int = 3,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        attn_dim: int = 128,
        mlp_hidden_dim: int = 128,
        num_classes: int = NUM_EMOTION_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture_name = "mfcc_cnn_bilstm"
        self.input_dim = input_dim
        self.conv_channels = conv_channels
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.repr_dim = lstm_hidden_dim * 2  # 256

        self.input_norm = nn.LayerNorm(input_dim)
        self.conv1 = MaskedConv1DBlock(
            in_channels=input_dim,
            out_channels=conv_channels[0],
            kernel_size=kernel_size,
            dropout=min(dropout, 0.25),
        )
        self.conv2 = MaskedConv1DBlock(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=kernel_size,
            dropout=min(dropout, 0.25),
        )

        self.bilstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.post_rnn_norm = nn.LayerNorm(self.repr_dim)
        self.attention_pool = MaskedAdditiveAttentionPool(
            input_dim=self.repr_dim,
            attn_dim=attn_dim,
            dropout=min(dropout, 0.2),
        )
        self.classifier = EmotionMLPClassifier(
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
        lengths_resolved, mask_resolved = resolve_mask_and_lengths(features, lengths=lengths, mask=mask)
        max_len = features.shape[1]
        mask_float = mask_resolved.unsqueeze(-1).to(dtype=features.dtype)

        x = self.input_norm(features) * mask_float
        x = self.conv1(x, mask_float)
        x = self.conv2(x, mask_float)  # (B, T, 256)

        packed = pack_padded_sequence(
            x,
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

        h_s, attn_weights = self.attention_pool(rnn_out, mask_resolved)  # (B, 256), (B, T)
        logits = self.classifier(h_s)  # (B, 7)

        return {
            "logits": logits,
            "embedding": h_s,
            "h_s": h_s,
            "attention_weights": attn_weights,
        }


def build_speech_model(
    architecture: str = "wavlm_bigru_attn",
    dropout: float = 0.3,
    num_classes: int = NUM_EMOTION_CLASSES,
    **kwargs: Any,
) -> nn.Module:
    """Factory function constructing a speech emotion recognition model by architecture name."""
    if architecture == "wavlm_bigru_attn":
        return WavLMBiGRUAttentionClassifier(
            dropout=dropout,
            num_classes=num_classes,
            **kwargs,
        )
    if architecture == "mfcc_cnn_bilstm":
        return MFCCCNNBiLSTMAttentionClassifier(
            dropout=dropout,
            num_classes=num_classes,
            **kwargs,
        )
    raise ValueError(
        f"Unknown speech architecture '{architecture}'. Expected one of {SPEECH_ARCHITECTURES}."
    )
