"""Text-only emotion recognition pipeline (DistilBERT + BiLSTM + Masked Attention)."""

from models.text_pipeline.model import (
    DistilBERTBiLSTMAttentionClassifier,
    ID_TO_EMOTION,
    NUM_EMOTION_CLASSES,
    TEXT_ARCHITECTURES,
    TextEmotionMLPClassifier,
    TextMaskedAdditiveAttentionPool,
    build_text_model,
    resolve_text_mask_and_lengths,
)

__all__ = [
    "DistilBERTBiLSTMAttentionClassifier",
    "ID_TO_EMOTION",
    "NUM_EMOTION_CLASSES",
    "TEXT_ARCHITECTURES",
    "TextEmotionMLPClassifier",
    "TextMaskedAdditiveAttentionPool",
    "build_text_model",
    "resolve_text_mask_and_lengths",
]
