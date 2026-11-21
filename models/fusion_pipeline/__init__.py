"""Multimodal fusion pipeline for Assignment 2b (Stage 08A)."""

from models.fusion_pipeline.model import (
    ConcatFusion,
    FUSION_ARCHITECTURES,
    FusionEmotionMLPClassifier,
    GatedMultimodalFusion,
    ID_TO_EMOTION,
    LateFusionEnsemble,
    NUM_EMOTION_CLASSES,
    build_fusion_model,
)

__all__ = [
    "ConcatFusion",
    "FUSION_ARCHITECTURES",
    "FusionEmotionMLPClassifier",
    "GatedMultimodalFusion",
    "ID_TO_EMOTION",
    "LateFusionEnsemble",
    "NUM_EMOTION_CLASSES",
    "build_fusion_model",
]
