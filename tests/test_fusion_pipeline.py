"""Unit tests for the Multimodal Fusion architectures (Stage 08A).

Uses synthetic tensors only; does not require access to the TESS feature cache,
frozen checkpoints, or training pipelines.
"""

import pytest
import torch

from models.fusion_pipeline import (
    ConcatFusion,
    FUSION_ARCHITECTURES,
    GatedMultimodalFusion,
    ID_TO_EMOTION,
    LateFusionEnsemble,
    NUM_EMOTION_CLASSES,
    build_fusion_model,
)
from src.data.build_manifest import EMOTION_TO_ID


def _count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# General & Factory Tests
# =============================================================================


def test_fusion_factory_and_canonical_mapping():
    """Verify factory builds all three fusion models and preserves the 7-class mapping."""
    assert FUSION_ARCHITECTURES == ("gated_fusion", "concat_fusion", "late_fusion")
    assert NUM_EMOTION_CLASSES == 7
    assert {v: k for k, v in ID_TO_EMOTION.items()} == EMOTION_TO_ID

    gated = build_fusion_model("gated_fusion")
    concat = build_fusion_model("concat_fusion")
    late = build_fusion_model("late_fusion", lambda_weight=0.7)

    assert isinstance(gated, GatedMultimodalFusion)
    assert isinstance(concat, ConcatFusion)
    assert isinstance(late, LateFusionEnsemble)
    assert late.lambda_weight == pytest.approx(0.7)

    with pytest.raises(ValueError, match="Unknown fusion architecture"):
        build_fusion_model("nonexistent_fusion_arch")


def test_parameter_counts_and_independent_weights():
    """Verify exact trainable parameter counts and ensure separate instances share no parameters."""
    gated_a = GatedMultimodalFusion()
    gated_b = GatedMultimodalFusion()
    concat_a = ConcatFusion()
    concat_b = ConcatFusion()
    late = LateFusionEnsemble()

    # Exact hand-calculated counts for defaults (256, 256 -> 256 -> 128 -> 7):
    # Classifier MLP: Linear(256,128) [32896] + LN(128) [256] + Linear(128,7) [903] = 34,055
    # Gated: speech_proj(256->256) [65792] + text_proj(256->256) [65792]
    #        + gate_proj(512->256) [131328] + classifier [34055] = 296,967
    # Concat: Linear(512->256) [131328] + LN(256) [512] + classifier [34055] = 165,895
    # Late: 0 trainable parameters
    assert _count_trainable_parameters(gated_a) == 296_967
    assert _count_trainable_parameters(concat_a) == 165_895
    assert _count_trainable_parameters(late) == 0

    # Ensure no parameter tensor objects are shared across models or instances
    gated_a_ids = {id(p) for p in gated_a.parameters()}
    gated_b_ids = {id(p) for p in gated_b.parameters()}
    concat_a_ids = {id(p) for p in concat_a.parameters()}
    concat_b_ids = {id(p) for p in concat_b.parameters()}

    assert gated_a_ids.isdisjoint(gated_b_ids)
    assert concat_a_ids.isdisjoint(concat_b_ids)
    assert gated_a_ids.isdisjoint(concat_a_ids)


# =============================================================================
# Architecture A — GatedMultimodalFusion Tests
# =============================================================================


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_gated_fusion_output_shapes_and_gate_bounds(batch_size: int):
    """Verify GatedMultimodalFusion keys, shapes, finiteness, and gate bounds for B=1 and B>1."""
    torch.manual_seed(42)
    model = GatedMultimodalFusion().eval()
    speech_emb = torch.randn(batch_size, 256)
    text_emb = torch.randn(batch_size, 256)

    with torch.no_grad():
        out = model(speech_emb, text_emb)

    assert set(out.keys()) == {"logits", "embedding", "gate"}
    assert out["logits"].shape == (batch_size, 7)
    assert out["embedding"].shape == (batch_size, 256)
    assert out["gate"].shape == (batch_size, 256)

    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["embedding"]).all()
    assert torch.isfinite(out["gate"]).all()
    assert (out["gate"] >= 0.0).all() and (out["gate"] <= 1.0).all()


def test_gated_fusion_deterministic_eval_and_train_mode():
    """Verify eval() mode is deterministic and train() mode executes cleanly."""
    torch.manual_seed(123)
    model = GatedMultimodalFusion(dropout=0.3)
    speech_emb = torch.randn(6, 256)
    text_emb = torch.randn(6, 256)

    model.eval()
    with torch.no_grad():
        out_1 = model(speech_emb, text_emb)
        out_2 = model(speech_emb, text_emb)

    assert torch.equal(out_1["logits"], out_2["logits"])
    assert torch.equal(out_1["embedding"], out_2["embedding"])
    assert torch.equal(out_1["gate"], out_2["gate"])

    model.train()
    out_train = model(speech_emb, text_emb)
    assert out_train["logits"].shape == (6, 7)
    assert torch.isfinite(out_train["logits"]).all()


def test_gated_fusion_gradient_flow():
    """Verify finite non-zero gradients flow through all trainable GatedMultimodalFusion parameters."""
    torch.manual_seed(42)
    model = GatedMultimodalFusion(dropout=0.1).train()
    # Frozen encoders produce detached embeddings (requires_grad=False)
    speech_emb = torch.randn(4, 256)
    text_emb = torch.randn(4, 256)
    labels = torch.tensor([0, 2, 4, 6], dtype=torch.long)

    out = model(speech_emb, text_emb)
    loss = torch.nn.functional.cross_entropy(out["logits"], labels)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.requires_grad
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


def test_gated_fusion_invalid_inputs_raise_value_error():
    """Verify invalid ranks, feature dimensions, and mismatched batch sizes raise ValueError."""
    model = GatedMultimodalFusion().eval()

    with pytest.raises(ValueError, match="Expected 2D `speech_embedding`"):
        model(torch.randn(256), torch.randn(2, 256))

    with pytest.raises(ValueError, match="Expected 2D `text_embedding`"):
        model(torch.randn(2, 256), torch.randn(2, 1, 256))

    with pytest.raises(ValueError, match="Batch size mismatch"):
        model(torch.randn(2, 256), torch.randn(3, 256))

    with pytest.raises(ValueError, match="Expected `speech_embedding` dimension 256"):
        model(torch.randn(2, 128), torch.randn(2, 256))

    with pytest.raises(ValueError, match="Expected `text_embedding` dimension 256"):
        model(torch.randn(2, 256), torch.randn(2, 512))


# =============================================================================
# Architecture B — ConcatFusion Tests
# =============================================================================


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_concat_fusion_output_shapes_and_finiteness(batch_size: int):
    """Verify ConcatFusion keys, shapes, and finiteness for B=1 and larger batches."""
    torch.manual_seed(42)
    model = ConcatFusion().eval()
    speech_emb = torch.randn(batch_size, 256)
    text_emb = torch.randn(batch_size, 256)

    with torch.no_grad():
        out = model(speech_emb, text_emb)

    assert set(out.keys()) == {"logits", "embedding"}
    assert out["logits"].shape == (batch_size, 7)
    assert out["embedding"].shape == (batch_size, 256)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["embedding"]).all()


def test_concat_fusion_deterministic_eval():
    """Verify ConcatFusion is deterministic in eval() mode."""
    torch.manual_seed(999)
    model = ConcatFusion(dropout=0.3).eval()
    speech_emb = torch.randn(5, 256)
    text_emb = torch.randn(5, 256)

    with torch.no_grad():
        out_1 = model(speech_emb, text_emb)
        out_2 = model(speech_emb, text_emb)

    assert torch.equal(out_1["logits"], out_2["logits"])
    assert torch.equal(out_1["embedding"], out_2["embedding"])


def test_concat_fusion_gradient_flow():
    """Verify finite non-zero gradients flow through all trainable ConcatFusion parameters."""
    torch.manual_seed(42)
    model = ConcatFusion(dropout=0.1).train()
    speech_emb = torch.randn(4, 256)
    text_emb = torch.randn(4, 256)
    labels = torch.tensor([1, 3, 5, 0], dtype=torch.long)

    out = model(speech_emb, text_emb)
    loss = torch.nn.functional.cross_entropy(out["logits"], labels)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.requires_grad
        assert param.grad is not None, f"Missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


def test_concat_fusion_invalid_inputs_raise_value_error():
    """Verify invalid ranks, dimensions, and batch size mismatches raise ValueError."""
    model = ConcatFusion().eval()

    with pytest.raises(ValueError, match="Expected 2D `speech_embedding`"):
        model(torch.randn(1, 2, 256), torch.randn(2, 256))

    with pytest.raises(ValueError, match="Batch size mismatch"):
        model(torch.randn(4, 256), torch.randn(2, 256))

    with pytest.raises(ValueError, match="Expected `speech_embedding` dimension 256"):
        model(torch.randn(2, 768), torch.randn(2, 256))

    with pytest.raises(ValueError, match="Expected `text_embedding` dimension 256"):
        model(torch.randn(2, 256), torch.randn(2, 64))


# =============================================================================
# Architecture C — LateFusionEnsemble Tests
# =============================================================================


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_late_fusion_shapes_probability_sum_and_zero_params(batch_size: int):
    """Verify LateFusionEnsemble output shapes, probability normalization, and zero trainable params."""
    torch.manual_seed(42)
    ensemble = LateFusionEnsemble(lambda_weight=0.7).eval()
    assert _count_trainable_parameters(ensemble) == 0
    assert list(ensemble.parameters()) == []

    speech_logits = torch.randn(batch_size, 7)
    text_logits = torch.randn(batch_size, 7)

    out = ensemble(speech_logits, text_logits)
    assert set(out.keys()) == {"probabilities", "predictions", "lambda_weight"}
    assert out["probabilities"].shape == (batch_size, 7)
    assert out["predictions"].shape == (batch_size,)
    assert out["predictions"].dtype == torch.long
    assert out["lambda_weight"] == pytest.approx(0.7)

    assert torch.isfinite(out["probabilities"]).all()
    assert (out["probabilities"] >= 0.0).all() and (out["probabilities"] <= 1.0).all()
    prob_sums = out["probabilities"].sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)
    assert torch.equal(out["predictions"], torch.argmax(out["probabilities"], dim=-1))


def test_late_fusion_lambda_extremes_and_midpoint():
    """Verify lambda=0 selects text probs, lambda=1 selects speech probs, and lambda=0.5 averages them."""
    torch.manual_seed(7)
    ensemble = LateFusionEnsemble(lambda_weight=0.5)
    speech_logits = torch.randn(6, 7)
    text_logits = torch.randn(6, 7)

    p_s = torch.softmax(speech_logits, dim=-1)
    p_t = torch.softmax(text_logits, dim=-1)

    out_text = ensemble(speech_logits, text_logits, lambda_weight=0.0)
    out_speech = ensemble(speech_logits, text_logits, lambda_weight=1.0)
    out_mid = ensemble(speech_logits, text_logits, lambda_weight=0.5)

    assert out_text["lambda_weight"] == 0.0
    assert out_speech["lambda_weight"] == 1.0
    assert out_mid["lambda_weight"] == 0.5

    assert torch.allclose(out_text["probabilities"], p_t, atol=1e-6)
    assert torch.allclose(out_speech["probabilities"], p_s, atol=1e-6)
    assert torch.allclose(out_mid["probabilities"], 0.5 * p_s + 0.5 * p_t, atol=1e-6)

    # Verify evaluate_lambdas helper matches individual calls
    sweep = ensemble.evaluate_lambdas(speech_logits, text_logits, [0.0, 0.5, 1.0])
    assert set(sweep.keys()) == {0.0, 0.5, 1.0}
    assert torch.equal(sweep[0.0]["probabilities"], out_text["probabilities"])
    assert torch.equal(sweep[0.5]["probabilities"], out_mid["probabilities"])
    assert torch.equal(sweep[1.0]["probabilities"], out_speech["probabilities"])


def test_late_fusion_invalid_lambda_and_inputs_raise_value_error():
    """Verify out-of-range lambdas, non-finite lambdas, and invalid logit shapes raise ValueError."""
    for bad_lambda in [-0.01, 1.01, float("nan"), float("inf"), True]:
        with pytest.raises(ValueError, match="lambda_weight"):
            LateFusionEnsemble(lambda_weight=bad_lambda)  # type: ignore[arg-type]

    ensemble = LateFusionEnsemble(lambda_weight=0.5)
    valid_logits = torch.randn(4, 7)

    with pytest.raises(ValueError, match="lambda_weight"):
        ensemble(valid_logits, valid_logits, lambda_weight=-0.5)

    with pytest.raises(ValueError, match="lambda_weight"):
        ensemble(valid_logits, valid_logits, lambda_weight=1.5)

    with pytest.raises(ValueError, match="Expected 2D `speech_logits`"):
        ensemble(torch.randn(7), valid_logits)

    with pytest.raises(ValueError, match="Batch size mismatch"):
        ensemble(torch.randn(4, 7), torch.randn(5, 7))

    with pytest.raises(ValueError, match="Expected `speech_logits` dimension 7"):
        ensemble(torch.randn(4, 5), valid_logits)

    with pytest.raises(ValueError, match="Expected `text_logits` dimension 7"):
        ensemble(valid_logits, torch.randn(4, 10))
