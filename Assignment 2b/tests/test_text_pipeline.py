"""Unit tests for the Text-Only Emotion Recognition model (Stage 07A).

Uses synthetic tensors only; does not require access to the TESS feature cache or run training.
"""

from pathlib import Path

import pytest
import torch

from models.text_pipeline.model import (
    DistilBERTBiLSTMAttentionClassifier,
    ID_TO_EMOTION,
    NUM_EMOTION_CLASSES,
    TEXT_ARCHITECTURES,
    build_text_model,
)
from src.data.build_manifest import EMOTION_TO_ID
from src.data.feature_dataset import build_padding_mask


def test_model_initialization_and_label_mapping():
    """Verify model initializes via class and factory and uses the canonical 7-class mapping."""
    model = build_text_model("distilbert_bilstm_attn", dropout=0.3)
    assert isinstance(model, DistilBERTBiLSTMAttentionClassifier)
    assert model.architecture_name == "distilbert_bilstm_attn"
    assert model.input_dim == 768
    assert model.repr_dim == 256
    assert model.num_classes == 7
    assert NUM_EMOTION_CLASSES == 7
    assert TEXT_ARCHITECTURES == ("distilbert_bilstm_attn",)
    assert ID_TO_EMOTION == {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "pleasant_surprise",
        6: "sad",
    }
    assert {v: k for k, v in ID_TO_EMOTION.items()} == EMOTION_TO_ID

    with pytest.raises(ValueError, match="Unknown text architecture"):
        build_text_model("nonexistent_text_arch")


def test_output_shapes_and_attention_properties():
    """Verify (B, 7) logits, (B, 256) embedding, (B, T) attention weights, sum==1, and 0 on padding."""
    torch.manual_seed(42)
    model = build_text_model("distilbert_bilstm_attn", dropout=0.2).eval()

    batch_size, max_len = 5, 10
    lengths = torch.tensor([10, 7, 6, 4, 2], dtype=torch.long)
    mask = build_padding_mask(lengths, max_len=max_len)
    features = torch.randn(batch_size, max_len, 768)

    with torch.no_grad():
        out = model(features, lengths=lengths, mask=mask)

    assert set(out.keys()) >= {"logits", "embedding", "h_t", "attention_weights"}
    assert out["logits"].shape == (batch_size, 7)
    assert out["embedding"].shape == (batch_size, 256)
    assert torch.equal(out["embedding"], out["h_t"])
    assert out["attention_weights"].shape == (batch_size, max_len)

    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["embedding"]).all()
    assert torch.isfinite(out["attention_weights"]).all()

    attn = out["attention_weights"]
    for i, valid_len in enumerate(lengths.tolist()):
        valid_sum = attn[i, :valid_len].sum().item()
        assert abs(valid_sum - 1.0) < 1e-5, f"Valid attention sum={valid_sum}, expected 1.0"
        if valid_len < max_len:
            padded_weights = attn[i, valid_len:]
            assert torch.all(padded_weights == 0.0), (
                f"Non-zero attention weight on padded tokens: {padded_weights}"
            )


def test_forward_variable_lengths_with_lengths_only_or_mask_only():
    """Verify forward pass works identically when given lengths only, mask only, or both."""
    torch.manual_seed(123)
    model = build_text_model("distilbert_bilstm_attn", dropout=0.0).eval()

    lengths = torch.tensor([9, 6, 7, 3], dtype=torch.long)
    mask = build_padding_mask(lengths, max_len=9)
    features = torch.randn(4, 9, 768)

    with torch.no_grad():
        out_both = model(features, lengths=lengths, mask=mask)
        out_len_only = model(features, lengths=lengths)
        out_mask_only = model(features, mask=mask)

    assert torch.allclose(out_both["logits"], out_len_only["logits"], atol=1e-6)
    assert torch.allclose(out_both["logits"], out_mask_only["logits"], atol=1e-6)
    assert torch.allclose(out_both["attention_weights"], out_len_only["attention_weights"], atol=1e-6)
    assert torch.allclose(out_both["attention_weights"], out_mask_only["attention_weights"], atol=1e-6)


def test_padding_invariance_on_noise_and_extra_trailing_tokens():
    """Verify altering padded feature values or appending extra padded tokens does not alter outputs."""
    torch.manual_seed(999)
    model = build_text_model("distilbert_bilstm_attn", dropout=0.0).eval()

    lengths = torch.tensor([6, 4], dtype=torch.long)
    base_len = 6
    x_base = torch.randn(2, base_len, 768)
    mask_base = build_padding_mask(lengths, max_len=base_len)

    # Case 1: Mutate padded positions [4..5] of sample 1 with large values
    x_mutated = x_base.clone()
    x_mutated[1, 4:, :] = 100.0

    # Case 2: Append 8 extra trailing padded tokens filled with random noise
    noise_tail = torch.randn(2, 8, 768) * 10.0
    x_extended = torch.cat([x_mutated, noise_tail], dim=1)
    mask_extended = build_padding_mask(lengths, max_len=base_len + 8)

    with torch.no_grad():
        out_base = model(x_base, lengths=lengths, mask=mask_base)
        out_mut = model(x_mutated, lengths=lengths, mask=mask_base)
        out_ext = model(x_extended, lengths=lengths, mask=mask_extended)

    assert torch.allclose(out_base["logits"], out_mut["logits"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_base["embedding"], out_mut["embedding"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_base["attention_weights"], out_mut["attention_weights"], atol=1e-5)

    assert torch.allclose(out_base["logits"], out_ext["logits"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_base["embedding"], out_ext["embedding"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        out_base["attention_weights"],
        out_ext["attention_weights"][:, :base_len],
        atol=1e-5,
    )
    assert torch.all(out_ext["attention_weights"][:, base_len:] == 0.0)


def test_backward_pass_and_gradient_flow():
    """Verify backward pass computes finite non-zero gradients for all trainable parameters."""
    torch.manual_seed(42)
    model = build_text_model("distilbert_bilstm_attn", dropout=0.1).train()

    lengths = torch.tensor([7, 6, 5], dtype=torch.long)
    mask = build_padding_mask(lengths, max_len=7)
    features = torch.randn(3, 7, 768, requires_grad=True)
    labels = torch.tensor([0, 4, 6], dtype=torch.long)

    out = model(features, lengths=lengths, mask=mask)
    loss = torch.nn.functional.cross_entropy(out["logits"], labels)
    loss.backward()

    assert features.grad is not None and torch.isfinite(features.grad).all()
    # Padded positions must receive zero gradient
    assert torch.all(features.grad[1, 6:, :] == 0.0)
    assert torch.all(features.grad[2, 5:, :] == 0.0)

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter {name}"
            assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for parameter {name}"


def test_invalid_input_dimensions_raise_value_error():
    """Verify wrong tensor rank or wrong feature dimension raises ValueError."""
    model = build_text_model("distilbert_bilstm_attn").eval()

    # 2D input instead of 3D
    with pytest.raises(ValueError, match="Expected 3D text features"):
        model(torch.randn(6, 768))

    # 4D input instead of 3D
    with pytest.raises(ValueError, match="Expected 3D text features"):
        model(torch.randn(2, 1, 6, 768))

    # Wrong feature dimension (e.g., 512 instead of 768)
    with pytest.raises(ValueError, match="Expected text feature dimension 768"):
        model(torch.randn(2, 6, 512), lengths=torch.tensor([6, 5]))


def test_invalid_sequence_lengths_raise_value_error():
    """Verify zero, negative, out-of-range, float, or wrong-shape lengths raise ValueError."""
    model = build_text_model("distilbert_bilstm_attn").eval()
    x = torch.randn(2, 6, 768)

    with pytest.raises(ValueError, match="All sequence lengths must satisfy"):
        model(x, lengths=torch.tensor([6, 0]))

    with pytest.raises(ValueError, match="All sequence lengths must satisfy"):
        model(x, lengths=torch.tensor([6, -1]))

    with pytest.raises(ValueError, match="All sequence lengths must satisfy"):
        model(x, lengths=torch.tensor([6, 7]))

    with pytest.raises(ValueError, match="Expected 1D `lengths`"):
        model(x, lengths=torch.tensor([6, 5, 4]))

    with pytest.raises(ValueError, match="Expected integer dtype"):
        model(x, lengths=torch.tensor([6.0, 5.0]))


def test_mask_length_inconsistency_raises_value_error():
    """Verify contradictory mask and lengths (or non-contiguous mask) raise ValueError."""
    model = build_text_model("distilbert_bilstm_attn").eval()
    x = torch.randn(2, 6, 768)

    lengths = torch.tensor([6, 4], dtype=torch.long)
    wrong_mask = build_padding_mask(torch.tensor([6, 5]), max_len=6)

    with pytest.raises(ValueError, match="Inconsistent `lengths` and `mask`"):
        model(x, lengths=lengths, mask=wrong_mask)

    # Wrong mask shape
    with pytest.raises(ValueError, match="Expected 2D `mask`"):
        model(x, mask=torch.ones(2, 5, dtype=torch.bool))

    # Non-contiguous mask (gap inside valid tokens)
    non_contig_mask = torch.tensor(
        [[True, True, False, True, False, False], [True, True, True, True, False, False]],
        dtype=torch.bool,
    )
    with pytest.raises(ValueError, match="non-contiguous"):
        model(x, mask=non_contig_mask)

    # All-False mask row
    empty_row_mask = torch.tensor(
        [[True, True, False, False, False, False], [False, False, False, False, False, False]],
        dtype=torch.bool,
    )
    with pytest.raises(ValueError, match="at least 1 valid token"):
        model(x, mask=empty_row_mask)


def test_batch_size_one_and_device_execution():
    """Verify batch size of 1 executes cleanly on CPU and available accelerator (MPS/CUDA)."""
    torch.manual_seed(7)
    model = build_text_model("distilbert_bilstm_attn", dropout=0.0).eval()

    x = torch.randn(1, 6, 768)
    lengths = torch.tensor([6], dtype=torch.long)
    mask = build_padding_mask(lengths, max_len=6)

    with torch.no_grad():
        out_cpu = model(x, lengths=lengths, mask=mask)

    assert out_cpu["logits"].shape == (1, 7)
    assert out_cpu["embedding"].shape == (1, 256)
    assert out_cpu["attention_weights"].shape == (1, 6)
    assert abs(out_cpu["attention_weights"].sum().item() - 1.0) < 1e-5
    assert torch.isfinite(out_cpu["logits"]).all()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        m_dev = model.to(device)
        with torch.no_grad():
            out_dev = m_dev(x.to(device), lengths=lengths.to(device), mask=mask.to(device))
        assert out_dev["logits"].device.type == "mps"
        assert torch.allclose(out_cpu["logits"], out_dev["logits"].cpu(), atol=1e-4, rtol=1e-4)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        m_dev = model.to(device)
        with torch.no_grad():
            out_dev = m_dev(x.to(device), lengths=lengths.to(device), mask=mask.to(device))
        assert out_dev["logits"].device.type == "cuda"
        assert torch.allclose(out_cpu["logits"], out_dev["logits"].cpu(), atol=1e-4, rtol=1e-4)


def test_state_dict_save_and_load(tmp_path: Path):
    """Verify model can be saved and loaded via state_dict with bit-identical outputs."""
    torch.manual_seed(11)
    model_orig = build_text_model("distilbert_bilstm_attn", dropout=0.0).eval()
    ckpt_file = tmp_path / "text_model_state.pt"
    torch.save(model_orig.state_dict(), ckpt_file)

    model_loaded = build_text_model("distilbert_bilstm_attn", dropout=0.0).eval()
    model_loaded.load_state_dict(torch.load(ckpt_file, map_location="cpu", weights_only=True))

    x = torch.randn(3, 7, 768)
    lengths = torch.tensor([7, 6, 4], dtype=torch.long)
    mask = build_padding_mask(lengths, max_len=7)

    with torch.no_grad():
        out_orig = model_orig(x, lengths=lengths, mask=mask)
        out_loaded = model_loaded(x, lengths=lengths, mask=mask)

    assert torch.equal(out_orig["logits"], out_loaded["logits"])
    assert torch.equal(out_orig["embedding"], out_loaded["embedding"])
    assert torch.equal(out_orig["attention_weights"], out_loaded["attention_weights"])
