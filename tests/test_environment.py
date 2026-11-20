"""Automated verification tests for the Assignment 2b Python environment (.venv)."""

import importlib
import sys
from pathlib import Path

import pytest


REQUIRED_PACKAGES = [
    "torch",
    "torchaudio",
    "transformers",
    "librosa",
    "soundfile",
    "scipy",
    "sklearn",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "kagglehub",
    "pytest",
]


def test_running_inside_project_venv():
    """Verify tests execute inside the isolated .venv environment and not Anaconda base."""
    exe_path = Path(sys.executable).resolve()
    assert ".venv" in exe_path.parts or ".venv" in str(Path(sys.prefix)), (
        f"Expected interpreter inside .venv, got sys.executable={exe_path}, sys.prefix={sys.prefix}"
    )
    assert "/opt/anaconda3" not in str(exe_path), (
        f"Interpreter must not use Anaconda base environment: {exe_path}"
    )


@pytest.mark.parametrize("package_name", REQUIRED_PACKAGES)
def test_required_packages_import(package_name: str):
    """Verify every required dependency imports without ABI or missing library errors."""
    mod = importlib.import_module(package_name)
    assert mod is not None


def test_numpy_scipy_sklearn_pandas_abi_compatibility():
    """Verify NumPy C-extension ABI compatibility with SciPy, scikit-learn, Pandas, and Matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from sklearn.metrics import accuracy_score, f1_score

    arr = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    mat = sp.csr_matrix(arr)
    assert mat.nnz == 2

    df = pd.DataFrame({"y_true": [0, 1, 2], "y_pred": [0, 1, 2]})
    assert accuracy_score(df["y_true"], df["y_pred"]) == 1.0
    assert f1_score(df["y_true"], df["y_pred"], average="macro") == 1.0
    assert plt is not None


def test_pytorch_mps_available_and_tensor_ops():
    """Verify PyTorch detects Apple MPS and executes basic tensor and autograd operations on MPS."""
    import torch

    assert not torch.cuda.is_available(), "CUDA should not be present on Apple Silicon macOS"
    assert torch.backends.mps.is_built(), "PyTorch must be built with MPS support"
    assert torch.backends.mps.is_available(), "PyTorch must detect Apple Silicon MPS device"

    device = torch.device("mps")
    a = torch.tensor([[1.5, -2.0], [3.0, 4.0]], device=device, requires_grad=True)
    b = torch.tensor([[2.0, 0.5], [-1.0, 1.0]], device=device)
    c = torch.matmul(a, b).relu().sum()
    c.backward()

    assert c.device.type == "mps"
    assert a.grad is not None
    assert a.grad.device.type == "mps"
    assert torch.isfinite(c).all().item()


def test_wavlm_and_distilbert_architectures_loadable():
    """Verify pretrained WavLM Base Plus and DistilBERT load and execute forward passes on MPS."""
    import torch
    from transformers import (
        DistilBertModel,
        DistilBertTokenizerFast,
        Wav2Vec2FeatureExtractor,
        WavLMModel,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus", local_files_only=True)
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", local_files_only=True).to(device).eval()
    dummy_audio = torch.randn(1, 16000, device=device)
    with torch.no_grad():
        wavlm_out = wavlm(dummy_audio)
    assert wavlm_out.last_hidden_state.shape == (1, 49, 768)
    assert fe.sampling_rate == 16000

    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased", local_files_only=True)
    distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased", local_files_only=True).to(device).eval()
    enc = tok(["Say the word back"], return_tensors="pt").to(device)
    with torch.no_grad():
        db_out = distilbert(**enc)
    assert db_out.last_hidden_state.shape == (1, 6, 768)

