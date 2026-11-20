# Assignment_2_Speech

## Assignment 2b: Multimodal Emotion Recognition on TESS

This repository implements **Assignment 2b: Multimodal Emotion Recognition** on the **Toronto Emotional Speech Set (TESS)** (`ejlok1/toronto-emotional-speech-set-tess`), alongside the untouched `Assignment 2a/` Forced Alignment materials.

> **Current Project Stage:** **STEP 04 In Progress (Colab Pro Migration & Pipeline Resilience Hardening Complete)** — 3 of 12 speech-only experiments (`wavlm_bigru_attn` on `word_indep_split`, seeds `42, 123, 999`) have been completed on Mac M2 (`99.76 ± 0.00%` test Accuracy & Macro-F1); the remaining 9 speech-only runs are configured for Google Colab Pro (Tesla T4 CUDA) with corrected Macro-F1 early stopping, epoch-level checkpoint resume (`last_checkpoint.pt`), completed-run protection, persistent Google Drive `--output-root` support, and graceful missing-checkpoint evaluation.

---

## 1. Repository Structure

```text
/Users/swaruppal/Downloads/Ass_2/
├── .venv/                                  # Isolated Python 3.11 virtual environment (created via uv)
├── Assignment 2a/                          # Read-only Assignment 2a (Montreal Forced Aligner) materials
├── Assignment 2b.pdf                       # Assignment 2b problem specification
├── PROJECT_AUDIT.md                        # Step 01 complete repository and architecture audit
├── README.md                               # Project documentation & setup guide
├── requirements.txt                        # Pinned/compatible dependencies for macOS Apple Silicon (M2) & Colab
├── data/
│   ├── features/
│   │   ├── distilbert/                     # 200 unique transcript .pt files (L, 768) + sample_to_transcript.json
│   │   ├── mfcc/                           # 2,800 utterance .pt files (T_m, 120) + metadata.json
│   │   └── wavlm/                          # 2,800 utterance .pt files (T_s, 768) + metadata.json
│   ├── splits/
│   │   └── word_splits.json                # Seed (42), emotion mapping, deduplication stats & exact word lists
│   ├── tess/
│   │   └── raw/                            # Raw extracted Kaggle TESS dataset (5,600 raw .wav files)
│   └── tess_manifest.csv                   # Canonical deduplicated manifest (2,800 unique recordings)
├── docs/
│   ├── IMPLEMENTATION_LOG.md               # Stage-by-stage implementation log
│   └── TESTING_REPORT.md                   # Automated verification & test results report
├── models/
│   └── speech_pipeline/
│       ├── __init__.py
│       ├── model.py                        # WavLM+BiGRU+Attn & MFCC+CNN-BiLSTM+Attn architectures
│       ├── train.py                        # Training pipeline with F1 early stopping, resume & skip guards
│       └── test.py                         # Test evaluation, embedding export & multi-seed aggregation
├── results/
│   └── speech/
│       ├── word_indep_split/
│       │   └── wavlm_bigru_attn/           # Completed on Mac M2 (seeds 42, 123, 999: 99.76% Test Macro-F1)
│       └── summaries/                      # Aggregated CSV & JSON summaries
├── src/
│   ├── data/
│   │   ├── download_tess.py                # Automated Kaggle dataset download script
│   │   ├── build_manifest.py               # Canonical filename parser, deduplicator & split generator
│   │   └── feature_dataset.py              # Lazy-loading PyTorch Dataset & dynamic-padding collate_fn
│   └── features/
│       ├── speech_preprocessing.py         # Native-SR loader, 16 kHz resampler, silence trimmer & normalizer
│       ├── extract_speech_features.py      # Checkpointable WavLM Base Plus (768-d) & MFCC (120-d) extractor
│       └── extract_text_features.py        # Unique-transcript DistilBERT (768-d) extractor & sample mapper
└── tests/
    ├── conftest.py                         # Pytest root path configuration
    ├── test_environment.py                 # Environment, ABI, MPS & transformer load verification
    ├── test_dataset.py                     # TESS 2,800-sample integrity & audio readability tests
    ├── test_splits.py                      # Word-independent & cross-speaker zero-leakage tests
    ├── test_preprocessing_and_features.py  # Smoke tests + full 2,800-sample cache & DataLoader verification
    └── test_speech_pipeline.py             # Speech architectures, padding invariance, early stopping & resume tests
```

---

## 2. Environment Setup & Installed Package Versions

To avoid the `numpy 2.2.6` ABI incompatibility present in `/opt/anaconda3` (`base`), an isolated Python 3.11 virtual environment (`.venv`) was created using `uv` without touching the Anaconda installation:

```bash
/opt/homebrew/bin/uv venv .venv --python /opt/homebrew/bin/python3.11
/opt/homebrew/bin/uv pip install --python .venv/bin/python -r requirements.txt
```

### Verified Installed Versions in `.venv` (Python 3.11.15, macOS ARM64 / Apple M2)

| Package | Installed Version | Purpose |
| :--- | :--- | :--- |
| `python` | `3.11.15` | Isolated Homebrew CPython interpreter (`aarch64`) |
| `numpy` | `2.0.2` | Array computing (ABI-compatible with all compiled scientific wheels) |
| `scipy` | `1.17.1` | Signal processing and sparse matrix utilities |
| `scikit-learn` | `1.9.1` | Evaluation metrics, stratified utilities, and dimensionality reduction |
| `pandas` | `3.1.1` | Manifest management and tabular results |
| `matplotlib` | `3.11.2` | Plotting and visualization |
| `seaborn` | `0.13.2` | Confusion matrices and statistical visualizations |
| `torch` | `2.15.0` | Deep learning framework with Apple Silicon **MPS** & Colab **CUDA** support |
| `torchaudio` | `2.11.1` | Audio tensor processing |
| `transformers` | `5.23.0` | Pretrained `microsoft/wavlm-base-plus` and `distilbert-base-uncased` |
| `librosa` | `0.11.0` | Audio resampling, silence trimming, and MFCC extraction |
| `soundfile` | `0.14.0` | Fast libsndfile-backed `.wav` header inspection and audio I/O |
| `tqdm` | `4.70.1` | Progress bars |
| `kagglehub` | `1.0.3` | Authenticated Kaggle dataset acquisition |
| `pytest` | `9.2.0` | Automated verification suite |

---

## 3. TESS Dataset Acquisition & Actual Extracted Directory Structure

The official TESS dataset (`ejlok1/toronto-emotional-speech-set-tess`) was downloaded into `data/tess/raw/` using existing `~/.kaggle/kaggle.json` authentication via:

```bash
.venv/bin/python -m src.data.download_tess
```

### Discovered Real-World Dataset Irregularities & Normalization

1. **Nested Duplicate Directory (5,600 raw files vs. 2,800 unique utterances):**
   - `data/tess/raw/TESS Toronto emotional speech set data/` contains 14 emotion folders (2,800 `.wav` files) plus an identical nested copy `TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/` (2,800 duplicate `.wav` files). `src/data/build_manifest.py` deduplicates by canonical `sample_id = "{speaker}_{word}_{emotion}"`, verifies byte-for-byte SHA-256 identity across all 2,800 duplicate pairs, and retains strictly one canonical entry per utterance without deleting raw files.
2. **Inconsistent Folder Naming & Filename Suffixes:**
   - Normalizes `OAF_Fear`/`YAF_fear` $\rightarrow$ `fear`, `OAF_Sad`/`YAF_sad` $\rightarrow$ `sad`, and `OAF_Pleasant_surprise`/`YAF_pleasant_surprised`/`_ps.wav` $\rightarrow$ `pleasant_surprise`.
3. **Speaker Prefix Typo in `OA_bite_neutral.wav`:**
   - Inside `OAF_neutral/`, `OA_bite_neutral.wav` uses prefix `OA_` instead of `OAF_`; normalized to `sample_id = OAF_bite_neutral`.
4. **Sampling Rate Outlier in `OAF_Fear/OAF_food_fear.wav`:**
   - **2,799 files** are sampled at **24,414 Hz** (16-bit mono PCM), while **1 file** (`OAF_Fear/OAF_food_fear.wav`) is sampled at **96,000 Hz** (145,172 frames, 1.512208 s).

---

## 4. Canonical Emotion Label Mapping & Splits

| `label_id` | Canonical `emotion` | Raw Folder Suffixes | Raw Filename Suffixes | Utterance Count |
| :---: | :--- | :--- | :--- | :---: |
| `0` | `angry` | `angry` | `_angry.wav` | 400 (200 OAF, 200 YAF) |
| `1` | `disgust` | `disgust` | `_disgust.wav` | 400 (200 OAF, 200 YAF) |
| `2` | `fear` | `Fear`, `fear` | `_fear.wav` | 400 (200 OAF, 200 YAF) |
| `3` | `happy` | `happy` | `_happy.wav` | 400 (200 OAF, 200 YAF) |
| `4` | `neutral` | `neutral` | `_neutral.wav` | 400 (200 OAF, 200 YAF) |
| `5` | `pleasant_surprise` | `Pleasant_surprise`, `pleasant_surprised` | `_ps.wav` | 400 (200 OAF, 200 YAF) |
| `6` | `sad` | `Sad`, `sad` | `_sad.wav` | 400 (200 OAF, 200 YAF) |

- **Primary Split (`word_indep_split`, seed `42`):** 140 train words (1,960 samples), 30 val words (420 samples), 30 test words (420 samples). Zero target-word overlap across partitions.
- **Cross-Speaker Split (`cross_speaker_split`, `OAF` $\rightarrow$ `YAF`):** 170 `OAF` train words (1,190 samples), 30 `OAF` val words (210 samples), 200 `YAF` test words (1,400 samples). Zero speaker overlap between train/val (`OAF`) and test (`YAF`).

---

## 5. Step 03: Preprocessing, Offline Feature Extraction & Caching

| Cache Directory | Modality & Backbone | Cached Files | Feature Shape | Sequence Length (Min / Mean / Max) | Device | Extraction Time | Disk Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `data/features/wavlm/` | **WavLM Base Plus** (`microsoft/wavlm-base-plus`, frozen final hidden state) | 2,800 `.pt` + `metadata.json` | `(T_s, 768)` `float32` | `51` / `96.01` / `147` frames | `mps` (`0` fallbacks) | `397.09 s` | `793.061 MB` |
| `data/features/mfcc/` | **MFCC + $\Delta$ + $\Delta\Delta$** (40+40+40 @ 16 kHz, 25 ms win, 10 ms hop, configurable CMVN) | 2,800 `.pt` + `metadata.json` | `(T_m, 120)` `float32` | `104` / `194.21` / `296` frames | `cpu` | `11.54 s` | `254.378 MB` |
| `data/features/distilbert/` | **DistilBERT** (`distilbert-base-uncased`, cached by unique transcript + 2,800-sample map) | 200 `.pt` + `sample_to_transcript.json` + `metadata.json` | `(L_t, 768)` `float32` | `6` / `6.05` / `7` tokens | `mps` (`0` fallbacks) | `1.75 s` | `4.794 MB` |

---

## 6. Step 04: Speech-Only Emotion Recognition Pipeline & Colab Execution Guide

### 6.1. Pipeline Safeguards (`models/speech_pipeline/train.py` & `test.py`)
1. **Decoupled Early Stopping & Loss Tie-Breaker:**
   - `f1_improved = val_f1 > (best_val_macro_f1 + 1e-6)` resets `epochs_without_improvement = 0` and saves `best_model.pt`.
   - `loss_tiebreak` updates `best_model.pt`, `best_val_loss`, and `best_epoch` when validation Macro-F1 is tied and validation loss improves by `> 1e-6`, **while still incrementing `epochs_without_improvement`** so training terminates after `patience` epochs on a Macro-F1 plateau.
2. **Epoch-Level Resume (`last_checkpoint.pt`):**
   - Saved atomically at the end of every completed epoch with full latest `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `last_epoch`, `best_val_macro_f1`, `best_val_loss`, `best_epoch`, `epochs_without_improvement`, `history`, and RNG states (`train_generator`, Python, NumPy, Torch CPU/CUDA).
   - Automatically detected and restored on restart (after validating configuration compatibility), and deleted only after `training_history.json`, `training_history.csv`, and `best_model.pt` are finalized.
3. **Completed-Run Protection:**
   - If `best_model.pt` and `training_history.json` already exist in a seed's output directory, `train_single_run()` skips training and returns the existing summary unless `--overwrite` is explicitly passed.
4. **Missing-Checkpoint Handling in `test.py`:**
   - `evaluate_and_aggregate()` logs a clear warning and skips any absent checkpoint combinations (such as the 3 Mac-completed runs when evaluating on Colab), evaluating all present checkpoints and aggregating available results cleanly.
5. **Persistent Google Drive Output Support (`--output-root`):**
   - Pass `--output-root /content/drive/MyDrive/Ass_2_Results/results/speech` on Google Colab so all checkpoints, histories, metrics, confusion matrices, predictions, and `(N_test, 256)` embeddings are persisted directly to Google Drive while reading cached features from fast local `/content/Ass_2/data/features/` storage.

### 6.2. Exact Colab Pro Commands for the 3 Pending Speech Batches (9 Runs)

```bash
# Set persistent Google Drive output root
DRIVE_OUT="/content/drive/MyDrive/Ass_2_Results/results/speech"

# Batch A — MFCC + CNN-BiLSTM on word_indep_split (Seeds 42, 123, 999)
python -m models.speech_pipeline.train \
  --splits word_indep_split \
  --architectures mfcc_cnn_bilstm \
  --seeds 42 123 999 \
  --output-root "$DRIVE_OUT"

# Batch B — WavLM + BiGRU + Attention on cross_speaker_split (Seeds 42, 123, 999)
python -m models.speech_pipeline.train \
  --splits cross_speaker_split \
  --architectures wavlm_bigru_attn \
  --seeds 42 123 999 \
  --output-root "$DRIVE_OUT"

# Batch C — MFCC + CNN-BiLSTM on cross_speaker_split (Seeds 42, 123, 999)
python -m models.speech_pipeline.train \
  --splits cross_speaker_split \
  --architectures mfcc_cnn_bilstm \
  --seeds 42 123 999 \
  --output-root "$DRIVE_OUT"

# Evaluate all available checkpoints and aggregate summaries
python -m models.speech_pipeline.test --output-root "$DRIVE_OUT"
```

---

## 7. Stage 07A–07B: Text-Only Emotion Recognition Pipeline (`models/text_pipeline/`)

`DistilBERTBiLSTMAttentionClassifier` (`distilbert_bilstm_attn`, `1,055,239` trainable parameters) consumes cached DistilBERT token embeddings `(B, T, 768)` via `TESSFeatureDataset` (`modalities=("distilbert",)`), applying `Linear(768 -> 256) + LayerNorm + GELU + Dropout`, a packed 2-layer BiLSTM (`hidden_size=128`), masked additive attention pooling (`h_t` in `R^256`), and a 2-layer MLP (`256 -> 128 -> 7`).

### Colab / Local Execution Commands for the 6 Text-Only Experiments (Ready for Stage 07C)

```bash
# Set persistent Google Drive output root on Colab (or omit for local results/text)
DRIVE_TEXT_OUT="/content/drive/MyDrive/Ass_2_Results/results/text"

# Word-independent split (Seeds 42, 123, 999)
python -m models.text_pipeline.train \
  --architectures distilbert_bilstm_attn \
  --splits word_independent \
  --seeds 42 123 999 \
  --output-root "$DRIVE_TEXT_OUT"

# Cross-speaker split (Seeds 42, 123, 999)
python -m models.text_pipeline.train \
  --architectures distilbert_bilstm_attn \
  --splits cross_speaker \
  --seeds 42 123 999 \
  --output-root "$DRIVE_TEXT_OUT"

# Evaluate all trained text checkpoints and aggregate summaries
python -m models.text_pipeline.test \
  --architectures distilbert_bilstm_attn \
  --splits word_independent cross_speaker \
  --seeds 42 123 999 \
  --output-root "$DRIVE_TEXT_OUT"
```

---

## 8. Verification & Automated Testing

Run the split, speech, and text pipeline regression suite with:

```bash
.venv/bin/python -m pytest tests/test_text_training_pipeline.py tests/test_text_pipeline.py tests/test_splits.py tests/test_speech_pipeline.py -v
```

- **Status:** **33 / 33 tests PASSED in 8.10s** (`6` text training/eval tests + `10` text model unit tests + `4` split tests + `13` speech pipeline tests).
