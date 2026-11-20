# Implementation Log — Assignment 2b: Multimodal Emotion Recognition

> **Current Status:** **STAGE 07B COMPLETED — TEXT-ONLY TRAINING & EVALUATION PIPELINE IMPLEMENTED & VERIFIED (NO FULL TEXT EXPERIMENTS LAUNCHED)**  
> All 12 speech-only experiments are completed, integrated, and verified (see `docs/SPEECH_RESULTS_INTEGRATION_REPORT.md`). The text-only model (`models/text_pipeline/model.py`), training script (`models/text_pipeline/train.py`), evaluation script (`models/text_pipeline/test.py`), and synthetic smoke/unit tests (`tests/test_text_pipeline.py`, `tests/test_text_training_pipeline.py`) are implemented and verified (`33/33` tests passing across text, speech, and split suites). No full TESS text experiments have been started.

---

## Stage Log

### Stage 01: Complete Repository Audit (2026-11-18)
- **Status:** Completed
- **Actions Taken:**
  - Inspected full directory structure of `/Users/swaruppal/Downloads/Ass_2`, read `Assignment 2b.pdf` and `Assignment 2a/Assignment_2a.pdf`, verified Apple M2 MPS capabilities, and documented technical risks and architecture in `PROJECT_AUDIT.md`, `docs/IMPLEMENTATION_LOG.md`, and `docs/TESTING_REPORT.md`.

---

### Stage 02: Environment Setup, TESS Dataset Acquisition & Split Verification (2026-11-18)
- **Status:** Completed
- **Actions Taken:**
  - Created isolated Python 3.11.15 `.venv` via `uv`, downloaded official TESS dataset (`ejlok1/toronto-emotional-speech-set-tess`), deduplicated 5,600 raw `.wav` files to 2,800 unique utterances in `data/tess_manifest.csv`, generated deterministic `word_indep_split` (`1960 / 420 / 420`) and `cross_speaker_split` (`1190 / 210 / 1400`) in `data/splits/word_splits.json`, and verified with 29 automated tests.

---

### Stage 03: Speech and Text Feature Extraction, Preprocessing and Caching (2026-11-18)
- **Status:** Completed
- **Actions Taken:**
  - Implemented `src/features/speech_preprocessing.py`, `src/features/extract_speech_features.py`, `src/features/extract_text_features.py`, and `src/data/feature_dataset.py`.
  - Extracted and verified all 2,800 `WavLM Base Plus` (`(T_s, 768)`), 2,800 `MFCC + Δ + ΔΔ` (`(T_m, 120)`), and 200 unique `DistilBERT` transcript (`(L_t, 768)`) feature files (~1.03 GB total).

---

### Stage 04A: Speech-Only Pipeline Implementation & Batch 1 Execution on Mac M2 (2026-11-18 to 2026-11-19)
- **Status:** Batch 1 Completed (`3 / 12` runs); Remaining 9 Runs Migrated to Colab Pro
- **Actions Taken:**
  - Implemented `WavLMBiGRUAttentionClassifier` (`wavlm_bigru_attn`) and `MFCCCNNBiLSTMAttentionClassifier` (`mfcc_cnn_bilstm`) in `models/speech_pipeline/model.py`, along with `models/speech_pipeline/train.py`, `models/speech_pipeline/test.py`, and `tests/test_speech_pipeline.py`.
  - Completed and evaluated Batch 1 (`wavlm_bigru_attn` on `word_indep_split`, seeds `42, 123, 999`) on Mac M2 (`mps`):
    - `seed=42`: `best_epoch=49/50`, `val_macro_f1=1.0000`, **Test Acc = `99.76%` (`419/420`)**, **Test Macro-F1 = `99.76%`**
    - `seed=123`: `best_epoch=50/50`, `val_macro_f1=1.0000`, **Test Acc = `99.76%` (`419/420`)**, **Test Macro-F1 = `99.76%`**
    - `seed=999`: `best_epoch=44/50`, `val_macro_f1=1.0000`, **Test Acc = `99.76%` (`419/420`)**, **Test Macro-F1 = `99.76%`**
    - **3-Seed Mean ± Std:** **`99.76 ± 0.00%`** Accuracy, **`99.77 ± 0.00%`** Macro-Precision, **`99.76 ± 0.00%`** Macro-Recall, **`99.76 ± 0.00%`** Macro-F1.
  - Created external Colab migration archives (`pkg1`–`pkg4`) under `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/`.

---

### Stage 04B: Minimal Training Pipeline Corrections for Colab Pro Migration (2026-11-19)
- **Status:** Completed (Implementation & Regression Testing Only; No New Experiments Started)
- **Timestamped Backups Created Prior to Edits:**
  - `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/backups_20261119_2346/models/speech_pipeline/train.py.20261119_2346.bak`
  - `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/backups_20261119_2346/models/speech_pipeline/test.py.20261119_2346.bak`
  - `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/backups_20261119_2346/tests/test_speech_pipeline.py.20261119_2346.bak`
  - `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/backups_20261119_2346/README.md.20261119_2346.bak`
  - `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/backups_20261119_2346/docs/IMPLEMENTATION_LOG.md.20261119_2346.bak`
  - `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/backups_20261119_2346/docs/TESTING_REPORT.md.20261119_2346.bak`
  - `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/backups_20261119_2346/results_speech_sha256_snapshot.json`
- **Corrections Implemented:**
  1. **Decoupled Early Stopping & Loss Tie-Breaker (`models/speech_pipeline/train.py`):**
     - Separated `f1_improved = val_f1 > (best_val_macro_f1 + 1e-6)` from `loss_tiebreak = (not f1_improved) and (abs(val_f1 - best_val_macro_f1) <= 1e-6) and (val_loss < best_val_loss - 1e-6)`.
     - `epochs_without_improvement` resets to `0` **only** when `f1_improved` is `True`, and increments on every non-F1-improving epoch (including loss tie-breaks), while `best_model.pt`, `best_val_loss`, and `best_epoch` still update when `loss_tiebreak` is `True`.
     - Added per-epoch console progress logging reporting `val_f1`, `best_val_macro_f1`, `f1_imp`, `tiebreak`, and `patience`.
  2. **Safe Epoch-Level Resume (`last_checkpoint.pt`) & Completed-Run Guard (`models/speech_pipeline/train.py`):**
     - Added `is_run_completed(run_dir)` check at the start of `train_single_run()`: if `best_model.pt` and `training_history.json` exist and `overwrite=False`, the completed run is skipped without modifying its files.
     - Atomically writes `last_checkpoint.pt` (`last_checkpoint.pt.tmp` $\rightarrow$ `last_checkpoint.pt`) at the end of each epoch preserving latest `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `last_epoch`, `best_val_macro_f1`, `best_val_loss`, `best_epoch`, `epochs_without_improvement`, `history`, and RNG states (`train_generator_state`, Python, NumPy, PyTorch CPU, and CUDA RNG states).
     - Validates configuration and hyperparameter compatibility (`_validate_resume_compatibility`) before resuming from `last_epoch + 1`, and removes `last_checkpoint.pt` only after `training_history.json`, `training_history.csv`, and `best_model.pt` are finalized.
  3. **Persistent Google Drive Output Support & Missing-Checkpoint Handling (`models/speech_pipeline/train.py` & `test.py`):**
     - Preserved `--output-root` (`Path(output_root).expanduser()`) across `train.py` and `test.py` for writing directly to `/content/drive/MyDrive/...`.
     - Added `evaluate_and_aggregate()` in `test.py` so missing checkpoints emit an explicit `WARNING: Missing checkpoint...` message and are skipped gracefully while all available checkpoints are evaluated and aggregated.
  4. **Tesla T4 CUDA Safeguards (`models/speech_pipeline/train.py` & `test.py`):**
     - Enabled `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` inside `set_reproducible_seed()`.
     - Added `release_device_memory(device)` calling `torch.cuda.synchronize()` and `torch.cuda.empty_cache()` when running on CUDA.
  5. **Regression Tests (`tests/test_speech_pipeline.py`):**
     - Added `test_early_stopping_does_not_reset_patience_on_loss_only_tiebreak`, `test_checkpoint_resume_mid_run_and_skip_completed_run`, and `test_evaluate_main_skips_missing_mac_checkpoints_gracefully`.
     - Executed `.venv/bin/pytest tests/test_splits.py tests/test_speech_pipeline.py -v` (`17 passed in 8.94s`).
     - Verified SHA-256 checksums of all 36 existing files in `results/speech/` remain 100% identical.

---

### Stage 07A: Text-Only Emotion Recognition Model Architecture & Unit Tests (2026-11-20)
- **Status:** Completed (Model Architecture & Unit Tests Only; No Training or Evaluation Scripts Created)
- **Files Created:**
  - `models/text_pipeline/__init__.py`
  - `models/text_pipeline/model.py`
  - `tests/test_text_pipeline.py`
- **Architecture Summary (`DistilBERTBiLSTMAttentionClassifier` / `distilbert_bilstm_attn`):**
  - **Input:** Cached DistilBERT token embeddings `(B, T, 768)` (`float32`), valid token `lengths` `(B,)`, and/or boolean `mask` `(B, T)`.
  - **Projection (`197,376` params):** `Linear(768 -> 256)` $\rightarrow$ `LayerNorm(256)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(dropout)`, zero-masked at padded positions.
  - **Contextual Modelling (`790,528` + `512` params):** 2-layer Bidirectional LSTM (`input_size=256`, `hidden_size=128` per direction, output `256`) using `pack_padded_sequence(..., lengths.cpu(), batch_first=True, enforce_sorted=False)` and `pad_packed_sequence(..., total_length=T)`, followed by `LayerNorm(256)` masked to zero at padded positions.
  - **Masked Additive Attention Pooling (`33,024` params):** `TextMaskedAdditiveAttentionPool(256 -> 128 -> 1)` with `masked_fill(~mask_bool, float("-inf"))` before `softmax`, explicit float-mask multiplication and renormalization so padded positions receive exact `0.0` weight and valid positions sum to `1.0`. Produces `(B, 256)` representation `h_t` and `(B, T)` attention weights.
  - **Classification Head (`33,799` params):** `Linear(256 -> 128)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(dropout)` $\rightarrow$ `Linear(128 -> 7)` aligned with canonical `EMOTION_TO_ID` (`0..6`).
  - **Total Trainable Parameters:** **`1,055,239`**.
- **Masking & Validation (`resolve_text_mask_and_lengths`):**
  - Validates 3D tensor rank, non-empty batch/sequence dimensions, `feat_dim == 768`, valid integer `lengths` in `[1, T]`, contiguous right-padded boolean `mask` with $\ge 1$ valid token per sample, and strict equality when both `lengths` and `mask` are supplied.
- **Test Results:**
  - `.venv/bin/python -m pytest tests/test_text_pipeline.py -v`: **10 passed in 4.27s** (0 warnings; covers initialization, canonical label mapping, output & attention shapes, attention sum-to-1 and zero-on-padding, variable lengths, padding invariance under noise & extra trailing tokens, backward pass & zero-gradient on padding, invalid input/length/mask errors, batch size 1 on CPU & MPS, and `state_dict` save/load).
  - `.venv/bin/python -m pytest tests/test_splits.py tests/test_speech_pipeline.py -v`: **17 passed in 8.32s**.

---

### Stage 07B: Text-Only Model Training & Evaluation Pipeline (2026-11-20)
- **Status:** Completed (Pipeline Implementation, Unit Tests & Synthetic Smoke Tests Only; No Full TESS Experiments Launched)
- **Files Created:**
  - `models/text_pipeline/train.py`
  - `models/text_pipeline/test.py`
  - `tests/test_text_training_pipeline.py`
- **Training Pipeline (`models/text_pipeline/train.py`):**
  - Integrates `DistilBERTBiLSTMAttentionClassifier` with `TESSFeatureDataset(..., modalities=("distilbert",))` and `tess_feature_collate_fn`.
  - Defaults matching speech pipeline: `AdamW(lr=1e-3, weight_decay=1e-2)`, `CosineAnnealingLR(eta_min=1e-5)`, `CrossEntropyLoss(label_smoothing=0.1)`, `batch_size=32`, `max_epochs=50`, `patience=12`, `dropout=0.3`.
  - Supports `--splits word_independent` / `cross_speaker` (mapping cleanly to `word_indep_split` / `cross_speaker_split`) and `--seeds 42 123 999`.
  - Outputs organized under `results/text/{architecture}/{word_independent|cross_speaker}/seed_{seed}/` (or `--output-root`), saving `best_model.pt`, `last_checkpoint.pt`, `train_config.json`, `training_history.csv`, and `training_history.json`.
  - Implements decoupled Macro-F1 early stopping (`f1_improved` resets patience; `loss_tiebreak` updates `best_model.pt` without resetting patience), atomic epoch-level `last_checkpoint.pt` resume with full optimizer/scheduler/RNG/history restoration and config validation, and completed-run skip protection (`--overwrite` required to rerun).
- **Evaluation Pipeline (`models/text_pipeline/test.py`):**
  - Supports single-checkpoint (`--checkpoint`) and multi-seed batch evaluation (`evaluate_and_aggregate_text`) under `torch.inference_mode()`.
  - Saves `test_metrics.json`, `per_class_metrics.csv`, `confusion_matrix.csv`, `test_predictions.csv`, `test_embeddings.pt` (`(N_test, 256)` + `attention_weights`), and `test_embeddings.npz`, plus multi-seed `summaries/text_all_runs.*` and `summaries/text_summary_mean_std.*`.
  - Enforces overwrite protection (`FileExistsError` in single-checkpoint mode unless `--overwrite`; skip with log in batch mode) and graceful warning/skip for missing checkpoints.
- **Verification & Synthetic Smoke-Test Results:**
  - `.venv/bin/python -m pytest tests/test_text_training_pipeline.py tests/test_text_pipeline.py tests/test_splits.py tests/test_speech_pipeline.py -v`: **33 passed, 2 warnings in 8.10s** (`6` Stage 07B training/eval/smoke tests + `10` Stage 07A model tests + `4` split tests + `13` speech tests).
  - Controlled 1-epoch synthetic training & evaluation smoke test ran strictly inside pytest's `tmp_path`; `results/text/` was not created in the project workspace and no real TESS text experiment was started.
