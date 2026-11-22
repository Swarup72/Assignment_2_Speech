# Testing & Verification Report — Assignment 2b: Multimodal Emotion Recognition

> **Current Status:** **STEP 04 PIPELINE RESILIENCE & COLAB MIGRATION REGRESSION TESTS COMPLETED (17 / 17 SPLIT & SPEECH PIPELINE TESTS PASSED; 47 / 47 TOTAL TESTS PASSED)**  
> All existing unit tests plus the new regression tests for Macro-F1 early stopping, epoch-level `last_checkpoint.pt` resume, completed-run skip protection, and missing-checkpoint evaluation handling pass. The 3 completed Mac runs (`word_indep_split / wavlm_bigru_attn / seed_{42,123,999}`) remain 100% untouched (SHA-256 verified).

---

## 1. Stage 01–03 Verification Summary (2026-11-18)

- **Stage 01 (Pre-Implementation Audit):** 5 / 5 checks passed.
- **Stage 02 (Environment, TESS Dataset & Splits):** 29 / 29 pytest tests passed (`test_environment.py`, `test_dataset.py`, `test_splits.py`).
- **Stage 03 (Preprocessing, Feature Extraction & 2,800-Sample Cache):** 5 / 5 pytest tests passed (`test_preprocessing_and_features.py`).

---

## 2. Stage 04: Speech-Only Pipeline & Colab Migration Resilience Tests (2026-11-19)

### 2.1. Test Command Executed
```bash
.venv/bin/pytest tests/test_splits.py tests/test_speech_pipeline.py -v
```

### 2.2. Actual Pytest Output
```text
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.2.0, pluggy-1.7.0 -- /Users/swaruppal/Downloads/Ass_2/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/swaruppal/Downloads/Ass_2
plugins: platformdirs-4.14.7, anyio-4.16.0
collecting ... collected 17 items

tests/test_splits.py::test_word_splits_metadata_file_exists_and_matches_manifest PASSED [  5%]
tests/test_splits.py::test_word_independent_split_zero_word_leakage_and_counts PASSED [ 11%]
tests/test_splits.py::test_word_independent_split_speaker_and_emotion_balance PASSED [ 17%]
tests/test_splits.py::test_cross_speaker_split_zero_speaker_leakage PASSED [ 23%]
tests/test_speech_pipeline.py::test_forward_output_shapes_and_attention_properties[wavlm_bigru_attn-768] PASSED [ 29%]
tests/test_speech_pipeline.py::test_forward_output_shapes_and_attention_properties[mfcc_cnn_bilstm-120] PASSED [ 35%]
tests/test_speech_pipeline.py::test_backward_pass_and_gradient_flow[wavlm_bigru_attn-768] PASSED [ 41%]
tests/test_speech_pipeline.py::test_backward_pass_and_gradient_flow[mfcc_cnn_bilstm-120] PASSED [ 47%]
tests/test_speech_pipeline.py::test_padding_mask_invariance[wavlm_bigru_attn-768] PASSED [ 52%]
tests/test_speech_pipeline.py::test_padding_mask_invariance[mfcc_cnn_bilstm-120] PASSED [ 58%]
tests/test_speech_pipeline.py::test_cpu_and_mps_execution_and_seed_reproducibility[wavlm_bigru_attn-768] PASSED [ 64%]
tests/test_speech_pipeline.py::test_cpu_and_mps_execution_and_seed_reproducibility[mfcc_cnn_bilstm-120] PASSED [ 70%]
tests/test_speech_pipeline.py::test_checkpoint_save_load_and_embedding_export[wavlm_bigru_attn] PASSED [ 76%]
tests/test_speech_pipeline.py::test_checkpoint_save_load_and_embedding_export[mfcc_cnn_bilstm] PASSED [ 82%]
tests/test_speech_pipeline.py::test_early_stopping_does_not_reset_patience_on_loss_only_tiebreak PASSED [ 88%]
tests/test_speech_pipeline.py::test_checkpoint_resume_mid_run_and_skip_completed_run PASSED [ 94%]
tests/test_speech_pipeline.py::test_evaluate_main_skips_missing_mac_checkpoints_gracefully PASSED [100%]

======================== 17 passed, 1 warning in 8.94s =========================
```

### 2.3. New Regression Test Coverage Summary

| Test Function | Verified Behavior | Result |
| :--- | :--- | :---: |
| `test_early_stopping_does_not_reset_patience_on_loss_only_tiebreak` | When `val_macro_f1` stays constant (`0.8500`) across epochs while `val_loss` decreases (`0.90 -> 0.85 -> 0.80 -> 0.75`), `loss_tiebreak` updates `best_model.pt` (`best_epoch=4`, `best_val_loss=0.75`), while `patience_counter` increments (`0 -> 1 -> 2 -> 3`) and stops training after `1 + patience = 4` epochs instead of running `max_epochs=10`. | **PASS** |
| `test_checkpoint_resume_mid_run_and_skip_completed_run` | Simulated disconnect during Epoch 3 leaves `last_checkpoint.pt` at `last_epoch=2`; incompatible config (`lr` mismatch) is rejected with `ValueError`; resuming with matching config runs only Epochs 3 and 4, writes contiguous history `[1, 2, 3, 4]`, removes `last_checkpoint.pt`, and skips training on subsequent calls without `overwrite=True`. | **PASS** |
| `test_evaluate_main_skips_missing_mac_checkpoints_gracefully` | When evaluating across multiple architectures where one is absent on disk, `evaluate_and_aggregate()` logs `WARNING: Missing checkpoint...`, skips the missing combination, evaluates present checkpoints, and writes `speech_all_runs.json` and `speech_summary_mean_std.json` without error. | **PASS** |
