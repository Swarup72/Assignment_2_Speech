# Colab Results Integration Report — Frozen Speech & Text Experiments

**Integration date:** 2026-11-21
**Scope:** Results integration only. No training, evaluation, feature extraction, or code changes were performed. All results are treated as frozen, immutable artifacts.

## 1. Archive information

| Item | Value |
|---|---|
| Archive | `/Users/swaruppal/Downloads/Ass_2_frozen_speech_text_results.zip` (original unmodified) |
| Size | 148,000,406 bytes |
| SHA-256 | `f8a439f5f8112b59b39fba39d32e03c896f4f5029b8d4c057d146d513cbc2961` (identical before and after integration) |
| `unzip -t` | No errors detected |

## 2. Paths

- **Project (discovered, unique):** `/Users/swaruppal/Downloads/Ass_2`
- **Staging (outside project):** `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/frozen_integration_staging/` (`extracted/`, `pre_integration_results_sha256.json`, `plan.json`)
- **Destination:** `/Users/swaruppal/Downloads/Ass_2/results/` (original directory structure preserved)

## 3. File accounting

| Metric | Count |
|---|---:|
| Files inspected in archive | 163 |
| Local result files before integration | 124 |
| Identical to existing local file (SHA-256) — left untouched | 90 |
| **New files copied** | **73** |
| **Conflicts (different content at same path)** | **0** |
| Unexpected files (`.DS_Store`, `__MACOSX`, paths outside `results/speech|text`) | 0 |
| Missing expected files | 0 |

Copies were made with `shutil.copy2` (never move); each destination was confirmed absent beforehand, and each copy was SHA-256-verified against its source.

**Archive composition note:** the archive contains 15 run directories — 9 speech runs (MFCC word-independent ×3, WavLM cross-speaker ×3, MFCC cross-speaker ×3) and 6 text runs. It does **not** contain the 3 Mac WavLM word-independent runs; those already exist locally and were left untouched. The 90 identical files are the 9 speech runs that were already integrated earlier. The 73 new files are the 66 text-run files (6 runs × 11 files) plus 7 summary files (3 speech `colab_*` summaries, 4 text summaries).

## 4. Speech verification (12 / 12 runs present)

Matrix: WavLM + BiGRU + Attention and MFCC + CNN-BiLSTM, each on word-independent and cross-speaker splits, seeds 42/123/999.

- All 10 artifacts present and non-empty in every run (`best_model.pt`, `train_config.json`, `training_history.{json,csv}`, `test_metrics.json`, `per_class_metrics.csv`, `confusion_matrix.csv`, `test_predictions.csv`, `test_embeddings.{pt,npz}`).
- All JSON/CSV parse; checkpoint architecture and seed agree with config, metrics file, and folder.
- Test embeddings are `(N, 256)` (N = 420 word-independent, 1400 cross-speaker); `.pt` and `.npz` agree.
- Accuracy and macro precision/recall/F1 recomputed from `test_predictions.csv` match `test_metrics.json` (tolerance 2e-6).
- The 3 Mac runs are byte-identical to their pre-integration hashes.

## 5. Text verification (6 / 6 runs present)

DistilBERT + BiLSTM + Attention, word-independent and cross-speaker, seeds 42/123/999.

- All 11 artifacts present and non-empty in every run (the 10 above plus `last_checkpoint.pt`).
- All JSON/CSV parse; checkpoint identity agrees with config, metrics, and folder.
- Test embeddings are `(N, 256)`; `.pt` and `.npz` agree; attention weights present in both `test_embeddings.pt` and `.npz` with N rows.
- Metrics recomputed from `test_predictions.csv` match `test_metrics.json`.
- Files were read only; no text metric or prediction file was modified.

### Text results as stored (test partition; reported, not interpreted beyond what was checked)

| Split | Seed | Accuracy | Macro-F1 | Best epoch | Epochs run |
|---|---:|---:|---:|---:|---:|
| word-independent | 42 / 123 / 999 | 14.29% (all) | 3.57% (all) | 13 / 13 / 13 | 13 each |
| cross-speaker | 42 / 123 / 999 | 14.29% (all) | 3.57% (all) | 13 / 13 / 10 | 13 each |

Mean ± std over seeds: 14.29 ± 0.00% accuracy and 3.57 ± 0.00% macro-F1 for both splits (matches the stored text summaries).

Checks run on these stored results:

- **Every run predicts exactly one class for the whole test set** (word-independent: `pleasant_surprise`, `sad`, `disgust`; cross-speaker: `disgust`, `angry`, `disgust`). 14.29% accuracy and 3.57% macro-F1 are exactly what a constant single-class predictor gives on a balanced 7-class test set.
- **The input features for a given word are one identical file** shared by all 14 recordings of that word (checked for the word "back": 1 distinct feature file across 14 samples), so the text input carries no information that distinguishes the emotion labels within a word.
- Validation macro-F1 is also 3.57% in every run, and the best epoch is mostly 13 with 13 epochs run per seed; training accuracy at the last epoch is about 12.5–14.4%.

These results are chance-level. No claim is made beyond the checks above, and no results were altered.

## 6. Summary files

- Speech: `results/speech/summaries/speech_all_runs.{csv,json}` and `speech_summary_mean_std.{csv,json}` (12 rows / 4 rows; unchanged, identical to the earlier consolidated versions), plus new `colab_speech_individual_results.csv`, `colab_speech_mean_std.{csv,json}` from the archive (3 combinations covering the Colab-trained runs only).
- Text: `results/text/summaries/text_all_runs.{csv,json}` (6 rows) and `text_summary_mean_std.{csv,json}` (2 rows).

## 7. Final integrity verification

- **Validation checks: 148 passed, 0 failed.**
- All 124 pre-existing result files: **0 changed, 0 missing** (SHA-256 vs pre-integration snapshot).
- All 163 incoming files match their local copies: **0 mismatches**.
- No files outside `results/` were modified (models, tests, data caches, manifests, splits, `Assignment 2a/` untouched).
- Archive SHA-256 unchanged.

## 8. Confirmation

No model training, evaluation, inference, or feature extraction was launched during this task. Only file copying, hashing, and read-only parsing of results were performed.

## 9. Items for your attention

1. The archive omits the 3 Mac WavLM word-independent runs; they remain intact locally, so all 12 speech runs are present.
2. The text-only results are at chance level with single-class predictions (Section 5). Whether and how to address this is a project decision; nothing was changed.
3. The Mac vs Colab protocol difference noted in `docs/SPEECH_RESULTS_INTEGRATION_REPORT.md` still applies.
