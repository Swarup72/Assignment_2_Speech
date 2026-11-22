# Speech Results Integration Report — Assignment 2b

**Integration date:** 2026-11-20
**Scope:** Organizing and verifying the 12 completed speech-model experiments only. No training, evaluation, feature extraction, or code changes were performed. No text-only or fusion work was started.

## 1. Sources

| Source | Location | Runs |
|---|---|---|
| Mac M2 (MPS) — pre-existing | `results/speech/word_indep_split/wavlm_bigru_attn/seed_{42,123,999}/` | 3 |
| Google Colab (T4 CUDA) | `/Users/swaruppal/Downloads/Ass_2_speech_results_colab.zip` (original preserved, unmodified) | 9 |

- Colab ZIP size: 52,205,069 bytes; SHA-256 `63feeb6c26d6ddbd5e16adfb94d72abca170b1e9392962444134b12c9f702004`; `unzip -t`: no errors.
- Extracted to a staging directory first: `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/integration_staging/colab/`.
- Copied (not moved) 90 files (9 runs × 10 artifacts) into `results/speech/`. Every destination was verified absent before copying; **0 conflicts, 0 overwrites**. Copied files are byte-identical to the staged originals (0 differences).

## 2. Final experiment matrix (12 / 12 present)

| Architecture | Split | Seeds | Origin |
|---|---|---|---|
| WavLM + BiGRU + Attention | word_indep_split | 42, 123, 999 | Mac (preserved) |
| MFCC + CNN-BiLSTM | word_indep_split | 42, 123, 999 | Colab |
| WavLM + BiGRU + Attention | cross_speaker_split | 42, 123, 999 | Colab |
| MFCC + CNN-BiLSTM | cross_speaker_split | 42, 123, 999 | Colab |

Run identity was established from `train_config.json`, the checkpoint payload, and `test_metrics.json` (not just folder names); all three agree with the folder for all 12 runs. The existing directory convention `results/speech/{split}/{architecture}/seed_{N}/` was kept; nothing was renamed.

## 3. Artifact verification (all 12 runs)

All 10 required artifacts exist and are non-empty in every run: `best_model.pt`, `train_config.json`, `training_history.json`, `training_history.csv`, `test_metrics.json`, `per_class_metrics.csv`, `confusion_matrix.csv`, `test_predictions.csv`, `test_embeddings.pt`, `test_embeddings.npz`.

- All JSON files parse; all CSV files are readable.
- All 12 checkpoints load on CPU for inspection (no inference or training was run).
- `test_embeddings.pt` and `.npz` agree and have shape `(N_test, 256)`: N = 420 (word-independent) and 1400 (cross-speaker).
- As an independent check, accuracy and macro precision/recall/F1 were recomputed from each run's `test_predictions.csv`; all match `test_metrics.json` (tolerance 2e-6).
- No `last_checkpoint.pt`, `.tmp`, or other stray files in `results/speech/`.
- **Missing artifacts: none.**

## 4. Consolidated results

Test metrics are on held-out test partitions. The `val_best_macro_f1` column is a validation metric and must not be read as test performance. Mean ± std use sample std (ddof=1) across 3 seeds, computed from unrounded values.

| Architecture | Split | Test Acc (%) | Test Macro-P (%) | Test Macro-R (%) | Test Macro-F1 (%) | Val best Macro-F1 (%) |
|---|---|---:|---:|---:|---:|---:|
| WavLM | word-independent | 99.76 ± 0.00 | 99.77 ± 0.00 | 99.76 ± 0.00 | 99.76 ± 0.00 | 100.00 |
| MFCC | word-independent | 99.68 ± 0.14 | 99.69 ± 0.14 | 99.68 ± 0.14 | 99.68 ± 0.14 | 100.00 |
| WavLM | cross-speaker | 40.12 ± 10.07 | 34.51 ± 8.96 | 40.12 ± 10.07 | 33.59 ± 10.95 | 100.00 |
| MFCC | cross-speaker | 43.67 ± 7.78 | 53.44 ± 6.42 | 43.67 ± 7.78 | 42.62 ± 7.26 | 100.00 |

**Comparison with the reference values you supplied:** all eight (accuracy and Macro-F1 for four combinations) match the recalculated values to two decimals. **No discrepancies.** The 3-combination summary inside the Colab ZIP also agrees with the recalculated Colab rows.

Individual cross-speaker test results vary widely by seed:

| Architecture | seed 42 (Acc / F1) | seed 123 (Acc / F1) | seed 999 (Acc / F1) |
|---|---|---|---|
| WavLM | 31.00 / 23.43 | 50.93 / 45.19 | 38.43 / 32.14 |
| MFCC | 41.79 / 41.17 | 52.21 / 50.49 | 37.00 / 36.18 |

### Scientific reporting notes

- **The cross-speaker (OAF → YAF) results are poor and should not be described as good generalization.** Both architectures are far below their word-independent scores, with large seed-to-seed variance (std 7–11 points). With 7 balanced classes, chance accuracy is about 14.3%.
- Mean per-class test F1 on cross-speaker shows the `happy` class at ≈0.005 (WavLM) and ≈0.003 (MFCC), i.e. essentially never correct. This is reported as an observation only; no cause has been established.
- **No explanation for the word-independent vs. cross-speaker gap is offered here**, because no experiment in this task isolates one.
- The very high word-independent scores include both speakers (OAF and YAF) in train and test, so that protocol does not measure speaker generalization.
- The cross-speaker validation split is OAF-only by design (same speaker as training), so its 100% validation Macro-F1 does not indicate cross-speaker performance. Checkpoint selection and early stopping in those runs used this same-speaker validation signal.

## 5. Differences between Mac and Colab runs (comparability caveats)

These are recorded as facts; they were not changed.

| Item | Mac runs (WavLM, word-indep) | Colab runs (other 9) |
|---|---|---|
| Device | MPS | CUDA (T4) |
| `patience` in `train_config.json` | 10 | 12 |
| Early-stopping logic | Original (val-loss tie-break could reset patience) | Corrected Stage 04B logic (patience resets on Macro-F1 improvement only) |
| Epochs trained | 50 / 50 for all 3 seeds | 18–27 (stopped early) |
| `overwrite` key in config | absent | present |

Consequences: (a) training times (Mac 1,994–2,776 s vs. Colab ~50–123 s) are not comparable across devices or stopping rules; (b) the three WavLM word-independent runs used a different stopping rule and patience than the other nine. Model architecture and the other hyperparameters recorded in the configs are the same. If a strictly uniform protocol is required for the final report, this difference should be disclosed (or those three runs re-run, which was explicitly not done here).

## 6. Summary files

Located in `results/speech/summaries/` (regenerated to cover all 12 runs by `integration_staging/consolidate_speech_summaries.py`, which only reads existing artifacts):

- `speech_all_runs.csv` / `speech_all_runs.json` — 12 rows: architecture, split, seed, best epoch, epochs trained, validation best Macro-F1/loss, test accuracy/macro-P/macro-R/macro-F1, per-class test F1, training time, training hardware, run directory.
- `speech_summary_mean_std.csv` / `speech_summary_mean_std.json` — 4 rows (one per architecture/split), mean and std across 3 seeds.

The previous 3-run versions of these four files were backed up (not deleted) to `/Users/swaruppal/Downloads/Ass_2_Colab_Migration/integration_staging/old_summaries_3runs/`. No individual per-run metric file was altered.

## 7. SHA-256 integrity of the original Mac runs

A SHA-256 snapshot of all `results/speech/` files was taken before integration (`integration_staging/pre_integration_sha256.json`, 34 files). After integration, all **30 Mac run files** (3 runs × 10 artifacts) have identical hashes: **0 changed**. (The 4 summary files were intentionally regenerated and are excluded from this comparison.)

## 8. Issues needing your attention

1. **Protocol difference** between the 3 Mac runs and the 9 Colab runs (patience 10 vs 12, old vs corrected early stopping) — see §5. Decide whether disclosure suffices.
2. **Cross-speaker performance is low and unstable** (see §4); the report should present it as such.
3. Cross-speaker model selection used a same-speaker validation set (see §4 note).

No missing files, no conflicts, no failed checks.
