# Assignment 2B — Stage 09 Comparative Analysis & Visualisation Audit Report

**Project Directory:** `/Users/swaruppal/Downloads/Ass_2`
**Stage:** Stage 09 — Unified Comparative Analysis & Visualisation
**Mode:** Strictly Read-Only Over All 27 Frozen Experiments (zero training executed, zero checkpoints modified, zero commits/pushes)

---

## 1. Deliverable Status Checklist

| Deliverable | Status | Output Path(s) |
| :--- | :---: | :--- |
| **Table 1 — Overall Modality Comparison** | **COMPLETE** | `results/fusion/summaries/overall_comparison.csv`, `overall_comparison.md` |
| **Table 2 — Speech Architecture Comparison** | **COMPLETE** | `results/fusion/summaries/speech_comparison.csv` |
| **Table 3 — Text-Only Results & Limitation** | **COMPLETE** | `results/fusion/summaries/text_comparison.csv`, `overall_comparison.md` |
| **Table 4 — Multimodal Fusion Comparison** | **COMPLETE** | `results/fusion/summaries/fusion_comparison.csv` |
| **Table 5 — Cross-Speaker Comparison** | **COMPLETE** | `results/fusion/summaries/cross_speaker_comparison.csv` |
| **Machine-Readable Unified JSON Summary** | **COMPLETE** | `results/fusion/summaries/stage09_unified_summary.json` |
| **Required 3-Panel t-SNE Visualisation** | **COMPLETE** | `results/plots/tsne_three_panel_comparison.png`, `tsne_config_and_sources.json` |
| **Confusion Matrices (Word-Indep & Cross-Speaker)** | **COMPLETE** | `results/plots/confusion_matrices_word_independent.png`, `confusion_matrices_cross_speaker.png` |
| **Training & Validation Curves** | **COMPLETE** | `results/plots/training_validation_curves.png` |
| **Per-Class F1 Comparison** | **COMPLETE** | `results/plots/per_class_f1_comparison.png` |
| **Gated Fusion Gate Activation Analysis** | **COMPLETE** | `results/plots/gated_fusion_gate_analysis.png` |
| **Late Fusion $\lambda$ Calibration Curve** | **COMPLETE** | `results/plots/late_fusion_lambda_calibration.png` |
| **Cross-Speaker Speech Performance Plot** | **COMPLETE** | `results/plots/cross_speaker_speech_comparison.png` |
| **Error Analysis (5 Cases CSV & Plot)** | **COMPLETE** | `results/fusion/summaries/error_analysis_five_cases.csv`, `results/plots/error_analysis_five_cases.png` |

---

## 2. Files Inspected

All **27 frozen experiments** were inspected directly from disk (no manual transcription from prior reports):
- **Speech-only (12 runs):** `results/speech/{word_indep_split,cross_speaker_split}/{wavlm_bigru_attn,mfcc_cnn_bilstm}/seed_{42,123,999}/` (`test_metrics.json`, `test_predictions.csv`, `confusion_matrix.csv`, `training_history.csv`, `test_embeddings.pt`).
- **Text-only (6 runs):** `results/text/distilbert_bilstm_attn/{word_independent,cross_speaker}/seed_{42,123,999}/` (`test_metrics.json`, `test_predictions.csv`, `confusion_matrix.csv`, `training_history.csv`, `test_embeddings.pt`).
- **Multimodal Fusion (9 word-independent runs):** `results/fusion/{gated_fusion,concat_fusion,late_fusion}/word_indep_split/seed_{42,123,999}/` (`metrics.json`, `test_predictions.csv`, `confusion_matrix.csv`, `training_history.csv`, `test_embeddings.npz`, `gate_activations.npz`, `gate_summary.json`, `lambda_search.csv`, `selected_lambda.json`).

---

## 3. Aggregation Methodology & Schema Notes

1. **Three-Seed Mean ± Standard Deviation:** Computed across seeds `42`, `123`, and `999` using sample standard deviation (`ddof=1`).
2. **Weighted-F1 & Log-Loss Computation for Speech/Text:** Speech and text `test_metrics.json` store `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `per_class_f1`, and `confusion_matrix`, while fusion `metrics.json` additionally stores `weighted_f1` and `log_loss`. For uniform cross-table comparison, `weighted_f1` and `log_loss` for the 18 unimodal runs were computed directly from their saved `test_predictions.csv` (`true_label_id`, `pred_label_id`, and the 7 `prob_<emotion>` columns, clipped to `[1e-12, 1.0]` and row-normalized).
3. **Trainable Parameter Counts:** Reported from actual model instantiations (`wavlm_bigru_attn`: `857,863`; `mfcc_cnn_bilstm`: `1,076,279`; `distilbert_bilstm_attn`: `1,055,239`; `gated_fusion`: `296,967`; `concat_fusion`: `165,895`; `late_fusion`: `0`).
4. **Missing / Optional Artifacts:** Cross-speaker multimodal fusion runs were not executed (optional per official assignment specification) and are explicitly omitted from Table 5 rather than fabricated.

---

## 4. Source Artifacts Used for Each Visualisation

| Visualisation | Dimensions (300 DPI) | Source Artifacts |
| :--- | :---: | :--- |
| `results/plots/tsne_three_panel_comparison.png` | `4684 × 1610 px` (`522.2 KB`) | Temporal: `results/speech/word_indep_split/wavlm_bigru_attn/seed_42/test_embeddings.pt` (`block='temporal_modelling_block'`); Contextual: `results/text/distilbert_bilstm_attn/word_independent/seed_42/test_embeddings.pt` (`block='contextual_modelling_block'`); Fusion: `results/fusion/gated_fusion/word_indep_split/seed_42/test_embeddings.npz` |
| `results/plots/confusion_matrices_word_independent.png` | `4534 × 2862 px` (`585.7 KB`) | Seed 42 `confusion_matrix.csv` across all 6 `word_indep_split` architectures |
| `results/plots/confusion_matrices_cross_speaker.png` | `4534 × 1482 px` (`328.6 KB`) | Seed 42 `confusion_matrix.csv` across WavLM, MFCC, and DistilBERT on `cross_speaker_split` |
| `results/plots/training_validation_curves.png` | `4684 × 1468 px` (`251.5 KB`) | Seed 42 `training_history.csv` for WavLM, MFCC, DistilBERT, Gated Fusion, and Concat Fusion (Late Fusion excluded as parameter-free) |
| `results/plots/per_class_f1_comparison.png` | `4084 × 1535 px` (`138.6 KB`) | 3-seed mean `per_class_f1` from all 6 `word_indep_split` architectures |
| `results/plots/gated_fusion_gate_analysis.png` | `4084 × 1468 px` (`209.8 KB`) | `gate_activations.npz` (`420 × 256` × 3 seeds) and `gate_summary.json` from `results/fusion/gated_fusion/word_indep_split/seed_{42,123,999}/` |
| `results/plots/late_fusion_lambda_calibration.png` | `4084 × 1468 px` (`289.7 KB`) | `lambda_search.csv` and `selected_lambda.json` from `results/fusion/late_fusion/word_indep_split/seed_{42,123,999}/` |
| `results/plots/cross_speaker_speech_comparison.png` | `4234 × 1530 px` (`214.7 KB`) | 3-seed aggregated metrics from `results/speech/cross_speaker_split/{wavlm_bigru_attn,mfcc_cnn_bilstm}/seed_{42,123,999}/` |
| `results/plots/error_analysis_five_cases.png` | `3209 × 3531 px` (`416.5 KB`) | Saved `test_predictions.csv` for `OAF_hall_pleasant_surprise` (Gated WI s42), `OAF_rush_disgust` (MFCC WI s123), `OAF_bite_angry` (Text WI s42), `YAF_back_happy` (WavLM CS s42), `YAF_back_fear` (MFCC CS s123) |

---

## 5. Validation Results

- **Table Validation:** Verified exact row counts (`overall_comparison.csv`: 9 rows; `speech_comparison.csv`: 4 rows; `text_comparison.csv`: 6 rows; `fusion_comparison.csv`: 3 rows; `cross_speaker_comparison.csv`: 3 rows; `error_analysis_five_cases.csv`: 5 rows) and **0 unexpected NaNs** across all core metric columns.
- **3-Panel t-SNE Validation:** Confirmed `sample_ids` and `labels` match **100% identically (`N = 420`)** across Temporal, Contextual, and Fusion embeddings; all embedding tensors verified `(420, 256)` with 0 NaN/Inf values; identical t-SNE settings (`perplexity=30`, `init="pca"`, `learning_rate="auto"`, `random_state=42`, `max_iter=1000`) applied to all three panels.
- **Plot Integrity:** All 9 PNG figures verified non-empty (`138.6 KB`–`585.7 KB`), 300 DPI, zero overlapping text.
- **Frozen Artifact Integrity:** All **27 frozen experiment anchors** (24 `best_model.pt` checkpoints + 3 `late_fusion` `selected_lambda.json` files) matched pre-execution SHA-256 hashes **byte-for-byte**.

---

## 6. Limitations

1. **Identical Carrier Phrase in TESS:** Because all 200 TESS words appear across all 7 emotions identically, text-only models achieve chance-level `14.29%` accuracy and fusion cannot improve upon the `99.76%` acoustic ceiling on `word_indep_split` (though Gated, Concat, and Late Fusion all successfully preserve `99.76%` accuracy without degrading under text noise).
2. **Gate Activation Interpretation:** Gated Fusion activation statistics (`mean = 0.6795 ± 0.0194` across seeds) are reported strictly as empirical layer statistics, not as proof of causal modality importance.
3. **Cross-Speaker Fusion:** Optional cross-speaker multimodal fusion runs are not included in this stage.

---

## 7. Exact Paths of All Newly Created Files (Stage 09)

- `scripts/run_stage09_analysis_and_plots.py`
- `results/fusion/summaries/overall_comparison.csv`
- `results/fusion/summaries/overall_comparison.md`
- `results/fusion/summaries/speech_comparison.csv`
- `results/fusion/summaries/text_comparison.csv`
- `results/fusion/summaries/fusion_comparison.csv`
- `results/fusion/summaries/cross_speaker_comparison.csv`
- `results/fusion/summaries/error_analysis_five_cases.csv`
- `results/fusion/summaries/stage09_unified_summary.json`
- `results/plots/tsne_three_panel_comparison.png`
- `results/plots/tsne_config_and_sources.json`
- `results/plots/confusion_matrices_word_independent.png`
- `results/plots/confusion_matrices_cross_speaker.png`
- `results/plots/training_validation_curves.png`
- `results/plots/per_class_f1_comparison.png`
- `results/plots/gated_fusion_gate_analysis.png`
- `results/plots/late_fusion_lambda_calibration.png`
- `results/plots/cross_speaker_speech_comparison.png`
- `results/plots/error_analysis_five_cases.png`
- `docs/STAGE_09_COMPARATIVE_ANALYSIS_REPORT.md`
