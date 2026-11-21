# Assignment 2B — Final Completion Audit & Remaining Tasks

**Project Directory:** `/Users/swaruppal/Downloads/Ass_2`
**GitHub Repository:** `https://github.com/Swarup72/Assignment_2_Speech`
**Audit Date:** 2026-11-21
**Audit Mode:** Strictly Read-Only Inspection & Planning (zero experiments modified, zero training executed, zero commits/pushes)

---

## 1. Official Assignment Requirements (`Assignment 2b.pdf`)

Extracted verbatim from [Assignment 2b.pdf](file:///Users/swaruppal/Downloads/Ass_2/Assignment%202b.pdf) (3 pages):

### 1.1. Explicitly Mandatory Requirements
1. **Objective & Functional Blocks (3 Pipelines, 5 Blocks):**
   - **Speech-only pipeline:** Preprocessing $\rightarrow$ Feature Extraction $\rightarrow$ Temporal Modelling $\rightarrow$ Classifier $\rightarrow$ Emotion Label.
   - **Text-only pipeline:** Preprocessing $\rightarrow$ Feature Extraction $\rightarrow$ Contextual Modelling $\rightarrow$ Classifier $\rightarrow$ Emotion Label.
   - **Multimodal (Speech & Text) pipeline:** Parallel Speech & Text branches $\rightarrow$ Fusion $\rightarrow$ Classifier $\rightarrow$ Emotion Label.
   - **Dataset:** Toronto Emotional Speech Set (TESS, Kaggle).
2. **Code Deliverables:**
   - `models/speech_pipeline/train.py` and `test.py`
   - `models/text_pipeline/train.py` and `test.py`
   - `models/fusion_pipeline/train.py` and `test.py`
   - `Results/` (implemented as `results/`) containing **accuracy tables for all 3 model variants** and **`plots/`**
   - `README.md` (with setup & execution instructions)
   - `requirements.txt`
3. **Technical Report Deliverable:**
   - **Section A — Architecture Decisions:** For each of the 5 blocks: *What architecture? And why?*
   - **Section B — Experiments:** Quantitative comparison across **Speech-only**, **Text-only**, and **Multimodal** variants.
   - **Section C — Analysis:**
     - Which emotions are easiest/hardest to classify? Why?
     - When does fusion help most?
     - **Error analysis:** 3–5 concrete failure cases.
     - **Cluster separability visualization:** Visualize the separability of emotion clusters using the learned representations from:
       1. Temporal Modelling block (Speech-only)
       2. Contextual Modelling block (Text-only)
       3. The Fusion block (Multimodal)
4. **Public GitHub Repository:**
   - Push all scripts, setup instructions, report, and results to a public repository; ensure all links (GitHub and Drive, if any) are publicly accessible.

### 1.2. Recommended Analysis & Visualisations
- Multi-seed mean $\pm$ standard deviation aggregation (3 seeds: `42`, `123`, `999`) for every model variant.
- Confusion matrix heatmaps, training/validation loss and Macro-F1 curves, per-class F1 bar plots, fusion gate activation distributions, and late-fusion $\lambda$-calibration curves.
- Speech baseline ablation (`WavLM + BiGRU + Attention` vs. `MFCC + CNN-BiLSTM`) and Fusion ablation (`Gated` vs. `Concatenation` vs. `Late-Logit`).

### 1.3. Optional Additional Experiments
- **Cross-Speaker Split (`OAF` $\rightarrow$ `YAF`) for Multimodal Fusion:** Already completed for Speech-only (6 runs) and Text-only (3 runs); optional for the official PDF but strongly recommended for full experimental parity across all 3 modalities (takes ~2 minutes locally on M2 MPS).

---

## 2. Deliverable Status Summary

| Deliverable | Status | Notes |
| :--- | :---: | :--- |
| **Speech-Only Pipeline & 12 Experiments** | **COMPLETE** | `models/speech_pipeline/`, 12/12 runs (`wavlm_bigru_attn`, `mfcc_cnn_bilstm` $\times$ 2 splits $\times$ 3 seeds) |
| **Text-Only Pipeline & 6 Experiments** | **COMPLETE** | `models/text_pipeline/`, 6/6 runs (`distilbert_bilstm_attn` $\times$ 2 splits $\times$ 3 seeds) |
| **Multimodal Fusion Pipeline Code & Tests** | **COMPLETE** | `models/fusion_pipeline/{model,train,test}.py`, 28 passing unit tests |
| **Word-Independent Fusion Experiments** | **PARTIALLY COMPLETE** | 7/9 runs complete (`gated_fusion` 3/3, `concat_fusion` 3/3, `late_fusion` 1/3 — missing seeds `123`, `999`) |
| **Cross-Speaker Fusion Experiments** | **OPTIONAL** | 0/9 runs executed; all frozen speech/text cross-speaker checkpoints exist and are ready |
| **Unified 3-Variant Accuracy & Summary Tables** | **PARTIALLY COMPLETE** | Individual summaries exist; `results/fusion/summaries/` needs re-aggregation across all architectures; unified cross-modality table missing |
| **Mandatory & Recommended Plots (`results/plots/`)** | **MISSING** | **0 plot image files** currently exist anywhere in `results/` |
| **Error Analysis (3–5 Failure Cases)** | **PARTIALLY COMPLETE** | Saved predictions contain full data for failure cases; written analysis not yet compiled |
| **Final Technical Report (Sections A, B, C)** | **MISSING** | No consolidated Assignment 2B technical report exists yet |
| **Updated `README.md` & Submission Docs** | **PARTIALLY COMPLETE** | `README.md` still reflects Step 04 status; missing Stage 07C/08 results, fusion CLI instructions, and report/plot links |
| **Final GitHub Sync** | **PARTIALLY COMPLETE** | Base commit `1b2f244` pushed; Stage 08 code, tests, fusion results, plots, report, and updated `README.md` not yet committed/pushed |

---

## 3. Audit of Completed Experiments & Artifact Inventory

### 3.1. Speech-Only Experiments (`results/speech/` — 12/12 COMPLETE)
- **Paths:** `results/speech/{word_indep_split,cross_speaker_split}/{wavlm_bigru_attn,mfcc_cnn_bilstm}/seed_{42,123,999}/`
- **Artifacts per run (10 files each, 100% present):** `best_model.pt`, `test_metrics.json`, `test_predictions.csv`, `per_class_metrics.csv`, `confusion_matrix.csv`, `test_embeddings.npz`, `test_embeddings.pt`, `training_history.csv`, `training_history.json`, `train_config.json`.
- **Summaries (`results/speech/summaries/`):** `speech_all_runs.{csv,json}`, `speech_summary_mean_std.{csv,json}`.
- **Note on Attention Weights:** Speech `test_embeddings.npz/.pt` do not store frame-level attention weights (only `(N, 256)` embeddings); text `test_embeddings.npz/.pt` store `(N, 7)` token attention weights.
- **Results Overview:**
  - `wavlm_bigru_attn` (`word_indep_split`): `99.76%` Acc / `99.76%` Macro-F1 across all 3 seeds (`42, 123, 999`).
  - `mfcc_cnn_bilstm` (`word_indep_split`): `99.76%` (`42`), `99.52%` (`123`), `99.76%` (`999`).
  - `wavlm_bigru_attn` (`cross_speaker_split`): `31.00%` (`42`), `50.93%` (`123`), `38.43%` (`999`) Acc; `23.43%` / `45.19%` / `32.14%` Macro-F1.
  - `mfcc_cnn_bilstm` (`cross_speaker_split`): `41.79%` (`42`), `52.21%` (`123`), `37.00%` (`999`) Acc; `41.17%` / `50.49%` / `36.18%` Macro-F1.

### 3.2. Text-Only Experiments (`results/text/` — 6/6 COMPLETE)
- **Paths:** `results/text/distilbert_bilstm_attn/{word_independent,cross_speaker}/seed_{42,123,999}/`
- **Artifacts per run (11 files each, 100% present):** `best_model.pt`, `last_checkpoint.pt`, `test_metrics.json`, `test_predictions.csv`, `per_class_metrics.csv`, `confusion_matrix.csv`, `test_embeddings.npz`, `test_embeddings.pt` (includes `attention_weights`), `training_history.csv`, `training_history.json`, `train_config.json`.
- **Summaries (`results/text/summaries/`):** `text_all_runs.{csv,json}`, `text_summary_mean_std.{csv,json}`.
- **Results Overview:** `14.29 ± 0.00%` Acc and `3.57 ± 0.00%` Macro-F1 across all 6 runs (exact $1/7$ theoretical chance bound on TESS because every target word occurs equally across all 7 emotions in `"Say the word <word>"`).

### 3.3. Multimodal Fusion Experiments (`results/fusion/` — 7 COMPLETE, 2 WORD-INDEP MISSING, 9 CROSS-SPEAKER OPTIONAL)
- **`gated_fusion` | `word_indep_split` | seeds `42, 123, 999` (COMPLETE — 3/3):**
  - 13 artifacts per run (`best_model.pt`, `last_checkpoint.pt`, `metrics.json`, `test_predictions.csv`, `per_class_metrics.csv`, `confusion_matrix.csv`, `classification_report.json`, `test_embeddings.npz`, `gate_activations.npz`, `gate_summary.json`, `training_history.csv`, `run_summary.json`, `config.json`).
  - Results: `99.76 ± 0.00%` Acc / `99.76 ± 0.00%` Macro-F1; mean gate $\bar{g} \in [0.658, 0.695]$.
- **`concat_fusion` | `word_indep_split` | seeds `42, 123, 999` (COMPLETE — 3/3):**
  - 11 artifacts per run (all above except gate files).
  - Results: `99.76 ± 0.00%` Acc / `99.76 ± 0.00%` Macro-F1.
- **`late_fusion` | `word_indep_split` | seed `42` (PARTIALLY COMPLETE — 1/3):**
  - 9 artifacts (`selected_lambda.json`, `lambda_search.csv`, `metrics.json`, `test_predictions.csv`, `per_class_metrics.csv`, `confusion_matrix.csv`, `classification_report.json`, `run_summary.json`, `config.json`).
  - Result: $\lambda = 1.00$, `99.76%` Acc / `99.76%` Macro-F1.
  - **Missing:** `late_fusion` | `word_indep_split` | seeds `123` and `999`.
- **Summary Discrepancy (`NEEDS VERIFICATION` / Re-aggregation):**
  - [results/fusion/summaries/fusion_summary_mean_std.json](file:///Users/swaruppal/Downloads/Ass_2/results/fusion/summaries/fusion_summary_mean_std.json) currently contains only `late_fusion` (1 seed) because `test.py` overwrote the summary when invoked for a single architecture. Re-running `aggregate_fusion_results()` across all completed architectures will regenerate the complete fusion summary without retraining.

---

## 4. Comparative Analysis Readiness

| Comparison | Status | Gaps / Notes |
| :--- | :---: | :--- |
| Speech-only vs. Text-only | **COMPLETE** | All 3 seeds available on both splits |
| WavLM vs. MFCC | **COMPLETE** | All 3 seeds available on both splits |
| Speech-only vs. Multimodal Fusion (`word_indep_split`) | **PARTIALLY COMPLETE** | Complete for Gated (3/3) & Concat (3/3); needs Late Fusion seeds `123` & `999` for 3-seed mean $\pm$ std |
| Gated vs. Concat vs. Late Fusion (`word_indep_split`) | **PARTIALLY COMPLETE** | Needs Late Fusion seeds `123` & `999` + summary re-aggregation |
| Word-Independent vs. Cross-Speaker (Unimodal) | **COMPLETE** | Complete for WavLM, MFCC, and DistilBERT |
| Word-Independent vs. Cross-Speaker (Multimodal) | **OPTIONAL / MISSING** | Cross-speaker fusion runs (9 runs) not yet executed; recommended to test whether fusion/gating behaves differently under speaker shift |
| Per-Emotion Performance Comparison | **COMPLETE** | `per_class_metrics.csv` present for all 25 completed runs |

---

## 5. Visualisation Audit

**Existing visualisations:** **0** (no `.png`, `.svg`, or `.pdf` plot files exist under `results/` or `results/plots/`).
**Available plotting environment:** `matplotlib`, `seaborn`, and `scikit-learn` (for `TSNE`) are installed in `.venv`; `umap-learn` is not installed (use `sklearn.manifold.TSNE` without adding dependencies).

| Visualisation | Status | Mandatory / Recommended | Source Artifacts (No Retraining) | Proposed Output Path |
| :--- | :---: | :---: | :--- | :--- |
| **3-Panel Emotion Cluster Separability (t-SNE): Temporal vs. Contextual vs. Fusion** | **MISSING** | **MANDATORY** (PDF Section 2.C) | `results/speech/word_indep_split/wavlm_bigru_attn/seed_42/test_embeddings.npz`, `results/text/distilbert_bilstm_attn/word_independent/seed_42/test_embeddings.npz`, `results/fusion/gated_fusion/word_indep_split/seed_42/test_embeddings.npz` | `results/plots/tsne_cluster_separability_3panel.png` |
| **Confusion Matrices (Speech, Text, Gated, Concat, Late)** | **MISSING** | Recommended | `confusion_matrix.csv` from seed 42 (or 3-seed summed) in `results/{speech,text,fusion}/` | `results/plots/confusion_matrices_comparison.png` |
| **Training & Validation Loss / Macro-F1 Curves** | **MISSING** | Recommended | `training_history.csv` from Speech, Text, Gated, and Concat runs | `results/plots/training_curves_loss_f1.png` |
| **Per-Class F1 Comparison Across Pipelines** | **MISSING** | Recommended | `per_class_metrics.csv` across WavLM, MFCC, DistilBERT, Gated, Concat, Late | `results/plots/per_class_f1_comparison.png` |
| **Gated Fusion Activation Distributions & Per-Emotion Mean Gate** | **MISSING** | Recommended | `results/fusion/gated_fusion/word_indep_split/seed_{42,123,999}/gate_activations.npz` & `gate_summary.json` | `results/plots/gated_fusion_gate_activations.png` |
| **Late Fusion $\lambda$ Calibration Curve (Val Macro-F1 & Log-Loss)** | **MISSING** | Recommended | `results/fusion/late_fusion/word_indep_split/seed_*/lambda_search.csv` | `results/plots/late_fusion_lambda_calibration.png` |
| **Text Token Attention Weights Visualization** | **MISSING** | Optional | `results/text/distilbert_bilstm_attn/word_independent/seed_42/test_embeddings.npz` (`attention_weights`) | `results/plots/text_attention_weights.png` |
| **Word-Independent vs. Cross-Speaker Performance Comparison** | **MISSING** | Recommended | `results/{speech,text,fusion}/summaries/*_summary_mean_std.csv` | `results/plots/word_indep_vs_cross_speaker.png` |
| **Error Analysis Confidence & Probability Breakdown** | **MISSING** | Recommended | `test_predictions.csv` for selected 3–5 failure cases | `results/plots/error_analysis_failure_cases.png` |

---

## 6. Error Analysis Readiness (3–5 Concrete Failure Cases from Saved Predictions)

Sufficient data **already exists** in saved `test_predictions.csv` files to perform a rigorous 3–5 failure case analysis without retraining:

1. **`OAF_hall_pleasant_surprise` (Word-Independent Split — Consistent Failure Across All Acoustic & Fusion Models):**
   - **True:** `pleasant_surprise` $\rightarrow$ **Predicted:** `disgust`
   - **Probabilities:** WavLM (`disgust=0.6586`, `pleasant_surprise=0.2112`), MFCC (`disgust=0.4917`, `pleasant_surprise=0.3403`), Gated Fusion (`disgust=0.8320`, `pleasant_surprise=0.0779`), Concat Fusion (`disgust=0.8501`, `pleasant_surprise=0.0636`), Late Fusion (`disgust=0.6586`).
   - **Data-supported observation:** Both independent acoustic backbones (WavLM and MFCC) and all three fusion models misclassify this exact utterance as `disgust` with `pleasant_surprise` as the runner-up class, indicating acoustic ambiguity in this specific recording rather than a single-model quirk.
2. **`OAF_rush_disgust` (Word-Independent Split — MFCC Seed 123 Failure):**
   - **True:** `disgust` $\rightarrow$ **Predicted:** `pleasant_surprise` (`conf=0.4706`, runner-up `disgust=0.3748`).
   - **Data-supported observation:** Low-margin confusion between `disgust` and `pleasant_surprise` (mirroring the `OAF_hall_pleasant_surprise` confusion pair), whereas WavLM classifies it correctly.
3. **`OAF_bite_angry` (Word-Independent Split — Text-Only Degeneracy Case):**
   - **True:** `angry` $\rightarrow$ **Predicted:** `pleasant_surprise` (`conf=0.1450`, `disgust=0.1439`, `sad=0.1437` — near-uniform $1/7 \approx 0.1429$).
   - **Data-supported observation:** Demonstrates the exact information-theoretic ceiling of text-only classification on TESS (`"Say the word bite"` appears identically across all 7 emotions).
4. **`YAF_back_happy` (Cross-Speaker Split — High-Confidence Speaker-Shift Failure):**
   - **True:** `happy` $\rightarrow$ **Predicted:** `angry` (WavLM s42 `conf=0.9113`, `happy=0.0191`; MFCC s123 `conf=0.9139`, `disgust=0.0153`).
   - **Data-supported observation:** Both WavLM and MFCC trained only on older female speaker `OAF` confidently misclassify younger female speaker `YAF`'s high-arousal `happy` utterance as high-arousal `angry`, illustrating pitch/timbre domain shift across speakers.
5. **`YAF_back_angry` (Cross-Speaker Split — WavLM Seed 42 Failure):**
   - **True:** `angry` $\rightarrow$ **Predicted:** `neutral` (`conf=0.8859`, `happy=0.0230`, `fear=0.0217`).
   - **Data-supported observation:** Illustrates severe cross-speaker calibration degradation when generalizing from a single training speaker (`OAF`) to an unseen speaker (`YAF`).

---

## 7. Documentation & Submission Package Audit

- **[README.md](file:///Users/swaruppal/Downloads/Ass_2/README.md) (`PARTIALLY COMPLETE`):**
  - *Present:* Project overview, environment setup, TESS irregularities, canonical label mapping, preprocessing & feature cache summary, speech & text CLI commands.
  - *Missing / Outdated:* Banner still states *"STEP 04 In Progress"*; repository tree omits `models/text_pipeline/`, `models/fusion_pipeline/`, `results/text/`, `results/fusion/`, and `results/plots/`; missing final comparative results tables, Stage 08 fusion CLI commands, limitations, conclusion, and link to final technical report.
- **Technical Report (`MISSING`):**
  - No consolidated report (`docs/ASSIGNMENT_2B_REPORT.md` / PDF) covering Sections A (Architecture Decisions), B (Experiments), and C (Analysis: easiest/hardest emotions, when fusion helps most, 3–5 failure cases, and 3-block cluster separability).
- **Unified Results Table (`MISSING`):**
  - Official PDF specifies `Results/` containing *"All 3 model variants accuracy tables"* and `plots/`. Currently results live in `results/{speech,text,fusion}/summaries/` separately without a unified master CSV/Markdown table.
- **[requirements.txt](file:///Users/swaruppal/Downloads/Ass_2/requirements.txt) (`NEEDS VERIFICATION`):**
  - Contains core packages (`torch`, `torchaudio`, `transformers`, `librosa`, `soundfile`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tqdm`, `kagglehub`, `pytest`) — sufficient for all code and t-SNE plots without new installs.

---

## 8. Prioritised Execution Plan (Final Checklist in Execution Order)

1. **[ ] Complete Remaining Word-Independent Late Fusion Seeds (`123`, `999`)** — *~20 seconds total*
   Run `late_fusion` on `word_indep_split` for seeds `123` and `999` so all 3 fusion architectures have complete 3-seed mean $\pm$ std results.
2. **[ ] (Optional / Recommended) Execute Cross-Speaker Fusion Runs (9 Runs)** — *~2 minutes total*
   Run `gated_fusion`, `concat_fusion`, and `late_fusion` on `cross_speaker_split` for seeds `42, 123, 999` to achieve full 2-split $\times$ 3-seed parity with Speech-only and Text-only baselines.
3. **[ ] Re-Aggregate Fusion Summaries & Build Unified 3-Variant Master Table**
   Run `aggregate_fusion_results()` over all completed fusion architectures/splits and generate `results/summaries/all_variants_comparison.{csv,json,md}` comparing Speech-only, Text-only, and Multimodal Fusion.
4. **[ ] Generate Mandatory & Recommended Visualisations (`results/plots/`)**
   Write a clean read-only plotting script (e.g. `scripts/generate_plots.py`) that reads existing `.npz`/`.csv`/`.json` artifacts (zero retraining) to generate:
   - **Mandatory:** `tsne_cluster_separability_3panel.png` (Temporal vs. Contextual vs. Fusion blocks).
   - **Recommended:** Confusion matrices, training curves, per-class F1 comparison, gate activation distributions, late-fusion $\lambda$ curve, cross-speaker comparison, and error analysis plot.
5. **[ ] Author Final Assignment 2B Technical Report**
   Create `docs/ASSIGNMENT_2B_REPORT.md` (and optionally export PDF) covering Sections A, B, and C (architecture rationale for all 5 blocks, full comparative tables, easiest/hardest emotion analysis, TESS modality utility & gating analysis, 5 concrete failure cases, embedded t-SNE and diagnostic plots, limitations, and conclusion).
6. **[ ] Update `README.md` for Final Submission**
   Update stage banner, repository structure, Stage 08 fusion CLI commands, unified results table, plot links, and report link.
7. **[ ] Final Verification & GitHub Push (with User Approval)**
   Run full regression test suite, verify `.gitignore` (excluding `last_checkpoint.pt` / caches / `Assignment 2a/`), review staged files, commit with user-supplied message, and push to `https://github.com/Swarup72/Assignment_2_Speech`.

---

## 9. Final Safety Verification

- **Frozen Checkpoints & Results:** **0 modified** (all 18 unimodal experiments and 7 completed fusion experiments remain untouched).
- **Assignment 2a:** **Untouched.**
- **Files Created During This Audit:** Only `docs/ASSIGNMENT_2B_FINAL_COMPLETION_AUDIT.md`.
- **Git Status:** Verified via `git status --short` and `git diff --stat` (0 tracked files modified; no commits or pushes executed).
