# Assignment 2B — Multimodal Speech Emotion Recognition on TESS

This directory contains the complete implementation, frozen experimental results, visualisations, and technical report for **IIITH Speech Processing — Assignment 2B: Multimodal Emotion Recognition** on the **Toronto Emotional Speech Set (TESS)** (`ejlok1/toronto-emotional-speech-set-tess`).

> - **Combined Assignment 2 Report (Parts 2A & 2B):** [../Report.pdf](../Report.pdf) *(11 pages, 3.93 MB)*
> - **Full Standalone Part 2B Technical Report:** [docs/ASSIGNMENT_2B_REPORT.md](docs/ASSIGNMENT_2B_REPORT.md)
> - **Comparative Analysis & Audits:** [docs/STAGE_09_COMPARATIVE_ANALYSIS_REPORT.md](docs/STAGE_09_COMPARATIVE_ANALYSIS_REPORT.md) | [docs/ASSIGNMENT_2B_FINAL_COMPLETION_AUDIT.md](docs/ASSIGNMENT_2B_FINAL_COMPLETION_AUDIT.md) | [docs/TESTING_REPORT.md](docs/TESTING_REPORT.md)

---

## 1. Project Objective

Classify seven discrete emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `pleasant_surprise`, `sad`) using three modular pipelines built around five functional blocks (Preprocessing, Feature Extraction, Temporal/Contextual Modelling, Fusion, and Classification):

1. **Speech-Only Pipeline (`models/speech_pipeline/`):** Raw waveform → Preprocessing → Feature Extraction (`WavLM Base Plus` or `MFCC + Δ + ΔΔ`) → Temporal Modelling (`WavLMBiGRUAttentionClassifier` / `MFCCCNNBiLSTMAttentionClassifier`) → MLP Classifier.
2. **Text-Only Pipeline (`models/text_pipeline/`):** Utterance transcript → Cleaning & Tokenization → Feature Extraction (`DistilBERT`) → Contextual Modelling (`DistilBERTBiLSTMAttentionClassifier`) → MLP Classifier.
3. **Multimodal Fusion Pipeline (`models/fusion_pipeline/`):** Parallel frozen same-split, same-seed Speech & Text encoders → Fusion Block (`GatedMultimodalFusion`, `ConcatFusion`, or parameter-free validation-calibrated `LateFusionEnsemble`) → Classifier.

---

## 2. TESS Dataset & Evaluation Protocols

- **Dataset:** Toronto Emotional Speech Set (TESS) — **2,800 canonical utterances** (deduplicated from 5,600 raw Kaggle files via SHA-256 verification), **2 female speakers** (`OAF`, `YAF`), **200 target words** in the identical neutral carrier phrase `"Say the word <word>"`, and **7 balanced emotion classes** (400 utterances per class).
- **Preprocessing (`src/features/speech_preprocessing.py`):** Mono conversion, dynamic resampling to **16 kHz** (handling both 24,414 Hz files and the 96,000 Hz outlier `OAF_food_fear.wav`), silence trimming (`librosa.effects.trim`, `top_db=30.0` with fallback), and DC-centred loudness-preserving normalization (`dc_center_headroom`).
- **Evaluation Splits (`data/splits/word_splits.json`, `data/tess_manifest.csv`):**
  1. **Word-Independent (`word_indep_split`, seed 42):** 1,960 train (140 words), 420 val (30 words), 420 test (30 words) — zero target-word overlap.
  2. **Cross-Speaker (`cross_speaker_split`, `OAF -> YAF`):** 1,190 `OAF` train (170 words), 210 `OAF` val (30 words), 1,400 `YAF` test (200 words) — zero speaker overlap.

---

## 3. Summary of Experimental Results (27 Frozen Runs, Seeds `42, 123, 999`)

Full unified tables are stored in [Results/fusion/summaries/overall_comparison.csv](Results/fusion/summaries/overall_comparison.csv) and [Results/fusion/summaries/stage09_unified_summary.json](Results/fusion/summaries/stage09_unified_summary.json).

| Split | Architecture | Trainable Params | Test Accuracy (Mean ± Std) | Test Macro-F1 (Mean ± Std) | Test Log-Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `word_indep_split` | **WavLM + BiGRU + Attn (Speech)** | `857,863` | **99.76 ± 0.00%** | **99.76 ± 0.00%** | `0.0935 ± 0.0018` |
| `word_indep_split` | MFCC + CNN-BiLSTM (Speech) | `1,076,279` | 99.68 ± 0.14% | 99.68 ± 0.14% | `0.0921 ± 0.0044` |
| `word_indep_split` | DistilBERT + BiLSTM + Attn (Text) | `1,055,239` | 14.29 ± 0.00% | 3.57 ± 0.00% | `1.9459 ± 0.0000` |
| `word_indep_split` | **Gated Multimodal Fusion** | `296,967` | **99.76 ± 0.00%** | **99.76 ± 0.00%** | `0.0941 ± 0.0022` |
| `word_indep_split` | **Concatenation Fusion** | `165,895` | **99.76 ± 0.00%** | **99.76 ± 0.00%** | **`0.0888 ± 0.0037`** |
| `word_indep_split` | **Late-Logit Fusion** ($\lambda^* = 1.00$) | `0` | **99.76 ± 0.00%** | **99.76 ± 0.00%** | `0.0935 ± 0.0018` |
| `cross_speaker_split` | WavLM + BiGRU + Attn (Speech) | `857,863` | 40.12 ± 10.07% | 33.59 ± 10.95% | `1.9507 ± 0.4001` |
| `cross_speaker_split` | **MFCC + CNN-BiLSTM (Speech)** | `1,076,279` | **43.67 ± 7.78%** | **42.62 ± 7.26%** | `2.0644 ± 0.1748` |
| `cross_speaker_split` | DistilBERT + BiLSTM + Attn (Text) | `1,055,239` | 14.29 ± 0.00% | 3.57 ± 0.00% | `1.9460 ± 0.0000` |

### Key Visualisations (`Results/plots/`)
- **Mandatory 3-Panel Emotion Cluster Separability (t-SNE):** [Results/plots/tsne_three_panel_comparison.png](Results/plots/tsne_three_panel_comparison.png)
- **Confusion Matrices:** [word-independent](Results/plots/confusion_matrices_word_independent.png) | [cross-speaker](Results/plots/confusion_matrices_cross_speaker.png)
- **Training Curves & Per-Class F1:** [training_validation_curves.png](Results/plots/training_validation_curves.png) | [per_class_f1_comparison.png](Results/plots/per_class_f1_comparison.png)
- **Fusion Diagnostics:** [gated_fusion_gate_analysis.png](Results/plots/gated_fusion_gate_analysis.png) | [late_fusion_lambda_calibration.png](Results/plots/late_fusion_lambda_calibration.png)
- **Cross-Speaker & Error Analysis:** [cross_speaker_speech_comparison.png](Results/plots/cross_speaker_speech_comparison.png) | [error_analysis_five_cases.png](Results/plots/error_analysis_five_cases.png)

---

## 4. Folder Structure

```text
Assignment 2b/
├── README.md                            # Assignment 2B overview & usage guide
├── requirements.txt                     # Python dependencies
├── Assignment 2b.pdf                    # Official Assignment 2B problem specification
├── PROJECT_AUDIT.md                     # Initial architecture & dataset audit
├── data/
│   ├── tess_manifest.csv                # Canonical 2,800-row deduplicated manifest
│   └── splits/word_splits.json          # Word-independent & cross-speaker split definitions
├── models/
│   ├── speech_pipeline/                 # WavLM+BiGRU+Attn & MFCC+CNN-BiLSTM (model, train, test)
│   ├── text_pipeline/                   # DistilBERT+BiLSTM+Attn (model, train, test)
│   └── fusion_pipeline/                 # Gated, Concat & Late Fusion (model, train, test)
├── src/
│   ├── data/                            # Kaggle downloader, manifest builder, bimodal Dataset
│   └── features/                        # Audio preprocessing, WavLM/MFCC & DistilBERT extractors
├── Results/
│   ├── speech/                          # 12 frozen speech experiments + summaries
│   ├── text/                            # 6 frozen text experiments + summaries
│   ├── fusion/                          # 9 frozen fusion experiments + unified comparison tables
│   └── plots/                           # 9 publication-quality 300-DPI figures
├── docs/                                # Final technical report, Stage 09 analysis, logs & audits
├── scripts/
│   └── run_stage09_analysis_and_plots.py # Reproducible table & 300-DPI figure generator
└── tests/                               # Automated unit & regression test suite
```

*(Raw TESS audio `data/tess/` and cached features `data/features/` are excluded via root `.gitignore` and can be regenerated with the commands below.)*

---

## 5. Installation & Reproducibility Commands

Run all commands from inside `Assignment 2b/`:

```bash
cd "Assignment 2b"
python3.11 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
```

### Dataset Preparation & Offline Feature Caching
```bash
python -m src.data.download_tess
python -m src.data.build_manifest
python -m src.features.extract_speech_features
python -m src.features.extract_text_features
```

### Training & Evaluation Commands
```bash
# Speech-Only Pipeline
python -m models.speech_pipeline.train --splits word_indep_split cross_speaker_split --architectures wavlm_bigru_attn mfcc_cnn_bilstm --seeds 42 123 999
python -m models.speech_pipeline.test

# Text-Only Pipeline
python -m models.text_pipeline.train --splits word_independent cross_speaker --seeds 42 123 999
python -m models.text_pipeline.test --splits word_independent cross_speaker --seeds 42 123 999

# Multimodal Fusion Pipeline (Frozen Base Encoders)
python -m models.fusion_pipeline.train --architectures gated_fusion concat_fusion late_fusion --splits word_indep_split --seeds 42 123 999
python -m models.fusion_pipeline.test --architectures gated_fusion concat_fusion late_fusion --splits word_indep_split --seeds 42 123 999
```

### Unified Tables, Visualisations & Automated Tests
```bash
python -m scripts.run_stage09_analysis_and_plots
pytest tests/test_splits.py tests/test_speech_pipeline.py tests/test_text_pipeline.py tests/test_text_training_pipeline.py tests/test_fusion_pipeline.py tests/test_fusion_training.py -v
```

---

## 6. Limitations & Future Work

- **Neutral Carrier Phrase in TESS:** Identical transcripts across all 7 emotions yield chance-level text accuracy (`14.29%`); fusion cannot exceed the `99.76%` acoustic ceiling on `word_indep_split`, though all three fusion methods preserve accuracy without degrading under lexical noise.
- **Cross-Speaker Shift:** Single-speaker training (`OAF`) drops sharply on unseen speaker `YAF` (`40.12%` WavLM, `43.67%` MFCC). Optional cross-speaker multimodal fusion runs were not executed and remain future work alongside multi-speaker expressive corpora (e.g., IEMOCAP, MELD).
