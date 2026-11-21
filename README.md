# Assignment_2_Speech

## Assignment 2B: Multimodal Emotion Recognition on TESS

This repository implements **IIITH Speech Processing — Assignment 2B: Multimodal Emotion Recognition** on the **Toronto Emotional Speech Set (TESS)** (`ejlok1/toronto-emotional-speech-set-tess`).

> **Project Status:** **Complete (27 Frozen Experiments + Unified Comparative Analysis, Visualisations & Final Technical Report)** — 12 Speech-Only runs (`wavlm_bigru_attn`, `mfcc_cnn_bilstm`), 6 Text-Only runs (`distilbert_bilstm_attn`), and 9 Word-Independent Multimodal Fusion runs (`gated_fusion`, `concat_fusion`, `late_fusion`) evaluated across 3 random seeds (`42`, `123`, `999`).

---

## 1. Problem Statement & System Overview

The objective is to classify seven discrete emotions (`angry`, `disgust`, `fear`, `happy`, `neutral`, `pleasant_surprise`, `sad`) using three modular pipelines built around five functional blocks:

1. **Speech-Only Pipeline:** Raw Audio $\rightarrow$ Preprocessing $\rightarrow$ Feature Extraction (`WavLM Base Plus` / `MFCC`) $\rightarrow$ Temporal Modelling (`BiGRU+Attn` / `CNN-BiLSTM+Attn`) $\rightarrow$ MLP Classifier.
2. **Text-Only Pipeline:** Transcript $\rightarrow$ Cleaning & Tokenization $\rightarrow$ Feature Extraction (`DistilBERT`) $\rightarrow$ Contextual Modelling (`BiLSTM+Attn`) $\rightarrow$ MLP Classifier.
3. **Multimodal Fusion Pipeline:** Parallel Frozen Speech & Text Encoders $\rightarrow$ Fusion Block (`Gated Multimodal Fusion` / `Concatenation Fusion` / `Late-Logit Fusion`) $\rightarrow$ Classifier.

Full architectural rationale, methodology, visual analysis, and error analysis are documented in the [Final Technical Report](docs/ASSIGNMENT_2B_REPORT.md).

---

## 2. Dataset & Evaluation Protocols

- **Dataset:** Toronto Emotional Speech Set (TESS) — **2,800 canonical utterances** (deduplicated from 5,600 raw Kaggle files via SHA-256 verification), **2 speakers** (`OAF`, `YAF`), **200 target words** in the neutral carrier phrase `"Say the word <word>"`, and **7 balanced emotion classes** (400 utterances per class).
- **Preprocessing (`src/features/speech_preprocessing.py`):** Mono conversion, dynamic resampling to **16 kHz** (handling both 24,414 Hz files and the 96,000 Hz outlier `OAF_food_fear.wav`), silence trimming (`librosa.effects.trim`, `top_db=30.0` with fallback), and DC-centred loudness-preserving normalization (`dc_center_headroom`).
- **Cached Features:**
  - `data/features/wavlm/`: 2,800 utterance `.pt` tensors `(T_s, 768)` from `microsoft/wavlm-base-plus`.
  - `data/features/mfcc/`: 2,800 utterance `.pt` tensors `(T_m, 120)` (40 MFCC + $\Delta$ + $\Delta\Delta$).
  - `data/features/distilbert/`: 200 unique transcript `.pt` tensors `(L_t, 768)` from `distilbert-base-uncased` + `sample_to_transcript.json`.
- **Evaluation Protocols (`data/splits/word_splits.json`):**
  1. **Word-Independent (`word_indep_split`, seed 42):** 140 train words (1,960), 30 val words (420), 30 test words (420) — zero word overlap.
  2. **Cross-Speaker (`cross_speaker_split`, `OAF -> YAF`):** 170 `OAF` train words (1,190), 30 `OAF` val words (210), 200 `YAF` test words (1,400) — zero speaker overlap.

---

## 3. Summary of Experimental Results (27 Frozen Runs, 3 Seeds: `42, 123, 999`)

Full tables are available in [results/fusion/summaries/overall_comparison.md](results/fusion/summaries/overall_comparison.md) and [overall_comparison.csv](results/fusion/summaries/overall_comparison.csv).

| Evaluation Split | Architecture | Trainable Params | Test Accuracy (Mean ± Std) | Test Macro-F1 (Mean ± Std) | Test Log-Loss |
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

### Key Visualisations (`results/plots/`)
- **Mandatory 3-Panel Emotion Cluster Separability (t-SNE):** [results/plots/tsne_three_panel_comparison.png](results/plots/tsne_three_panel_comparison.png)
- **Confusion Matrices:** [word-independent](results/plots/confusion_matrices_word_independent.png) | [cross-speaker](results/plots/confusion_matrices_cross_speaker.png)
- **Training Curves & Per-Class F1:** [training_validation_curves.png](results/plots/training_validation_curves.png) | [per_class_f1_comparison.png](results/plots/per_class_f1_comparison.png)
- **Fusion Diagnostics:** [gated_fusion_gate_analysis.png](results/plots/gated_fusion_gate_analysis.png) | [late_fusion_lambda_calibration.png](results/plots/late_fusion_lambda_calibration.png)
- **Cross-Speaker & Error Analysis:** [cross_speaker_speech_comparison.png](results/plots/cross_speaker_speech_comparison.png) | [error_analysis_five_cases.png](results/plots/error_analysis_five_cases.png)

---

## 4. Repository Structure

```text
Assignment_2_Speech/
├── Assignment 2b.pdf                    # Official Assignment 2B problem specification
├── README.md                            # Project overview, results & usage guide
├── PROJECT_AUDIT.md                     # Initial architecture & dataset audit
├── requirements.txt                     # Python package dependencies
├── data/
│   ├── tess_manifest.csv                # Canonical 2,800-row deduplicated manifest
│   └── splits/word_splits.json          # Word-independent & cross-speaker split definitions
├── docs/
│   ├── ASSIGNMENT_2B_REPORT.md          # Final Assignment 2B Technical Report (Sections A–P)
│   ├── ASSIGNMENT_2B_FINAL_COMPLETION_AUDIT.md
│   ├── STAGE_09_COMPARATIVE_ANALYSIS_REPORT.md
│   ├── IMPLEMENTATION_LOG.md
│   └── TESTING_REPORT.md
├── models/
│   ├── speech_pipeline/                 # WavLM+BiGRU+Attn & MFCC+CNN-BiLSTM (model, train, test)
│   ├── text_pipeline/                   # DistilBERT+BiLSTM+Attn (model, train, test)
│   └── fusion_pipeline/                 # Gated, Concat & Late Fusion (model, train, test)
├── scripts/
│   └── run_stage09_analysis_and_plots.py # Reproducible table & 300-DPI figure generator
├── src/
│   ├── data/                            # Kaggle downloader, manifest builder, bimodal Dataset
│   └── features/                        # Audio preprocessing, WavLM/MFCC & DistilBERT extractors
├── results/
│   ├── speech/                          # 12 frozen speech experiments + summaries
│   ├── text/                            # 6 frozen text experiments + summaries
│   ├── fusion/                          # 9 frozen fusion experiments + unified comparison tables
│   └── plots/                           # 9 publication-quality 300-DPI figures
└── tests/                               # Automated unit & regression test suite
```

*(Note: Raw TESS audio `data/tess/raw/` and cached features `data/features/` are excluded from Git via `.gitignore` and can be regenerated using the commands below.)*

---

## 5. Environment Setup & Reproducibility

Create an isolated Python 3.11 virtual environment and install dependencies from [requirements.txt](requirements.txt):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Data Acquisition, Manifest & Feature Extraction
```bash
python -m src.data.download_tess
python -m src.data.build_manifest
python -m src.features.extract_speech_features
python -m src.features.extract_text_features
```

### Step 2: Speech-Only Pipeline
```bash
python -m models.speech_pipeline.train --splits word_indep_split cross_speaker_split --architectures wavlm_bigru_attn mfcc_cnn_bilstm --seeds 42 123 999
python -m models.speech_pipeline.test
```

### Step 3: Text-Only Pipeline
```bash
python -m models.text_pipeline.train --splits word_independent cross_speaker --seeds 42 123 999
python -m models.text_pipeline.test --splits word_independent cross_speaker --seeds 42 123 999
```

### Step 4: Multimodal Fusion Pipeline (Frozen Base Encoders)
```bash
python -m models.fusion_pipeline.train --architectures gated_fusion concat_fusion late_fusion --splits word_indep_split --seeds 42 123 999
python -m models.fusion_pipeline.test --architectures gated_fusion concat_fusion late_fusion --splits word_indep_split --seeds 42 123 999
```

### Step 5: Unified Tables, Visualisations & Automated Tests
```bash
python -m scripts.run_stage09_analysis_and_plots
pytest tests/test_splits.py tests/test_speech_pipeline.py tests/test_text_pipeline.py tests/test_text_training_pipeline.py tests/test_fusion_pipeline.py tests/test_fusion_training.py -v
```

---

## 6. Limitations & Future Work

- **Identical Carrier Phrase in TESS:** Because every target word appears identically across all 7 emotions, text-only models achieve chance accuracy (`14.29%`) and fusion cannot exceed the `99.76%` acoustic ceiling on `word_indep_split`, though all three fusion methods preserve accuracy without degrading under lexical noise.
- **Cross-Speaker Generalization:** Single-speaker training (`OAF`) drops substantially on unseen speaker `YAF` (`40.12%` WavLM, `43.67%` MFCC), demonstrating that word-independent accuracy does not imply speaker-independent generalization.
- **Unexecuted Optional Experiments:** Optional cross-speaker multimodal fusion runs were not executed and are left for future work alongside multi-speaker expressive corpora (e.g., IEMOCAP, MELD).
