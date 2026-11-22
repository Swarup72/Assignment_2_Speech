# Assignment_2_Speech — IIITH Speech Processing (Assignment 2)

This repository contains the complete submission for **IIITH Speech Processing — Assignment 2**, organized cleanly into two standalone assignment folders plus the unified final report at the repository root:

- **[Assignment 2a/](Assignment%202a/)** — Forced Alignment using Montreal Forced Aligner (MFA `3.4.2`)
- **[Assignment 2b/](Assignment%202b/)** — Multimodal Speech Emotion Recognition on TESS (27 Frozen Experiments)
- **[Report.pdf](Report.pdf)** — Combined Final Technical Report covering both Parts 2A and 2B *(11 pages, 3.93 MB)*

---

## Quick Navigation

| Resource | Link |
|---|---|
| **Combined Final PDF Report (Parts 2A & 2B)** | [Report.pdf](Report.pdf) |
| **Part 2A README (Setup, MFA Commands & Reproducibility)** | [Assignment 2a/README.md](Assignment%202a/README.md) |
| **Part 2A Detailed Analysis** | [Assignment 2a/ALIGNMENT_ANALYSIS.md](Assignment%202a/ALIGNMENT_ANALYSIS.md) |
| **Part 2A Aligned TextGrids (6 files)** | [Assignment 2a/aligned_textgrids/](Assignment%202a/aligned_textgrids/) |
| **Part 2B README (Pipelines, Results & Commands)** | [Assignment 2b/README.md](Assignment%202b/README.md) |
| **Part 2B Detailed Technical Report** | [Assignment 2b/docs/ASSIGNMENT_2B_REPORT.md](Assignment%202b/docs/ASSIGNMENT_2B_REPORT.md) |
| **Part 2B Results & 300-DPI Visualisations** | [Assignment 2b/Results/](Assignment%202b/Results/) |

---

## Repository Structure

```text
Assignment_2_Speech/
├── README.md                            # Combined Assignment 2 landing page
├── Report.pdf                           # Combined Assignment 2 PDF report (Parts 2A & 2B, 11 pp., 3.93 MB)
├── .gitignore                           # Excludes raw audio, feature caches, corpora & temp build files
├── Assignment 2a/                       # Part 2A — Forced Alignment using MFA 3.4.2
│   ├── README.md
│   ├── ALIGNMENT_ANALYSIS.md
│   ├── custom_oov_entries.dict
│   ├── alignment_statistics.csv
│   ├── alignment_statistics.json
│   ├── generate_stage06_outputs.py
│   ├── generate_pdf_report.py
│   ├── aligned_textgrids/               # 6 validated .TextGrid files (F2B/ & ISLE_SESS0131/)
│   └── visualizations/                  # 6 overview + 7 detailed programmatic Python plots
└── Assignment 2b/                       # Part 2B — Multimodal Emotion Recognition on TESS
    ├── README.md
    ├── requirements.txt
    ├── Assignment 2b.pdf
    ├── PROJECT_AUDIT.md
    ├── data/                            # Canonical manifest & train/val/test split definitions
    ├── models/                          # speech_pipeline/, text_pipeline/, fusion_pipeline/
    ├── src/                             # Data acquisition, manifest builder & feature extraction
    ├── Results/                         # 27 frozen runs (speech/, text/, fusion/) + 9 plots/
    ├── docs/                            # Technical report, Stage 09 analysis, logs & audits
    ├── scripts/                         # Stage 09 unified analysis & plotting script
    └── tests/                           # Automated unit & regression test suite
```

---

## Part 2A — Forced Alignment using Montreal Forced Aligner (MFA)

- **Toolchain:** MFA `3.4.2` (`mfa_env`, Python `3.10.22`), `english_us_arpa` acoustic model (`v3.0.0a2`), pretrained dictionary & G2P (`v2.0.0rc4`), extended with 5 resolved OOV entries ([Assignment 2a/custom_oov_entries.dict](Assignment%202a/custom_oov_entries.dict): `dukakis`, `maffy`, `melnicove`, `politicize`, and manually constructed letter-by-letter `wbur`).
- **Results:** All 6 English recordings (`F2B` broadcast news & `ISLE_SESS0131` read speech; `97.16 s` total) aligned with `0` OOVs, `0` errors, `0` `<unk>`/`spn` tokens, and `0.0000 s` duration mismatch (`241` non-empty words, `1,063` non-empty phones).
- **Key Findings:** Fast broadcast speech (`F2B`, 160.2 wpm) exhibits `77` phones nominally 30 ms or shorter (`≤30.01 ms` under MFA's `±0.01 ms` frame-grid tolerance), whereas slow read speech (`ISLE_SESS0131`, 72.0 wpm) exhibits `0` short phones, `1` phone at least nominally 300 ms (`OW1`, 350 ms), and post-release pause absorption into word-final stops (`240–260 ms`).
- **Disclosure:** All 13 plots in [Assignment 2a/visualizations/](Assignment%202a/visualizations/) were generated programmatically in Python; Praat GUI screenshots and manual Praat auditory inspection were not performed.
- **Full Guide:** See [Assignment 2a/README.md](Assignment%202a/README.md) and [Assignment 2a/ALIGNMENT_ANALYSIS.md](Assignment%202a/ALIGNMENT_ANALYSIS.md).

---

## Part 2B — Multimodal Speech Emotion Recognition on TESS

- **Pipelines:** Speech-Only (`WavLM+BiGRU+Attn`, `MFCC+CNN-BiLSTM`), Text-Only (`DistilBERT+BiLSTM+Attn`), and Multimodal Fusion (`GatedMultimodalFusion`, `ConcatFusion`, parameter-free `LateFusionEnsemble`) evaluated across **27 frozen runs** (3 seeds: `42, 123, 999`) on canonical TESS (`2,800` utterances, 7 balanced emotions).
- **Key Results:**
  - **Word-Independent Split (420 test samples):** WavLM speech-only and all three multimodal fusion models achieve **99.76 ± 0.00%** Accuracy and Macro-F1 (MFCC: **99.68 ± 0.14%**; Concat Fusion achieves lowest test log-loss **0.0888 ± 0.0037**), while DistilBERT text-only stays at the **14.29 ± 0.00%** chance bound due to TESS's identical neutral carrier phrase.
  - **Cross-Speaker Split (`OAF -> YAF`, 1,400 test samples):** Exposes strong speaker shift (MFCC **43.67 ± 7.78%** Accuracy / **42.62 ± 7.26%** Macro-F1 vs. WavLM **40.12 ± 10.07%** Accuracy / **33.59 ± 10.95%** Macro-F1; `neutral` transfers best, `happy` transfers worst).
- **Full Guide & Results:** See [Assignment 2b/README.md](Assignment%202b/README.md), [Assignment 2b/docs/ASSIGNMENT_2B_REPORT.md](Assignment%202b/docs/ASSIGNMENT_2B_REPORT.md), and [Assignment 2b/Results/plots/tsne_three_panel_comparison.png](Assignment%202b/Results/plots/tsne_three_panel_comparison.png).
