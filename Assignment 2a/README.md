# Assignment 2A — Forced Alignment using Montreal Forced Aligner (MFA)

This directory contains the reproducible outputs, custom OOV pronunciation entries, statistical summaries, programmatic visualisations, and qualitative analysis for **IIITH Speech Processing — Assignment 2A: Forced Alignment using Montreal Forced Aligner (MFA)**.

> **Combined Report:** See [../Report.pdf](../Report.pdf) for the combined Assignment 2 (Parts 2A & 2B) PDF report, and [ALIGNMENT_ANALYSIS.md](ALIGNMENT_ANALYSIS.md) for the full standalone Part 2A technical analysis.

---

## 1. Objective & Role of Montreal Forced Aligner (MFA)

The objective is to set up and execute an end-to-end **forced alignment** pipeline that automatically aligns speech audio recordings with their known orthographic transcripts at both the **word** and **phoneme** levels.

**Montreal Forced Aligner (MFA v3.4.2)** expands each transcript word into canonical phoneme sequences using a pronunciation dictionary and aligns those sequences to acoustic frames (10 ms frame grid) via Kaldi HMM-GMM acoustic models, outputting Praat-compatible `.TextGrid` files.

---

## 2. Dataset Overview & Audio Normalization

The provided dataset consists of **6 English audio recordings and 6 matching transcripts** (`97.1633 s` total duration) across two distinct speaker domains:

| Speaker Group | Recordings | Count | Total Duration | Source Audio Format | Speaking Style |
|---|---|---:|---:|---|---|
| `F2B` | `F2BJRLP1`, `F2BJRLP2`, `F2BJRLP3` | 3 | `84.6633 s` | NIST SPHERE (`.wav` extension) | Continuous, fast broadcast news prose (~160.2 wpm) |
| `ISLE_SESS0131` | `ISLE_SESS0131_BLOCKD02_01_sprt1`, `_02_sprt1`, `_03_sprt1` | 3 | `12.5000 s` | RIFF WAV (PCM-16, mono, 16 kHz) | Short, deliberate minimal-pair read speech (~72.0 wpm) |

**Preprocessing & Format Normalization (performed locally without altering originals):**
- Original `F2B` NIST SPHERE recordings were losslessly converted to standard **RIFF WAV (16-bit PCM, mono, 16 kHz)**; `ISLE_SESS0131` recordings were already in that format.
- Transcripts were normalized into single-line `.lab` files matching each `.wav` basename.
- Audio/transcript pairs were organized into speaker subdirectories (`alignment_corpus/F2B/` and `alignment_corpus/ISLE_SESS0131/`) so MFA could perform independent per-speaker feature normalization.

*(Note: Source `.wav` files, raw transcripts, `working_corpus/`, and `alignment_corpus/` are intentionally excluded from Git publication via `.gitignore`. Users reproducing the alignment must obtain the assignment dataset separately.)*

---

## 3. Acoustic Model, Pronunciation Dictionary & Custom OOV Handling

- **MFA Version & Environment:** `montreal-forced-aligner 3.4.2` (`conda-forge`), Python `3.10.22`, isolated Conda environment `mfa_env` on macOS Apple Silicon (M2, `arm64`).
- **Acoustic Model:** `english_us_arpa` (`v3.0.0a2`, ARPAbet phone set).
- **Base Dictionary & G2P Model:** `english_us_arpa` pretrained dictionary and `english_us_arpa` G2P (`v2.0.0rc4`).
- **Custom OOV Entries ([custom_oov_entries.dict](custom_oov_entries.dict)):** Initial corpus validation identified **5 OOV word types** (all in `F2B`), resolved as follows:

| OOV Word | Resolution Method | Custom ARPAbet Pronunciation |
|---|---|---|
| `dukakis` | `mfa g2p` (`english_us_arpa`) | `D AH0 K AA1 K IH0 S` |
| `maffy` | `mfa g2p` (`english_us_arpa`) | `M AE1 F IY0` |
| `melnicove` | `mfa g2p` (`english_us_arpa`) | `M EH1 L N IH0 K OW0 V` |
| `politicize` | `mfa g2p` (`english_us_arpa`) | `P AH0 L IH1 T IH0 S AY2 Z` |
| `wbur` | Manual letter-by-letter construction* | `D AH1 B AH0 L Y UW0 B IY1 Y UW1 AA1 R` |

*\*`mfa g2p` initially proposed word-like pronunciations (`B ER0` / `W AH0 B ER0`) for the radio call sign `wbur`; it was manually constructed from verified base-dictionary letter entries (`W`, `B`, `U`, `R`).*

> **Recreating the Full Combined Dictionary:** The 6.9 MB combined dictionary (`english_us_arpa_custom.dict`) is excluded from Git to keep the repository lightweight. It can be recreated identically by concatenating MFA's downloaded `english_us_arpa.dict` with [custom_oov_entries.dict](custom_oov_entries.dict) (see Section 7).

---

## 4. Alignment Procedure & Output TextGrids

Post-OOV validation (`mfa validate --ignore_acoustics --clean`) confirmed **0 OOVs and 0 errors**. Forced alignment completed in `44.836 s` with **zero alignment errors**, producing **6 Praat `.TextGrid` files** under `aligned_textgrids/`:

- `aligned_textgrids/F2B/F2BJRLP1.TextGrid`
- `aligned_textgrids/F2B/F2BJRLP2.TextGrid`
- `aligned_textgrids/F2B/F2BJRLP3.TextGrid`
- `aligned_textgrids/ISLE_SESS0131/ISLE_SESS0131_BLOCKD02_01_sprt1.TextGrid`
- `aligned_textgrids/ISLE_SESS0131/ISLE_SESS0131_BLOCKD02_02_sprt1.TextGrid`
- `aligned_textgrids/ISLE_SESS0131/ISLE_SESS0131_BLOCKD02_03_sprt1.TextGrid`

**Tier Structure:** Every `.TextGrid` contains exactly two interval tiers:
1. **`words` tier:** Word-level start/end boundaries and empty (`""`) silence intervals (`241` non-empty words + `45` silence intervals across all 6 files).
2. **`phones` tier:** ARPAbet phoneme-level start/end boundaries and empty (`""`) silence intervals (`1,063` non-empty phones + `45` silence intervals across all 6 files).

All 6 files passed structural validation: `0` `<unk>` tokens, `0` `spn` fallbacks, `0` interval gaps/overlaps, and exact `0.0000 s` duration match against their source WAV files.

---

## 5. Alignment Statistics & Visualisations

Full per-recording and group metrics are saved in [alignment_statistics.csv](alignment_statistics.csv) and [alignment_statistics.json](alignment_statistics.json):

| Group | Recordings | Total Dur (s) | Non-Empty Words | Non-Empty Phones | Mean / Med Phone (ms) | `<29.99 ms` | `Nominal 30 ms` (`29.99–30.01 ms`) | `≤30.01 ms` | `≥299.99 ms` | Rate (wpm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`F2B`** | 3 | 84.66 | 226 | 1,025 | 73.7 / 70.0 | 2 | 75 | 77 | 0 | 160.2 |
| **`ISLE_SESS0131`** | 3 | 12.50 | 15 | 38 | 133.4 / 115.0 | 0 | 0 | 0 | 1 | 72.0 |
| **All 6** | 6 | 97.16 | 241 | 1,063 | 75.9 / 70.0 | 2 | 75 | 77 | 1 | 148.8 |

*(Reconciled counts on non-empty phones only using MFA's `±0.01 ms` frame-grid tolerance to accommodate float32 TextGrid serialization jitter; the 75 nominally 30 ms intervals include 12 serialized as `30.001 ms`, and the 2 intervals shorter than nominal 30 ms are the two 10 ms `AH0` intervals in `F2BJRLP1`.)*

**Programmatic Visualisations (`visualizations/`):**
- `visualizations/overview/` — 6 full-utterance 300-DPI figures (waveform, 0–8 kHz wideband spectrogram, `words` tier, `phones` tier).
- `visualizations/detailed/` — 7 zoomed 300-DPI figures inspecting `wbur's`, `S.J.C.'s`, `Dukakis`, `de-politicize`, `Maffy`, `ISLE 02` final `OW1`, and `ISLE 03` word-final stops.

> **Important Disclosure:** All visualisations in `visualizations/` were generated **programmatically in Python** (`praatio`, `soundfile`, `scipy`, `matplotlib`). **Praat GUI screenshots and manual Praat auditory inspection were not performed**, as Praat was not installed on the analysis machine.

---

## 6. Included Scripts & Scope

- [generate_stage06_outputs.py](generate_stage06_outputs.py): Read-only analysis and plotting script (requires `praatio`, `soundfile`, `scipy`, `numpy`, `matplotlib`, and local `alignment_corpus/` WAVs + `aligned_textgrids/` TextGrids). Computes `alignment_statistics.{csv,json}` and renders all 13 plots. **It does not run MFA alignment itself.**
- [generate_pdf_report.py](generate_pdf_report.py): Renders the standalone Part 2A PDF report via headless Google Chrome from the generated figures and statistics.

---

## 7. Reproducibility: Installation & Execution Commands

### Step 1 — Install MFA 3.4.2 & Download Pretrained Models
```bash
conda create -n mfa_env -c conda-forge montreal-forced-aligner=3.4.2 python=3.10
conda activate mfa_env
mfa version   # verified: 3.4.2

mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

### Step 2 — Recreate the Custom Dictionary
```bash
cat ~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict \
    "Assignment 2a/custom_oov_entries.dict" \
    > "Assignment 2a/english_us_arpa_custom.dict"
```

### Step 3 — Validate Corpus & Run Forced Alignment
*(Requires preparing `Assignment 2a/alignment_corpus/{F2B,ISLE_SESS0131}/` with 16 kHz mono PCM-16 `.wav` and matching `.lab` files from the course dataset.)*
```bash
mfa validate "Assignment 2a/alignment_corpus" \
  "Assignment 2a/english_us_arpa_custom.dict" \
  --ignore_acoustics --clean

mfa align "Assignment 2a/alignment_corpus" \
  "Assignment 2a/english_us_arpa_custom.dict" \
  english_us_arpa \
  "Assignment 2a/aligned_textgrids" --clean
```

### Step 4 — Regenerate Stage 06 Statistics & Visualisations (Optional)
```bash
python "Assignment 2a/generate_stage06_outputs.py"
```

---

## 8. Limitations

1. **No Praat GUI / Auditory Verification:** Technical validation and visual spectrogram inspection were completed, but direct Praat GUI inspection and human auditory verification remain pending.
2. **Single-Pronunciation OOV Entries:** Providing only one canonical pronunciation for `wbur` and `dukakis` led to 10 ms single-frame `AH0` compression in `wbur's` (`20.73–20.74 s`).
3. **Domain-Specific Boundary Artifacts:** Fast broadcast speech (`F2B`) exhibits frame-floor compression (`77` phones `≤30.01 ms`), whereas slow read speech (`ISLE_SESS0131`) exhibits right-edge vowel trailing (`OW1` 350 ms) and post-release pause absorption into word-final stops (`240–260 ms`).
