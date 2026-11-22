# Forced Alignment Qualitative & Quantitative Analysis — IIITH Assignment 2A

## 1. Objective

This document presents a quantitative and qualitative evaluation of the automatic word-level and phoneme-level forced alignments generated for IIITH Assignment 2A using the **Montreal Forced Aligner (MFA)**. Forced alignment maps known orthographic transcripts to their corresponding speech waveforms by estimating start and end timestamps for every word and constituent phoneme, outputting Praat-compatible `.TextGrid` files.

---

## 2. Environment & Toolchain Configuration

| Component | Verified Value |
|---|---|
| Operating System / Hardware | macOS 27.0.1 (`arm64`, Apple Silicon M2) |
| Conda Environment | `mfa_env` (`/opt/anaconda3/envs/mfa_env`) |
| Python Version | `3.10.22` |
| Montreal Forced Aligner | `3.4.2` (`conda-forge`) |
| Acoustic Model | `english_us_arpa` (`v3.0.0a2`, ARPAbet phone set) |
| Base Pronunciation Dictionary | `english_us_arpa` (pretrained MFA dictionary) |
| Custom Pronunciation Dictionary | `Assignment 2a/english_us_arpa_custom.dict` |
| G2P Model | `english_us_arpa` (`v2.0.0rc4`) |
| Praat Desktop Application | **Not installed** on this machine (`which praat` and `/Applications/Praat.app` absent) |
| Visualisation & Analysis Stack | Python (`praatio`, `soundfile`, `scipy`, `numpy`, `matplotlib`) at 300 DPI |

> **Important Methodological Note on Praat:** Because Praat is not installed in this environment, direct interactive Praat GUI inspection and auditory playback verification were **not** performed during this stage. All waveform, wideband spectrogram, word-tier, and phone-tier visualisations were generated programmatically in Python directly from the unmodified Stage 05 `.TextGrid` files and 16 kHz `.wav` recordings. Direct Praat inspection and auditory listening remain recommended as a final human verification step.

---

## 3. Dataset Preparation & Speaker Organization

The provided dataset comprises **6 English speech recordings** (total duration `97.1633 s`) spanning two distinct corpora/speaker groups:

1. **`F2B` (3 recordings, `84.6633 s` total):** Continuous, fast-paced broadcast news speech (`F2BJRLP1`, `F2BJRLP2`, `F2BJRLP3`; `25.31–30.71 s` each). The original `.wav` files were stored in NIST SPHERE format and were losslessly converted to standard RIFF WAV (16-bit PCM, mono, 16 kHz) in `working_corpus/` and `alignment_corpus/F2B/` while preserving the original files byte-for-byte.
2. **`ISLE_SESS0131` (3 recordings, `12.5000 s` total):** Short, deliberate read-speech minimal-pair contrast utterances (`ISLE_SESS0131_BLOCKD02_01_sprt1`, `_02_sprt1`, `_03_sprt1`; `3.88–4.50 s` each), already in RIFF WAV PCM-16 mono 16 kHz format.

Recordings were organized into speaker-specific subdirectories (`Assignment 2a/alignment_corpus/F2B/` and `Assignment 2a/alignment_corpus/ISLE_SESS0131/`) so MFA could perform per-speaker feature normalization (CMVN/fMLLR) independently across the two acoustic domains.

---

## 4. Pronunciation Dictionary, G2P & Custom OOV Resolution

Initial corpus validation against the base `english_us_arpa` dictionary identified **5 out-of-vocabulary (OOV) word types** (all occurring in the `F2B` broadcast news transcripts):

| OOV Word | Source | Custom ARPAbet Pronunciation |
|---|---|---|
| `dukakis` | `mfa g2p` (`english_us_arpa`) | `D AH0 K AA1 K IH0 S` |
| `maffy` | `mfa g2p` (`english_us_arpa`) | `M AE1 F IY0` |
| `melnicove` | `mfa g2p` (`english_us_arpa`) | `M EH1 L N IH0 K OW0 V` |
| `politicize` | `mfa g2p` (`english_us_arpa`) | `P AH0 L IH1 T IH0 S AY2 Z` |
| `wbur` | Manual letter-by-letter construction | `D AH1 B AH0 L Y UW0 B IY1 Y UW1 AA1 R` |

- `mfa g2p` initially produced word-like pronunciations for the radio call sign `wbur` (`B ER0` / `W AH0 B ER0`), which would misalign the spoken letter sequence "W-B-U-R". Therefore, `wbur` was manually constructed from verified base-dictionary letter pronunciations (`W`, `B`, `U`, `R`).
- All 5 entries were saved to `Assignment 2a/custom_oov_entries.dict` and appended to a local copy of the base dictionary (`Assignment 2a/english_us_arpa_custom.dict`), leaving the downloaded base dictionary untouched.
- Post-resolution validation (`mfa validate --ignore_acoustics --clean`) confirmed **0 OOVs and 0 errors**.

---

## 5. Forced Alignment Execution Summary

Forced alignment was executed in Stage 05 via:

```bash
mfa align "Assignment 2a/alignment_corpus" \
  "Assignment 2a/english_us_arpa_custom.dict" \
  english_us_arpa \
  "Assignment 2a/aligned_textgrids" --clean
```

- **Execution time:** `44.836 s`
- **Outputs generated:** 6 `.TextGrid` files (`3` in `F2B/`, `3` in `ISLE_SESS0131/`), each containing `words` and `phones` tiers.
- **Technical integrity checks:** `0` `<unk>` tokens, `0` `spn` (spoken noise) fallbacks, `0` interval gaps or overlaps, `0` non-positive durations, and exact `0.0000 s` duration match between every `.TextGrid` and its corresponding `.wav` file.

---

## 6. Quantitative Alignment Statistics

Full statistics are stored in `Assignment 2a/alignment_statistics.csv` and `Assignment 2a/alignment_statistics.json`.

> **Counting Definition (MFA 10 ms Frame-Grid Tolerance):** All duration metrics and threshold counts below apply only to **non-empty phone intervals** (empty/silence intervals are excluded and counted separately). The `±0.01 ms` (`±10 µs`) tolerance accommodates timestamp serialization jitter around MFA's 10 ms frame grid (e.g., `29.998–30.001 ms`), yielding reconciled counts for four tolerance-based classifications:
> - **Shorter than nominal 30 ms (`<29.99 ms`):** `2` intervals (the two 10 ms `AH0` intervals in `F2BJRLP1`).
> - **Nominally 30 ms (`29.99–30.01 ms`, inclusive):** `75` intervals (including the 12 intervals serialized as `30.001 ms`).
> - **Nominally 30 ms or shorter (`≤30.01 ms`):** `77` intervals.
> - **At least nominally 300 ms (`≥299.99 ms`):** `1` interval (`OW1`, `350.0 ms`, in `ISLE_SESS0131_BLOCKD02_02_sprt1`).

### 6.1 Per-Recording Statistics

| Recording | Speaker | Audio Dur (s) | Words (Non-Empty / Total) | Phones (Non-Empty / Total) | Mean / Med Word (ms) | Mean / Med Phone (ms) | Min Phone (ms) | Max Phone (ms) | `<29.99 ms` | `Nominal 30 ms` | `≤30.01 ms` | `≥299.99 ms` | Rate (wpm) | Anomalies |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `F2BJRLP1` | `F2B` | 25.3091 | 70 / 84 | 308 / 322 | 318.0 / 305.0 | 72.3 / 60.0 | 10.0 (`AH0`) | 230.0 (`IY0`) | 2 | 27 | 29 | 0 | 165.95 | 0 |
| `F2BJRLP2` | `F2B` | 28.6474 | 74 / 84 | 345 / 355 | 348.5 / 360.0 | 74.8 / 70.0 | 30.0 (`UW1`) | 240.0 (`AY2`) | 0 | 25 | 25 | 0 | 154.99 | 0 |
| `F2BJRLP3` | `F2B` | 30.7068 | 82 / 95 | 372 / 385 | 335.7 / 320.0 | 74.0 / 70.0 | 30.0 (`IH1`) | 290.0 (`OW1`) | 0 | 23 | 23 | 0 | 160.23 | 0 |
| `ISLE_..._01_sprt1` | `ISLE_SESS0131` | 4.1250 | 5 / 8 | 13 / 16 | 332.0 / 390.0 | 127.7 / 130.0 | 40.0 (`B`) | 210.0 (`EY1`) | 0 | 0 | 0 | 0 | 72.73 | 0 |
| `ISLE_..._02_sprt1` | `ISLE_SESS0131` | 3.8750 | 5 / 8 | 12 / 15 | 314.0 / 320.0 | 130.8 / 100.0 | 50.0 (`D`) | 350.0 (`OW1`) | 0 | 0 | 0 | 1 | 77.42 | 0 |
| `ISLE_..._03_sprt1` | `ISLE_SESS0131` | 4.5000 | 5 / 7 | 13 / 15 | 368.0 / 400.0 | 141.5 / 140.0 | 40.0 (`B`) | 260.0 (`D`) | 0 | 0 | 0 | 0 | 66.67 | 0 |

### 6.2 Group-Level Aggregates

| Group | Recordings | Total Dur (s) | Words (Non-Empty / Sil) | Phones (Non-Empty / Sil) | Mean / Med Word (ms) | Mean / Med Phone (ms) | Min–Max Phone (ms) | `<29.99 ms` | `Nominal 30 ms` | `≤30.01 ms` | `≥299.99 ms` | Rate (wpm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All 6** | 6 | 97.1633 | 241 / 45 | 1,063 / 45 | 334.7 / 320.0 | 75.9 / 70.0 | 10.0 – 350.0 | 2 (0.2%) | 75 (7.1%) | 77 (7.2%) | 1 (0.1%) | 148.82 |
| **F2B** | 3 | 84.6633 | 226 / 37 | 1,025 / 37 | 334.4 / 320.0 | 73.7 / 70.0 | 10.0 – 290.0 | 2 (0.2%) | 75 (7.3%) | 77 (7.5%) | 0 (0.0%) | 160.16 |
| **ISLE_SESS0131** | 3 | 12.5000 | 15 / 8 | 38 / 8 | 338.0 / 390.0 | 133.4 / 115.0 | 40.0 – 350.0 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (2.6%) | 72.00 |

---

## 7. Visualisation Inventory & Qualitative Acoustic Findings

Thirteen 300-DPI figures were generated:
- **6 full-utterance overviews:** `Assignment 2a/visualizations/overview/<recording>_overview.png`
- **7 zoomed inspection regions:** `Assignment 2a/visualizations/detailed/*.png`

Each figure aligns the time-domain waveform, a wideband spectrogram (0–8 kHz, ~8 ms Hann window), the `words` tier, and the `phones` tier on a shared time axis.

### 7.1 Cleanly Aligned Landmarks (Spectrogram-Verified)
- **`F2BJRLP1` — Initialism `S.J.C.'s` (`4.790–5.430 s`, `F2BJRLP1_sjcs_zoom.png`):** Tokenized into `s` (`150 ms`), `j` (`180 ms`), and `c's` (`310 ms`). Sibilant fricatives `S [4.860–4.940 s]`, `S [5.120–5.250 s]`, and `Z [5.350–5.430 s]` coincide closely with dense 4–8 kHz frication bands; `JH [4.940–5.030 s]` captures both closure and postalveolar affricate release; and `EY1` / `IY1` align with clear voiced formant intervals.
- **`F2BJRLP2` — Hyphenated compound `de-politicize` (`5.420–6.510 s`, `F2BJRLP2_depoliticize_zoom.png`):** Split cleanly into `de` (`230 ms`) and `politicize` (`860 ms`). Stop closures/bursts (`D`, `P`, `T`), high-frequency sibilants (`S`, `Z`), and the prominent diphthong `AY2 [6.150–6.390 s]` (`240 ms`) match visible spectrogram transitions.
- **`F2BJRLP2` — Proper name `Maffy` (`23.940–24.330 s`, `F2BJRLP2_maffy_zoom.png`):** `M` (`50 ms`), `AE1` (`120 ms`), `F` (`100 ms`), and `IY0` (`120 ms`) align well with the transition out of `thomas`'s final `S`, the voiced vowel core, diffuse labiodental frication, and the final vowel before `now`.

### 7.2 Suspicious Boundaries & Acoustic Anomalies (Exact Timestamps)

1. **Single-frame phone compression in `WBUR's` (`F2BJRLP1`, `20.560–21.440 s`, `F2BJRLP1_wbur_zoom.png`):**
   - **Flagged interval:** `AH0 [20.730–20.740 s]` (`10.0 ms`, the minimum single-frame step of MFA's 10 ms frame shift), immediately followed by `L [20.740–20.770 s]` (`30.0 ms`).
   - **Spectrogram evidence:** After the bilabial dip of `B [20.680–20.730 s]`, voicing resumes directly into the `-ble-` liquid/glide transition (`L Y UW0`) without a distinct steady-state schwa. In fluent American English broadcast speech, the letter "W" is commonly reduced (`/dʌbju/` or `/dʌbl̩ju/`). Because the custom entry specified a single canonical pronunciation containing `AH0`, MFA was forced to emit `AH0` and compressed it to 10 ms. By contrast, the surrounding landmarks (`D` burst, second `B` closure, `Y UW1 AA1 R` smooth formant trajectory, and final `Z` 4–8 kHz frication at `21.360–21.440 s`) align cleanly.

2. **Utterance-final vowel trailing in `NO` (`ISLE_SESS0131_BLOCKD02_02_sprt1`, `1.750–2.220 s`, `ISLE_02_no_OW1_zoom.png`):**
   - **Flagged interval:** `OW1 [1.870–2.220 s]` (`350.0 ms`, the single phone at least nominally 300 ms (`≥299.99 ms`) across the entire corpus).
   - **Spectrogram evidence:** Vowel voicing onset at `1.870 s` is crisp, but periodic waveform amplitude and low-frequency formant energy visibly decay to near silence around `~2.14–2.15 s`. MFA extends the right boundary of `OW1` by approximately `70–80 ms` into low-energy post-vocalic decay/background floor before transitioning to trailing silence.

3. **Inter-word and utterance-final pause absorption into word-final stops (`ISLE_SESS0131_BLOCKD02_03_sprt1`, `ISLE_03_final_stops_zoom.png`):**
   - **Flagged intervals:**
     - `bad` final `D [0.920–1.170 s]` (`250.0 ms`)
     - `not` final `T [1.330–1.570 s]` (`240.0 ms`)
     - `bed` final `D [1.800–2.060 s]` (`260.0 ms`)
   - **Spectrogram evidence:** Each stop shows a clear closure followed by a sharp release burst (`D` burst at `~1.02–1.06 s`, `T` aspiration burst at `~1.42–1.46 s`, `D` burst at `~1.93–1.98 s`), followed by `~80–110 ms` of near-silence before the next word (`not`, `bed`) or trailing silence. Because MFA did not insert optional inter-word silence intervals between `bad`–`not` or `not`–`bed` (total silence intervals = 2, only leading and trailing), those post-release pauses were absorbed into the final stop intervals.

4. **High-rate phone compression in `F2B` (`77` phones nominally 30 ms or shorter (`≤30.01 ms`) across `F2BJRLP1–3`):**
   - Across `F2B`, `77` of `1,025` non-empty phones (`7.5%`) are nominally 30 ms or shorter (`≤30.01 ms`) — comprising `2` intervals shorter than nominal 30 ms (`<29.99 ms`, both `10.0 ms` single-frame `AH0` vowels in `F2BJRLP1`: `wbur's` `[20.730–20.740 s]` and `will` `[23.440–23.450 s]`) and `75` intervals nominally 30 ms (`29.99–30.01 ms`, inclusive, including the 12 intervals serialized as `30.001 ms`) — typically occurring on unstressed vowels, alveolar/velar taps/stops, and fast consonant clusters in rapid broadcast delivery.

---

## 8. Proper-Name & OOV Pronunciation Limitations

- **Single-pronunciation rigidity (`wbur`, `dukakis`, `maffy`, `melnicove`):** Each custom OOV entry provided only one canonical pronunciation. For `wbur`, omitting a reduced variant without `AH0` directly caused the `10 ms` `AH0` compression.
- **G2P vowel-quality assumptions (`Dukakis`, `F2BJRLP2_dukakis_zoom.png`):** `mfa g2p` mapped the first unstressed syllable of `dukakis` (`2.720–3.360 s`, `640 ms`) to `AH0` (`D AH0 K AA1 K IH0 S`) rather than `UW0` (`/duːˈkɑːkɪs/`). While the spectrogram confirms that `D`, `K`, `AA1`, `K`, `IH0`, and final `S [3.230–3.360 s]` align well to the major acoustic events (stop closures, bursts, stressed vowel, and sibilant), a speaker producing a rounded `/u/` in the first syllable would be better modeled by including both `UW0` and `AH0` variants.
- **Unverified proper-name phonology:** Proper surnames (`Maffy`, `Melnicove`) often follow non-standard or heritage orthography-to-sound rules that generic English G2P models cannot guarantee without speaker-specific listening verification.

---

## 9. Comparative Analysis: `F2B` vs `ISLE_SESS0131`

| Dimension | `F2B` (Broadcast News) | `ISLE_SESS0131` (Minimal-Pair Read Speech) |
|---|---|---|
| Speaking style | Continuous, fluent, fast broadcast prose | Short, deliberate 5-word contrast sentences |
| Effective speaking rate | **160.2 wpm** (`155–166 wpm`) | **72.0 wpm** (`67–77 wpm`, ~2.2× slower) |
| Mean / median phone duration | **73.7 / 70.0 ms** | **133.4 / 115.0 ms** (~1.8× longer) |
| Mean / median word duration | **334.4 / 320.0 ms** | **338.0 / 390.0 ms** (short monosyllables stretched) |
| Short phones (nominally 30 ms or shorter, `≤30.01 ms`) | **77** (`7.5%`: `2` `<29.99 ms`, `75` nominally 30 ms `[29.99–30.01 ms]`) — fast coarticulation & reduction | **0** (`0.0%`) |
| Long phones (at least nominally 300 ms, `≥299.99 ms`) | **0** (`0.0%`) | **1** (`2.6%`, `OW1` = `350 ms`) |
| Dominant alignment risk | Under-duration / frame-floor compression (`10–30 ms`) during fast reduction | Over-duration / boundary trailing into silence and unmodeled inter-word pauses (`240–350 ms`) |

Interestingly, mean word durations are nearly identical (`334.4 ms` vs `338.0 ms`) despite a 2.2× difference in speaking rate because `F2B` contains many multisyllabic broadcast words (`massachusetts`, `administration`, `de-politicize`; `4.54` phones/word on average), whereas `ISLE_SESS0131` consists exclusively of short monosyllables (`i said bad not bed`; `2.53` phones/word on average) pronounced slowly and emphatically.

---

## 10. Technical Validation vs. Perceptual Accuracy

It is essential to distinguish between two levels of alignment quality:
- **Technical / Structural Validation (Passed 100%):** All 6 `.TextGrid` files parse cleanly, contain exact `words` and `phones` tiers, match audio durations to `0.0000 s`, have `0` `<unk>`/`spn` tokens, and have `0` gaps or overlaps.
- **Perceptual / Phonetic Accuracy (Partially Evaluated Visually; Auditory Check Pending):** Visual spectrogram inspection confirms strong macro-level and landmark-level alignment on sibilants, stop bursts, and stressed vowels, while revealing specific micro-level boundary artifacts (`10 ms` `AH0` compression in `wbur's`, `~70–80 ms` right-edge trailing on `OW1` in `ISLE 02`, and `~80–110 ms` post-release pause absorption on final stops in `ISLE 03`). True perceptual accuracy can only be established by listening to each aligned segment in Praat (or an equivalent audio editor).

---

## 11. Limitations & Recommended Improvements

1. **Direct Praat auditory & visual verification:** Open all 6 `.wav` + `.TextGrid` pairs in Praat to listen to flagged intervals (`wbur's`, `dukakis`, `maffy`, `melnicove`, `ISLE 02` `NO`, `ISLE 03` `BAD`/`NOT`/`BED`) and confirm whether any manual boundary refinement is warranted.
2. **Add pronunciation variants to the custom dictionary:** Include reduced/alternative pronunciations (e.g., `wbur` without `AH0`: `D AH1 B L Y UW0 B IY1 Y UW1 AA1 R`; `dukakis` with `UW0`: `D UW0 K AA1 K IH0 S`) so MFA can select the acoustically best-matching variant.
3. **Evaluate a second acoustic model (extra-credit direction):** Compare `english_us_arpa` against `english_mfa` (IPA phone set with finer allophonic modeling, such as flaps and syllabic consonants) to test whether frame-floor compressions in `F2B` and stop-release trailing in `ISLE` decrease.
4. **Silence-modeling sensitivity for slow read speech:** For corpora like `ISLE_SESS0131` with deliberate pauses between words, evaluating MFA's silence probability / beam parameters or explicitly marking short pauses can prevent final stops from absorbing `80–110 ms` of post-release silence.
