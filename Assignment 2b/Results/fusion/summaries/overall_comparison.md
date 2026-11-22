# Assignment 2B — Unified Comparative Tables (27 Frozen Experiments)

All metrics are computed directly from saved experiment artifacts across 3 random seeds (`42`, `123`, `999`) with sample standard deviation (`ddof=1`). Weighted-F1 and Log-Loss for speech and text runs are computed directly from saved `test_predictions.csv`.

## Table 1 — Overall Modality Comparison (Mean ± Std across 3 Seeds)

| Split | Architecture | Trainable Params | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Log-Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `word_indep_split` | WavLM + BiGRU + Attn (Speech) | `857,863` | 99.76 ± 0.00% | 99.77 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 0.0935 ± 0.0018 |
| `word_indep_split` | MFCC + CNN-BiLSTM (Speech) | `1,076,279` | 99.68 ± 0.14% | 99.69 ± 0.14% | 99.68 ± 0.14% | 99.68 ± 0.14% | 99.68 ± 0.14% | 0.0921 ± 0.0044 |
| `word_indep_split` | DistilBERT + BiLSTM + Attn (Text) | `1,055,239` | 14.29 ± 0.00% | 2.04 ± 0.00% | 14.29 ± 0.00% | 3.57 ± 0.00% | 3.57 ± 0.00% | 1.9459 ± 0.0000 |
| `word_indep_split` | Gated Multimodal Fusion | `296,967` | 99.76 ± 0.00% | 99.77 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 0.0941 ± 0.0022 |
| `word_indep_split` | Concatenation Fusion | `165,895` | 99.76 ± 0.00% | 99.77 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 0.0888 ± 0.0037 |
| `word_indep_split` | Late-Logit Fusion | `0` | 99.76 ± 0.00% | 99.77 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 0.0935 ± 0.0018 |
| `cross_speaker_split` | WavLM + BiGRU + Attn (Speech) | `857,863` | 40.12 ± 10.07% | 34.51 ± 8.96% | 40.12 ± 10.07% | 33.59 ± 10.95% | 33.59 ± 10.95% | 1.9507 ± 0.4001 |
| `cross_speaker_split` | MFCC + CNN-BiLSTM (Speech) | `1,076,279` | 43.67 ± 7.78% | 53.44 ± 6.42% | 43.67 ± 7.78% | 42.62 ± 7.26% | 42.62 ± 7.26% | 2.0644 ± 0.1748 |
| `cross_speaker_split` | DistilBERT + BiLSTM + Attn (Text) | `1,055,239` | 14.29 ± 0.00% | 2.04 ± 0.00% | 14.29 ± 0.00% | 3.57 ± 0.00% | 3.57 ± 0.00% | 1.9460 ± 0.0000 |

## Table 2 — Speech Architecture Comparison (WavLM vs. MFCC)

| Split | Speech Architecture | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Log-Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `word_indep_split` | WavLM + BiGRU + Attn (Speech) | 99.76 ± 0.00% | 99.77 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 0.0935 ± 0.0018 |
| `word_indep_split` | MFCC + CNN-BiLSTM (Speech) | 99.68 ± 0.14% | 99.69 ± 0.14% | 99.68 ± 0.14% | 99.68 ± 0.14% | 99.68 ± 0.14% | 0.0921 ± 0.0044 |
| `cross_speaker_split` | WavLM + BiGRU + Attn (Speech) | 40.12 ± 10.07% | 34.51 ± 8.96% | 40.12 ± 10.07% | 33.59 ± 10.95% | 33.59 ± 10.95% | 1.9507 ± 0.4001 |
| `cross_speaker_split` | MFCC + CNN-BiLSTM (Speech) | 43.67 ± 7.78% | 53.44 ± 6.42% | 43.67 ± 7.78% | 42.62 ± 7.26% | 42.62 ± 7.26% | 2.0644 ± 0.1748 |

## Table 3 — Text-Only Results (`DistilBERT + BiLSTM + Attention`)

Every one of the 200 TESS target words is spoken identically across all 7 emotions inside the carrier phrase `"Say the word <word>"`. Consequently, lexical transcripts carry zero mutual information with acted emotion ($I(\text{Text}; \text{Emotion}) = 0$), yielding exact $1/7 \approx 14.29\%$ chance accuracy and near-uniform predicted probabilities (~`0.1429` per class, log-loss $\approx \ln(7) \approx 1.9459$) across every seed and split.

| Split | Seed | Test Samples | Accuracy | Macro-F1 | Weighted-F1 | Log-Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `word_indep_split` | `42` | `420` | `14.29%` | `3.57%` | `3.57%` | `1.9460` |
| `word_indep_split` | `123` | `420` | `14.29%` | `3.57%` | `3.57%` | `1.9459` |
| `word_indep_split` | `999` | `420` | `14.29%` | `3.57%` | `3.57%` | `1.9459` |
| `cross_speaker_split` | `42` | `1400` | `14.29%` | `3.57%` | `3.57%` | `1.9460` |
| `cross_speaker_split` | `123` | `1400` | `14.29%` | `3.57%` | `3.57%` | `1.9460` |
| `cross_speaker_split` | `999` | `1400` | `14.29%` | `3.57%` | `3.57%` | `1.9461` |

## Table 4 — Multimodal Fusion Comparison (`word_indep_split`, 3 Seeds)

> *Note:* Gate activation statistics for `GatedMultimodalFusion` describe empirical layer outputs on test embeddings and are not interpreted as definitive causal modality importance.

| Fusion Architecture | Trainable Params | Accuracy | Macro-F1 | Weighted-F1 | Log-Loss | Extra Calibration / Gate Stats |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Gated Multimodal Fusion | `296,967` | 99.76 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 0.0941 ± 0.0022 | Mean gate across seeds: `0.6795 ± 0.0194` |
| Concatenation Fusion | `165,895` | 99.76 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 0.0888 ± 0.0037 | — |
| Late-Logit Fusion | `0` | 99.76 ± 0.00% | 99.76 ± 0.00% | 99.76 ± 0.00% | 0.0935 ± 0.0018 | Selected $\lambda$: `1.00,1.00,1.00` |

## Table 5 — Cross-Speaker Comparison (`OAF -> YAF`, Existing Unimodal Runs Only)

*Cross-speaker multimodal fusion experiments were not executed (optional) and are omitted rather than fabricated.*

| Architecture | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Log-Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| WavLM + BiGRU + Attn (Speech) | 40.12 ± 10.07% | 34.51 ± 8.96% | 40.12 ± 10.07% | 33.59 ± 10.95% | 33.59 ± 10.95% | 1.9507 ± 0.4001 |
| MFCC + CNN-BiLSTM (Speech) | 43.67 ± 7.78% | 53.44 ± 6.42% | 43.67 ± 7.78% | 42.62 ± 7.26% | 42.62 ± 7.26% | 2.0644 ± 0.1748 |
| DistilBERT + BiLSTM + Attn (Text) | 14.29 ± 0.00% | 2.04 ± 0.00% | 14.29 ± 0.00% | 3.57 ± 0.00% | 3.57 ± 0.00% | 1.9460 ± 0.0000 |
