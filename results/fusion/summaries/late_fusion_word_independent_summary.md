# Late Fusion — Word-Independent 3-Seed Summary (`word_indep_split`)

Parameter-free decision-level probability ensemble (`WavLM + BiGRU + Attention` & `DistilBERT + BiLSTM + Attention`) calibrated strictly on validation data across 21 $\lambda \in [0.00, 1.00]$ candidates (step `0.05`).

## Per-Seed Results (`N_test = 420`)

| Seed | Selected $\lambda$ | Val Macro-F1 | Val Log-Loss | Test Accuracy | Test Macro-P | Test Macro-R | Test Macro-F1 | Test Weighted-F1 | Test Log-Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `42` | `1.00` | `1.000000` | `0.088125` | `0.997619` (`99.76%`) | `0.997658` | `0.997619` | `0.997619` (`99.76%`) | `0.997619` | `0.092664` |
| `123` | `1.00` | `1.000000` | `0.087420` | `0.997619` (`99.76%`) | `0.997658` | `0.997619` | `0.997619` (`99.76%`) | `0.997619` | `0.095605` |
| `999` | `1.00` | `1.000000` | `0.087870` | `0.997619` (`99.76%`) | `0.997658` | `0.997619` | `0.997619` (`99.76%`) | `0.997619` | `0.092344` |

## Three-Seed Aggregate (Mean ± Sample Std)

- **Test Accuracy:** **99.76 ± 0.00%** (`0.997619 ± 0.000000`)
- **Test Macro-F1:** **99.76 ± 0.00%** (`0.997619 ± 0.000000`)
- **Test Log-Loss:** `0.093538 ± 0.001797`
- **Selected $\lambda$:** `1.00` across all 3 seeds (validation log-loss tie-breaker consistently suppresses chance-level TESS text probability noise).
