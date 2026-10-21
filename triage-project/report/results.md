# Triage results

## Calibration (fit on the calibration split)

| Approach | n | Accuracy | Temperature | ECE before → after | Naive threshold | Risk-controlled threshold | Conformal q̂ (α=0.1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | 495 | 77.0 | 0.58 | 0.113 → 0.031 | 0.761 | 0.975 | 0.778 |
| Embedding + LR | 495 | 68.7 | 0.64 | 0.090 → 0.040 | 0.803 | 0.910 | 0.795 |
| LLM zero-shot | 495 | 68.3 | 10.11 | 0.296 → 0.050 | 0.831 | inf | 0.873 |
| LLM few-shot (retrieval) | 495 | 73.3 | 7.61 | 0.233 → 0.076 | 0.814 | 0.921 | 0.851 |

## Accuracy

Macro-F1 and cross-repo F1 (mean of per-repository macro-F1, the NLBSE'24 ranking metric).

| Approach | Condition | n | Accuracy [95% CI] | Macro-F1 | Cross-repo F1 | ECE (calibrated) | Prompt tokens | s / issue |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | NLBSE'24 test | 1500 | 76.2 [74.0, 78.3] | 0.762 | 0.760 | 0.042 | 0 | 0.00 |
| TF-IDF + LR | Fresh 2025-26 | 514 | 67.5 [63.6, 71.8] | 0.670 | 0.584 | 0.081 | 0 | 0.00 |
| TF-IDF + LR | New repository | 750 | 60.3 [56.7, 63.7] | 0.596 | 0.589 | 0.066 | 0 | 0.00 |
| Embedding + LR | NLBSE'24 test | 1500 | 69.6 [67.3, 71.9] | 0.695 | 0.694 | 0.026 | 0 | 0.00 |
| Embedding + LR | Fresh 2025-26 | 514 | 68.1 [64.2, 72.2] | 0.669 | 0.640 | 0.037 | 0 | 0.00 |
| Embedding + LR | New repository | 750 | 58.8 [55.2, 62.4] | 0.583 | 0.576 | 0.088 | 0 | 0.00 |
| LLM zero-shot | NLBSE'24 test | 1500 | 68.5 [66.1, 70.7] | 0.669 | 0.663 | 0.036 | 414 | 2.47 |
| LLM zero-shot | Fresh 2025-26 | 514 | 71.2 [67.1, 74.9] | 0.670 | 0.650 | 0.063 | 477 | 2.98 |
| LLM zero-shot | New repository | 750 | 68.4 [65.1, 71.7] | 0.668 | 0.661 | 0.036 | 409 | 2.45 |
| LLM few-shot (retrieval) | NLBSE'24 test | 1500 | 70.9 [68.6, 73.3] | 0.696 | 0.694 | 0.049 | 1062 | 5.91 |
| LLM few-shot (retrieval) | Fresh 2025-26 | 514 | 75.3 [71.6, 79.0] | 0.723 | 0.682 | 0.034 | 1137 | 6.90 |
| LLM few-shot (retrieval) | New repository | 750 | 68.0 [64.7, 71.3] | 0.662 | 0.657 | 0.061 | 1088 | 5.80 |

## Selective automation (target: at most 10% errors among auto-labeled issues)

Naive threshold = lowest threshold with ≤10% calibration error; risk-controlled = Learn-then-Test with a 95% Clopper-Pearson bound. Thresholds and conformal sets are calibrated on the calibration split and applied unchanged. Agreement = auto-label only when the LLM few-shot and embedding classifiers agree. Cells: coverage % / risk % (errors among auto-labeled).

| Approach | Policy | NLBSE'24 test | Fresh 2025-26 | New repository |
|---|---|---:|---:|---:|
| TF-IDF + LR | auto-all | 100.0 / 23.8 | 100.0 / 32.5 | 100.0 / 39.7 |
| TF-IDF + LR | threshold | 54.9 / 12.5 | 48.8 / 19.1 | 33.5 / 26.3 |
| TF-IDF + LR | risk-controlled | 15.6 / 4.3 | 12.5 / 9.4 | 1.5 / 18.2 |
| TF-IDF + LR | conformal | 61.9 / 14.5 | 60.7 / 23.7 | 46.5 / 28.9 |
| Embedding + LR | auto-all | 100.0 / 30.4 | 100.0 / 31.9 | 100.0 / 41.2 |
| Embedding + LR | threshold | 33.7 / 11.3 | 29.8 / 13.7 | 22.3 / 22.8 |
| Embedding + LR | risk-controlled | 13.5 / 5.9 | 11.5 / 5.1 | 8.0 / 11.7 |
| Embedding + LR | conformal | 47.7 / 15.7 | 44.7 / 18.3 | 38.5 / 28.7 |
| LLM zero-shot | auto-all | 100.0 / 31.5 | 100.0 / 28.8 | 100.0 / 31.6 |
| LLM zero-shot | threshold | 13.1 / 12.2 | 24.9 / 7.8 | 10.9 / 14.6 |
| LLM zero-shot | risk-controlled | 0.0 / nan | 0.0 / nan | 0.0 / nan |
| LLM zero-shot | conformal | 29.5 / 17.8 | 39.3 / 11.4 | 27.1 / 17.2 |
| LLM few-shot (retrieval) | auto-all | 100.0 / 29.1 | 100.0 / 24.7 | 100.0 / 32.0 |
| LLM few-shot (retrieval) | threshold | 42.9 / 14.9 | 51.9 / 14.2 | 40.3 / 20.5 |
| LLM few-shot (retrieval) | risk-controlled | 6.9 / 3.9 | 8.0 / 0.0 | 1.2 / 11.1 |
| LLM few-shot (retrieval) | conformal | 52.8 / 17.9 | 59.5 / 16.3 | 50.5 / 22.7 |
| LLM few-shot + Embedding | agreement | 67.0 / 17.8 | 71.4 / 17.7 | 63.6 / 26.4 |

Conformal prediction-set coverage (target ≥ 90%): TF-IDF + LR: NLBSE'24 test 89.3%, Fresh 2025-26 82.7%, New repository 80.0%; Embedding + LR: NLBSE'24 test 88.8%, Fresh 2025-26 87.7%, New repository 83.5%; LLM zero-shot: NLBSE'24 test 88.1%, Fresh 2025-26 88.7%, New repository 88.5%; LLM few-shot (retrieval): NLBSE'24 test 87.5%, Fresh 2025-26 87.7%, New repository 86.3%

## Robustness to meaning-preserving rewording

Flip = the label differs from the label on the original issue. Auto flips = both versions auto-labeled (calibrated threshold) with different labels. Accuracy: original → variant.

| Approach | Rewording | n | Flip % | Auto flips % | Accuracy % | → question % |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | no code / stack trace | 77 | 13.0 | 1.3 | 76.6 → 75.3 | 3.9 |
| TF-IDF + LR | no template | 112 | 13.4 | 0.0 | 75.9 → 80.4 | 6.2 |
| TF-IDF + LR | newcomer cue | 210 | 7.1 | 0.0 | 78.6 → 75.7 | 0.5 |
| TF-IDF + LR | non-native English | 159 | 13.2 | 0.0 | 78.0 → 77.4 | 9.4 |
| TF-IDF + LR | non-native + no code | 159 | 17.0 | 0.6 | 78.0 → 74.2 | 11.9 |
| Embedding + LR | no code / stack trace | 77 | 5.2 | 0.0 | 64.9 → 59.7 | 2.6 |
| Embedding + LR | no template | 112 | 12.5 | 0.0 | 70.5 → 70.5 | 4.5 |
| Embedding + LR | newcomer cue | 210 | 6.2 | 0.0 | 72.9 → 75.7 | 3.3 |
| Embedding + LR | non-native English | 159 | 8.2 | 0.0 | 71.1 → 68.6 | 3.1 |
| Embedding + LR | non-native + no code | 159 | 9.4 | 0.0 | 71.1 → 67.3 | 4.4 |
| LLM zero-shot | no code / stack trace | 77 | 3.9 | 0.0 | 70.1 → 66.2 | 1.3 |
| LLM zero-shot | no template | 112 | 0.0 | 0.0 | 66.1 → 66.1 | 0.0 |
| LLM zero-shot | newcomer cue | 210 | 5.7 | 0.0 | 70.0 → 67.6 | 4.8 |
| LLM zero-shot | non-native English | 159 | 6.9 | 0.0 | 71.1 → 64.2 | 3.1 |
| LLM zero-shot | non-native + no code | 159 | 10.1 | 0.0 | 71.1 → 61.0 | 4.4 |
| LLM few-shot (retrieval) | no code / stack trace | 77 | 6.5 | 0.0 | 68.8 → 68.8 | 2.6 |
| LLM few-shot (retrieval) | no template | 112 | 3.6 | 0.0 | 70.5 → 68.8 | 3.6 |
| LLM few-shot (retrieval) | newcomer cue | 210 | 16.2 | 0.0 | 73.3 → 70.0 | 14.8 |
| LLM few-shot (retrieval) | non-native English | 159 | 7.5 | 0.0 | 74.2 → 69.8 | 3.1 |
| LLM few-shot (retrieval) | non-native + no code | 159 | 9.4 | 0.0 | 74.2 → 69.2 | 3.1 |
