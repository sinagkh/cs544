# Triage results

## Calibration

Calibration rows are split into a fit part (temperature, threshold grid) and a certification part (thresholds, conformal quantile). Risk-controlled thresholds: Learn-then-Test with fixed-sequence testing over the pre-specified grid and exact binomial tests at δ = 0.05.

| Approach | n fit / cert | Accuracy (cert) | Temperature | ECE raw → calibrated (cert) | Naive threshold | Risk-controlled (10%) | Conformal q̂ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | 165 / 330 | 79.1 | 0.69 | 0.137 → 0.072 | 0.673 | 0.971 | 0.712 |
| Embedding + LR | 165 / 330 | 71.5 | 0.70 | 0.112 → 0.084 | 0.778 | inf | 0.773 |
| LLM zero-shot | 165 / 330 | 67.9 | 10.35 | 0.303 → 0.046 | 0.826 | inf | 0.862 |
| LLM few-shot (retrieval) | 165 / 330 | 74.5 | 7.86 | 0.232 → 0.074 | 0.827 | inf | 0.831 |

## Accuracy

Cross-repo F1 = mean of per-repository macro-F1 (the NLBSE'24 ranking metric).

| Approach | Condition | n | Accuracy [95% CI] | Macro-F1 | Cross-repo F1 | ECE | Prompt tokens | s / issue |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | NLBSE'24 test | 1500 | 75.5 [73.4, 77.6] | 0.755 | 0.754 | 0.032 | 0 | 0.00 |
| TF-IDF + LR | Fresh 2025-26 | 514 | 66.3 [62.4, 70.4] | 0.660 | 0.583 | 0.054 | 0 | 0.00 |
| TF-IDF + LR | New repository | 750 | 60.4 [56.7, 63.6] | 0.599 | 0.592 | 0.038 | 0 | 0.00 |
| Embedding + LR | NLBSE'24 test | 1500 | 69.3 [67.0, 71.6] | 0.692 | 0.691 | 0.017 | 0 | 0.00 |
| Embedding + LR | Fresh 2025-26 | 514 | 68.5 [64.4, 72.4] | 0.670 | 0.630 | 0.050 | 0 | 0.00 |
| Embedding + LR | New repository | 750 | 57.5 [53.9, 61.2] | 0.569 | 0.561 | 0.068 | 0 | 0.00 |
| LLM zero-shot | NLBSE'24 test | 1500 | 68.5 [66.1, 70.7] | 0.669 | 0.663 | 0.043 | 414 | 2.47 |
| LLM zero-shot | Fresh 2025-26 | 514 | 71.2 [67.1, 74.9] | 0.670 | 0.650 | 0.079 | 477 | 2.98 |
| LLM zero-shot | New repository | 750 | 68.4 [65.1, 71.7] | 0.668 | 0.661 | 0.041 | 409 | 2.45 |
| LLM few-shot (retrieval) | NLBSE'24 test | 1500 | 70.9 [68.6, 73.3] | 0.696 | 0.694 | 0.041 | 1062 | 5.91 |
| LLM few-shot (retrieval) | Fresh 2025-26 | 514 | 75.3 [71.6, 79.0] | 0.723 | 0.682 | 0.035 | 1137 | 6.90 |
| LLM few-shot (retrieval) | New repository | 750 | 68.0 [64.7, 71.3] | 0.662 | 0.657 | 0.055 | 1088 | 5.80 |

Fresh 2025-26 sample label mix: bug 188, feature 169, question 157 (the benchmark is balanced).

## Selective automation (target: at most 10% errors among auto-labeled issues)

Cells: coverage % / risk % (n auto-labeled, errors). 'New repository' reuses the in-distribution calibration; 'LORO calib.' recalibrates on the other repositories' calibration issues scored cross-repository.

| Approach | Policy | NLBSE'24 test | Fresh 2025-26 | New repository | New repo (LORO calib.) |
|---|---|---:|---:|---:|---:|
| TF-IDF + LR | auto-all | 100.0 / 24.5 (1500, 367) | 100.0 / 33.7 (514, 173) | 100.0 / 39.6 (750, 297) | 100.0 / 39.6 (750, 297) |
| TF-IDF + LR | naive threshold | 60.1 / 12.5 (902, 113) | 53.1 / 20.1 (273, 55) | 39.2 / 27.6 (294, 81) | 10.0 / 24.0 (75, 18) |
| TF-IDF + LR | risk-controlled | 11.3 / 3.5 (170, 6) | 6.8 / 5.7 (35, 2) | 0.4 / 33.3 (3, 1) | 0.0 / – (0, 0) |
| TF-IDF + LR | conformal | 68.1 / 15.7 (1021, 160) | 67.9 / 23.5 (349, 82) | 58.3 / 32.0 (437, 140) | 25.1 / 23.9 (188, 45) |
| Embedding + LR | auto-all | 100.0 / 30.7 (1500, 460) | 100.0 / 31.5 (514, 162) | 100.0 / 42.5 (750, 319) | 100.0 / 42.5 (750, 319) |
| Embedding + LR | naive threshold | 31.9 / 10.2 (479, 49) | 27.8 / 11.9 (143, 17) | 20.5 / 22.1 (154, 34) | 0.0 / – (0, 0) |
| Embedding + LR | risk-controlled | 0.0 / – (0, 0) | 0.0 / – (0, 0) | 0.0 / – (0, 0) | 0.0 / – (0, 0) |
| Embedding + LR | conformal | 48.9 / 16.3 (734, 120) | 44.2 / 18.1 (227, 41) | 38.1 / 28.7 (286, 82) | 17.1 / 19.5 (128, 25) |
| LLM zero-shot | auto-all | 100.0 / 31.5 (1500, 473) | 100.0 / 28.8 (514, 148) | 100.0 / 31.6 (750, 237) | 100.0 / 31.6 (750, 237) |
| LLM zero-shot | naive threshold | 12.4 / 12.4 (186, 23) | 23.7 / 8.2 (122, 10) | 10.5 / 13.9 (79, 11) | 13.1 / 15.3 (98, 15) |
| LLM zero-shot | risk-controlled | 0.0 / – (0, 0) | 0.0 / – (0, 0) | 0.0 / – (0, 0) | 0.0 / – (0, 0) |
| LLM zero-shot | conformal | 33.1 / 19.1 (497, 95) | 43.8 / 12.4 (225, 28) | 30.5 / 18.8 (229, 43) | 31.5 / 19.9 (236, 47) |
| LLM few-shot (retrieval) | auto-all | 100.0 / 29.1 (1500, 437) | 100.0 / 24.7 (514, 127) | 100.0 / 32.0 (750, 240) | 100.0 / 32.0 (750, 240) |
| LLM few-shot (retrieval) | naive threshold | 36.6 / 13.3 (549, 73) | 44.6 / 10.0 (229, 23) | 32.0 / 19.2 (240, 46) | 19.2 / 20.8 (144, 30) |
| LLM few-shot (retrieval) | risk-controlled | 0.0 / – (0, 0) | 0.0 / – (0, 0) | 0.0 / – (0, 0) | 0.0 / – (0, 0) |
| LLM few-shot (retrieval) | conformal | 57.2 / 18.5 (858, 159) | 63.6 / 17.1 (327, 56) | 55.3 / 23.9 (415, 99) | 38.5 / 20.1 (289, 58) |
| LLM few-shot + Embedding | agreement | 67.4 / 18.2 | 72.8 / 17.6 | 62.0 / 26.9 | – |

Conformal prediction-set coverage (target ≥ 90%): TF-IDF + LR: NLBSE'24 test 87.5%, Fresh 2025-26 80.0%, New repository 75.6%, New repo (LORO calib.) 87.9%; Embedding + LR: NLBSE'24 test 88.0%, Fresh 2025-26 87.9%, New repository 83.5%, New repo (LORO calib.) 92.0%; LLM zero-shot: NLBSE'24 test 87.7%, Fresh 2025-26 88.5%, New repository 88.3%, New repo (LORO calib.) 88.1%; LLM few-shot (retrieval): NLBSE'24 test 86.9%, Fresh 2025-26 86.4%, New repository 84.3%, New repo (LORO calib.) 88.3%

### Risk-control frontier

Coverage % / actual risk % when the risk-controlled threshold is certified for each target (δ = 0.05).

| Approach | Target | NLBSE'24 test | Fresh 2025-26 | New repository | New repo (LORO calib.) |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | 10% | 11.3 / 3.5 | 6.8 / 5.7 | 0.4 / 33.3 | 0.0 / – |
| TF-IDF + LR | 15% | 17.4 / 4.2 | 14.4 / 9.5 | 1.7 / 15.4 | 0.0 / – |
| TF-IDF + LR | 20% | 79.9 / 19.6 | 69.6 / 23.5 | 62.3 / 32.3 | 2.1 / 12.5 |
| TF-IDF + LR | 25% | 99.9 / 24.4 | 100.0 / 33.7 | 99.6 / 39.6 | 17.6 / 27.3 |
| Embedding + LR | 10% | 0.0 / – | 0.0 / – | 0.0 / – | 0.0 / – |
| Embedding + LR | 15% | 0.0 / – | 0.0 / – | 0.0 / – | 0.0 / – |
| Embedding + LR | 20% | 38.1 / 12.9 | 33.9 / 16.1 | 27.9 / 23.0 | 0.0 / – |
| Embedding + LR | 25% | 68.4 / 22.4 | 63.8 / 24.1 | 56.1 / 33.5 | 3.1 / 21.7 |
| LLM zero-shot | 10% | 0.0 / – | 0.0 / – | 0.0 / – | 0.0 / – |
| LLM zero-shot | 15% | 0.0 / – | 0.0 / – | 0.0 / – | 0.0 / – |
| LLM zero-shot | 20% | 0.0 / – | 0.0 / – | 0.0 / – | 2.8 / 4.8 |
| LLM zero-shot | 25% | 58.9 / 23.2 | 71.6 / 20.1 | 57.6 / 23.6 | 22.7 / 26.5 |
| LLM few-shot (retrieval) | 10% | 0.0 / – | 0.0 / – | 0.0 / – | 0.0 / – |
| LLM few-shot (retrieval) | 15% | 0.0 / – | 0.0 / – | 0.0 / – | 0.0 / – |
| LLM few-shot (retrieval) | 20% | 62.9 / 19.1 | 69.8 / 17.3 | 60.3 / 24.3 | 0.0 / – |
| LLM few-shot (retrieval) | 25% | 78.0 / 22.4 | 82.1 / 19.7 | 78.0 / 26.7 | 11.6 / 24.1 |

## Robustness to rewording

Flip = label differs from the original's. Auto flips = both versions auto-labeled (naive threshold) with different labels. Auto-labeled = share auto-labeled under the naive threshold, original → variant (the rest goes to a maintainer). 'No code' removes information and is reported as reduced input, not as an invariance test.

| Approach | Rewording | n | Flip % [95% CI] | Auto flips % | Accuracy % | → question % | Auto-labeled % |
|---|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | no code / stack trace | 77 | 11.7 [5.2, 19.5] | 1.3 | 77.9 → 75.3 | 2.6 | 63.6 → 61.0 |
| TF-IDF + LR | no template | 112 | 12.5 [7.1, 18.8] | 0.0 | 76.8 → 77.7 | 4.5 | 70.5 → 65.2 |
| TF-IDF + LR | newcomer sentence (full) | 210 | 6.2 [3.3, 9.5] | 0.0 | 78.6 → 75.7 | 2.4 | 59.5 → 55.2 |
| TF-IDF + LR | non-native English | 159 | 13.2 [8.2, 18.9] | 0.0 | 78.6 → 74.2 | 9.4 | 62.9 → 59.7 |
| TF-IDF + LR | non-native + no code | 159 | 17.0 [11.3, 22.6] | 1.9 | 78.6 → 71.7 | 13.2 | 62.9 → 61.6 |
| TF-IDF + LR | newcomer status only | 210 | 1.4 [0.0, 3.3] | 0.0 | 78.6 → 77.6 | 0.0 | 59.5 → 57.6 |
| TF-IDF + LR | apology only | 210 | 6.7 [3.3, 10.0] | 0.0 | 78.6 → 76.7 | 1.0 | 59.5 → 59.5 |
| TF-IDF + LR | 'silly question' only | 210 | 3.8 [1.4, 6.7] | 0.0 | 78.6 → 76.7 | 3.3 | 59.5 → 59.0 |
| TF-IDF + LR | neutral sentence | 210 | 7.1 [3.8, 10.5] | 0.0 | 78.6 → 74.8 | 6.7 | 59.5 → 59.0 |
| Embedding + LR | no code / stack trace | 77 | 5.2 [1.3, 10.4] | 0.0 | 64.9 → 59.7 | 3.9 | 32.5 → 42.9 |
| Embedding + LR | no template | 112 | 14.3 [8.0, 21.4] | 0.0 | 70.5 → 70.5 | 6.2 | 37.5 → 39.3 |
| Embedding + LR | newcomer sentence (full) | 210 | 8.6 [5.2, 12.4] | 0.0 | 72.9 → 75.7 | 5.7 | 35.7 → 37.6 |
| Embedding + LR | non-native English | 159 | 10.1 [5.7, 15.1] | 0.0 | 71.7 → 69.2 | 5.7 | 32.7 → 33.3 |
| Embedding + LR | non-native + no code | 159 | 10.7 [6.3, 15.7] | 0.0 | 71.7 → 67.3 | 6.3 | 32.7 → 39.0 |
| Embedding + LR | newcomer status only | 210 | 8.6 [4.8, 12.4] | 0.0 | 72.9 → 73.8 | 5.2 | 35.7 → 37.6 |
| Embedding + LR | apology only | 210 | 5.2 [2.4, 8.6] | 0.0 | 72.9 → 74.3 | 2.9 | 35.7 → 36.7 |
| Embedding + LR | 'silly question' only | 210 | 6.2 [3.3, 10.0] | 0.0 | 72.9 → 73.8 | 4.8 | 35.7 → 35.2 |
| Embedding + LR | neutral sentence | 210 | 8.1 [4.8, 11.9] | 0.0 | 72.9 → 73.3 | 5.2 | 35.7 → 36.7 |
| LLM zero-shot | no code / stack trace | 77 | 3.9 [0.0, 9.1] | 0.0 | 70.1 → 66.2 | 1.3 | 16.9 → 15.6 |
| LLM zero-shot | no template | 112 | 0.0 [0.0, 0.0] | 0.0 | 66.1 → 66.1 | 0.0 | 16.1 → 13.4 |
| LLM zero-shot | newcomer sentence (full) | 210 | 5.7 [2.9, 9.0] | 0.0 | 70.0 → 67.6 | 4.8 | 10.0 → 4.3 |
| LLM zero-shot | non-native English | 159 | 6.9 [3.1, 10.7] | 0.0 | 71.1 → 64.2 | 3.1 | 10.7 → 11.9 |
| LLM zero-shot | non-native + no code | 159 | 10.1 [5.7, 15.1] | 0.0 | 71.1 → 61.0 | 4.4 | 10.7 → 8.8 |
| LLM zero-shot | newcomer status only | 210 | 1.9 [0.5, 3.8] | 0.0 | 70.0 → 69.0 | 0.5 | 10.0 → 10.0 |
| LLM zero-shot | apology only | 210 | 1.4 [0.0, 3.3] | 0.0 | 70.0 → 68.6 | 0.0 | 10.0 → 9.5 |
| LLM zero-shot | 'silly question' only | 210 | 7.6 [4.3, 11.4] | 0.0 | 70.0 → 66.7 | 6.7 | 10.0 → 3.8 |
| LLM zero-shot | neutral sentence | 210 | 1.9 [0.5, 3.8] | 0.0 | 70.0 → 68.1 | 0.0 | 10.0 → 10.5 |
| LLM few-shot (retrieval) | no code / stack trace | 77 | 6.5 [1.3, 13.0] | 0.0 | 68.8 → 68.8 | 2.6 | 36.4 → 29.9 |
| LLM few-shot (retrieval) | no template | 112 | 3.6 [0.9, 7.1] | 0.0 | 70.5 → 68.8 | 3.6 | 44.6 → 42.0 |
| LLM few-shot (retrieval) | newcomer sentence (full) | 210 | 16.2 [11.4, 21.4] | 0.0 | 73.3 → 70.0 | 14.8 | 33.8 → 16.2 |
| LLM few-shot (retrieval) | non-native English | 159 | 7.5 [3.8, 11.9] | 0.0 | 74.2 → 69.8 | 3.1 | 35.8 → 32.1 |
| LLM few-shot (retrieval) | non-native + no code | 159 | 9.4 [5.0, 13.8] | 0.0 | 74.2 → 69.2 | 3.1 | 35.8 → 29.6 |
| LLM few-shot (retrieval) | newcomer status only | 210 | 6.2 [2.9, 9.5] | 0.0 | 73.3 → 71.9 | 4.3 | 33.8 → 29.5 |
| LLM few-shot (retrieval) | apology only | 210 | 3.8 [1.4, 6.7] | 0.0 | 73.3 → 72.4 | 1.4 | 33.8 → 31.0 |
| LLM few-shot (retrieval) | 'silly question' only | 210 | 17.1 [12.4, 22.4] | 0.0 | 73.3 → 68.6 | 16.2 | 33.8 → 13.3 |
| LLM few-shot (retrieval) | neutral sentence | 210 | 4.3 [1.9, 7.1] | 0.0 | 73.3 → 70.0 | 1.4 | 33.8 → 32.4 |

### Which part of the newcomer sentence matters?

Paired by issue: difference from the neutral sentence, in percentage points [95% bootstrap CI].

| Approach | Sentence | Flip % | Δ flip vs neutral | → question % | Δ → question vs neutral | Auto-labeled % |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | newcomer sentence (full) | 6.2 | -1.0 [-4.3, +2.4] | 2.4 | -4.3 [-7.1, -1.9] | 55.2 |
| TF-IDF + LR | newcomer status only | 1.4 | -5.7 [-9.0, -2.4] | 0.0 | -6.7 [-10.0, -3.8] | 57.6 |
| TF-IDF + LR | apology only | 6.7 | -0.5 [-4.3, +3.3] | 1.0 | -5.7 [-9.5, -2.9] | 59.5 |
| TF-IDF + LR | 'silly question' only | 3.8 | -3.3 [-5.7, -1.0] | 3.3 | -3.3 [-5.7, -1.0] | 59.0 |
| TF-IDF + LR | neutral sentence | 7.1 | +0.0 [+0.0, +0.0] | 6.7 | +0.0 [+0.0, +0.0] | 59.0 |
| Embedding + LR | newcomer sentence (full) | 8.6 | +0.5 [-2.9, +3.8] | 5.7 | +0.5 [-2.4, +3.3] | 37.6 |
| Embedding + LR | newcomer status only | 8.6 | +0.5 [-1.4, +2.4] | 5.2 | +0.0 [-1.4, +1.4] | 37.6 |
| Embedding + LR | apology only | 5.2 | -2.9 [-5.2, -1.0] | 2.9 | -2.4 [-4.8, -0.5] | 36.7 |
| Embedding + LR | 'silly question' only | 6.2 | -1.9 [-4.8, +0.5] | 4.8 | -0.5 [-2.9, +1.9] | 35.2 |
| Embedding + LR | neutral sentence | 8.1 | +0.0 [+0.0, +0.0] | 5.2 | +0.0 [+0.0, +0.0] | 36.7 |
| LLM zero-shot | newcomer sentence (full) | 5.7 | +3.8 [+1.0, +7.1] | 4.8 | +4.8 [+1.9, +7.6] | 4.3 |
| LLM zero-shot | newcomer status only | 1.9 | +0.0 [-1.9, +1.9] | 0.5 | +0.5 [+0.0, +1.4] | 10.0 |
| LLM zero-shot | apology only | 1.4 | -0.5 [-2.4, +1.0] | 0.0 | +0.0 [+0.0, +0.0] | 9.5 |
| LLM zero-shot | 'silly question' only | 7.6 | +5.7 [+1.9, +10.0] | 6.7 | +6.7 [+3.3, +10.5] | 3.8 |
| LLM zero-shot | neutral sentence | 1.9 | +0.0 [+0.0, +0.0] | 0.0 | +0.0 [+0.0, +0.0] | 10.5 |
| LLM few-shot (retrieval) | newcomer sentence (full) | 16.2 | +11.9 [+7.1, +16.7] | 14.8 | +13.3 [+9.0, +18.6] | 16.2 |
| LLM few-shot (retrieval) | newcomer status only | 6.2 | +1.9 [+0.0, +4.3] | 4.3 | +2.9 [+1.0, +5.2] | 29.5 |
| LLM few-shot (retrieval) | apology only | 3.8 | -0.5 [-3.3, +2.4] | 1.4 | +0.0 [-1.9, +2.4] | 31.0 |
| LLM few-shot (retrieval) | 'silly question' only | 17.1 | +12.9 [+8.1, +18.1] | 16.2 | +14.8 [+10.0, +20.0] | 13.3 |
| LLM few-shot (retrieval) | neutral sentence | 4.3 | +0.0 [+0.0, +0.0] | 1.4 | +0.0 [+0.0, +0.0] | 32.4 |
