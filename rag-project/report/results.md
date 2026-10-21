# Results

Scoring: string matching plus the audited equivalence judge (the judge accepted 3059 answer instances that string matching rejected).

## Llama-3.2-3B-Instruct-4bit (n = 150 questions per condition)

### Success rate by condition (%)

Success = correct answer in answerable conditions, `INSUFFICIENT` in `absent`.

| Policy | clean | encoder | truncation | overflow | stale | absent | encoder+truncation | Faults pooled [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | 71.3 | 26.0 | 20.7 | 30.0 | 50.0 | 38.7 | 7.3 | 26.8 [22.7, 31.2] |
| Retry | 71.3 | 26.7 | 21.3 | 30.7 | 50.7 | 34.0 | 8.7 | 27.6 [23.3, 31.9] |
| Self-check (LLM) | 55.3 | 20.0 | 14.7 | 25.3 | 38.7 | 60.7 | 4.7 | 20.7 [16.9, 24.5] |
| Self-check (MiniCheck) | 64.0 | 20.0 | 16.0 | 28.0 | 42.7 | 55.3 | 5.3 | 22.4 [18.8, 26.3] |
| Repair-all | 63.3 | 48.0 | 63.3 | 6.0 | 66.7 | 20.0 | 48.0 | 46.4 [41.1, 52.0] |
| Trace checks only | 71.3 | 44.7 | 71.3 | 51.3 | 67.3 | 38.0 | 44.7 | 55.9 [49.7, 62.0] |
| Diagnose (text match) | 71.3 | 48.7 | 70.7 | 48.0 | 68.7 | 28.0 | 48.7 | 56.9 [50.7, 63.1] |
| Diagnose (LLM verifier) | 67.3 | 48.7 | 70.7 | 48.0 | 70.0 | 28.0 | 48.7 | 57.2 [50.9, 63.3] |
| Diagnose (MiniCheck) | 68.7 | 52.0 | 71.3 | 50.7 | 70.7 | 29.3 | 52.0 | 59.3 [53.1, 65.6] |
|   − no re-check | 70.7 | 41.3 | 68.0 | 48.0 | 66.7 | 30.0 | 41.3 | 53.1 [46.5, 59.1] |
|   − no abstention | 73.3 | 52.0 | 72.7 | 51.3 | 71.3 | 23.3 | 52.0 | 59.9 [53.9, 66.0] |
|   − no trace checks | 70.7 | 45.3 | 18.0 | 29.3 | 52.0 | 30.0 | 14.7 | 31.9 [27.6, 36.3] |

### Damage, abstention and cost

Clean damage = clean questions answered correctly at first and wrong after the policy. Calls include the initial call; GPU s is generation time; MiniCheck columns are support checks per case and their time.

| Policy | Repair % | Clean damage % | Unneeded abstain % | Stale value used % | Calls | Prompt tokens | GPU s | MiniCheck checks | MiniCheck s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | 0.0 | 0.0 | 0.0 | 18.7 | 1.00 | 660 | 1.5 | 0.00 | 0.00 |
| Retry | 1.1 | 0.0 | 0.0 | 18.7 | 1.33 | 857 | 2.0 | 0.00 | 0.00 |
| Self-check (LLM) | 4.0 | 24.3 | 16.8 | 16.7 | 2.21 | 1491 | 3.3 | 0.00 | 0.00 |
| Self-check (MiniCheck) | 2.0 | 12.1 | 12.8 | 19.3 | 1.42 | 950 | 2.2 | 0.67 | 0.26 |
| Repair-all | 38.6 | 15.9 | 3.1 | 2.0 | 2.00 | 2380 | 5.2 | 0.00 | 0.00 |
| Trace checks only | 48.1 | 1.9 | 4.2 | 2.0 | 1.72 | 1154 | 2.7 | 0.00 | 0.00 |
| Diagnose (text match) | 50.1 | 4.7 | 5.6 | 2.0 | 1.96 | 1477 | 3.4 | 0.00 | 0.00 |
| Diagnose (LLM verifier) | 50.5 | 11.2 | 5.7 | 0.7 | 2.94 | 2335 | 5.1 | 0.00 | 0.00 |
| Diagnose (MiniCheck) | 52.6 | 7.5 | 3.6 | 2.7 | 2.42 | 2110 | 4.7 | 1.40 | 0.28 |
|   − no re-check | 45.7 | 5.6 | 7.6 | 2.0 | 1.82 | 1306 | 3.0 | 0.00 | 0.00 |
|   − no abstention | 52.5 | 1.9 | 2.9 | 2.0 | 1.96 | 1477 | 3.4 | 0.00 | 0.00 |
|   − no trace checks | 9.3 | 3.7 | 3.4 | 18.7 | 1.72 | 1336 | 3.0 | 0.00 | 0.00 |

### Verifier × repair (2 × 2)

Rows: what decides that an answer is unsupported. Columns: what happens next. Cells: success under faults / clean damage (%).

| Verifier | Regenerate on the same passages | Repair the context (trace checks + retrieval ladder) |
|---|---:|---:|
| LLM self-verifier | 20.7 / 24.3 | 57.2 / 11.2 |
| MiniCheck | 22.4 / 12.1 | 59.3 / 7.5 |
| Text match | – | 56.9 / 4.7 |

### Diagnose (text match) vs. each policy (paired by question)

Success difference in percentage points (reference minus row). Pooled CI: bootstrap over questions. Per-condition p: exact McNemar.

| Policy | Faults pooled Δ [95% CI] | clean Δ (p) | encoder Δ (p) | truncation Δ (p) | overflow Δ (p) | stale Δ (p) | absent Δ (p) | encoder+truncation Δ (p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | +30.1 [+24.8, +35.7] | +0.0 (1.000) | +22.7 (0.000) | +50.0 (0.000) | +18.0 (0.002) | +18.7 (0.000) | -10.7 (0.004) | +41.3 (0.000) |
| Retry | +29.3 [+24.0, +34.9] | +0.0 (1.000) | +22.0 (0.000) | +49.3 (0.000) | +17.3 (0.003) | +18.0 (0.000) | -6.0 (0.108) | +40.0 (0.000) |
| Self-check (LLM) | +36.3 [+30.3, +42.1] | +16.0 (0.000) | +28.7 (0.000) | +56.0 (0.000) | +22.7 (0.000) | +30.0 (0.000) | -32.7 (0.000) | +44.0 (0.000) |
| Self-check (MiniCheck) | +34.5 [+28.8, +40.0] | +7.3 (0.043) | +28.7 (0.000) | +54.7 (0.000) | +20.0 (0.000) | +26.0 (0.000) | -27.3 (0.000) | +43.3 (0.000) |
| Repair-all | +10.5 [+6.4, +14.9] | +8.0 (0.023) | +0.7 (1.000) | +7.3 (0.035) | +42.0 (0.000) | +2.0 (0.678) | +8.0 (0.008) | +0.7 (1.000) |
| Trace checks only | +1.1 [-2.0, +4.0] | +0.0 (1.000) | +4.0 (0.180) | -0.7 (1.000) | -3.3 (0.062) | +1.3 (0.727) | -10.0 (0.006) | +4.0 (0.180) |
| Diagnose (LLM verifier) | -0.3 [-0.7, +0.0] | +4.0 (0.070) | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) | -1.3 (0.500) | +0.0 (1.000) | +0.0 (1.000) |
| Diagnose (MiniCheck) | -2.4 [-5.1, +0.1] | +2.7 (0.388) | -3.3 (0.227) | -0.7 (1.000) | -2.7 (0.219) | -2.0 (0.549) | -1.3 (0.804) | -3.3 (0.227) |
|   − no re-check | +3.9 [+1.9, +6.0] | +0.7 (1.000) | +7.3 (0.001) | +2.7 (0.125) | +0.0 (1.000) | +2.0 (0.250) | -2.0 (0.250) | +7.3 (0.001) |
|   − no abstention | -2.9 [-5.3, -1.1] | -2.0 (0.250) | -3.3 (0.062) | -2.0 (0.250) | -3.3 (0.062) | -2.7 (0.125) | +4.7 (0.016) | -3.3 (0.062) |
|   − no trace checks | +25.1 [+20.3, +30.0] | +0.7 (1.000) | +3.3 (0.500) | +52.7 (0.000) | +18.7 (0.001) | +16.7 (0.000) | -2.0 (0.250) | +34.0 (0.000) |

### Fault delivery and detection (% of cases)

Delivery columns use the gold answer (evaluation only); flag columns are the trace checks of `diagnose` on the initial answer.

| Condition | Answer text not seen by model | Both gold passages retrieved | Stale version in prompt | Prompt cut by server | Flag: truncation | Flag: version | Flag: overflow | Flag: low retrieval | Flag: unsupported | Abstained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 | 8.7 | 3.3 |
| encoder | 68.7 | 18.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 9.3 |
| truncation | 54.7 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 4.0 | 0.0 | 4.0 |
| overflow | 48.7 | 100.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 4.0 | 0.0 | 25.3 |
| stale | 4.7 | 93.3 | 90.7 | 0.0 | 0.0 | 86.0 | 0.0 | 4.0 | 3.3 | 4.0 |
| absent | 91.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 12.0 | 33.3 | 28.0 |
| encoder+truncation | 85.3 | 18.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 9.3 |

### Support checks on initial answers

700 non-abstaining initial answers; positive = correct and the answer text was in what the model saw (285 positives). Good kept = share of positives judged supported; bad caught = share of negatives judged unsupported.

| Check | Good kept % | Bad caught % | Balanced accuracy |
|---|---:|---:|---:|
| LLM self-verifier | 26.3 | 86.7 | 0.565 |
| MiniCheck ≥ 0.5 (AUROC 0.67) | 52.6 | 72.8 | 0.627 |
| MiniCheck ≥ 0.05 (abstain gate) | 81.1 | 38.1 | 0.596 |
