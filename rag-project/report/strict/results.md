# Results

Scoring: string matching only (the judge accepted 0 answer instances that string matching rejected).

## Llama-3.2-3B-Instruct-4bit (n = 150 questions per condition)

### Success rate by condition (%)

Success = correct answer in answerable conditions, `INSUFFICIENT` in `absent`.

| Policy | clean | encoder | truncation | overflow | stale | absent | encoder+truncation | Faults pooled [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | 52.0 | 14.7 | 13.3 | 18.7 | 34.0 | 38.7 | 5.3 | 17.2 [13.7, 20.9] |
| Retry | 52.0 | 14.7 | 14.0 | 19.3 | 34.0 | 34.0 | 5.3 | 17.5 [14.0, 21.2] |
| Self-check (LLM) | 38.7 | 10.7 | 7.3 | 14.7 | 26.7 | 60.7 | 3.3 | 12.5 [9.5, 15.9] |
| Self-check (MiniCheck) | 46.7 | 10.7 | 8.7 | 16.0 | 28.7 | 55.3 | 4.0 | 13.6 [10.7, 16.7] |
| Repair-all | 45.3 | 32.0 | 45.3 | 3.3 | 46.7 | 20.0 | 32.0 | 31.9 [26.9, 37.3] |
| Trace checks only | 52.7 | 31.3 | 52.7 | 36.0 | 50.7 | 38.0 | 31.3 | 40.4 [34.5, 46.7] |
| Diagnose (text match) | 53.3 | 34.7 | 53.3 | 34.7 | 51.3 | 28.0 | 34.7 | 41.7 [35.6, 48.0] |
| Diagnose (LLM verifier) | 48.7 | 34.7 | 53.3 | 34.7 | 52.7 | 28.0 | 34.7 | 42.0 [35.9, 48.4] |
| Diagnose (MiniCheck) | 48.7 | 34.7 | 52.0 | 35.3 | 49.3 | 29.3 | 34.7 | 41.2 [35.1, 47.3] |
|   − no re-check | 53.3 | 29.3 | 52.0 | 34.7 | 50.7 | 30.0 | 29.3 | 39.2 [33.3, 45.5] |
|   − no abstention | 54.0 | 36.0 | 54.0 | 36.0 | 51.3 | 23.3 | 36.0 | 42.7 [36.7, 48.9] |
|   − no trace checks | 52.7 | 30.0 | 12.0 | 19.3 | 36.0 | 30.0 | 8.7 | 21.2 [17.3, 25.3] |

### Damage, abstention and cost

Clean damage = clean questions answered correctly at first and wrong after the policy. Calls include the initial call; GPU s is generation time; MiniCheck columns are support checks per case and their time.

| Policy | Repair % | Clean damage % | Unneeded abstain % | Stale value used % | Calls | Prompt tokens | GPU s | MiniCheck checks | MiniCheck s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | 0.0 | 0.0 | 0.0 | 18.7 | 1.00 | 660 | 1.5 | 0.00 | 0.00 |
| Retry | 0.3 | 0.0 | 0.0 | 18.7 | 1.33 | 857 | 2.0 | 0.00 | 0.00 |
| Self-check (LLM) | 2.6 | 26.9 | 16.8 | 16.7 | 2.21 | 1491 | 3.3 | 0.00 | 0.00 |
| Self-check (MiniCheck) | 1.1 | 11.5 | 12.8 | 19.3 | 1.42 | 950 | 2.2 | 0.67 | 0.26 |
| Repair-all | 26.2 | 20.5 | 3.1 | 2.0 | 2.00 | 2380 | 5.2 | 0.00 | 0.00 |
| Trace checks only | 34.3 | 1.3 | 4.2 | 2.0 | 1.72 | 1154 | 2.7 | 0.00 | 0.00 |
| Diagnose (text match) | 35.9 | 2.6 | 5.6 | 2.0 | 1.96 | 1477 | 3.4 | 0.00 | 0.00 |
| Diagnose (LLM verifier) | 36.2 | 12.8 | 5.7 | 0.7 | 2.94 | 2335 | 5.1 | 0.00 | 0.00 |
| Diagnose (MiniCheck) | 35.7 | 10.3 | 3.6 | 2.7 | 2.42 | 2110 | 4.7 | 1.40 | 0.28 |
|   − no re-check | 33.3 | 2.6 | 7.6 | 2.0 | 1.82 | 1306 | 3.0 | 0.00 | 0.00 |
|   − no abstention | 36.4 | 1.3 | 2.9 | 2.0 | 1.96 | 1477 | 3.4 | 0.00 | 0.00 |
|   − no trace checks | 5.8 | 1.3 | 3.4 | 18.7 | 1.72 | 1336 | 3.0 | 0.00 | 0.00 |

### Verifier × repair (2 × 2)

Rows: what decides that an answer is unsupported. Columns: what happens next. Cells: success under faults / clean damage (%).

| Verifier | Regenerate on the same passages | Repair the context (trace checks + retrieval ladder) |
|---|---:|---:|
| LLM self-verifier | 12.5 / 26.9 | 42.0 / 12.8 |
| MiniCheck | 13.6 / 11.5 | 41.2 / 10.3 |
| Text match | – | 41.7 / 2.6 |

### Diagnose (text match) vs. each policy (paired by question)

Success difference in percentage points (reference minus row). Pooled CI: bootstrap over questions. Per-condition p: exact McNemar.

| Policy | Faults pooled Δ [95% CI] | clean Δ (p) | encoder Δ (p) | truncation Δ (p) | overflow Δ (p) | stale Δ (p) | absent Δ (p) | encoder+truncation Δ (p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | +24.5 [+19.5, +29.9] | +1.3 (0.688) | +20.0 (0.000) | +40.0 (0.000) | +16.0 (0.003) | +17.3 (0.000) | -10.7 (0.004) | +29.3 (0.000) |
| Retry | +24.3 [+19.2, +29.5] | +1.3 (0.688) | +20.0 (0.000) | +39.3 (0.000) | +15.3 (0.004) | +17.3 (0.000) | -6.0 (0.108) | +29.3 (0.000) |
| Self-check (LLM) | +29.2 [+23.7, +34.9] | +14.7 (0.000) | +24.0 (0.000) | +46.0 (0.000) | +20.0 (0.000) | +24.7 (0.000) | -32.7 (0.000) | +31.3 (0.000) |
| Self-check (MiniCheck) | +28.1 [+22.9, +33.9] | +6.7 (0.021) | +24.0 (0.000) | +44.7 (0.000) | +18.7 (0.000) | +22.7 (0.000) | -27.3 (0.000) | +30.7 (0.000) |
| Repair-all | +9.9 [+5.7, +13.9] | +8.0 (0.017) | +2.7 (0.572) | +8.0 (0.017) | +31.3 (0.000) | +4.7 (0.210) | +8.0 (0.008) | +2.7 (0.572) |
| Trace checks only | +1.3 [-0.9, +3.5] | +0.7 (1.000) | +3.3 (0.227) | +0.7 (1.000) | -1.3 (0.500) | +0.7 (1.000) | -10.0 (0.006) | +3.3 (0.227) |
| Diagnose (LLM verifier) | -0.3 [-0.7, +0.0] | +4.7 (0.039) | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) | -1.3 (0.500) | +0.0 (1.000) | +0.0 (1.000) |
| Diagnose (MiniCheck) | +0.5 [-1.2, +2.3] | +4.7 (0.039) | +0.0 (1.000) | +1.3 (0.688) | -0.7 (1.000) | +2.0 (0.508) | -1.3 (0.804) | +0.0 (1.000) |
|   − no re-check | +2.5 [+1.1, +4.3] | +0.0 (1.000) | +5.3 (0.008) | +1.3 (0.500) | +0.0 (1.000) | +0.7 (1.000) | -2.0 (0.250) | +5.3 (0.008) |
|   − no abstention | -0.9 [-1.9, -0.3] | -0.7 (1.000) | -1.3 (0.500) | -0.7 (1.000) | -1.3 (0.500) | +0.0 (1.000) | +4.7 (0.016) | -1.3 (0.500) |
|   − no trace checks | +20.5 [+15.6, +25.5] | +0.7 (1.000) | +4.7 (0.281) | +41.3 (0.000) | +15.3 (0.004) | +15.3 (0.000) | -2.0 (0.250) | +26.0 (0.000) |

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

700 non-abstaining initial answers; positive = correct and the answer text was in what the model saw (200 positives). Good kept = share of positives judged supported; bad caught = share of negatives judged unsupported.

| Check | Good kept % | Bad caught % | Balanced accuracy |
|---|---:|---:|---:|
| LLM self-verifier | 27.5 | 85.0 | 0.562 |
| MiniCheck ≥ 0.5 (AUROC 0.63) | 52.5 | 68.4 | 0.605 |
| MiniCheck ≥ 0.05 (abstain gate) | 80.0 | 34.4 | 0.572 |
