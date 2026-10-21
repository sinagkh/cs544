# Results

Scoring: string matching plus the audited equivalence judge (the judge accepted 863 answer instances that string matching rejected).

## Llama-3.2-3B-Instruct-4bit (n = 30 questions per condition)

### Success rate by condition (%)

Success = correct answer in answerable conditions, `INSUFFICIENT` in `absent`.

| Policy | clean | encoder | truncation | overflow | stale | absent | encoder+truncation | Faults pooled [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | 76.7 | 26.7 | 30.0 | 30.0 | 56.7 | 36.7 | 6.7 | 30.0 [21.3, 39.3] |
| Retry | 76.7 | 26.7 | 33.3 | 33.3 | 56.7 | 30.0 | 10.0 | 32.0 [23.3, 41.3] |
| Self-check (LLM) | 66.7 | 23.3 | 26.7 | 16.7 | 40.0 | 66.7 | 6.7 | 22.7 [14.7, 31.3] |
| Self-check (MiniCheck) | 73.3 | 23.3 | 26.7 | 23.3 | 53.3 | 66.7 | 6.7 | 26.7 [18.7, 35.3] |
| Repair-all | 76.7 | 53.3 | 76.7 | 10.0 | 73.3 | 16.7 | 53.3 | 53.3 [42.0, 64.0] |
| Trace checks only | 76.7 | 50.0 | 76.7 | 60.0 | 76.7 | 36.7 | 50.0 | 62.7 [48.0, 76.7] |
| Diagnose (text match) | 76.7 | 56.7 | 76.7 | 60.0 | 76.7 | 13.3 | 56.7 | 65.3 [50.7, 78.7] |
| Diagnose (LLM verifier) | 70.0 | 56.7 | 76.7 | 56.7 | 73.3 | 13.3 | 56.7 | 64.0 [49.3, 77.3] |
| Diagnose (MiniCheck) | 73.3 | 50.0 | 70.0 | 53.3 | 70.0 | 26.7 | 50.0 | 58.7 [43.3, 72.7] |
|   − no re-check | 76.7 | 50.0 | 76.7 | 60.0 | 76.7 | 16.7 | 50.0 | 62.7 [48.0, 76.7] |
|   − no abstention | 76.7 | 56.7 | 76.7 | 60.0 | 76.7 | 13.3 | 56.7 | 65.3 [50.7, 78.7] |
|   − no trace checks | 80.0 | 50.0 | 23.3 | 30.0 | 60.0 | 16.7 | 16.7 | 36.0 [26.0, 45.3] |

### Damage, abstention and cost

Clean damage = clean questions answered correctly at first and wrong after the policy. Calls include the initial call; GPU s is generation time; MiniCheck columns are support checks per case and their time.

| Policy | Repair % | Clean damage % | Unneeded abstain % | Stale value used % | Calls | Prompt tokens | GPU s | MiniCheck checks | MiniCheck s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | 0.0 | 0.0 | 0.0 | 13.3 | 1.00 | 650 | 1.6 | 0.00 | 0.00 |
| Retry | 2.9 | 0.0 | 0.0 | 13.3 | 1.35 | 857 | 2.1 | 0.00 | 0.00 |
| Self-check (LLM) | 2.9 | 17.4 | 15.0 | 20.0 | 2.21 | 1477 | 3.5 | 0.00 | 0.00 |
| Self-check (MiniCheck) | 1.0 | 13.0 | 12.8 | 13.3 | 1.44 | 955 | 2.4 | 0.65 | 0.23 |
| Repair-all | 43.8 | 4.3 | 3.3 | 0.0 | 2.00 | 2323 | 5.4 | 0.00 | 0.00 |
| Trace checks only | 51.4 | 0.0 | 3.9 | 3.3 | 1.73 | 1153 | 2.8 | 0.00 | 0.00 |
| Diagnose (text match) | 55.2 | 0.0 | 2.8 | 3.3 | 1.90 | 1375 | 3.3 | 0.00 | 0.00 |
| Diagnose (LLM verifier) | 55.2 | 8.7 | 3.9 | 3.3 | 2.97 | 2304 | 5.4 | 0.00 | 0.00 |
| Diagnose (MiniCheck) | 48.6 | 4.3 | 4.4 | 0.0 | 2.57 | 2283 | 5.4 | 1.51 | 0.00 |
|   − no re-check | 51.4 | 0.0 | 3.9 | 3.3 | 1.80 | 1259 | 3.1 | 0.00 | 0.00 |
|   − no abstention | 55.2 | 0.0 | 2.8 | 3.3 | 1.90 | 1375 | 3.3 | 0.00 | 0.00 |
|   − no trace checks | 12.4 | 0.0 | 2.2 | 13.3 | 1.66 | 1241 | 3.0 | 0.00 | 0.00 |

### Verifier × repair (2 × 2)

Rows: what decides that an answer is unsupported. Columns: what happens next. Cells: success under faults / clean damage (%).

| Verifier | Regenerate on the same passages | Repair the context (trace checks + retrieval ladder) |
|---|---:|---:|
| LLM self-verifier | 22.7 / 17.4 | 64.0 / 8.7 |
| MiniCheck | 26.7 / 13.0 | 58.7 / 4.3 |
| Text match | – | 65.3 / 0.0 |

### Diagnose (text match) vs. each policy (paired by question)

Success difference in percentage points (reference minus row). Pooled CI: bootstrap over questions. Per-condition p: exact McNemar.

| Policy | Faults pooled Δ [95% CI] | clean Δ (p) | encoder Δ (p) | truncation Δ (p) | overflow Δ (p) | stale Δ (p) | absent Δ (p) | encoder+truncation Δ (p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No repair | +35.3 [+24.7, +46.7] | +0.0 (1.000) | +30.0 (0.012) | +46.7 (0.001) | +30.0 (0.035) | +20.0 (0.031) | -23.3 (0.016) | +50.0 (0.000) |
| Retry | +33.3 [+23.3, +44.0] | +0.0 (1.000) | +30.0 (0.012) | +43.3 (0.001) | +26.7 (0.057) | +20.0 (0.031) | -16.7 (0.062) | +46.7 (0.000) |
| Self-check (LLM) | +42.7 [+29.3, +55.3] | +10.0 (0.375) | +33.3 (0.006) | +50.0 (0.000) | +43.3 (0.001) | +36.7 (0.003) | -53.3 (0.000) | +50.0 (0.000) |
| Self-check (MiniCheck) | +38.7 [+26.7, +50.7] | +3.3 (1.000) | +33.3 (0.006) | +50.0 (0.000) | +36.7 (0.007) | +23.3 (0.039) | -53.3 (0.000) | +50.0 (0.000) |
| Repair-all | +12.0 [+4.0, +20.0] | +0.0 (1.000) | +3.3 (1.000) | +0.0 (1.000) | +50.0 (0.000) | +3.3 (1.000) | -3.3 (1.000) | +3.3 (1.000) |
| Trace checks only | +2.7 [+0.0, +6.7] | +0.0 (1.000) | +6.7 (0.500) | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) | -23.3 (0.016) | +6.7 (0.500) |
| Diagnose (LLM verifier) | +1.3 [+0.0, +3.3] | +6.7 (0.500) | +0.0 (1.000) | +0.0 (1.000) | +3.3 (1.000) | +3.3 (1.000) | +0.0 (1.000) | +0.0 (1.000) |
| Diagnose (MiniCheck) | +6.7 [+1.3, +13.3] | +3.3 (1.000) | +6.7 (0.500) | +6.7 (0.500) | +6.7 (0.500) | +6.7 (0.500) | -13.3 (0.125) | +6.7 (0.500) |
|   − no re-check | +2.7 [+0.0, +6.7] | +0.0 (1.000) | +6.7 (0.500) | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) | -3.3 (1.000) | +6.7 (0.500) |
|   − no abstention | +0.0 [+0.0, +0.0] | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) | +0.0 (1.000) |
|   − no trace checks | +29.3 [+20.0, +39.3] | -3.3 (1.000) | +6.7 (0.688) | +53.3 (0.000) | +30.0 (0.035) | +16.7 (0.062) | -3.3 (1.000) | +40.0 (0.000) |

### Fault delivery and detection (% of cases)

Delivery columns use the gold answer (evaluation only); flag columns are the trace checks of `diagnose` on the initial answer.

| Condition | Answer text not seen by model | Both gold passages retrieved | Stale version in prompt | Prompt cut by server | Flag: truncation | Flag: version | Flag: overflow | Flag: low retrieval | Flag: unsupported | Abstained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.7 | 3.3 | 3.3 |
| encoder | 60.0 | 23.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 3.3 |
| truncation | 60.0 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 6.7 | 0.0 | 3.3 |
| overflow | 53.3 | 100.0 | 0.0 | 96.7 | 0.0 | 0.0 | 96.7 | 6.7 | 0.0 | 20.0 |
| stale | 0.0 | 96.7 | 96.7 | 0.0 | 0.0 | 96.7 | 0.0 | 3.3 | 0.0 | 3.3 |
| absent | 90.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 | 30.0 | 13.3 |
| encoder+truncation | 83.3 | 23.3 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 | 3.3 |

### Support checks on initial answers

136 non-abstaining initial answers; positive = correct and the answer text was in what the model saw (63 positives). Good kept = share of positives judged supported; bad caught = share of negatives judged unsupported.

| Check | Good kept % | Bad caught % | Balanced accuracy |
|---|---:|---:|---:|
| LLM self-verifier | 14.3 | 89.0 | 0.517 |
| MiniCheck ≥ 0.5 (AUROC 0.65) | 41.3 | 76.7 | 0.590 |
| MiniCheck ≥ 0.05 (abstain gate) | 77.8 | 39.7 | 0.588 |
