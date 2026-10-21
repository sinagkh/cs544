# Monitoring simulation (Qwen2.5-7B few-shot)

300 streams per shift: 400 NLBSE'24 test requests, then 200 shifted requests. Window 50; alarm when the windowed value exceeds mean + 3 SD of a separate clean calibration stream (disjoint half of the test issues). Cells: median delay in requests [IQR], share of runs detected, false-alarm rate on the evaluation stream's clean windows.

| Shift | handed to human | low confidence | LLM-classifier disagreement | novelty vs. history | predicted-label shift |
|---|---:|---:|---:|---:|---:|
| fresh | 52 [32–73], 1%, 0.3% FA | 1 [1–80], 1%, 0.8% FA | 8 [1–32], 15%, 7.0% FA | 44 [32–60], 96%, 0.4% FA | 88 [48–142], 36%, 1.4% FA |
| newrepo | 84 [47–143], 24%, 0.2% FA | 87 [49–135], 27%, 0.9% FA | 64 [16–107], 63%, 6.0% FA | 13 [10–16], 100%, 0.7% FA | 74 [31–125], 25%, 1.3% FA |
| non_native | 76 [44–128], 25%, 0.5% FA | 74 [49–136], 17%, 1.0% FA | 76 [20–136], 41%, 5.0% FA | 83 [26–110], 14%, 0.5% FA | 70 [17–128], 26%, 1.7% FA |

Mean signal per request in each pool:

| Pool | error | handed to human | low confidence | LLM-classifier disagreement | novelty vs. history |
|---|---:|---:|---:|---:|---:|
| clean_cal (n=750) | 28.9 | 0.647 | 0.266 | 0.300 | 0.195 |
| clean (n=750) | 29.3 | 0.621 | 0.262 | 0.352 | 0.191 |
| fresh (n=514) | 24.7 | 0.554 | 0.238 | 0.272 | 0.219 |
| newrepo (n=750) | 32.0 | 0.680 | 0.276 | 0.380 | 0.297 |
| non_native (n=159) | 30.2 | 0.679 | 0.269 | 0.346 | 0.196 |
