# Monitoring simulation (Qwen2.5-7B few-shot)

300 streams per shift: 400 NLBSE'24 test requests, then 200 shifted requests. Window 50; alarm when the windowed value exceeds clean mean + 3 SD. Cells: median delay in requests [IQR], share of runs detected, false-alarm rate on clean windows.

| Shift | handed to human | low confidence | LLM-classifier disagreement | novelty vs. history | predicted-label shift |
|---|---:|---:|---:|---:|---:|
| fresh | 98 [42–113], 2%, 0.0% FA | 58 [11–113], 3%, 0.0% FA | 36 [11–114], 5%, 0.0% FA | 38 [29–52], 98%, 0.0% FA | 71 [39–122], 47%, 0.2% FA |
| newrepo | 83 [50–128], 23%, 0.0% FA | 74 [43–124], 27%, 0.0% FA | 90 [44–139], 30%, 0.0% FA | 11 [8–15], 100%, 0.0% FA | 99 [45–140], 28%, 0.2% FA |
| non_native | 93 [52–142], 37%, 0.0% FA | 88 [46–145], 21%, 0.0% FA | 75 [29–130], 19%, 0.0% FA | 88 [53–142], 24%, 0.0% FA | 78 [38–115], 34%, 0.2% FA |

Mean signal per request in each pool:

| Pool | error | handed to human | low confidence | LLM-classifier disagreement | novelty vs. history |
|---|---:|---:|---:|---:|---:|
| clean (n=1500) | 29.1 | 0.571 | 0.256 | 0.330 | 0.193 |
| fresh (n=514) | 24.7 | 0.481 | 0.230 | 0.286 | 0.219 |
| newrepo (n=750) | 32.0 | 0.597 | 0.268 | 0.364 | 0.297 |
| non_native (n=159) | 30.2 | 0.623 | 0.260 | 0.340 | 0.196 |
