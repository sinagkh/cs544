# Monitoring simulation (Llama-3.2-3B-Instruct-4bit)

300 random streams per fault: 400 clean requests, then 200 after the deploy (stale affects 20% of post-deploy requests, the other faults all of them). Rolling window 50; alarm when the windowed rate exceeds clean mean + 3 SD. Delay = requests after the deploy until the first alarm (median [IQR]); detected = share of runs with an alarm within the fault segment; false alarms = share of clean windows above the threshold.

| Fault | trace checks | MiniCheck | LLM verifier | abstention |
|---|---:|---:|---:|---:|
| encoder | 4 [3–6], 100% det., 0.1% FA | 23 [17–30], 100% det., 0.0% FA | 63 [41–109], 67% det., 0.0% FA | 10 [7–14], 100% det., 0.1% FA |
| truncation | 4 [3–5], 100% det., 0.1% FA | 19 [15–23], 100% det., 0.0% FA | 39 [30–47], 92% det., 0.0% FA | 10 [7–14], 100% det., 0.1% FA |
| overflow | 4 [3–6], 100% det., 0.1% FA | 35 [26–46], 100% det., 0.0% FA | 88 [51–146], 34% det., 0.0% FA | 56 [31–102], 89% det., 0.1% FA |
| stale | 24 [14–35], 100% det., 0.1% FA | 75 [38–128], 22% det., 0.0% FA | 93 [69–138], 20% det., 0.0% FA | 77 [41–136], 25% det., 0.1% FA |
| encoder+truncation | 4 [3–6], 100% det., 0.0% FA | 18 [13–23], 100% det., 0.0% FA | 34 [25–41], 99% det., 0.0% FA | 7 [5–9], 100% det., 0.1% FA |

Signal rates per condition (% of requests):

| Signal | clean | encoder | truncation | overflow | stale | encoder+truncation |
|---|---:|---:|---:|---:|---:|---:|
| trace checks | 4.0 | 100.0 | 100.0 | 100.0 | 86.0 | 100.0 |
| MiniCheck | 44.0 | 84.7 | 94.7 | 70.7 | 52.0 | 97.3 |
| LLM verifier | 74.7 | 86.0 | 94.7 | 80.7 | 90.0 | 99.3 |
| abstention | 5.3 | 50.0 | 49.3 | 12.0 | 5.3 | 72.7 |
