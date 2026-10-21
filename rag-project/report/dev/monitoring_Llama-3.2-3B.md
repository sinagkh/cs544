# Monitoring simulation (Llama-3.2-3B-Instruct-4bit)

300 random streams per fault: 400 clean requests, then 200 under the fault. Rolling window 50; alarm when the windowed rate exceeds clean mean + 3 SD. Delay = requests after the deploy until the first alarm (median [IQR]); detected = share of runs with an alarm within the fault segment; false alarms = share of clean windows above the threshold.

| Fault | trace checks | MiniCheck | LLM verifier | abstention |
|---|---:|---:|---:|---:|
| encoder | 6 [4–7], 100% det., 0.1% FA | 35 [26–44], 100% det., 0.0% FA | 44 [33–63], 88% det., 0.0% FA | 11 [7–14], 100% det., 0.1% FA |
| truncation | 5 [4–7], 100% det., 0.1% FA | 24 [19–28], 100% det., 0.0% FA | 35 [27–41], 94% det., 0.0% FA | 12 [7–16], 100% det., 0.0% FA |
| overflow | 5 [4–7], 100% det., 0.1% FA | 69 [44–116], 64% det., 0.0% FA | 76 [51–118], 21% det., 0.0% FA | 72 [37–116], 58% det., 0.1% FA |
| stale | 6 [4–7], 100% det., 0.1% FA | 95 [44–158], 5% det., 0.0% FA | 38 [31–46], 94% det., 0.0% FA | 60 [35–107], 81% det., 0.1% FA |
| encoder+truncation | 5 [4–7], 100% det., 0.0% FA | 25 [19–30], 100% det., 0.0% FA | 35 [27–41], 97% det., 0.0% FA | 7 [5–9], 100% det., 0.1% FA |

Signal rates per condition (% of requests):

| Signal | clean | encoder | truncation | overflow | stale | encoder+truncation |
|---|---:|---:|---:|---:|---:|---:|
| trace checks | 6.7 | 100.0 | 100.0 | 96.7 | 96.7 | 100.0 |
| MiniCheck | 60.0 | 86.7 | 100.0 | 70.0 | 56.7 | 100.0 |
| LLM verifier | 76.7 | 93.3 | 100.0 | 80.0 | 96.7 | 100.0 |
| abstention | 6.7 | 53.3 | 46.7 | 10.0 | 13.3 | 80.0 |
