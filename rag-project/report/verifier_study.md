# Verifier study (240 initial answers of Llama-3.2-3B-Instruct-4bit; 92 correct and grounded)

Good kept = correct, grounded answers judged supported. Bad caught = other answers judged unsupported.

| Verifier | Good kept % | Bad caught % | Balanced accuracy |
|---|---:|---:|---:|
| Llama-3.2-3B-Instruct-4bit (self-verifier) | 20.7 | 91.2 | 0.559 |
| claude-haiku-4-5 (same prompt) | 71.7 | 82.4 | 0.771 |
| MiniCheck-RoBERTa-L, score >= 0.5 | 45.7 | 81.1 | 0.634 |

By condition (balanced accuracy):

| Condition | n | self-verifier | strong verifier | MiniCheck |
|---|---:|---:|---:|---:|
| clean | 40 | 0.53 | 0.80 | 0.63 |
| encoder | 40 | 0.59 | 0.86 | 0.62 |
| truncation | 40 | 0.49 | 0.53 | 0.55 |
| stale | 40 | 0.57 | 0.74 | 0.60 |
| absent | 40 | 0.67 | 0.45 | 0.62 |
| encoder+truncation | 40 | 0.56 | 0.62 | 0.52 |
