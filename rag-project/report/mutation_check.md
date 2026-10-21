# Mutation check of the regression suite (35 cases, Llama-3.2-3B-Instruct-4bit)

| Mutation | Failing cases | By condition |
|---|---:|---|
| none (original) | 0 / 35 | – |
| no truncation check | 10 / 35 | encoder+truncation: 5, truncation: 5 |
| no version check | 4 / 35 | stale: 4 |
| no overflow check | 5 / 35 | overflow: 5 |
| no low-retrieval check | 8 / 35 | encoder: 4, encoder+truncation: 4 |
| no re-check after repair | 3 / 35 | encoder: 1, encoder+truncation: 2 |
