# Mutation check of the triage regression suite (40 cases)

| Injected bug | Failing cases | By case kind |
|---|---:|---|
| none (original) | 0 / 40 | – |
| retrieval dropped (zero-shot prompt) | 6 / 40 | auto_correct: 1, must_defer: 5 |
| examples from other repositories | 8 / 40 | auto_correct: 5, must_defer: 1, stable_rewording: 2 |
| temperature scaling missing | 13 / 40 | must_defer: 13 |
| label definitions removed | 2 / 40 | must_defer: 2 |
| issue body cut to 300 characters | 1 / 40 | auto_correct: 1 |
