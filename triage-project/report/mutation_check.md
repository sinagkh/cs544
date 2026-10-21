# Mutation check of the triage regression suite (40 cases)

| Injected bug | Failing cases | By case kind |
|---|---:|---|
| none (original) | 0 / 40 | – |
| retrieval dropped (zero-shot prompt) | 5 / 40 | auto_correct: 2, must_defer: 3 |
| examples from other repositories | 10 / 40 | auto_correct: 5, must_defer: 3, stable_rewording: 2 |
| temperature scaling missing | 14 / 40 | must_defer: 14 |
| label definitions removed | 1 / 40 | auto_correct: 1 |
| issue body cut to 300 characters | 6 / 40 | auto_correct: 5, must_defer: 1 |
