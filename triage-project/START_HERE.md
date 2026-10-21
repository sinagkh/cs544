# Selective LLM issue triage

## Review the project

- [Full project report](paper/main.pdf) and [LaTeX source](paper/main.tex).
- [Two-page proposal PDF](documents/Two_Page_Project_Proposal.pdf) and [editable Word document](documents/Two_Page_Project_Proposal.docx), including 10 linked papers.
- [Recorded results and tables](report/results.md).
- [Original project README](README.md) for the method and experiment commands.

## Download the reproducibility snapshot

Download [CS544_Triage_Project_Artifact.zip](https://raw.githubusercontent.com/sinagkh/cs544/main/artifacts/CS544_Triage_Project_Artifact.zip) from the [artifact downloads](../artifacts/README.md). Extract it and open its `START_HERE.md` for setup, cached replay, and analysis instructions.

This Git checkout contains source, tests, reports, proposals, and audit sheets. The artifact ZIP supplies the prepared data, predictions, embeddings, response caches, and provenance files needed to reproduce the recorded analysis. To work in this clone, copy `data/`, `cache/`, `results/`, and (for RAG) `archive/` from the extracted snapshot into this project folder.

## Current status and next work

The revised seven-page report uses separate temperature/threshold fitting and statistical certification, matched classifier inputs, wording controls, and independent clean monitoring streams. The 10% risk target is certified only for TF-IDF at low automation coverage; the few-shot LLM is first certified at a 20% target. The explanation deletion test is inconclusive. The mutation results detect all five injected bugs, with little redundancy for prompt-definition removal and body truncation.

Review follow-ups remain: exclude the target repository from every step of the new-repository calibration-score construction; complete the 40-pair human rewrite audit; report monitoring false alarms both per window and per clean stream; and include `calib_xrepo` explicitly when rebuilding the LLM scores. The two-page proposal is the earlier project plan. The full report, README, and result files contain the revised measurements.

## Environment and validation

Live local inference uses Apple silicon through MLX. Each teammate creates a local Python environment using `pyproject.toml`; `environment-snapshot.txt` records the original package versions. The artifact includes cached responses for replay tests. First-use tokenizer downloads may require network access.

[VALIDATION.md](VALIDATION.md) records checks on the complete artifact snapshot, including unit tests, cached replay tests, and exact reproduction of the JSON result summary. Read the data-source and provenance notes in the ZIP's `START_HERE.md` before redistributing datasets.
