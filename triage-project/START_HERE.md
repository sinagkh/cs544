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

The full report contains the current Qwen2.5-7B and classical-classifier measurements. The two-page proposal includes planned methodological validation. Existing numerical results describe the current implementation; those validation changes remain to be completed.

Before final submission, separate temperature fitting from statistical certification, use source-only calibration for held-out repositories, complete the human rewrite audit, and add controls for the newcomer sentence's question-like wording. Include changes in human-review workload in the fairness analysis and evaluate false alarms on independent clean traffic. Small selected subsets need uncertainty intervals.


## Environment and validation

Live local inference uses Apple silicon through MLX. Each teammate creates a local Python environment using `pyproject.toml`; `environment-snapshot.txt` records the original package versions. The artifact includes cached responses for replay tests. First-use tokenizer downloads may require network access.

[VALIDATION.md](VALIDATION.md) records checks on the complete artifact snapshot, including unit tests, cached replay tests, and exact reproduction of the JSON result summary. Read the data-source and provenance notes in the ZIP's `START_HERE.md` before redistributing datasets.
