# RAG diagnosis and repair

## Review the project

- [Full project report](paper/main.pdf) and [LaTeX source](paper/main.tex).
- [Two-page proposal PDF](documents/Two_Page_Project_Proposal.pdf) and [editable Word document](documents/Two_Page_Project_Proposal.docx), including 10 linked papers.
- [Recorded results and tables](report/results.md).
- [Original project README](README.md) for the method and experiment commands.

## Download the reproducibility snapshot

Download [CS544_RAG_Project_Artifact.zip](https://raw.githubusercontent.com/sinagkh/cs544/main/artifacts/CS544_RAG_Project_Artifact.zip) from the [artifact downloads](../artifacts/README.md). Extract it and open its `START_HERE.md` for setup, cached replay, and analysis instructions.

This Git checkout contains source, tests, reports, proposals, and audit sheets. The artifact ZIP supplies the prepared data, predictions, embeddings, response caches, and provenance files needed to reproduce the recorded analysis. To work in this clone, copy `data/`, `cache/`, `results/`, and (for RAG) `archive/` from the extracted snapshot into this project folder.

## Current status and next work

The full report contains the current Llama-3.2-3B measurements. The two-page proposal describes the study and planned validation. The Qwen run in `results/pending/` is incomplete. Trace checks encode predefined fault families and repair rules.

Before final submission, complete the independent human answer audit in `report/audit/`, validate the missing-evidence cases, and evaluate monitoring false alarms on independent clean traffic. A second completed generator run would broaden the evidence.


## Environment and validation

Live local inference uses Apple silicon through MLX. Each teammate creates a local Python environment using `pyproject.toml`; `environment-snapshot.txt` records the original package versions. The artifact includes cached responses for replay tests. First-use tokenizer downloads may require network access.

[VALIDATION.md](VALIDATION.md) records checks on the complete artifact snapshot, including unit tests, cached replay tests, and exact reproduction of the JSON result summary. Read the data-source and provenance notes in the ZIP's `START_HERE.md` before redistributing datasets.
