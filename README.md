# CS544 research project candidates

Two UIC CS544 Machine Learning in Production projects for group review and collaboration. Each project includes its full report, source code, results, and a two-page proposal with 10 linked papers and all three group members.

| Project | Start here | Full report | Two-page proposal | Editable proposal |
| --- | --- | --- | --- | --- |
| RAG component diagnosis and repair | [Project guide](rag-project/START_HERE.md) | [PDF](rag-project/paper/main.pdf) | [PDF](rag-project/documents/Two_Page_Project_Proposal.pdf) | [Word](rag-project/documents/Two_Page_Project_Proposal.docx) |
| Reliable issue triage and human deferral | [Project guide](triage-project/START_HERE.md) | [PDF](triage-project/paper/main.pdf) | [PDF](triage-project/documents/Two_Page_Project_Proposal.pdf) | [Word](triage-project/documents/Two_Page_Project_Proposal.docx) |

## Complete downloadable artifacts

The [artifact downloads](artifacts/README.md) contain the October 21, 2026 handoff:

- [RAG project ZIP](https://raw.githubusercontent.com/sinagkh/cs544/main/artifacts/CS544_RAG_Project_Artifact.zip) (27.4 MB).
- [Triage project ZIP](https://raw.githubusercontent.com/sinagkh/cs544/main/artifacts/CS544_Triage_Project_Artifact.zip) (15.4 MB).
- SHA-256 checksum files for both ZIPs.

The artifact ZIPs include prepared datasets, embeddings, recorded predictions, response caches, source, tests, and both reports. Open `START_HERE.md` inside each extracted archive for setup and the remaining validation tasks. The approximately 1.3 GB Python environments are recreated locally; downloaded model weights are obtained separately for live inference.

## Validation and project status

The prepared RAG snapshot passed 42 tests; the triage snapshot passed 49. Both main analysis scripts reproduced their supplied JSON result summaries exactly. Each proposal PDF has two pages and its Word version has 10 unique linked paper references. The ZIP files were checked for integrity and against their file hashes.

These are research project candidates with current measurements and planned validation. Each project guide identifies the remaining human audits and methodological checks. Live inference uses Apple silicon through MLX; the report PDFs can be reviewed directly.

## Working together

Clone this repository for editing source, reports, or proposals. Download the relevant artifact ZIP for cached replay and recorded-result analysis. Copy its `data/`, `cache/`, `results/`, and RAG `archive/` folders into the matching project folder when working in the clone. The Git ignore rules exclude the expanded data, caches, results, and local environments; the prepared ZIPs are tracked in `artifacts/`.
