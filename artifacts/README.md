# Complete project artifacts

Prepared October 21, 2026 for the CS544 group.

| Project | Download ZIP | Size | Checksum |
| --- | --- | --- | --- |
| RAG diagnosis and repair | [RAG artifact](https://raw.githubusercontent.com/sinagkh/cs544/main/artifacts/CS544_RAG_Project_Artifact.zip) | 27.4 MB | [SHA-256](https://raw.githubusercontent.com/sinagkh/cs544/main/artifacts/CS544_RAG_Project_Artifact.zip.sha256) |
| Selective LLM issue triage | [Triage artifact](https://raw.githubusercontent.com/sinagkh/cs544/main/artifacts/CS544_Triage_Project_Artifact.zip) | 25.0 MB | [SHA-256](https://raw.githubusercontent.com/sinagkh/cs544/main/artifacts/CS544_Triage_Project_Artifact.zip.sha256) |

Each ZIP includes the full project report, two-page proposal in Word and PDF with 10 linked papers, source, tests, prepared data, embeddings, recorded results, useful response caches, and a START_HERE guide. Python environments and downloaded model weights are recreated locally.

The RAG package passed 42 unit and cached replay tests; triage passed 51. Both main analysis scripts reproduced the original JSON result summaries exactly. Each ZIP passed integrity and file-hash checks. The included guides identify the remaining scientific validation work.

To verify a downloaded ZIP on macOS or Linux, keep its checksum file alongside it and run:

```bash
shasum -a 256 -c CS544_RAG_Project_Artifact.zip.sha256
shasum -a 256 -c CS544_Triage_Project_Artifact.zip.sha256
```

Extract the desired ZIP and open its START_HERE.md. If working from the Git clone, copy the expanded data, cache, results, and RAG archive folders into the corresponding project folder as described in each project guide.
