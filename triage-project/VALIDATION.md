# Artifact validation

Checked October 21, 2026 against this revised snapshot, using the original project's installed dependencies on Apple silicon.

- 11 unit tests and 40 cached replay tests passed (51 total), importing the staged source.
- The staged analysis regenerated the supplied JSON result summary exactly from the packaged predictions.
- The full project report has seven pages; the existing proposal has two pages and 10 linked papers.
- SQLite snapshots passed database integrity checks.
- ZIP contents passed CRC checks and were verified against the included SHA-256 manifest.

The local dependency environment and tokenizer cache were available during validation. Live model inference and mutation experiments were not rerun during packaging. The revised mutation results are included in `report/mutation_check.md`. Scientific follow-ups are listed in `START_HERE.md`.
