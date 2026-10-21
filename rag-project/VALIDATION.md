# Artifact validation

Checked on October 21, 2026, using the staged source and files that are included in this ZIP, with the original project's installed Python dependencies on Apple silicon.

- 7 unit tests and 35 cached replay tests passed (42 total).
- The main analysis script ran using the packaged data and recorded predictions.
- The regenerated JSON result summary matched the original report summary exactly.
- The full project report PDF has 7 pages.
- The proposal PDF has exactly two pages; its Word version contains 10 unique linked paper references.
- Included SQLite snapshots passed their database integrity checks.

The local dependency environment and tokenizer cache were available during these checks. Each teammate must install dependencies, and initial tokenizer downloads may require network access. Live model inference was outside this packaging validation. `START_HERE.md` identifies the scientific validation work still pending.

The ZIP is also checked for CRC errors and verified against `SHA256SUMS.txt` after creation.
