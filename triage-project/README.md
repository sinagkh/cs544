# When should an LLM issue triager stay silent?

[Start here for the report, proposal, data download, and current validation status](START_HERE.md).

CS 544 (Machine Learning in Production) project. An issue-triage service labels new GitHub issues
as bug, feature or question, or leaves them for a maintainer. We evaluate it as a deployed,
selective service: in distribution, a year later, and on a newly onboarded repository; under
meaning-preserving rewording; with evidence explanations; and with monitoring and regression tests.
Everything runs locally on an Apple-silicon laptop (Qwen2.5-7B 4-bit through MLX).

* Paper draft: [paper/main.pdf](paper/main.pdf)
* Generated tables: [report/results.md](report/results.md)
* Monitoring: [report/monitoring.md](report/monitoring.md) · Explanations: [report/explanations.md](report/explanations.md)
* Regression-suite mutation check: [report/mutation_check.md](report/mutation_check.md)

## Results at a glance

| Approach | NLBSE'24 test | 2025-26 issues | New repository |
|---|---:|---:|---:|
| TF-IDF + logistic regression | **76.2%** | 67.5% | 60.3% |
| Embedding + logistic regression | 69.6% | 68.1% | 58.8% |
| LLM zero-shot | 68.5% | 71.2% | 68.4% |
| LLM few-shot (6 retrieved examples) | 70.9% | **75.3%** | **68.0%** |

* **Deferral.** Target: at most 10% errors among auto-labeled issues. The naive calibrated
  threshold overshoots everywhere (LLM few-shot: 14.9% in distribution, 20.5% on a new repository);
  the risk-controlled (Learn-then-Test) threshold keeps the target in distribution and a year later
  but auto-labels only 7% of issues; no threshold carries over to a new repository.
* **Rewording.** A newcomer apology flips 16.2% of the few-shot LLM's labels (mostly toward
  *question*); non-native English costs the zero-shot LLM up to 10 points. Deferral keeps almost
  all flips from being auto-applied (0–1.3%).
* **Explanations.** 93% of the LLM's evidence quotes are verbatim, but deleting them moves the
  decision little more than deleting random text (p = 0.12).
* **Monitoring.** Input novelty (distance to the closest labeled history issue) detects a new
  repository within 11 requests; model-confidence signals rarely notice.

## Data

* **NLBSE'24** issue report classification: 3,000 issues from facebook/react, tensorflow/tensorflow,
  microsoft/vscode, bitcoin/bitcoin, opencv/opencv (2016–2023), from
  <https://github.com/nlbse2024/issue-report-classification> (see that repository for its license).
  The training half is split into a labeled history (1,005) and a calibration split (495); the test
  half (1,500) is the in-distribution test set.
* **Fresh 2025–26:** `scripts/fetch_fresh_issues.py` collects issues created since 2025-01-01 from the
  same repositories through the public GitHub API and maps each repository's label names to the
  three types (1,430 single-type issues; balanced sample of 514).
* **Variants:** `scripts/make_variants.py` rewrites 210 probe issues five ways (no code, no template,
  newcomer cue, non-native English by Llama-3.2-3B, non-native + no code).

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
mkdir -p data/raw/nlbse24 && for f in issues_train.csv issues_test.csv; do curl -sSL -o data/raw/nlbse24/$f https://raw.githubusercontent.com/nlbse2024/issue-report-classification/main/data/$f; done
.venv/bin/python -I scripts/fetch_fresh_issues.py      # unauthenticated GitHub API, ~35 requests
.venv/bin/python -I -c "from triage.data import build; print(build()['meta'])"
.venv/bin/python -I scripts/make_variants.py
.venv/bin/python -I scripts/run_llm.py                 # ~6 h on an M4 MacBook Pro
.venv/bin/python -I scripts/explain.py
.venv/bin/python -I scripts/run_classical.py
.venv/bin/python -I scripts/analyze.py
.venv/bin/python -I scripts/monitor_sim.py
.venv/bin/python -I scripts/make_regression_suite.py
.venv/bin/python -I -m pytest                          # unit tests
.venv/bin/python -I -m pytest -m replay                # 40 recorded cases, no GPU
.venv/bin/python -I scripts/mutation_check.py          # injects bugs, replays live
.venv/bin/python -I scripts/make_paper_tables.py && (cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main)
```

Model scores and generations are cached in `cache/llm_cache.sqlite` (the Llama chat template's date
is pinned so cache keys do not change from day to day).

## Team tasks

* **Rewrite audit:** two annotators rate 40 original/rewritten pairs in `report/audit/rewrite_audit.csv`
  (same issue type? information preserved?), blind to each other.
* **More models:** rerun `scripts/run_llm.py --model ...` (for example Llama-3.2-3B or Gemma-2-9B).
* **Larger calibration set:** the risk-controlled threshold needs thousands of labeled issues to
  reach useful coverage; the 2025–26 collection script can supply more.

## Layout

```
src/triage/   data  llm  prompts  models (embeddings, retrieval, classical)  perturb  selective
scripts/      fetch_fresh_issues make_variants run_llm run_classical explain analyze monitor_sim
              make_regression_suite mutation_check make_paper_tables
tests/        test_components.py (unit)  test_regression.py (replay)  regression_cases.json
results/      one JSONL record per (issue, condition, approach); explanations.jsonl
report/       results.md summary.json monitoring.* explanations.md mutation_check.md figures/
paper/        main.tex sections/ tables/ refs.bib main.pdf
```
