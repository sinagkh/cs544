# When should an LLM issue triager stay silent?

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

All classifiers see the same 1,500 characters of each issue.

| Approach | NLBSE'24 test | 2025-26 issues | New repository |
|---|---:|---:|---:|
| TF-IDF + logistic regression | **75.5%** | 66.3% | 60.4% |
| Embedding + logistic regression | 69.3% | 68.5% | 57.5% |
| LLM zero-shot | 68.5% | 71.2% | 68.4% |
| LLM few-shot (6 retrieved examples) | 70.9% | **75.3%** | **68.0%** |

* **Deferral.** Target: at most 10% errors among auto-labeled issues. The naive threshold
  overshoots everywhere (10.2–13.3% in distribution, up to 27.6% under shift). With the temperature
  and threshold candidates fitted on one third of the calibration issues and Learn-then-Test
  certification on the other two thirds, 10% is certified only for TF-IDF (11.3% coverage, 6/170
  errors). The few-shot LLM is first certified at a 20% target (62.9% coverage, 19.1% risk); nothing
  certified carries over to a new repository.
* **New repository.** Leave-one-repository-out (LORO) calibration restores conformal coverage (TF-IDF
  75.6% → 87.9%, few-shot 84.3% → 88.3%) but not threshold risk (few-shot naive: 20.8%).
* **Rewording.** The newcomer sentence flips 16.2% of the few-shot LLM's labels. Controls show the
  phrase "sorry if this is a silly question" drives it (+14.8 points toward *question* vs. a neutral
  sentence, 95% CI [10.0, 20.0]); newcomer status alone +2.9, apology alone 0. Deferral keeps flips
  from being auto-applied (0–1.9%) but halves the share auto-labeled (33.8% → 16.2%).
* **Explanations.** 93% of the LLM's evidence quotes are verbatim; whether they drive the decision is
  inconclusive (deletion test, p = 0.115).
* **Monitoring** (independent clean calibration stream). Input novelty detects a new repository in
  13 requests (100% of runs) and 2025–26 traffic in 44 (96%), with under 1% false alarms;
  model-confidence signals detect at most 27% of runs; disagreement has 5–7% false alarms.

## Data

* **NLBSE'24** issue report classification: 3,000 issues from facebook/react, tensorflow/tensorflow,
  microsoft/vscode, bitcoin/bitcoin, opencv/opencv (2016–2023), from
  <https://github.com/nlbse2024/issue-report-classification> (see that repository for its license).
  The training half is split into a labeled history (1,005) and a calibration split (495: one third
  fits the temperature and threshold candidates, two thirds certify); the test half (1,500) is the
  in-distribution test set. For the new-repository condition, calibration issues are also scored
  with the target repository held out (`calib_xrepo`) for LORO calibration.
* **Fresh 2025–26:** `scripts/fetch_fresh_issues.py` collects issues created since 2025-01-01 from the
  same repositories through the public GitHub API and maps each repository's label names to the
  three types (1,430 single-type issues; balanced sample of 514).
* **Variants:** `scripts/make_variants.py` rewrites 210 probe issues: no template, newcomer sentence,
  non-native English by Llama-3.2-3B (161 valid after automatic checks), the reduced-input
  conditions no code and non-native + no code, and four cue controls (newcomer status only, apology
  only, "silly question" only, neutral sentence).

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

* **Rewrite audit (pending, cited as a threat in the paper):** two annotators rate 40
  original/rewritten pairs in `report/audit/rewrite_audit.csv` (same issue type? information
  preserved?), blind to each other.
* **More models:** rerun `scripts/run_llm.py --model ...` (for example Llama-3.2-3B or Gemma-2-9B).
* **Larger calibration set:** with 330 certification issues only TF-IDF certifies a 10% target;
  the 2025–26 collection script can supply more labeled issues.
* **Regression-suite gaps:** add cases that catch prompt-content and preprocessing bugs (label
  definitions removed: 2/40 caught; body cut to 300 characters: 1/40).

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
