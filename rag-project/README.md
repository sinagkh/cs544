# What Reaches the Model? Trace-based diagnosis and repair of component faults in RAG

[Start here for the report, proposal, data download, and current validation status](START_HERE.md).

CS 544 (Machine Learning in Production) project. A small but complete retrieval-augmented QA
service runs locally on an Apple-silicon laptop. Six production-motivated faults are injected at
four pipeline stages, and twelve runtime repair policies are compared on success, damage to
answers that were already right, abstention, cost, and how quickly each signal detects a faulty
deploy.

* Paper draft: [paper/main.pdf](paper/main.pdf) (source in `paper/`)
* Generated tables: [report/results.md](report/results.md), string-match-only version in
  [report/strict/results.md](report/strict/results.md)
* Monitoring simulation: [report/monitoring_Llama-3.2-3B.md](report/monitoring_Llama-3.2-3B.md)
* Regression-suite mutation check: [report/mutation_check.md](report/mutation_check.md)
* Earlier prototype (BM25, synthetic rank demotion, two models): `archive/v1/`

## Results at a glance (Llama-3.2-3B, 150 held-out questions)

| Policy | Success under faults [95% CI] | Correct clean answers broken | Prompt tokens per case |
|---|---:|---:|---:|
| No repair | 26.8% [22.7, 31.2] | – | 660 |
| Retry | 27.6% | 0.0% | 857 |
| Self-check (model verifies itself) | 20.7% | 24.3% | 1,491 |
| Repair-all (every repair, every request) | 46.4% | 15.9% | 2,380 |
| Trace checks only | 55.9% [49.7, 62.0] | 1.9% | 1,154 |
| **Diagnose (trace checks + text-match support)** | **56.9% [50.7, 63.1]** | 4.7% | 1,477 |
| Diagnose with MiniCheck support | 59.3% | 7.5% | 2,110 |

Trace checks detect each global fault a median of **4 requests** after a faulty deploy (0.1% false
alarms); the model's own verifier needs 34–93 requests and misses most overflow deploys.

## The system under test

```
question -> dense retrieval (bge-small, top 5) -> prompt template -> serving runtime (local LLM) -> "Answer / Evidence"
                 \-> BM25 index (fallback)
```

Every call is logged: retrieved documents with similarity scores and dates, the passages exactly
as rendered, tokens sent vs. tokens the server processed, and the raw output.

## Injected faults (`src/ragrepair/faults.py`)

| Condition | What breaks | Production cause |
|---|---|---|
| `clean` | nothing | control |
| `encoder` | queries embedded with all-MiniLM-L6-v2, index built with bge-small-en-v1.5 | query service falls back to a library default |
| `truncation` | every passage cut to 100 characters | `text[:max_chars]` misconfiguration |
| `overflow` | server keeps only the last 512 prompt tokens | unset context window (`num_ctx` default) |
| `stale` | outdated version of the answer passage indexed beside the current one | index not deduplicated after an update |
| `absent` | answer passage missing from the index (abstaining is correct) | content gap |
| `encoder+truncation` | two faults together | independent faults co-occur |

## Repair policies (`src/ragrepair/policies.py`)

All start from the same initial answer and may make at most two extra model calls.

* **Generic:** no repair; retry; self-check with the LLM verifier or with MiniCheck; repair-all.
* **Trace checks** (no model call): rendered passage shorter than retrieved text → render in full;
  two versions of a document → keep the newest; server processed fewer tokens than sent → fit to
  the reported window; top similarity below the 5th percentile of clean traffic → BM25 fallback.
* **Diagnose:** trace checks, then a support check (text match, LLM verifier, or MiniCheck) with
  re-check after each repair, a retrieval ladder (widen to 10, then add BM25), and abstention.
* **Ablations** of Diagnose: no re-check, no abstention, no trace checks.

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/python -c "from huggingface_hub import hf_hub_download as d; d('hotpotqa/hotpot_qa','distractor/validation-00000-of-00001.parquet',repo_type='dataset',local_dir='data/raw')"
.venv/bin/python -I scripts/build_dataset.py          # corpus, dense-retrieval filter, splits
.venv/bin/python -I scripts/make_stale_values.py      # outdated values for the stale fault
.venv/bin/python -I scripts/embed_store.py            # embeddings + retrieval-confidence threshold
.venv/bin/python -I -m pytest                          # unit tests
.venv/bin/python -I scripts/run_experiment.py --split test --model mlx-community/Llama-3.2-3B-Instruct-4bit
.venv/bin/python -I scripts/judge_answers.py results/*.jsonl
.venv/bin/python -I scripts/analyze.py results/Llama-3.2-3B-Instruct-4bit_test.jsonl --out report
.venv/bin/python -I scripts/analyze.py results/Llama-3.2-3B-Instruct-4bit_test.jsonl --out report/strict --strict
.venv/bin/python -I scripts/monitor_sim.py results/Llama-3.2-3B-Instruct-4bit_test.jsonl
.venv/bin/python -I scripts/verifier_study.py results/Llama-3.2-3B-Instruct-4bit_test.jsonl   # uses the Claude CLI
.venv/bin/python -I scripts/make_regression_suite.py results/Llama-3.2-3B-Instruct-4bit_test.jsonl --repo mlx-community/Llama-3.2-3B-Instruct-4bit
.venv/bin/python -I -m pytest -m replay               # replays the 35 saved cases from recorded responses
.venv/bin/python -I scripts/mutation_check.py          # disables each diagnostic and replays live
.venv/bin/python -I scripts/make_paper_tables.py && (cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main)
```

The test run takes about 3 hours on an M4 MacBook Pro. All model responses and MiniCheck scores
are cached in `cache/` (the Llama chat template's date is pinned in `llm.py`, so cache keys do not
change from day to day), so reruns, shared prompts across policies, and the regression suite replay
without regenerating. To add models later, rerun `run_experiment.py` with `--model`; a partial
Qwen2.5-7B run is kept in `results/pending/` and resumes where it stopped.

## Team tasks

* **Judge audit:** fill `report/audit/judge_audit.csv` (two annotators, blind), then run
  `scripts/score_audit.py` for agreement and Cohen's kappa.
* **More models:** Qwen2.5-7B and Gemma-2-9B with the same harness.
* **Demo:** deploy a bad template mid-stream, show the trace-check alarm, run the regression suite.

## Layout

```
src/ragrepair/  data embed retriever store faults prompts llm system policies support judge evaluate
scripts/        build_dataset make_stale_values embed_store run_experiment judge_answers analyze
                monitor_sim verifier_study make_regression_suite mutation_check make_audit_sheet
                score_audit make_paper_tables
tests/          test_components.py (unit)  test_regression.py (replay)  regression_cases.json
results/        one JSONL record per (model, question, condition, policy)
report/         results.md summary.json figures/ monitoring_* mutation_check.md audit/ strict/ dev/
paper/          main.tex sections/ tables/ refs.bib main.pdf
archive/v1/     prototype data, results and report
.github/workflows/ci.yml   unit tests + replayed regression suite
```
