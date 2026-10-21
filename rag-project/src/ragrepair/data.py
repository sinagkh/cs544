"""Build the evaluation set: HotpotQA questions, a shared corpus, and outdated document versions."""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .text import contains_phrase, normalize_answer, tokens

MONTHS = (
    "january february march april may june july august september october november december".split()
)


def load_hotpot(parquet_path: str | Path) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    rows = []
    for r in df.itertuples(index=False):
        ctx = [
            (str(t), [str(s) for s in sents])
            for t, sents in zip(r.context["title"], r.context["sentences"])
        ]
        sf = [(str(t), int(s)) for t, s in zip(r.supporting_facts["title"], r.supporting_facts["sent_id"])]
        rows.append(
            dict(id=str(r.id), question=str(r.question), answer=str(r.answer), type=str(r.type),
                 supporting_facts=sf, context=ctx)
        )
    return rows


def para_text(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())


def answer_kind(ans: str) -> str:
    a = ans.strip()
    if re.fullmatch(r"\d{4}", a) and 1000 <= int(a) <= 2100:
        return "year"
    if any(m in a.lower() for m in MONTHS) and re.search(r"\d{4}", a):
        return "date"
    if re.search(r"\d", a):
        return "number"
    return "entity"


def _stable_rng(*parts: str) -> random.Random:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def eligible(row: dict) -> dict | None:
    """Return answer-location info if the question suits controlled evidence faults."""
    ans = row["answer"]
    # Bridge questions only: in comparison questions the answer is usually the title of its own
    # article, so an "outdated version" of that article cannot carry a different answer value.
    if row["type"] != "bridge":
        return None
    if normalize_answer(ans) in {"yes", "no"} or not (1 <= len(tokens(ans)) <= 6):
        return None
    # Annotation artifacts where two fields were glued together, e.g. "1755 or 1757July".
    if re.search(r"\d[A-Za-z]{3,}|[a-z]{2,}[A-Z][a-z]", ans):
        return None
    gold_titles = sorted({t for t, _ in row["supporting_facts"]})
    if len(gold_titles) != 2:
        return None
    holders = [t for t, sents in row["context"] if contains_phrase(para_text(sents), ans)]
    # The answer must appear in exactly one of the ten paragraphs, and that paragraph must be gold.
    if len(holders) != 1 or holders[0] not in gold_titles:
        return None
    title = holders[0]
    sents = dict(row["context"])[title]
    sup_ids = [s for t, s in row["supporting_facts"] if t == title]
    hit = [s for s in sup_ids if s < len(sents) and contains_phrase(sents[s], ans)]
    if not hit:
        return None
    return dict(gold_titles=gold_titles, answer_title=title, answer_sent_id=hit[0])


def _looks_like_name(s: str) -> bool:
    toks = s.split()
    return 1 <= len(toks) <= 5 and all(t[:1].isupper() for t in toks if t.isalpha()) and not re.search(r"\d", s)


def make_stale_answer(ans: str, kind: str, candidates: list[str], para: str, rng: random.Random) -> str | None:
    """Pick an outdated value of the same kind as the answer.

    Years, dates and numbers are shifted. Entities are replaced by the title of one of the
    question's own distractor articles, which HotpotQA chose to resemble the gold entities.
    """
    if kind == "year":
        y = int(ans)
        return str(y + rng.choice([-1, 1]) * rng.randint(3, 15))
    if kind == "date":
        m = re.search(r"\d{4}", ans)
        y = int(m.group())
        return ans[: m.start()] + str(y + rng.choice([-1, 1]) * rng.randint(2, 9)) + ans[m.end():]
    if kind == "number":
        m = re.search(r"\d[\d,]*(\.\d+)?", ans)
        raw = m.group().replace(",", "")
        if re.fullmatch(r"\d{4}", raw) and 1000 <= int(raw) <= 2100:  # a year inside a phrase
            y = int(raw) + rng.choice([-1, 1]) * rng.randint(3, 15)
            return ans[: m.start()] + str(y) + ans[m.end():]
        val = float(raw)
        new = val * rng.choice([0.6, 0.7, 1.4, 1.8]) if val >= 5 else val + rng.randint(2, 6)
        new_s = f"{int(round(new)):,}" if "," in m.group() else str(int(round(new)))
        if new_s == m.group():
            new_s = str(int(round(new)) + 1)
        return ans[: m.start()] + new_s + ans[m.end():]
    n = len(tokens(ans))
    cands = [
        e for e in candidates
        if abs(len(tokens(e)) - n) <= 2
        and normalize_answer(e) != normalize_answer(ans)
        and not contains_phrase(e, ans) and not contains_phrase(ans, e)
        and not contains_phrase(para, e)
    ]
    if _looks_like_name(ans):
        named = [e for e in cands if _looks_like_name(e)]
        cands = named or cands
    cands.sort(key=lambda e: (abs(len(tokens(e)) - n), e))
    return cands[0] if cands else None


def replace_answer(sentences: list[str], ans: str, new: str) -> list[str]:
    pat = re.compile(re.escape(ans), re.IGNORECASE)
    return [pat.sub(new, s) for s in sentences]


def _date_str(d: date) -> str:
    return d.isoformat()


def build(parquet_path: str | Path, out_path: str | Path, pool_size: int = 1200, n_dev: int = 30,
          n_test: int = 150, k: int = 5, seed: int = 544, exclude_ids: set[str] | None = None) -> dict:
    """`exclude_ids`: questions used by an earlier prototype; they stay in the corpus pool but are
    never selected for dev or test (they remain available for threshold calibration)."""
    rows = load_hotpot(parquet_path)
    rng = random.Random(seed)
    rng.shuffle(rows)

    pool = []
    for r in rows:
        info = eligible(r)
        if info:
            pool.append({**r, **info})
        if len(pool) >= pool_size:
            break

    # Shared corpus: every paragraph of every pooled question, deduplicated by title.
    corpus, by_title = [], {}
    conflicts = set()
    for q in pool:
        for t, sents in q["context"]:
            text = para_text(sents)
            if t in by_title:
                if corpus[by_title[t]]["text"] != text:
                    conflicts.add(t)
                continue
            by_title[t] = len(corpus)
            corpus.append(dict(doc_id=f"d{len(corpus):05d}", title=t, sentences=[s.strip() for s in sents],
                               text=text))
    for d in corpus:
        r_ = _stable_rng("date", d["title"])
        d["date"] = _date_str(date(2023, 1, 1) + timedelta(days=r_.randint(0, 729)))
        d["stale"] = False

    # Keep questions whose clean dense top-k holds both gold paragraphs, so that in the clean
    # condition the evidence is present and each injected fault is the only reason it goes missing.
    import numpy as np

    from .embed import INDEX_ENCODER, embed_docs, embed_queries

    E = embed_docs(corpus, INDEX_ENCODER)
    np.save(Path(out_path).with_name("corpus_emb.npy"), E)
    usable = [q for q in pool if not any(t in conflicts for t, _ in q["context"])]
    Q = embed_queries([q["question"] for q in usable], INDEX_ENCODER)
    S = Q @ E.T
    kept = []
    for qi, q in enumerate(usable):
        top = {corpus[i]["title"] for i in np.argsort(-S[qi])[:k]}
        if all(t in top for t in q["gold_titles"]):
            kept.append(q)

    selected = []
    for q in kept:
        doc = corpus[by_title[q["answer_title"]]]
        kind = answer_kind(q["answer"])
        r_ = _stable_rng("stale", q["id"])
        distractor_titles = [re.sub(r"\s*\(.*?\)\s*$", "", t) for t, _ in q["context"] if t not in q["gold_titles"]]
        new = make_stale_answer(q["answer"], kind, distractor_titles, doc["text"], r_)
        if not new:
            continue
        stale_sents = replace_answer(doc["sentences"], q["answer"], new)
        stale_text = para_text(stale_sents)
        if contains_phrase(stale_text, q["answer"]) or not contains_phrase(stale_text, new):
            continue
        cur = date.fromisoformat(doc["date"])
        stale_date = cur - timedelta(days=r_.randint(3 * 365, 6 * 365))
        stale_doc = dict(doc_id=doc["doc_id"] + "_old", title=doc["title"], sentences=stale_sents,
                         text=stale_text, date=_date_str(stale_date), stale=True, base_id=doc["doc_id"])
        selected.append(dict(
            id=q["id"], question=q["question"], answer=q["answer"], type=q["type"],
            gold_titles=q["gold_titles"], answer_title=q["answer_title"],
            answer_doc_id=doc["doc_id"], answer_sent_id=q["answer_sent_id"],
            answer_kind=kind, stale_answer=new, stale_doc=stale_doc,
        ))

    rng.shuffle(selected)
    exclude_ids = exclude_ids or set()
    selected = [q for q in selected if q["id"] not in exclude_ids] + [q for q in selected if q["id"] in exclude_ids]
    for q in selected:
        q["prototype_overlap"] = q["id"] in exclude_ids
    dev, test = selected[:n_dev], selected[n_dev:n_dev + n_test]
    out = dict(
        meta=dict(seed=seed, pool_size=len(pool), corpus_size=len(corpus), title_conflicts=len(conflicts),
                  retrieval_filter_kept=len(kept), selectable=len(selected), k=k, n_dev=n_dev, n_test=n_test),
        corpus=corpus, selected=selected, dev=dev, test=test,
    )
    Path(out_path).write_text(json.dumps(out, indent=1))
    return out


STALE_PROMPT = (
    "Question: {question}\nCorrect answer: {answer}\n\n"
    "Write one different answer that is plausible but wrong, of exactly the same type and format as the "
    "correct answer (for example another person's full name for a person, another band for a band, "
    "another place for a place, another quantity in the same units for a quantity). "
    "Reply with only that answer."
)


def valid_stale_value(new: str | None, q: dict, para: str) -> bool:
    if not new or "\n" in new or not (1 <= len(tokens(new)) <= 8) or "insufficient" in new.lower():
        return False
    if normalize_answer(new) == normalize_answer(q["answer"]):
        return False
    if contains_phrase(new, q["answer"]) or contains_phrase(q["answer"], new) or contains_phrase(para, new):
        return False
    return True


def with_stale_value(q: dict, doc: dict, new: str) -> dict | None:
    sents = replace_answer(doc["sentences"], q["answer"], new)
    text = para_text(sents)
    if contains_phrase(text, q["answer"]) or not contains_phrase(text, new):
        return None
    stale_doc = dict(q["stale_doc"], sentences=sents, text=text)
    return dict(q, stale_answer=new, stale_doc=stale_doc, stale_source="llm")


def refresh_entity_stale(ds: dict, generate) -> dict:
    """Replace rule-based outdated values for entity answers with model-written ones.

    `generate(messages, temperature, seed) -> str`. Questions with no valid value are dropped,
    and dev/test are re-cut from the remaining questions in their original shuffled order.
    """
    corpus = {d["doc_id"]: d for d in ds["corpus"]}
    out, dropped = [], 0
    need = ds["meta"]["n_dev"] + ds["meta"]["n_test"]
    rest = []
    for q in ds["selected"]:
        if len(out) >= need or q.get("prototype_overlap"):
            rest.append(q)  # held out (used only to calibrate thresholds)
            continue
        if q["answer_kind"] != "entity":
            out.append(dict(q, stale_source="rule"))
            continue
        doc = corpus[q["answer_doc_id"]]
        msgs = [{"role": "system", "content": "You help build test data. Reply with only the requested text."},
                {"role": "user", "content": STALE_PROMPT.format(question=q["question"], answer=q["answer"])}]
        new_q = None
        for temp, seed in ((0.0, 0), (0.8, 1), (0.8, 2)):
            cand = generate(msgs, temp, seed).strip().strip('"').strip().rstrip(".")
            if valid_stale_value(cand, q, doc["text"]):
                new_q = with_stale_value(q, doc, cand)
                if new_q:
                    break
        if new_q:
            out.append(new_q)
        else:
            dropped += 1
    nd = ds["meta"]["n_dev"]
    ds = dict(ds, dev=out[:nd], test=out[nd:need], selected=out + rest)
    ds["meta"] = dict(ds["meta"], stale_refresh_dropped=dropped)
    return ds


def load_dataset(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
