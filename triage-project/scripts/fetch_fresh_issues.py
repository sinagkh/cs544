"""Fetch issues created since 2025-01-01 from the five NLBSE'24 repositories (unauthenticated GitHub API).

Writes data/raw/github/fresh_issues.jsonl. Uses at most 2 pages of 100 per (repo, label); stops
early when the rate limit runs low. Pull requests are skipped.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

LABELS = {
    "facebook/react": {"bug": "Type: Bug", "feature": "Type: Feature Request", "question": "Type: Question"},
    "tensorflow/tensorflow": {"bug": "type:bug", "feature": "type:feature", "question": "type:support"},
    "microsoft/vscode": {"bug": "bug", "feature": "feature-request", "question": "*question"},
    "opencv/opencv": {"bug": "bug", "feature": "feature", "question": "question (invalid tracker)"},
    "bitcoin/bitcoin": {"bug": "Bug", "feature": "Feature", "question": "Questions and Help"},
}
SINCE = "2025-01-01"
out = Path("data/raw/github/fresh_issues.jsonl")
seen, rows = set(), []


def get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "cs544-triage-study"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()), int(r.headers.get("X-RateLimit-Remaining", "0"))


for repo, labels in LABELS.items():
    for ours, theirs in labels.items():
        for page in (1, 2):
            q = urllib.parse.urlencode(dict(labels=theirs, state="all", sort="created", direction="desc",
                                            per_page=100, page=page))
            data, remaining = get(f"https://api.github.com/repos/{repo}/issues?{q}")
            older = False
            for it in data:
                if "pull_request" in it:
                    continue
                if it["created_at"][:10] < SINCE:
                    older = True
                    continue
                if it["id"] in seen:
                    continue
                seen.add(it["id"])
                rows.append(dict(repo=repo, label=ours, number=it["number"], created_at=it["created_at"],
                                 title=it["title"], body=it.get("body") or "",
                                 labels=[l["name"] for l in it.get("labels", [])], url=it["html_url"]))
            print(repo, ours, "page", page, "->", len(data), "items; remaining", remaining, flush=True)
            if older or len(data) < 100 or remaining < 3:
                break
            time.sleep(1)
        if remaining < 3:
            break
out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(len(rows), "issues written to", out)
