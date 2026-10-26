"""open_pr for the Git path: one draft pull request per topic, with the report, the gate statuses and the run record.

Runs inside the alignr workflow with the workflow's GITHUB_TOKEN. That account can open pull requests but can't
merge them (the open-questions status blocks the merge, and only a writer can close the questions).

    python scripts/open_prs.py OUT_DIR --issue N --dita-ot success|failure
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alignr.agent import report_markdown  # noqa: E402


def sh(*args, check=True, capture=True) -> str:
    r = subprocess.run(args, cwd=ROOT, text=True, capture_output=capture)
    if check and r.returncode:
        raise SystemExit(f"{' '.join(args)} failed: {r.stderr}")
    return (r.stdout or "").strip()


def gh_api(path, *fields, method="POST"):
    args = ["gh", "api", "-X", method, path]
    for k, v in fields:
        args += ["-f", f"{k}={v}"]
    return sh(*args)


def status(repo, sha, context, ok, desc, url=""):
    gh_api(f"repos/{repo}/statuses/{sha}", ("state", "success" if ok else "failure"), ("context", context),
           ("description", desc[:139]), *([("target_url", url)] if url else []))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("out")
    a.add_argument("--issue", type=int, default=0)
    a.add_argument("--dita-ot", default="skipped")
    a.add_argument("--open-prs", default="true")
    args = a.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    out = ROOT / args.out
    r = json.loads((out / "report.json").read_text())
    gen, run = r["generation"], r["run"]
    issue_created = None
    if args.issue:
        issue_created = json.loads(sh("gh", "api", f"repos/{repo}/issues/{args.issue}"))["created_at"]
    sh("git", "config", "user.name", "github-actions[bot]")
    sh("git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    sh("git", "fetch", "-q", "origin", "main")
    prs = []
    for t in r["topics"]:
        if t["withheld"] or args.open_prs != "true":
            continue
        tid, slot = t["topic_id"], t["slot"]
        branch = f"alignr/{gen}-{slot}-{run}"
        sh("git", "checkout", "-q", "-B", branch, "origin/main")
        dst = ROOT / "drafts" / gen / "topics" / f"{tid}.dita"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((out / "topics" / f"{tid}.dita").read_text())
        rep = ROOT / "drafts" / gen / "reports" / f"{tid}.md"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text((out / "reports" / f"{tid}.md").read_text())
        sh("git", "add", str(dst), str(rep))
        sh("git", "commit", "-q", "-m", f"AlignR draft: {tid} ({gen}, run {run})")
        sh("git", "push", "-q", "-f", "origin", branch)
        sha = sh("git", "rev-parse", "HEAD")
        body = (out / "reports" / f"{tid}.md").read_text()
        if args.issue:
            body += f"\nFrom issue #{args.issue} · [workflow run]({run_url})\n"
        url = sh("gh", "pr", "create", "--draft", "--base", "main", "--head", branch,
                 "--title", f"AlignR draft: {tid} ({gen})", "--body", body)
        number = int(url.rstrip("/").split("/")[-1])
        sh("gh", "pr", "edit", str(number), "--add-label", "alignr:draft", check=False)
        g = t["grounding"]
        status(repo, sha, "alignr/grammar", t["grammar"]["valid"], t["grammar"].get("grammar", "DITA grammar"), run_url)
        status(repo, sha, "alignr/rules", not t["rules"].get("errors"), f"{len(t['rules'].get('errors', []))} errors, "
               f"{len(t['rules'].get('warnings', []))} warnings", run_url)
        status(repo, sha, "alignr/grounding", g["grounded"] == g["total"], f"{g['grounded']}/{g['total']} values sourced", run_url)
        status(repo, sha, "alignr/dita-ot", args.dita_ot == "success", f"DITA-OT strict build of the set: {args.dita_ot}", run_url)
        q = len(t["questions"])
        status(repo, sha, "alignr/open-questions", q == 0, f"{q} open draft comment(s): close them before merging", run_url)
        prs.append({"topic": tid, "number": number, "url": url, "questions": q,
                    "opened": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    sh("git", "checkout", "-q", "origin/main")

    done = dt.datetime.now(dt.timezone.utc)
    seconds = None
    if issue_created and prs:
        seconds = int((done - dt.datetime.fromisoformat(issue_created.replace("Z", "+00:00"))).total_seconds())
    record = r | {"source": "github", "issue": args.issue or None, "issue_created": issue_created, "prs": prs,
                  "issue_to_prs_seconds": seconds, "workflow_run": run_url, "dita_ot": args.dita_ot,
                  "finished": done.isoformat(timespec="seconds")}
    if args.issue:
        lines = [report_markdown(r), "", "| Topic | Pull request | Open questions |", "|---|---|---|"]
        lines += [f"| `{p['topic']}` | #{p['number']} | {p['questions']} |" for p in prs]
        if seconds is not None:
            lines += ["", f"Issue to draft pull requests: **{seconds // 60} min {seconds % 60} s** · DITA-OT build of the set: "
                          f"{args.dita_ot} · [workflow run]({run_url})"]
        sh("gh", "issue", "comment", str(args.issue), "--body", "\n".join(lines))
    # The run record: what the scorecard on the AlignR page reads.
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"{run}.json").write_text(json.dumps(record, indent=1))
    (runs / "latest-github.json").write_text(json.dumps(record, indent=1))
    for attempt in range(3):
        sh("git", "checkout", "-q", "-B", "main", "origin/main")
        sh("git", "add", "runs")
        if not sh("git", "status", "--porcelain", "runs"):
            break
        sh("git", "commit", "-q", "-m", f"Run record {run} ({gen}, {len(prs)} draft pull requests)")
        if subprocess.run(["git", "push", "-q", "origin", "main"], cwd=ROOT).returncode == 0:
            break
        sh("git", "fetch", "-q", "origin", "main")
        sh("git", "reset", "-q", "--soft", "origin/main")
    print(json.dumps({"prs": prs, "issue_to_prs_seconds": seconds}, indent=1))


if __name__ == "__main__":
    main()
