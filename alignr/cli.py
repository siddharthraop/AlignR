"""alignr draft | check | stage: the command line the workflows and writers use."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .agent import draft_set, report_markdown, write_outputs
from .checks import check_rules, open_questions, validate_grammar
from .corpus import corpus_dir
from .model import make_model


def cmd_draft(a):
    model = make_model(a.model)
    def emit(e):
        if a.verbose:
            print(json.dumps(e), file=sys.stderr)
    r = draft_set(a.facts, model, emit=emit, parallel=a.parallel, source=a.source)
    out = write_outputs(r, a.out)
    print(report_markdown(r))
    print(f"Wrote {out}")
    m = r["metrics"]
    return 1 if m["grammar_failures_reaching_writer"] else 0


def cmd_check(a):
    """Gates for draft files: grammar, rules, and (with --merge) no open questions."""
    bad = 0
    for f in sorted(Path(p) for pat in a.files for p in (Path().glob(pat) if any(c in pat for c in "*?") else [pat])):
        xml = f.read_bytes()
        g = validate_grammar(xml)
        r = check_rules(xml) if g["valid"] else {"pass": False, "errors": []}
        q = open_questions(xml)
        ok = g["valid"] and r["pass"] and (not a.merge or not q)
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {f}: grammar {'valid' if g['valid'] else 'invalid'}, "
              f"rules {len(r.get('errors', []))} errors, {len(q)} open questions")
        for e in g["errors"][:5]:
            print(f"::error file={f},line={e['line']}::{e['message']}")
        for e in r.get("errors", []):
            print(f"::error file={f}::{e['id']}: {e['message']}")
        if a.merge:
            for t in q:
                print(f"::error file={f}::open question: {t}")
    return 1 if bad else 0


def cmd_stage(a):
    """Copy the TPU set and drop the drafts in place of the held-out topics, for a DITA-OT build of the whole set."""
    src = corpus_dir() / "tpu"
    dst = Path(a.to)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("media"))
    for f in Path(a.drafts).glob("*.dita"):
        shutil.copy(f, dst / "topics" / f.name)
    print(f"Staged {dst} with {len(list(Path(a.drafts).glob('*.dita')))} drafts")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="alignr")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft", help="draft a generation's topic set from its siblings and launch facts")
    d.add_argument("--facts", default="facts/tpu7x.yaml")
    d.add_argument("--model", default="auto", choices=["auto", "gemini", "baseline"])
    d.add_argument("--out", default="out/tpu7x")
    d.add_argument("--parallel", type=int, default=5)
    d.add_argument("--source", default="cli")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(fn=cmd_draft)
    c = sub.add_parser("check", help="run the gates on draft files")
    c.add_argument("files", nargs="+")
    c.add_argument("--merge", action="store_true", help="also fail on open questions (the merge gate)")
    c.set_defaults(fn=cmd_check)
    s = sub.add_parser("stage", help="stage drafts into a copy of the TPU set for DITA-OT")
    s.add_argument("drafts")
    s.add_argument("--to", default="build/tpu")
    s.set_defaults(fn=cmd_stage)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
