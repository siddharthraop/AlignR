"""The agent loop: draft each topic, gate it, revise failed fields up to two times, then ask instead of guessing."""
from __future__ import annotations

import copy
import datetime as dt
import io
import json
import re
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from . import __version__
from .checks import check_grounding, check_rules, open_questions, validate_grammar
from .compare import compare
from .corpus import Corpus
from .facts import Facts, load_facts
from .model import Usage
from .schema import build_schema, validate_shape
from .serialise import safe_fixes, serialise
from .skeleton import Plan, plan_topic

MAX_REVISIONS = 2
EDIT_RATIO_TARGET = 0.40


# ---- paths into the JSON -------------------------------------------------------------------------------------
def _split(path: str) -> list:
    out = []
    for part in path.split("."):
        m = re.fullmatch(r"([^\[]+)((?:\[\d+\])*)", part)
        out.append(m.group(1))
        out += [int(i) for i in re.findall(r"\[(\d+)\]", m.group(2))]
    return out


def _get(doc, path):
    cur = doc
    for p in _split(path):
        cur = cur[p]
    return cur


def _set(doc, path, value):
    parts = _split(path)
    cur = doc
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def merge(old: dict, new: dict, failures: list[dict]) -> dict:
    """Take only the failed fields from the revision; everything else stays as it was. Questions are merged."""
    if any(f["path"] in ("", "xml") for f in failures):
        return new
    out = copy.deepcopy(old)
    for f in failures:
        path = re.sub(r"\.(text|unit|label)$", "", f["path"])
        try:
            _set(out, path, copy.deepcopy(_get(new, path)))
        except (KeyError, IndexError, TypeError):
            try:
                _set(out, path, copy.deepcopy(_get(new, path.rsplit(".", 1)[0] if "." in path else path)))
            except (KeyError, IndexError, TypeError):
                pass
    seen = {(q.get("about"), q.get("text")) for q in out.get("questions", [])}
    for q in new.get("questions", []) or []:
        if (q.get("about"), q.get("text")) not in seen:
            out.setdefault("questions", []).append(q)
    return out


def resolve(doc: dict, failures: list[dict], plan: Plan, facts: Facts) -> list[str]:
    """After two revisions, a field that still fails becomes a question for a person. Nothing is dropped silently."""
    notes = []
    for f in failures:
        path, problem = f["path"], f["problem"]
        try:
            val = _get(doc, path)
        except (KeyError, IndexError, TypeError):
            val = None
        if re.search(r"\.paras\[\d+\]\.text$", path) and isinstance(val, str):
            _set(doc, path, "")
            doc.setdefault("questions", []).append({"for": "writer", "about": "gap", "text":
                f"AlignR removed a draft sentence that failed the checks ({problem}): \"{val}\". Write it from a source, or leave it out."})
            notes.append(f"{path}: removed, now a question")
        elif path.endswith(".unit"):
            row = _get(doc, path.rsplit(".", 1)[0])
            v = next((v for v in facts.spec_values() if v.key == row.get("key")), None)
            if v:
                row["unit"] = v.unit or "none"
                doc.setdefault("questions", []).append({"for": "writer", "about": "gap", "text":
                    f"AlignR set the unit for {v.key} to {v.unit or 'none'} from {v.source} after the draft failed twice ({problem}). Confirm it."})
                notes.append(f"{path}: unit set from the facts, now a question")
        elif isinstance(val, str) and (path in ("title", "shortdesc") or re.search(r"\.(title|label|columns\[\d+\])$", path)):
            clean = re.sub(r"\{(?!" + "|".join(map(re.escape, facts.allowed_keys())) + r")[a-z0-9-]+\}", "", val or "")
            fallback = "{" + facts.product_key + "}" if path == "title" else ""
            if path == "shortdesc" and plan.stub:
                fallback = plan.about[:1].upper() + plan.about[1:] + "." if plan.about else ""
            _set(doc, path, " ".join(clean.split()) or fallback)
            doc.setdefault("questions", []).append({"for": "writer", "about": "gap",
                                                    "text": f"Check {path}: {problem}. AlignR removed what had no source."})
            notes.append(f"{path}: cleaned, now a question")
        else:
            doc.setdefault("questions", []).append({"for": "writer", "about": "gap", "text": f"{path}: {problem}"})
            notes.append(f"{path}: now a question")
    return notes


# ---- one topic -------------------------------------------------------------------------------------------------
def check_all(doc, plan, facts, corpus, schema, run):
    shape = [{"path": p, "problem": m, "check": "schema"} for p, m in validate_shape(doc, schema)]
    g = check_grounding(doc, plan, facts, corpus) if not shape else None
    xml = serialise(doc, plan, facts, run) if not shape else b""
    gram = validate_grammar(xml) if xml else {"valid": False, "errors": [{"line": 0, "message": "not serialised"}]}
    rules = check_rules(xml, plan) if gram["valid"] else {"pass": False, "errors": [], "warnings": []}
    failures = list(shape)
    if g:
        failures += [f | {"check": "grounding"} for f in g.failures]
    if xml and not gram["valid"]:
        failures += [{"path": "xml", "problem": e["message"], "check": "grammar"} for e in gram["errors"][:5]]
    for e in rules.get("errors", []):
        path = "shortdesc" if e["id"] == "SD-1" else next(
            (f"body.{s['id']}.table.columns" for s in plan.sections if e["id"] == "ALR-2" and s.get("fixed_columns")), "xml")
        failures.append({"path": path, "problem": f"{e['id']}: {e['message']}", "check": "rules"})
    return failures, xml, gram, rules, g


def draft_topic(plan: Plan, facts: Facts, corpus: Corpus, model, run: dict, emit: Callable = lambda e: None) -> dict:
    t0 = time.time()
    schema, h = build_schema(plan, facts, corpus)
    run = run | {"schema": h, "model": getattr(model, "id", model.name)}
    usage = Usage()
    stages = []

    def stage(name, **kw):
        e = {"topic": plan.topic_id, "stage": name, **kw}
        stages.append(e)
        emit(e)

    stage("find_siblings", siblings=[h_ for _, h_ in plan.siblings], stub=plan.stub)
    stage("load_facts", values=len(facts.values), conflicts=[c.id + " " + c.kind for c in plan.conflicts])
    stage("build_schema", hash=h)
    error = None
    try:
        doc = model.generate(plan, facts, corpus, schema, usage)
    except Exception as e:  # a model failure still yields a stub with a question, not a crash
        error = str(e)[:300]
        doc = {"title": "{" + facts.product_key + "} " + plan.slot.replace("-", " "), "shortdesc": "",
               "outline": [{"title": "Overview", "facts": [facts.statements[0]["id"]]}], "questions": []}
        plan = Plan(**{**plan.__dict__, "stub": True})
        schema, h = build_schema(plan, facts, corpus)
    stage("generate_topic", model=getattr(model, "id", model.name), calls=usage.calls, error=error)
    fixes_all, revisions, history = [], 0, []
    while True:
        fixes = safe_fixes(doc, plan, facts)
        fixes_all += fixes
        failures, xml, gram, rules, g = check_all(doc, plan, facts, corpus, schema, run)
        history.append({"attempt": revisions + 1, "failures": failures, "fixes": fixes})
        stage("checks", attempt=revisions + 1, failures=len(failures), fixes=fixes)
        if not failures or revisions >= MAX_REVISIONS or error:
            break
        revisions += 1
        try:
            new = model.revise(plan, facts, corpus, schema, usage, doc, failures)
            doc = merge(doc, new, failures)
        except Exception as e:
            error = str(e)[:300]
            break
        stage("revise", attempt=revisions, fields=sorted({f["path"] for f in failures}))
    resolved = []
    if failures:
        resolved = resolve(doc, [f for f in failures if f["check"] != "grammar"], plan, facts)
        safe_fixes(doc, plan, facts)
        failures, xml, gram, rules, g = check_all(doc, plan, facts, corpus, schema, run)
    withheld = not gram["valid"]
    questions = open_questions(xml) if xml else []
    stage("open_pr" if not withheld else "withheld", questions=len(questions))
    published = f"topics/{plan.topic_id}.dita"
    cmp = compare(xml, published, corpus) if xml and not withheld else {"edit_ratio": None}
    return {
        "topic_id": plan.topic_id, "slot": plan.slot, "type": plan.type, "stub": plan.stub,
        "has_sibling": plan.has_sibling, "siblings": [h_ for _, h_ in plan.siblings],
        "schema_hash": h, "schema": schema, "model": getattr(model, "id", model.name), "model_error": error,
        "answer": doc, "xml": xml.decode() if xml else "", "withheld": withheld,
        "grammar": gram, "rules": {k: v for k, v in rules.items()},
        "grounding": g.as_dict() if g else {"total": 0, "grounded": 0, "failures": [], "values": []},
        "remaining_failures": failures, "fixes": fixes_all, "revisions": revisions, "resolved": resolved,
        "history": history, "questions": questions, "conflicts": [c.__dict__ for c in plan.conflicts],
        "usage": usage.as_dict(), "ms": int((time.time() - t0) * 1000), "stages": stages,
        "compare": {k: v for k, v in cmp.items() if k not in ("draft_text", "published_text")},
        "published_href": published if cmp.get("published") else None,
    }


# ---- the set ---------------------------------------------------------------------------------------------------
def target_slots(corpus: Corpus, facts: Facts) -> list[str]:
    """The new set: every slot the latest sibling generation has, then the topics new in this generation."""
    slots: list[str] = []
    for g in facts.siblings:
        for s in corpus.families[g].slots:
            if s not in slots:
                slots.append(s)
    return slots + [t["slot"] for t in facts.new_topics if t["slot"] not in slots]


def draft_set(facts_path: str | Path, model, emit: Callable = lambda e: None, run_id: str | None = None,
              parallel: int = 5, source: str = "cli") -> dict:
    t0 = time.time()
    started = dt.datetime.now(dt.timezone.utc)
    corpus = Corpus()
    facts = load_facts(facts_path, corpus)
    corpus = Corpus(holdout=facts.prefix)
    run = {"id": run_id or started.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4], "date": started.date().isoformat()}
    slots = target_slots(corpus, facts)
    plans = [plan_topic(corpus, facts, s) for s in slots]
    emit({"stage": "plan", "topics": [p.topic_id for p in plans], "run": run["id"]})
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        topics = list(ex.map(lambda p: draft_topic(p, facts, corpus, model, run, emit), plans))
    report = summarise(topics, facts, model, run, started, time.time() - t0, source)
    report["proposals"] = facts.proposals
    return report


def summarise(topics, facts, model, run, started, seconds, source) -> dict:
    sib = [t for t in topics if t["has_sibling"] and t["compare"].get("edit_ratio") is not None]
    total = sum(t["grounding"]["total"] for t in topics)
    grounded = sum(t["grounding"]["grounded"] for t in topics)
    usage = {k: sum(t["usage"][k] for t in topics) for k in ("calls", "tokens_in", "tokens_out")}
    usage["cost_usd"] = round(sum(t["usage"]["cost_usd"] for t in topics), 5)
    ratios = [t["compare"]["edit_ratio"] for t in sib]
    reached = [t for t in topics if not t["withheld"]]
    return {
        "alignr": __version__, "run": run["id"], "source": source, "generation": facts.generation,
        "model": getattr(model, "id", model.name), "model_kind": model.name,
        "started": started.isoformat(timespec="seconds"), "seconds": round(seconds, 1), "facts_page": facts.page,
        "metrics": {
            "grammar_failures_reaching_writer": sum(1 for t in reached if not t["grammar"]["valid"]),
            "withheld": sum(1 for t in topics if t["withheld"]),
            "values_total": total, "values_sourced": grounded,
            "values_sourced_pct": round(100 * grounded / total, 1) if total else 100.0,
            "edit_ratio_mean": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "edit_ratio_max": max(ratios) if ratios else None,
            "edit_ratio_target": EDIT_RATIO_TARGET,
            "edit_ratio_met": bool(ratios) and max(ratios) < EDIT_RATIO_TARGET,
            "topics": len(topics), "topics_with_sibling": len(sib),
            "open_questions": sum(len(t["questions"]) for t in topics),
            "model_calls": usage["calls"], "calls_per_topic": round(usage["calls"] / len(topics), 2) if topics else 0,
            "cost_logged": True,
        },
        "usage": usage,
        "topics": topics,
    }


# ---- files -----------------------------------------------------------------------------------------------------
def report_markdown(r: dict, t: dict | None = None) -> str:
    """The report a writer reads on a draft pull request (one topic) or for the whole set."""
    m = r["metrics"]
    if t is None:
        lines = [f"# AlignR run {r['run']}: {r['generation']} topic set", "",
                 f"Model: `{r['model']}` · {r['seconds']} s · {r['usage']['calls']} model calls · "
                 f"{r['usage']['tokens_in']:,} tokens in · {r['usage']['tokens_out']:,} out · ${r['usage']['cost_usd']:.4f}", "",
                 "| Topic | Grammar | Values sourced | Revisions | Questions | Edit ratio |", "|---|---|---|---|---|---|"]
        for t_ in r["topics"]:
            er = t_["compare"].get("edit_ratio")
            lines.append(f"| `{t_['topic_id']}`{' (stub, no sibling)' if t_['stub'] else ''} | "
                         f"{'valid' if t_['grammar']['valid'] else 'INVALID'} | {t_['grounding']['grounded']}/{t_['grounding']['total']} | "
                         f"{t_['revisions']} | {len(t_['questions'])} | {'' if er is None else f'{er:.0%}'} |")
        lines += ["", f"Values with a key or a source: {m['values_sourced_pct']}%. Mean edit ratio on topics with a sibling: "
                  f"{(m['edit_ratio_mean'] or 0):.0%} (target under {m['edit_ratio_target']:.0%})."]
        return "\n".join(lines) + "\n"
    g = t["grounding"]
    lines = [f"## AlignR draft: `{t['topic_id']}` ({t['type']})", "",
             f"**{'Ready for review' if not t['questions'] else str(len(t['questions'])) + ' open question(s)'}** · "
             f"grammar {'valid' if t['grammar']['valid'] else 'INVALID'} · {g['grounded']}/{g['total']} values sourced · "
             f"{t['usage']['calls']} model call(s) · ${t['usage']['cost_usd']:.4f}", "",
             "### Built from",
             f"- Siblings: {', '.join('`' + s.split('/')[-1] + '`' for s in t['siblings']) or 'none (new in this generation: stub)'}",
             f"- Facts: `facts/{r['generation']}.yaml` · schema `{t['schema_hash']}` · model `{t['model']}`", "",
             "### Checks",
             f"- Grammar: {t['grammar'].get('grammar', '')} {'valid' if t['grammar']['valid'] else 'invalid'}",
             f"- Rules: {len(t['rules'].get('errors', []))} errors, {len(t['rules'].get('warnings', []))} warnings"
             + "".join(f"\n  - {w['id']}: {w['message']}" for w in t['rules'].get('warnings', [])),
             f"- Grounding: {g['grounded']}/{g['total']} values"]
    if t["fixes"]:
        lines.append("- Safe fixes in code: " + "; ".join(f"`{f}`" for f in t["fixes"]))
    for h in t["history"]:
        if h["failures"]:
            lines.append(f"- Attempt {h['attempt']}: " + "; ".join(f"`{f['path']}` {f['problem']}" for f in h["failures"][:6]))
    if t["resolved"]:
        lines.append("- Became questions after two revisions: " + "; ".join(t["resolved"]))
    lines += ["", "### Questions"] + ([f"- {q}" for q in t["questions"]] or ["- None"])
    er = t["compare"].get("edit_ratio")
    if er is not None:
        lines += ["", f"### Compared with the published topic", f"Edit ratio {er:.0%} against `{t['published_href']}` "
                  f"(adapted from {r['facts_page']})."]
    lines += ["", f"Run `{r['run']}` · {t['usage']['tokens_in']:,} tokens in · {t['usage']['tokens_out']:,} out. "
              "AlignR can't merge. Close every draft comment before merging: the open-questions check blocks the merge."]
    return "\n".join(lines) + "\n"


def drafts_map(r: dict) -> str:
    refs = "\n".join(f'    <topicref href="topics/{t["topic_id"]}.dita"/>' for t in r["topics"] if not t["withheld"])
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE map PUBLIC "-//OASIS//DTD DITA Map//EN" "map.dtd">
<!-- AlignR run {r['run']}. Copy topics/ into corpus/tpu/topics to build these drafts with the rest of the set. -->
<map id="alignr-{r['generation']}-drafts" xml:lang="en-US">
  <title>{r['generation']} drafts (AlignR run {r['run']})</title>
  <mapref href="../keys/tpu-keys.ditamap" processing-role="resource-only"/>
  <topicref href="topics/{r['topics'][0]['topic_id']}.dita" platform="{r['generation']}" collection-type="family">
{refs}
  </topicref>
</map>
"""


def write_outputs(r: dict, out: str | Path) -> Path:
    out = Path(out)
    (out / "topics").mkdir(parents=True, exist_ok=True)
    for t in r["topics"]:
        if not t["withheld"]:
            (out / "topics" / f"{t['topic_id']}.dita").write_text(t["xml"])
        (out / "reports").mkdir(exist_ok=True)
        (out / "reports" / f"{t['topic_id']}.md").write_text(report_markdown(r, t))
    (out / "report.md").write_text(report_markdown(r))
    (out / "report.json").write_text(json.dumps(slim(r), indent=1))
    (out / f"{r['generation']}-drafts.ditamap").write_text(drafts_map(r))
    return out


def slim(r: dict) -> dict:
    """The run record without the bulky parts (schemas, reader views), for logs and the run history."""
    s = copy.deepcopy(r)
    for t in s["topics"]:
        t.pop("schema", None)
        t.pop("xml", None)
        t["compare"] = {"edit_ratio": t["compare"].get("edit_ratio")}
        t["grounding"].pop("values", None)
    return s


def zip_bytes(r: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        base = f"alignr-{r['generation']}-{r['run']}/"
        for t in r["topics"]:
            if not t["withheld"]:
                z.writestr(base + f"topics/{t['topic_id']}.dita", t["xml"])
            z.writestr(base + f"reports/{t['topic_id']}.md", report_markdown(r, t))
        z.writestr(base + "report.md", report_markdown(r))
        z.writestr(base + "report.json", json.dumps(slim(r), indent=1))
        z.writestr(base + f"{r['generation']}-drafts.ditamap", drafts_map(r))
        z.writestr(base + "README.txt", README_TXT.format(run=r["run"], gen=r["generation"]))
    return buf.getvalue()


README_TXT = """AlignR drafts for {gen}, run {run}

topics/      one DITA topic per draft. Every value is a key reference; every open question is a <draft-comment>.
reports/     the report for each topic, as it appears on its draft pull request
report.md    the set summary; report.json is the full run record (checks, grounding, cost)
{gen}-drafts.ditamap   a map of the drafts

To build the drafts with the rest of the set, copy topics/ into the dita-corpus tpu/topics folder and run DITA-OT on
tpu/tpu.ditamap. Close every draft comment before you publish: DITA processors hide them, and AlignR's merge check
fails while any is open.
"""
