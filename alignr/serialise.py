"""serialise_dita: plain Python writes the XML from the JSON. The model never writes XML.

Every {key} becomes <keyword keyref>, backticks become <codeph>, every question becomes a <draft-comment>, and the
safe fixes run here, before any model revision: a literal value that matches a key becomes a key reference, a
title in title case becomes sentence case, a missing unit becomes empty, and the topic ID follows the house pattern.
"""
from __future__ import annotations

import datetime as dt
import re

from lxml import etree

from . import __version__
from .facts import PLACEHOLDER, Facts
from .schema import NO_UNIT
from .skeleton import Plan

DOCTYPES = {"concept": ("-//OASIS//DTD DITA Concept//EN", "concept.dtd"),
            "reference": ("-//OASIS//DTD DITA Reference//EN", "reference.dtd")}
# Words that keep their capital in sentence case.
PROPER = {"tpu", "tpus", "vm", "vms", "hbm", "ici", "dcn", "gke", "google", "cloud", "compute", "engine", "kubernetes",
          "jax", "pytorch", "tensorflow", "tensorcore", "tensorcores", "sparsecore", "sparsecores", "ironwood", "trillium",
          "api", "cpu", "nic", "numa", "bf16", "fp8", "int8", "pod", "pods"}
TOKEN = re.compile(r"(\{[a-z0-9][a-z0-9-]*\}|`[^`]+`)")


def sentence_case(text: str) -> str:
    parts = TOKEN.split(text)
    out, first = [], True
    for p in parts:
        if TOKEN.fullmatch(p or "x") and p:
            out.append(p)
            first = False
            continue

        def low(m):
            nonlocal first
            w = m.group(0)
            if first:
                first = False
                return w
            if w[0].isupper() and w[1:].islower() and w.lower() not in PROPER:
                return w.lower()
            return w
        out.append(re.sub(r"[A-Za-z][A-Za-z0-9]*", low, p))
    return "".join(out)


def literal_to_keys(text: str, facts: Facts) -> tuple[str, list[str]]:
    """A number typed into prose that a key holds becomes the key: unique values of 3+ characters, or a value
    followed by the key's own unit."""
    fixes = []
    by_value: dict[str, list] = {}
    for v in facts.keyed.values():
        by_value.setdefault(v.value, []).append(v)

    def rep(m):
        num, unit = m.group(1), (m.group(2) or "").strip()
        cands = by_value.get(num.replace(",", ""), [])
        pick = None
        if unit:
            pick = next((v for v in cands if v.unit == unit), None)
        if not pick and len(cands) == 1 and len(num) >= 3 and (not unit or cands[0].unit in ("", unit)):
            pick = cands[0]
        if not pick:
            return m.group(0)
        fixes.append(f"literal {m.group(0).strip()} -> {{{pick.key}}}")
        return "{" + pick.key + "}" + (m.group(2) or "")

    pieces = TOKEN.split(text)
    for i, p in enumerate(pieces):
        if not TOKEN.fullmatch(p or "x"):
            pieces[i] = re.sub(r"(?<![\w.{/x-])(\d[\d,]*(?:\.\d+)?)(?![\w.x/-])(\s+(?:GiB|GB|GBps|TB/s|Gbps|TFLOPs|chips)\b)?",
                               rep, p)
    return "".join(pieces), fixes


def safe_fixes(doc: dict, plan: Plan, facts: Facts) -> list[str]:
    """Fix in code what code can fix, so the model is asked again only for what it can't."""
    fixes: list[str] = []

    def fix_text(holder, key, title=False):
        t = holder.get(key, "")
        if not isinstance(t, str):
            return
        new, f = literal_to_keys(t, facts)
        fixes.extend(f)
        if title:
            sc = sentence_case(new)
            if sc != new:
                fixes.append(f"title case -> sentence case: {new!r}")
                new = sc
        holder[key] = new

    fix_text(doc, "title", title=True)
    fix_text(doc, "shortdesc")
    for b in (doc.get("body") or {}).values():
        if not isinstance(b, dict):
            continue
        fix_text(b, "title", title=True)
        for p in b.get("paras") or []:
            if isinstance(p, dict):
                fix_text(p, "text")
    for o in doc.get("outline") or []:
        if isinstance(o, dict):
            fix_text(o, "title", title=True)
    return fixes


def mixed(el: etree._Element, text: str):
    """Write text with {key} and `code` into el as mixed content."""
    last = None
    for part in TOKEN.split(text or ""):
        if not part:
            continue
        if PLACEHOLDER.fullmatch(part):
            last = etree.SubElement(el, "keyword", keyref=part[1:-1])
            last.tail = ""
        elif part.startswith("`") and part.endswith("`") and len(part) > 1:
            last = etree.SubElement(el, "codeph")
            last.text = part[1:-1]
            last.tail = ""
        elif last is None:
            el.text = (el.text or "") + part
        else:
            last.tail = (last.tail or "") + part


def question_text(q: dict) -> str:
    who = "For the engineer" if q.get("for") == "engineer" else "For the writer"
    text = q.get("text", "").strip()
    return text if text.lower().startswith("for the ") else f"{who}: {text}"


def comment(parent: etree._Element, q: dict):
    dc = etree.SubElement(parent, "draft-comment", author="alignr", disposition="open")
    dc.set("otherprops", f"for({q.get('for', 'writer')}) about({q.get('about', 'gap')})")
    dc.text = question_text(q)


def serialise(doc: dict, plan: Plan, facts: Facts, run: dict) -> bytes:
    root = etree.Element(plan.type, id=plan.topic_id)
    root.set("{http://www.w3.org/XML/1998/namespace}lang", "en-US")
    mixed(etree.SubElement(root, "title"), doc.get("title") or "{" + facts.product_key + "}")
    mixed(etree.SubElement(root, "shortdesc"), doc.get("shortdesc") or "")
    prolog = etree.SubElement(root, "prolog")
    etree.SubElement(prolog, "author", type="contributor").text = "AlignR draft, for review by a technical writer"
    cd = etree.SubElement(prolog, "critdates")
    etree.SubElement(cd, "created", date=run.get("date", dt.date.today().isoformat()))
    md = etree.SubElement(prolog, "metadata")
    etree.SubElement(md, "audience", type="ml-engineer")
    etree.SubElement(md, "othermeta", name="generated-by",
                     content=f"alignr {__version__} · run {run.get('id', 'local')} · schema {run.get('schema', '----')} · "
                             f"model {run.get('model', '?')}")
    etree.SubElement(prolog, "data", name="source", value=facts.generation, href=facts.page, format="html", scope="external")
    etree.SubElement(prolog, "data", name="licence",
                     value="CC BY 4.0. Drafted from Google Cloud TPU documentation (adapted and modified) and launch facts.")
    if plan.siblings:
        etree.SubElement(prolog, "data", name="siblings", value=" ".join(h.split("/")[-1] for _, h in plan.siblings))

    body = etree.SubElement(root, "conbody" if plan.type == "concept" else "refbody")
    questions = list(doc.get("questions") or [])
    placed = set()

    if plan.stub:
        comment(body, {"for": "writer", "about": "gap", "text":
                       f"This topic is new in {facts.generation} and has no sibling to draft from. AlignR wrote an outline "
                       f"with the facts each section should cover. Write the prose, then close this comment."})
        # Questions go before the sections: DITA allows no loose blocks after a section in a concept body.
        for q in questions:
            comment(body, q)
        _ensure_conflicts(body, plan, questions)
        for i, o in enumerate(doc.get("outline") or [], 1):
            sec = etree.SubElement(body, "section", id=f"s{i}")
            mixed(etree.SubElement(sec, "title"), o.get("title", f"Section {i}"))
            covers = []
            for fid in o.get("facts", []):
                try:
                    s = facts.statement(fid)
                    covers.append(f"{fid} ({s['source']}): {s['text']}")
                except StopIteration:
                    pass
            dc = etree.SubElement(sec, "draft-comment", author="alignr", disposition="open", otherprops="for(writer) about(gap)")
            mixed(dc, "For the writer: write this section from " + " ".join(covers))
        return _bytes(root, plan.type)

    secs = {}
    for s in plan.sections:
        b = (doc.get("body") or {}).get(s["id"], {})
        if s["wrapped"]:
            el = etree.SubElement(body, "section", id=s["id"])
            mixed(etree.SubElement(el, "title"), b.get("title") or s["id"].replace("-", " ").capitalize())
        else:
            el = body
        secs[s["id"]] = el
        if s["wrapped"]:
            for n in b.get("notes", []):
                etree.SubElement(el, "note", conref=f"../reuse/tpu-warehouse.dita#tpu-warehouse/{n}")
        t = b.get("table")
        if s["table"] == "spec" and t:
            cols = s["fixed_columns"] or t.get("columns") or ["Specification", "Value", "Unit"]
            rows = [[r.get("label", ""), "{" + r.get("key", "") + "}", "" if r.get("unit") == NO_UNIT else r.get("unit", "")]
                    for r in t.get("rows", [])]
            _table(el, "spec", cols, rows, tid="spec-table")
        elif s["table"] in facts.tables and t:
            _table(el, s["table"], t.get("columns", []), facts.tables[s["table"]]["rows"])
        for p in b.get("paras", []):
            if p.get("text", "").strip():
                mixed(etree.SubElement(el, "p"), p["text"])
        if not s["wrapped"]:
            for n in b.get("notes", []):
                etree.SubElement(el, "note", conref=f"../reuse/tpu-warehouse.dita#tpu-warehouse/{n}")
    # Each question goes in the section that shows the value it is about.
    for q in questions:
        c = next((c for c in plan.conflicts if c.id == q.get("about")), None)
        target = body if plan.type == "concept" and "body" in secs else next(iter(secs.values()))
        if c and c.key:
            for el in secs.values():
                if el.find(f".//keyword[@keyref='{c.key}']") is not None:
                    target = el
                    break
        comment(target, q)
        placed.add(q.get("about"))
    _ensure_conflicts(next(iter(secs.values())), plan, questions)
    return _bytes(root, plan.type)


def _ensure_conflicts(parent, plan: Plan, questions: list[dict]):
    """Every conflict in the sources is a question, even if the model didn't ask it."""
    asked = {q.get("about") for q in questions}
    for c in plan.conflicts:
        if c.id not in asked:
            q = {"for": c.ask, "about": c.id, "text": c.detail + (" Which is correct?" if c.kind == "sources" else ""),
                 "added_by": "alignr"}
            questions.append(q)
            comment(parent, q)


def _table(parent, outputclass: str, cols: list[str], rows: list[list[str]], tid: str | None = None):
    t = etree.SubElement(parent, "table", outputclass=outputclass)
    if tid:
        t.set("id", tid)
    tg = etree.SubElement(t, "tgroup", cols=str(len(cols)))
    for i in range(len(cols)):
        etree.SubElement(tg, "colspec", colname=f"c{i + 1}")
    hr = etree.SubElement(etree.SubElement(tg, "thead"), "row")
    for c in cols:
        mixed(etree.SubElement(hr, "entry"), c)
    tb = etree.SubElement(tg, "tbody")
    for r in rows:
        row = etree.SubElement(tb, "row")
        for i in range(len(cols)):
            mixed(etree.SubElement(row, "entry"), r[i] if i < len(r) else "")


def _bytes(root, kind: str) -> bytes:
    pub, sysid = DOCTYPES[kind]
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          doctype=f'<!DOCTYPE {kind} PUBLIC "{pub}" "{sysid}">', pretty_print=True)
