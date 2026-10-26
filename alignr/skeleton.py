"""Read sibling topics into the JSON shape the model fills, and find where the siblings disagree.

The same shape is used three ways: as the sibling examples in the prompt, as the answer Gemini returns
(gate 1), and as the input to the serialiser. A sibling read back this way is also the "copy the last
generation" baseline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

from .corpus import Corpus
from .facts import Conflict, Facts


def inline(el: etree._Element | None) -> str:
    """Element content as text with {key} placeholders and `code`."""
    if el is None:
        return ""
    out = [el.text or ""]
    for c in el:
        if not isinstance(c.tag, str):
            out.append(c.tail or "")
            continue
        if c.get("keyref") and not (c.text or "").strip() and not len(c):
            out.append("{" + c.get("keyref") + "}")
        elif c.tag in ("codeph", "cmdname", "parmname", "option"):
            out.append("`" + "".join(c.itertext()) + "`")
        elif c.tag == "draft-comment":
            pass
        else:
            out.append(inline(c))
        out.append(c.tail or "")
    return re.sub(r"\s+", " ", "".join(out)).strip()


def body_of(root: etree._Element) -> etree._Element:
    return root.find("conbody") if root.tag == "concept" else root.find("refbody")


def blocks(root: etree._Element) -> list[tuple[str, etree._Element | None, list[etree._Element]]]:
    """[(section id, section element or None, child blocks)]; loose blocks in a concept body are section 'body'."""
    body = body_of(root)
    out = []
    loose = [c for c in body if isinstance(c.tag, str) and c.tag != "section"]
    if loose:
        out.append(("body", None, loose))
    for s in body.findall("section"):
        out.append((s.get("id") or f"s{len(out)}", s, [c for c in s if isinstance(c.tag, str) and c.tag != "title"]))
    return out


def to_json(root: etree._Element, generation: str) -> dict:
    """A topic as {title, shortdesc, body: {sid: {title, notes, table, paras}}, questions}."""
    doc = {"title": inline(root.find("title")), "shortdesc": inline(root.find("shortdesc")), "body": {}, "questions": []}
    for sid, sec, kids in blocks(root):
        b = {"title": inline(sec.find("title")) if sec is not None else "", "notes": [], "table": None, "paras": []}
        for k in kids:
            if k.tag == "note" and k.get("conref"):
                b["notes"].append(k.get("conref").split("/")[-1])
            elif k.tag == "table":
                b["table"] = table_json(k, generation)
            elif k.tag == "p":
                b["paras"].append({"text": inline(k), "sources": ["sibling"]})
        doc["body"][sid] = b
    return doc


def table_json(t: etree._Element, generation: str) -> dict:
    head = [inline(e) for e in t.find(".//thead/row").findall("entry")]
    rows = [[inline(e) for e in r.findall("entry")] for r in t.find(".//tbody").findall("row")]
    out = {"outputclass": t.get("outputclass", ""), "columns": head}
    if t.get("outputclass") == "spec":
        out["rows"] = []
        for r in rows:
            m = re.fullmatch(r"\{([^}]+)\}", r[1])
            out["rows"].append({"label": r[0], "key": m.group(1) if m else r[1], "unit": r[2] if len(r) > 2 else ""})
    else:
        out["data"] = rows
    return out


@dataclass
class Plan:
    """What the siblings fix for one target topic: its type, sections, tables and notes."""
    slot: str
    topic_id: str
    type: str
    siblings: list[tuple[str, str]]                    # (generation, href)
    examples: list[dict]                               # siblings as JSON
    sections: list[dict] = field(default_factory=list) # [{id, wrapped, table, fixed_columns}]
    conflicts: list[Conflict] = field(default_factory=list)
    stub: bool = False
    about: str = ""
    facts: list[str] = field(default_factory=list)     # statement IDs for a stub

    @property
    def has_sibling(self) -> bool:
        return bool(self.siblings)


def plan_topic(corpus: Corpus, facts: Facts, slot: str) -> Plan:
    sibs = corpus.find_siblings(slot, facts.siblings)
    tid = facts.prefix + slot
    if not sibs:
        nt = next((t for t in facts.new_topics if t["slot"] == slot), {"type": "concept", "about": slot, "facts": []})
        p = Plan(slot, tid, nt["type"], [], [], stub=True, about=nt.get("about", ""), facts=nt.get("facts", []))
        p.sections = [{"id": "body", "wrapped": False, "table": None, "fixed_columns": None}]
        p.conflicts = list(facts.conflicts)
        return p
    roots = [(g, corpus.parse(h)) for g, h in sibs]
    examples = [to_json(r, g) | {"generation": g} for g, r in roots]
    types = {r.tag for _, r in roots}
    if len(types) != 1:
        raise ValueError(f"siblings for {slot} have different topic types: {types}")
    order: list[str] = []
    for ex in examples:
        for sid in ex["body"]:
            if sid not in order:
                order.append(sid)
    sections = []
    for sid in order:
        tables = [ex["body"][sid]["table"] for ex in examples if sid in ex["body"] and ex["body"][sid]["table"]]
        oc = tables[0]["outputclass"] if tables else None
        heads = {tuple(t["columns"]) for t in tables}
        sections.append({"id": sid, "wrapped": sid != "body", "table": oc,
                         "fixed_columns": list(next(iter(heads))) if oc == "spec" and len(heads) == 1 else None,
                         "sibling_columns": [t["columns"] for t in tables]})
    p = Plan(slot, tid, types.pop(), sibs, examples, sections)
    p.conflicts = sibling_conflicts(examples, facts) + relevant(facts.conflicts, p, facts)
    for i, c in enumerate(p.conflicts, 1):
        c.id = f"C{i}"
    return p


def relevant(conflicts: list[Conflict], plan: Plan, facts: Facts) -> list[Conflict]:
    """A source conflict belongs to a topic that shows that quantity."""
    out = []
    for c in conflicts:
        uses_spec = any(s["table"] == "spec" for s in plan.sections)
        if uses_spec and any(v.quantity == c.quantity and v.table == "spec" for v in facts.values):
            out.append(Conflict(**{**c.__dict__}))
    return out


def sibling_conflicts(examples: list[dict], facts: Facts) -> list[Conflict]:
    """Siblings that give one quantity in different units (KI-1: v6e HBM in GB, v5p in GiB)."""
    units: dict[str, dict[str, str]] = {}
    for ex in examples:
        g = ex["generation"]
        for b in ex["body"].values():
            t = b["table"]
            if t and t.get("outputclass") == "spec":
                for r in t["rows"]:
                    q = r["key"][len(g) + 1:] if r["key"].startswith(g + "-") else r["key"]
                    if r["unit"]:
                        units.setdefault(q, {})[g] = r["unit"]
    out = []
    target = {v.quantity: v for v in facts.spec_values()}
    for q, by_gen in units.items():
        if len(set(by_gen.values())) > 1 and q in target:
            seen = ", ".join(f"{g} {u}" for g, u in by_gen.items())
            out.append(Conflict(
                id="", kind="siblings", quantity=q, key=target[q].key, ask="writer",
                known_issue="KI-1" if q == "hbm-capacity" else "",
                detail=(f"The sibling topics give {q} in different units ({seen}). The {facts.generation} facts use "
                        f"{target[q].unit or 'no unit'} ({target[q].source}). Confirm the unit, and correct the sibling "
                        f"that is wrong.")))
    return out
