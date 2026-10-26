"""The checks that decide pass or fail: validate_grammar (gate 2), check_rules, check_grounding, open questions."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from lxml import etree, isoschematron

from .corpus import ROOT, Corpus, corpus_dir
from .facts import PLACEHOLDER, UNIT_BASE, Facts, fill
from .skeleton import Plan

PARSER = etree.XMLParser(load_dtd=False, no_network=True, resolve_entities=False)
SVRL = "{http://purl.oclc.org/dsdl/svrl}"
UNIT_WORDS = sorted([u for u in UNIT_BASE] + ["chips"], key=len, reverse=True)
NUMBER = re.compile(r"(?<![\w.{/-])(\d[\d,]*(?:\.\d+)?(?:/\d+)?(?:x\d+)*)(?![\w/-])(?:\s+(" +
                    "|".join(re.escape(u) for u in UNIT_WORDS) + r")(?![\w/]))?")


# ---- Gate 2: the DITA grammar ----------------------------------------------------------------------------------
def grammar_dir() -> Path:
    return Path(os.environ.get("ALIGNR_GRAMMAR", corpus_dir() / ".grammar"))


@lru_cache(maxsize=4)
def relaxng(kind: str) -> etree.RelaxNG:
    """The OASIS DITA 1.3 RELAX NG modules, assembled in a document-type shell without the MathML and SVG domains.

    libxml2 (and so lxml) can't compile the OASIS shells as shipped: the MathML module overrides a define that the
    included grammar doesn't have. DITA expects teams to assemble their own shells from the modules, so AlignR's
    shell is the OASIS shell with those two domains (and equation, which needs MathML) left out.
    """
    d = grammar_dir() / "rng" / "technicalContent" / "rng"
    src = d / f"{kind}.rng"
    if not src.exists():
        raise FileNotFoundError(f"DITA grammar not found at {d}. Run corpus/tools/fetch_grammar.sh or set ALIGNR_GRAMMAR.")
    shell = d / f"alignr-{kind}.rng"
    if not shell.exists():
        s = src.read_text()
        s = re.sub(r'\s*<include href="(mathmlDomain|svgDomain|equationDomain)\.rng"[^>]*/>', "", s)
        s = re.sub(r"\s*\(topic (mathml|svg|equation)-d\)", "", s)
        s = s.replace('<nsName ns="http://www.w3.org/2000/svg"/>', "").replace(
            '<nsName ns="http://www.w3.org/1998/Math/MathML"/>', "")
        shell.write_text(s)
    return etree.RelaxNG(etree.parse(str(shell)))


def validate_grammar(xml: bytes) -> dict:
    try:
        doc = etree.fromstring(xml, PARSER)
    except etree.XMLSyntaxError as e:
        return {"valid": False, "grammar": "XML", "errors": [{"line": e.lineno, "message": str(e)}]}
    kind = doc.tag
    try:
        rng = relaxng(kind)
    except (FileNotFoundError, etree.RelaxNGParseError) as e:
        return {"valid": False, "grammar": kind, "errors": [{"line": 0, "message": f"no grammar for <{kind}>: {e}"}]}
    ok = rng.validate(doc.getroottree())
    errs = [{"line": e.line, "message": e.message} for e in rng.error_log] if not ok else []
    return {"valid": ok, "grammar": f"DITA 1.3 {kind} (RELAX NG)", "errors": errs}


# ---- House rules: ISO Schematron ------------------------------------------------------------------------------
ALIGNR_SCH = """<schema xmlns="http://purl.oclc.org/dsdl/schematron" queryBinding="xslt">
  <title>AlignR draft rules</title>
  <pattern id="alignr">
    <rule context="/concept | /reference">
      <assert id="ALR-1" role="error" test="prolog/metadata/othermeta[@name='generated-by']">A draft records its run,
        schema and model, so it can be traced to its row in the run log.</assert>
      <assert id="ALR-3" role="warning" test="string-length(normalize-space(shortdesc)) &lt;= 300">The short
        description is longer than 300 characters.</assert>
    </rule>
    <rule context="draft-comment">
      <assert id="ALR-4" role="error" test="@disposition">A question must say whether it is open, so the merge gate can
        find it.</assert>
    </rule>
    {sibling_rules}
  </pattern>
</schema>"""


def _sibling_rules(plan: Plan | None) -> str:
    """ALR-2 (same-columns-as-siblings): when every sibling uses one header for a table, the draft uses it too."""
    if not plan:
        return ""
    out = []
    for s in plan.sections:
        cols = s.get("fixed_columns")
        if not cols:
            continue
        want = "|".join(cols)
        out.append(f"""<rule context="table[@outputclass='{s['table']}']/tgroup/thead/row">
      <let name="h" value="normalize-space(concat({", '|', ".join(f"entry[{i + 1}]" for i in range(len(cols)))}))"/>
      <assert id="ALR-2" role="error" test="count(entry) = {len(cols)} and $h = '{want}'">The table header is
        "<value-of select="$h"/>"; every sibling uses "{want}".</assert>
    </rule>""")
    return "\n".join(out)


@lru_cache(maxsize=32)
def _schematron(text: str) -> isoschematron.Schematron:
    return isoschematron.Schematron(etree.fromstring(text.encode()), store_report=True)


def check_rules(xml: bytes, plan: Plan | None = None) -> dict:
    doc = etree.fromstring(xml, PARSER).getroottree()
    results = []
    corpus_sch = corpus_dir() / "tools" / "schematron" / "corpus.sch"
    for name, text in (("corpus", corpus_sch.read_text()), ("alignr", ALIGNR_SCH.replace("{sibling_rules}", _sibling_rules(plan)))):
        sch = _schematron(text)
        sch.validate(doc)
        for fa in sch.validation_report.iter(SVRL + "failed-assert", SVRL + "successful-report"):
            msg = " ".join("".join(fa.find(SVRL + "text").itertext()).split())
            results.append({"id": fa.get("id"), "role": fa.get("role", "error"), "location": fa.get("location"),
                            "message": msg, "set": name})
    errors = [r for r in results if r["role"] == "error"]
    return {"pass": not errors, "errors": errors, "warnings": [r for r in results if r["role"] != "error"],
            "rules": "dita-corpus house rules (corpus.sch) + AlignR rules ALR-1 to ALR-4"}


def open_questions(xml: bytes) -> list[str]:
    doc = etree.fromstring(xml, PARSER)
    return [" ".join("".join(d.itertext()).split()) for d in doc.iter("draft-comment") if d.get("disposition") == "open"]


# ---- Grounding ------------------------------------------------------------------------------------------------
@dataclass
class Grounding:
    total: int = 0
    grounded: int = 0
    failures: list[dict] = field(default_factory=list)
    values: list[dict] = field(default_factory=list)   # every value, with how it is sourced

    def as_dict(self):
        return {"total": self.total, "grounded": self.grounded, "failures": self.failures, "values": self.values}


class FactIndex:
    def __init__(self, facts: Facts, corpus: Corpus):
        self.facts = facts
        keys = {**corpus.keys, **{k: v.value for k, v in facts.keyed.items()}}
        self.keys = keys
        self.allowed = set(facts.allowed_keys())
        self.numbers: dict[str, set[str]] = {}         # number -> sources
        self.pairs: dict[tuple[str, str], set[str]] = {}  # (number, unit) -> sources
        self.code: dict[str, set[str]] = {}
        # A key is supported by its value's source and by every statement or table that uses it.
        self.key_source: dict[str, set[str]] = {}
        for v in facts.values:
            if v.key:
                self.key_source.setdefault(v.key, set()).add(v.source)
        for s in facts.statements:
            for k in PLACEHOLDER.findall(s["text"]):
                self.key_source.setdefault(k, set()).add(s["source"])
        for t in facts.tables.values():
            for r in t["rows"]:
                for k in PLACEHOLDER.findall(" ".join(r)):
                    self.key_source.setdefault(k, set()).add(t["source"])
        for v in facts.values:
            self._add(v.value, v.unit, v.source)
        for s in facts.statements:
            self._scan(fill(s["text"], keys), s["source"])
            for c in re.findall(r"`([^`]+)`", s["text"]):
                self.code.setdefault(c, set()).add(s["source"])
        for t in facts.tables.values():
            for r in t["rows"]:
                for cell in r:
                    self._scan(fill(cell, keys), t["source"])
        for k in facts.allowed_keys():
            self.code.setdefault(keys.get(k, ""), set()).update(self.key_source.get(k, {"key"}))

    def _add(self, num, unit, src):
        num = num.replace(",", "")
        self.numbers.setdefault(num, set()).add(src)
        if unit:
            self.pairs.setdefault((num, unit), set()).add(src)

    def _scan(self, text, src):
        for m in NUMBER.finditer(text):
            self._add(m.group(1), m.group(2) or "", src)


def _texts(doc: dict, plan: Plan):
    """Every model-written text field, with its path and the sources it cites."""
    yield "title", doc.get("title", ""), None
    yield "shortdesc", doc.get("shortdesc", ""), None
    for i, o in enumerate(doc.get("outline") or []):
        yield f"outline[{i}].title", o.get("title", ""), None
    for sid, b in (doc.get("body") or {}).items():
        if not isinstance(b, dict):
            continue
        yield f"body.{sid}.title", b.get("title", ""), None
        for i, p in enumerate(b.get("paras") or []):
            yield f"body.{sid}.paras[{i}].text", p.get("text", ""), set(p.get("sources") or [])
        t = b.get("table") or {}
        for i, c in enumerate(t.get("columns") or []):
            yield f"body.{sid}.table.columns[{i}]", c, None
        for i, r in enumerate(t.get("rows") or []):
            yield f"body.{sid}.table.rows[{i}].label", r.get("label", ""), None


def check_grounding(doc: dict, plan: Plan, facts: Facts, corpus: Corpus) -> Grounding:
    """Every number, unit, flag, command and key in the draft must match a fact. GiB against GB fails."""
    ix = FactIndex(facts, corpus)
    g = Grounding()

    def value(path, shown, ok, how, problem=""):
        g.total += 1
        g.grounded += ok
        g.values.append({"path": path, "value": shown, "sourced": bool(ok), "how": how})
        if not ok:
            g.failures.append({"path": path, "problem": problem or f"{shown} has no matching fact"})

    for path, text, cites in _texts(doc, plan):
        if not isinstance(text, str):
            continue
        for m in PLACEHOLDER.finditer(text):
            k = m.group(1)
            if k not in ix.allowed:
                value(path, "{" + k + "}", False, "key", f"{{{k}}} is not a key for {facts.generation}")
            elif not k.startswith("product-"):
                src = ix.key_source.get(k, set())
                ok = not cites or not src or bool(src & cites)
                shown = ", ".join(sorted(src))
                value(path, "{" + k + "}", ok, f"key {k} ({shown})",
                      "" if ok else f"{{{k}}} comes from {shown}, which the paragraph doesn't cite")
        for c in re.findall(r"`([^`]+)`", text):
            ok = c in ix.code or any(c in fill(s["text"], ix.keys) for s in facts.statements)
            value(path, f"`{c}`", ok, "code", f"`{c}` doesn't appear in any fact")
        bare = re.sub(r"`[^`]+`", " ", PLACEHOLDER.sub(" ", text))
        for m in NUMBER.finditer(bare):
            num, unit = m.group(1).replace(",", ""), m.group(2) or ""
            if unit:
                srcs = ix.pairs.get((num, unit), set())
                problem = (f"{num} {unit}: no fact gives {num} in {unit}" +
                           (f" (the facts give {num} in {', '.join(sorted(u for n, u in ix.pairs if n == num))})"
                            if any(n == num for n, _ in ix.pairs) else ""))
            else:
                srcs = ix.numbers.get(num, set())
                problem = f"{num}: no fact gives this number"
            ok = bool(srcs) and (not cites or bool(srcs & cites) or "key" in srcs)
            if srcs and cites and not ok:
                problem = f"{num}{' ' + unit if unit else ''} comes from {', '.join(sorted(srcs))}, which the paragraph doesn't cite"
            value(path, f"{num} {unit}".strip(), ok, f"literal ({', '.join(sorted(srcs))})" if srcs else "literal", problem)

    spec = {v.key: v for v in facts.spec_values()}
    for sid, b in (doc.get("body") or {}).items():
        if not isinstance(b, dict):
            continue
        t = b.get("table") or {}
        rows = t.get("rows")
        if rows is not None:
            seen = [r.get("key") for r in rows]
            for k in spec:
                if seen.count(k) != 1:
                    g.failures.append({"path": f"body.{sid}.table.rows", "problem": f"{k} appears {seen.count(k)} times; it must appear once"})
            for i, r in enumerate(rows):
                v = spec.get(r.get("key"))
                if not v:
                    continue
                unit = "" if r.get("unit") in (None, "none") else r.get("unit")
                ok = unit == (v.unit or "")
                value(f"body.{sid}.table.rows[{i}].unit", f"{{{v.key}}} {unit}".strip(), ok, f"key {v.key} ({v.source})",
                      f"unit {unit or 'none'} for {v.key}, but {v.source} gives {v.unit or 'no unit'}")
        sec = next((s for s in plan.sections if s["id"] == sid), None)
        if sec and sec["table"] in facts.tables and "columns" in t:
            ft = facts.tables[sec["table"]]
            for r in ft["rows"]:
                for cell in r:
                    if cell:
                        g.total += 1
                        g.grounded += 1
                        g.values.append({"path": f"body.{sid}.table", "value": cell, "sourced": True,
                                         "how": f"facts table {sec['table']} ({ft['source']})"})
    return g
