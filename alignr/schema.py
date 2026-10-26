"""build_schema: a small JSON Schema for one topic and one run (gate 1).

The enums hold only the keys, source IDs, notes and conflicts that exist for this run, so Gemini can't cite a
source that isn't there or invent a key name. The siblings and the facts fix the table layout: the spec table
has exactly one row per spec value, and a data table has exactly as many columns as the facts supply.
"""
from __future__ import annotations

import hashlib
import json

from .corpus import Corpus
from .facts import UNITS, Facts
from .skeleton import Plan

NO_UNIT = "none"


def _str():
    return {"type": "string"}


def _arr(items, lo=None, hi=None):
    a = {"type": "array", "items": items}
    if lo is not None:
        a["minItems"] = lo
    if hi is not None:
        a["maxItems"] = hi
    return a


def _obj(props: dict, required=None):
    return {"type": "object", "properties": props, "required": list(required or props), "additionalProperties": False}


def unit_enum() -> list[str]:
    return [u or NO_UNIT for u in UNITS]


def questions_schema(plan: Plan) -> dict:
    about = [c.id for c in plan.conflicts] + ["gap"]
    return _arr(_obj({"for": {"type": "string", "enum": ["engineer", "writer"]},
                      "about": {"type": "string", "enum": about}, "text": _str()}), 0, 8)


def build_schema(plan: Plan, facts: Facts, corpus: Corpus) -> tuple[dict, str]:
    sources = list(facts.sources)
    if plan.stub:
        stmt_ids = [s["id"] for s in facts.statements]
        schema = _obj({
            "title": _str(), "shortdesc": _str(),
            "outline": _arr(_obj({"title": _str(), "facts": _arr({"type": "string", "enum": stmt_ids}, 1, 4)}), 1, 5),
            "questions": questions_schema(plan)})
        return schema, schema_hash(schema)
    body = {}
    for s in plan.sections:
        notes = allowed_notes(plan, s["id"], facts, corpus)
        sec = {"title": _str(),
               "notes": _arr({"type": "string", "enum": notes}, 0, 3) if notes else _arr(_str(), 0, 0),
               "paras": _arr(_obj({"text": _str(), "sources": _arr({"type": "string", "enum": sources}, 1, 3)}), 0, 4)}
        if s["table"] == "spec":
            spec_keys = [v.key for v in facts.spec_values()]
            row = _obj({"label": _str(), "key": {"type": "string", "enum": spec_keys},
                        "unit": {"type": "string", "enum": unit_enum()}})
            tab = {"rows": _arr(row, len(spec_keys), len(spec_keys))}
            if not s["fixed_columns"]:
                tab["columns"] = _arr(_str(), 3, 3)
            sec["table"] = _obj(tab)
        elif s["table"] and s["table"] in facts.tables:
            n = len(facts.tables[s["table"]]["columns"])
            sec["table"] = _obj({"columns": _arr(_str(), n, n)})
        body[s["id"]] = _obj(sec)
    schema = _obj({"title": _str(), "shortdesc": _str(), "body": _obj(body), "questions": questions_schema(plan)})
    return schema, schema_hash(schema)


def allowed_notes(plan: Plan, sid: str, facts: Facts, corpus: Corpus) -> list[str]:
    """A section may reuse the notes its siblings reuse there. Notes about the new generation itself (they name its
    product key) belong in the family's first topic, the introduction, so they aren't repeated on every page."""
    used = [n for ex in plan.examples for n in ex["body"].get(sid, {}).get("notes", [])]
    first = plan.slot == (plan.family_head or plan.slot)
    about_gen = [n.id for n in corpus.notes.values() if f'keyref="{facts.product_key}"' in n.xml] if first else []
    return list(dict.fromkeys(used + about_gen))


def schema_hash(schema: dict) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:4]


def validate_shape(doc: dict, schema: dict, path: str = "") -> list[tuple[str, str]]:
    """Gate 1, re-checked locally: the Gemini API enforces the schema, and this proves it for every model,
    including the scripted and baseline ones. Returns [(path, problem)]."""
    errs: list[tuple[str, str]] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(doc, dict):
            return [(path, "expected an object")]
        for k in schema.get("required", []):
            if k not in doc:
                errs.append((f"{path}.{k}".lstrip("."), "missing"))
        if schema.get("additionalProperties") is False:
            for k in doc:
                if k not in schema["properties"]:
                    errs.append((f"{path}.{k}".lstrip("."), "not in the schema"))
        for k, sub in schema.get("properties", {}).items():
            if k in doc:
                errs += validate_shape(doc[k], sub, f"{path}.{k}".lstrip("."))
    elif t == "array":
        if not isinstance(doc, list):
            return [(path, "expected a list")]
        if "minItems" in schema and len(doc) < schema["minItems"]:
            errs.append((path, f"needs at least {schema['minItems']} items"))
        if "maxItems" in schema and len(doc) > schema["maxItems"]:
            errs.append((path, f"allows at most {schema['maxItems']} items"))
        for i, item in enumerate(doc):
            errs += validate_shape(item, schema["items"], f"{path}[{i}]")
    elif t == "string":
        if not isinstance(doc, str):
            errs.append((path, "expected a string"))
        elif "enum" in schema and doc not in schema["enum"]:
            errs.append((path, f"{doc!r} is not one of the allowed values"))
    return errs
