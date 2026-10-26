"""generate_topic: the one tool that calls a model.

GeminiModel fills the JSON Schema with structured output. BaselineModel copies the most recent sibling and
swaps the generation's keys, the way a writer starts today; it costs nothing and shows what the gates catch.
ScriptedModel replays fixed answers for tests.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time

from .corpus import Corpus
from .facts import Facts
from .schema import NO_UNIT
from .skeleton import Plan

# Gemini 3.8 Flash, introductory price to 31 December 2026, per 1M tokens (design doc §5.9, Agent Platform pricing).
PRICE_IN, PRICE_OUT = 0.75, 3.75

SYSTEM = """You draft one DITA topic for a new hardware generation. You return JSON that matches the schema. You never
write XML.

Rules:
1. Follow the structure, order, wording and style of the sibling topics: the same topic for earlier generations.
   Change only what the new generation's facts change.
2. Write every value that has a key as a {key} placeholder, using only the allowed keys. Never type a number that a
   key holds. Write the product name as its {key} placeholder.
3. Every other number, unit, flag or command you write must appear in the facts. Units must match the facts exactly:
   GiB is not GB, and GBps is not TB/s. Put flags, commands and shape patterns in backticks.
4. Each paragraph cites the source IDs of the facts it uses.
5. Spec table: one row per spec value, a label in the siblings' style, the key, and the unit from the facts
   ("none" for no unit).
6. Use a reusable note by its ID only when a fact supports it.
7. If the sources conflict, or the facts don't say something the siblings say, add a question instead of guessing:
   for the engineer about values, for the writer about wording or structure. Every listed conflict needs one question
   whose "about" is that conflict's ID. Don't repeat a sibling sentence that the facts don't support.
8. Sentence case for titles and headings. Google developer documentation style: present tense, active voice, short
   sentences."""

STUB = """No sibling topic exists for this one: it is new in this generation. Don't write prose. Return a title and a
short description in the style of the other topics in the set, an outline of sections with the statement IDs each one
should cover, and questions for the writer and engineer. Every listed conflict needs one question."""

REVISE = """Your last answer failed these checks. Return the full JSON again. Change only the fields listed as failing,
so that they pass, and keep every other field exactly as it was. If a field can't pass using the facts, replace the
value with a question in "questions" and remove the unsupported value."""


def prompt_for(plan: Plan, facts: Facts, corpus: Corpus) -> str:
    keyed = facts.keyed
    payload = {
        "target": {"generation": facts.generation, "topic_id": plan.topic_id, "slot": plan.slot, "type": plan.type,
                   "product": "{" + facts.product_key + "}"},
        "siblings": [] if plan.stub else plan.examples,
        "facts": {
            "sources": {k: v["title"] for k, v in facts.sources.items()},
            "values": [{"key": v.key or None, "quantity": v.quantity, "value": v.value, "unit": v.unit or NO_UNIT,
                        "source": v.source, "spec_table": v.table == "spec", **({"approximately": True} if v.approx else {})}
                       for v in facts.values],
            "statements": facts.statements,
            "tables": {k: {"columns": t["columns"], "rows": t["rows"], "source": t["source"]} for k, t in facts.tables.items()},
        },
        "allowed_keys": {k: (keyed[k].value if k in keyed else corpus.keys.get(k, "")) for k in facts.allowed_keys()},
        "reusable_notes": {n.id: n.text for n in corpus.notes.values()},
        "conflicts": [{"id": c.id, "ask": c.ask, "detail": c.detail} for c in plan.conflicts],
    }
    if plan.stub:
        payload["target"]["about"] = plan.about
        payload["target"]["cover_statements"] = plan.facts
    return json.dumps(payload, indent=1, ensure_ascii=False)


def _clean_key():
    """Secrets pasted from a clipboard can carry extra lines. Keep the one that looks like a key."""
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        raw = os.environ.get(var, "")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if lines and (len(lines) > 1 or lines[0] != raw):
            os.environ[var] = next((ln for ln in lines if re.fullmatch(r"(AIza[\w-]{30,}|AQ\.[\w.-]{20,})", ln)), lines[0])


def has_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


class Usage:
    def __init__(self):
        self.calls, self.tokens_in, self.tokens_out, self.ms = 0, 0, 0, 0

    @property
    def cost(self) -> float:
        return round(self.tokens_in / 1e6 * PRICE_IN + self.tokens_out / 1e6 * PRICE_OUT, 5)

    def as_dict(self):
        return {"calls": self.calls, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "cost_usd": self.cost, "model_ms": self.ms, "price": f"${PRICE_IN}/${PRICE_OUT} per 1M tokens in/out"}


class GeminiModel:
    name = "gemini"

    def __init__(self, model_id: str | None = None):
        _clean_key()
        from google import genai
        from google.genai import types
        self.id = model_id or os.environ.get("ALIGNR_MODEL", "gemini-3.8-flash")
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"),
                                   http_options=types.HttpOptions(timeout=int(os.environ.get("ALIGNR_TIMEOUT_MS", "90000"))))
        self.types = types
        self.thinking = os.environ.get("ALIGNR_THINKING", "low")   # structured filling needs little reasoning

    def _call(self, system: str, prompt: str, schema: dict, usage: Usage) -> dict:
        extra = {"thinking_config": self.types.ThinkingConfig(thinking_level=self.thinking)} if self.thinking else {}
        t0 = time.time()
        last = None
        for attempt in range(3):
            cfg = self.types.GenerateContentConfig(system_instruction=system, temperature=0.2, response_mime_type="application/json",
                                                   response_json_schema=schema, **extra)
            try:
                r = self.client.models.generate_content(model=self.id, contents=prompt, config=cfg)
                break
            except Exception as e:  # rate limits and transient errors: back off, then give up
                last = e
                if "thinking" in str(e).lower():
                    extra = {}            # a model that doesn't take a thinking level
                time.sleep(2 + 4 * attempt)
        else:
            raise RuntimeError(f"Gemini call failed: {last}")
        usage.calls += 1
        usage.ms += int((time.time() - t0) * 1000)
        um = getattr(r, "usage_metadata", None)
        if um:
            usage.tokens_in += um.prompt_token_count or 0
            usage.tokens_out += (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        text = (r.text or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        return json.loads(text)

    def generate(self, plan, facts, corpus, schema, usage):
        system = SYSTEM + ("\n\n" + STUB if plan.stub else "")
        return self._call(system, prompt_for(plan, facts, corpus), schema, usage)

    def revise(self, plan, facts, corpus, schema, usage, previous: dict, failures: list[dict]):
        prompt = prompt_for(plan, facts, corpus) + "\n\nYOUR LAST ANSWER:\n" + json.dumps(previous, ensure_ascii=False) + \
            "\n\nFAILING FIELDS:\n" + json.dumps(failures, ensure_ascii=False)
        system = SYSTEM + ("\n\n" + STUB if plan.stub else "") + "\n\n" + REVISE
        return self._call(system, prompt, schema, usage)


class BaselineModel:
    """Copy the most recent sibling and swap the generation's keys: what a writer does by hand today."""
    name = "baseline"
    id = "copy-last-generation"

    def generate(self, plan, facts, corpus, schema, usage):
        if plan.stub:
            return {"title": "{" + facts.product_key + "} " + plan.slot.replace("-", " "),
                    "shortdesc": "", "outline": [{"title": "Overview", "facts": plan.facts[:4] or [facts.statements[0]["id"]]}],
                    "questions": []}
        src = plan.examples[0]
        g = src["generation"]
        doc = copy.deepcopy({k: src[k] for k in ("title", "shortdesc", "body", "questions")})

        def swap(text: str) -> str:
            def rep(m):
                k = m.group(1)
                if k.startswith("product-"):
                    return "{" + facts.product_key + "}"
                if k.startswith(g + "-"):
                    return "{" + facts.generation + "-" + k[len(g) + 1:] + "}"
                return m.group(0)
            return re.sub(r"\{([a-z0-9-]+)\}", rep, text)

        doc["title"], doc["shortdesc"] = swap(doc["title"]), swap(doc["shortdesc"])
        default_src = next(iter(facts.sources))
        for sid, b in doc["body"].items():
            sec = next(s for s in plan.sections if s["id"] == sid)
            b["title"] = swap(b["title"])
            b["paras"] = [{"text": swap(p["text"]), "sources": [default_src]} for p in b["paras"]]
            t = b.get("table")
            if sec["table"] == "spec" and t:
                by_q = {r["key"][len(g) + 1:]: r for r in t["rows"]}
                rows = []
                for v in facts.spec_values():
                    r = by_q.get(v.quantity)
                    rows.append({"label": r["label"] if r else v.quantity.replace("-", " ").capitalize(),
                                 "key": v.key, "unit": (r["unit"] if r else v.unit) or NO_UNIT})
                b["table"] = {"rows": rows} | ({} if sec["fixed_columns"] else {"columns": t["columns"]})
            elif sec["table"] in facts.tables and t:
                n = len(facts.tables[sec["table"]]["columns"])
                cols = t["columns"][:n] + [c.capitalize() for c in facts.tables[sec["table"]]["columns"][len(t["columns"]):n]]
                b["table"] = {"columns": cols}
            else:
                b.pop("table", None)
            if sec["table"] is None:
                b.pop("table", None)
        for sid in list(doc["body"]):
            if sid not in schema["properties"]["body"]["properties"]:
                doc["body"].pop(sid)
        doc["questions"] = []
        return doc

    def revise(self, plan, facts, corpus, schema, usage, previous, failures):
        return copy.deepcopy(previous)


class ScriptedModel:
    """Replays fixed answers per topic ID: {topic_id: [first answer, revision 1, revision 2]}."""
    name = "scripted"
    id = "scripted"

    def __init__(self, answers: dict[str, list[dict]]):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.seen: dict[str, list] = {}

    def _next(self, plan):
        q = self.answers[plan.topic_id]
        return copy.deepcopy(q.pop(0) if len(q) > 1 else q[0])

    def generate(self, plan, facts, corpus, schema, usage):
        self.seen.setdefault(plan.topic_id, []).append(prompt_for(plan, facts, corpus))
        return self._next(plan)

    def revise(self, plan, facts, corpus, schema, usage, previous, failures):
        self.seen.setdefault(plan.topic_id, []).append(failures)
        return self._next(plan)


def make_model(kind: str = "auto"):
    if kind == "baseline" or (kind == "auto" and not has_key()):
        return BaselineModel()
    return GeminiModel()
