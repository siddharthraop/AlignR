# Architecture

AlignR is one Python package, `alignr`, used two ways: by the `alignr` command in GitHub Actions (the Git path) and by
a FastAPI service on Cloud Run (the web path). Both call `alignr.agent.draft_set`.

```text
facts/tpu7x.yaml ─┐
corpus/tpu (DITA) ─┼─> plan_topic ─> build_schema ─> generate_topic ─> safe_fixes ─> serialise ─> gates ─┬─> draft PR / page
OASIS RELAX NG ───┘   (siblings)     (gate 1)        (Gemini)                       (XML)     (gate 2,   │
                                                         ^                                   rules,     │
                                                         └──── revise failed fields, ≤ 2 ────grounding)─┘
                                                                         still failing -> <draft-comment>
```

## Modules

| Module | Tool in the design | What it does |
|---|---|---|
| `corpus.py` | `find_siblings` | Reads the map, keys and reuse warehouse. Enforces the holdout: `Corpus(holdout="tpu7x-")` raises `HoldoutError` on any held-out topic. |
| `facts.py` | `load_facts` | Loads the facts file, checks every value against its key, converts units and finds conflicting values (KI-2). |
| `skeleton.py` | `find_siblings` | Reads sibling topics into the JSON shape the model fills; plans the target topic; finds sibling unit drift (KI-1). |
| `schema.py` | `build_schema` | The per-topic JSON Schema (gate 1): enums of real keys, sources, notes and conflicts; row and column counts fixed by the facts and siblings. Also re-checks the shape locally. |
| `model.py` | `generate_topic` | `GeminiModel` (structured output with `response_json_schema`), `BaselineModel` (copy the last generation), `ScriptedModel` (tests). |
| `serialise.py` | `serialise_dita` | JSON to DITA XML with `lxml`. Keys become `<keyword keyref>`, questions become `<draft-comment>`. Safe fixes run here. |
| `checks.py` | `validate_grammar`, `check_rules`, `check_grounding` | OASIS DITA 1.3 RELAX NG (lxml), ISO Schematron (corpus rules + ALR rules), grounding of every value and unit. |
| `agent.py` | the loop, `open_pr` report | Drafts each topic, merges revisions field by field, resolves what still fails into questions, summarises the run, writes files and the .zip. |
| `compare.py` | evaluator | Renders draft and published topics as a reader sees them and computes the edit ratio. The only code that reads held-out topics. |
| `cli.py` | | `alignr draft`, `alignr check`, `alignr stage`. |
| `api/main.py` | | Streams a draft as NDJSON; stores runs and the daily count in Firestore. |
| `scripts/open_prs.py` | `open_pr` | Git path: one draft pull request per topic, five commit statuses, the issue comment, the run record. |

## Data

- Runs from the Git path are committed to `runs/` (`latest-github.json`, one file per run). Merges append to
  `runs/merges.jsonl`. The page reads both from `raw.githubusercontent.com`.
- Runs from the web are stored in Firestore (`alignr_runs`), with the daily draft count in `alignr_quota/<date>`.
- Nothing else persists. Every value can be rebuilt from the corpus and the facts file.

## Why the model never writes XML

The DITA grammar is too large and too nested for a structured-output schema, and a model that writes XML can write
invalid XML. AlignR gives the model a small schema per topic and writes the XML itself, so gate 2 checks code AlignR
controls. See [ADR 0002](../adr/0002-model-fills-json-python-writes-xml.md).
