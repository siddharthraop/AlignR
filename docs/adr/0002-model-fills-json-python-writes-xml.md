# ADR 0002: The model fills a per-topic JSON Schema; Python writes the XML

- Status: Accepted · Date: 2026-10-26

## Context
Structured output accepts a subset of JSON Schema and rejects very large or deeply nested schemas. The DITA grammar is
both.

## Decision
`build_schema` writes a small schema for each topic and run: title, short description, one object per sibling section,
and questions. Enums list only this run's keys, source IDs, reusable notes and conflicts. The spec table has exactly one
row per spec value. The serialiser writes the XML. The same JSON shape is used for the sibling examples in the prompt,
so the model sees what it must produce.

## Consequences
Gate 2 checks code AlignR controls, so a grammar failure is a bug, not a model error. The model can still put a valid
value in the wrong place, which is why grounding runs after gate 1.
