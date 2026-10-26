# ADR 0003: An AlignR document-type shell for RELAX NG validation in lxml

- Status: Accepted · Date: 2026-10-26

## Context
lxml 6 (libxml2 2.14) can't validate against the OASIS DITA 1.3 DTDs (its entity-amplification guard rejects them),
which is why the corpus uses `xmllint`. The design calls for RELAX NG with lxml. libxml2 also can't compile the OASIS
RELAX NG shells as shipped: the MathML domain overrides a define that the included grammar doesn't have.

## Decision
`checks.relaxng` assembles AlignR's own concept and reference shells from the OASIS modules, leaving out the MathML,
SVG and equation domains. DITA expects teams to assemble shells; no TPU topic uses those domains.

## Consequences
Every corpus concept and reference topic validates. lxml doesn't apply default attribute values, so DITA-OT still
builds every set in CI, in strict mode. Task topics need a task shell built the same way (not in 0.1).
