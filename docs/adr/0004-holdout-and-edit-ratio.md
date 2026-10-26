# ADR 0004: How the holdout is enforced and the edit ratio measured

- Status: Accepted · Date: 2026-10-26

## Decision
- The `Corpus` used for drafting raises `HoldoutError` on any topic with the facts file's prefix and on the comparison
  topic that lists every generation. A test checks that no published sentence reaches a prompt.
- The facts file holds values, terse statements and tables, not published prose.
- The edit ratio is token-level: both topics rendered as a reader sees them (keys resolved, conref notes expanded,
  draft comments removed), split into words and punctuation, and compared with `difflib`. Changed tokens divided by the
  longer length.
- Only topics with a sibling count toward the 40% target. Stubs are shown, not counted.

## Consequences
The number is reproducible from the run record and the corpus commit. It rewards structure and wording close to the
published page, which is what a writer would otherwise retype.
