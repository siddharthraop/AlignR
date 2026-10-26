# Extending AlignR

## Add a generation

1. Add its family to the corpus map with `@platform` and `collection-type="family"`, and its keys to the key map.
2. Write `facts/<generation>.yaml` (see the user help, *Launch facts file*). List its siblings, most recent first.
3. Run `alignr draft --facts facts/<generation>.yaml --model baseline`. Fix the facts file until the baseline passes
   the gates, then draft with Gemini.

The page lists every facts file the API finds in `facts/`.

## Add a topic type or a section shape

Sibling topics define the target's shape. `skeleton.to_json` reads a topic's title, short description, and each
section's notes, table and paragraphs. To support a new block:

1. Read it in `skeleton.to_json` and `skeleton.plan_topic`.
2. Add its field to `schema.build_schema`. Keep enums tight: list only values that exist for this run.
3. Write it in `serialise.serialise`, in the order the DITA grammar allows.
4. Check its values in `checks.check_grounding`.
5. Add a test with a scripted answer in `tests/answers.py`.

A task shell is the next type to add. The OASIS task shell doesn't compile in libxml2 as shipped (the strict task body
constraint redefines a pattern the included grammar doesn't have), so `checks.relaxng` needs a task shell assembled
the same way as the concept and reference shells.

## Add a house rule

Rules the information architect owns are ISO Schematron. Corpus-wide rules live in
`corpus/tools/schematron/corpus.sch`; AlignR's are in `checks.ALIGNR_SCH`, and `_sibling_rules` writes rules that
depend on the siblings for each run. Use `role="error"` to fail a draft and `role="warning"` to report only. Give each
rule an ID and add it to the help topic *Checks and rule IDs*.
