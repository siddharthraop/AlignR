# Testing

`python -m pytest -q` runs the suite with a scripted model: no network, no key, under a second.

| Test | Proves |
|---|---|
| `test_published_topics_are_held_out`, `test_no_published_text_reaches_the_model` | The holdout: no published TPU7x sentence reaches a prompt. |
| `test_good_answer_passes_every_gate[...]` | Each topic type (concept, specification reference, data-table reference) passes grammar, rules and grounding, with an edit ratio under 40%. |
| `test_wrong_unit_fails_and_is_revised` | GiB against GB fails, and a revision changes only the failed field. |
| `test_still_failing_after_two_revisions_becomes_a_question` | Nothing is dropped silently. |
| `test_safe_fixes_run_in_code` | Literal values become keys and titles become sentence case without a model call. |
| `test_missing_conflict_question_is_added` | Every conflict becomes a `<draft-comment>` even if the model doesn't ask. |
| `test_new_concept_is_a_stub_with_questions` | A topic with no sibling gets an outline, not prose. |
| `test_baseline_is_gated_too`, `test_set_with_scripted_model` | The whole set, the files, the .zip and the merge gate. |
| `tests/test_api.py` | The streamed draft, the run, the .zip and the scorecard. |

CI (`tests.yml`) also drafts the baseline set and runs `alignr check` on it. The `alignr` workflow builds every set
with DITA-OT in strict mode, because lxml validates the grammar but doesn't apply DITA's default attribute values.

Run only the tests you touched while working, for example `python -m pytest -q -k grounding`.
