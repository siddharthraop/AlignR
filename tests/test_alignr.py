import copy
import json

import pytest

from alignr.agent import draft_set, draft_topic, write_outputs, zip_bytes
from alignr.checks import check_grounding, validate_grammar
from alignr.cli import main
from alignr.compare import edit_ratio
from alignr.corpus import Corpus, HoldoutError
from alignr.facts import load_facts
from alignr.model import BaselineModel, ScriptedModel, prompt_for
from alignr.schema import build_schema, validate_shape
from alignr.skeleton import plan_topic

from answers import GOOD, SPEC, good

FACTS = "facts/tpu7x.yaml"


@pytest.fixture(scope="module")
def env():
    c = Corpus()
    f = load_facts(FACTS, c)
    return Corpus(holdout=f.prefix), f


def run(env, slot, answers):
    c, f = env
    p = plan_topic(c, f, slot)
    m = ScriptedModel({p.topic_id: answers})
    return draft_topic(p, f, c, m, {"id": "test"}), m


# ---- the holdout -------------------------------------------------------------------------------------------
def test_published_topics_are_held_out(env):
    c, _ = env
    with pytest.raises(HoldoutError):
        c.parse("topics/tpu7x-specifications.dita")
    with pytest.raises(HoldoutError):
        c.parse("topics/tpu-generations-compared.dita")


def test_no_published_text_reaches_the_model(env):
    c, f = env
    published_only = ["is the generation for the largest", "For the memory layout of each chip",
                      "require contacting your account team", "approximately 7.37 TB/s. The"]
    for slot in ("intro", "specifications", "configurations", "vm-types", "memory"):
        prompt = prompt_for(plan_topic(c, f, slot), f, c)
        for s in published_only:
            assert s not in prompt, (slot, s)


def test_siblings_come_from_the_map(env):
    c, f = env
    p = plan_topic(c, f, "vm-types")
    assert [h for _, h in p.siblings] == ["topics/tpu-v6e-vm-types.dita", "topics/tpu-v5p-vm-types.dita"]
    assert plan_topic(c, f, "memory").stub


# ---- facts and conflicts -----------------------------------------------------------------------------------
def test_conflicts_are_found(env):
    c, f = env
    p = plan_topic(c, f, "specifications")
    kinds = {(x.kind, x.known_issue) for x in p.conflicts}
    assert ("siblings", "KI-1") in kinds and ("sources", "KI-2") in kinds


def test_facts_match_the_keys(env):
    c, f = env
    assert not f.proposals
    assert all(c.keys[v.key] == v.value for v in f.values if v.key)


# ---- gate 1: the schema ------------------------------------------------------------------------------------
def test_schema_enums_hold_only_real_keys_and_sources(env):
    c, f = env
    p = plan_topic(c, f, "specifications")
    schema, h = build_schema(p, f, c)
    row = schema["properties"]["body"]["properties"]["system-architecture"]["properties"]["table"]["properties"]["rows"]
    assert row["minItems"] == row["maxItems"] == 9
    assert set(row["items"]["properties"]["key"]["enum"]) == {v.key for v in f.spec_values()}
    bad = copy.deepcopy(SPEC)
    bad["body"]["system-architecture"]["table"]["rows"][0]["key"] = "tpu7x-made-up"
    assert any("allowed values" in m for _, m in validate_shape(bad, schema))
    assert not validate_shape(SPEC, schema)


# ---- gate 2 and the full loop, per topic type ---------------------------------------------------------------
@pytest.mark.parametrize("slot", ["intro", "specifications", "configurations", "vm-types"])
def test_good_answer_passes_every_gate(env, slot):
    r, m = run(env, slot, good()[f"tpu7x-{slot}"])
    assert r["grammar"]["valid"], r["grammar"]
    assert r["rules"]["pass"], r["rules"]
    assert r["grounding"]["grounded"] == r["grounding"]["total"] > 0
    assert r["revisions"] == 0 and not r["remaining_failures"]
    assert r["compare"]["edit_ratio"] < 0.40, r["compare"]["edit_ratio"]


def test_spec_values_are_keys_and_conflicts_are_questions(env):
    r, _ = run(env, "specifications", good()["tpu7x-specifications"])
    xml = r["xml"]
    assert xml.count('keyref="tpu7x-') == 9 and "192" not in xml
    assert xml.count("<draft-comment") == 2 and 'about(C2)' in xml


def test_missing_conflict_question_is_added(env):
    a = copy.deepcopy(SPEC)
    a["questions"] = []
    r, _ = run(env, "specifications", [a])
    assert r["xml"].count("<draft-comment") == 2


def test_wrong_unit_fails_and_is_revised(env):
    bad = copy.deepcopy(SPEC)
    bad["body"]["system-architecture"]["table"]["rows"][3]["unit"] = "GB"   # GiB in the facts
    fixed = copy.deepcopy(SPEC)
    fixed["body"]["system-architecture"]["paras"][0]["text"] = "CHANGED BY THE MODEL"  # not a failed field: ignored
    r, m = run(env, "specifications", [bad, fixed])
    assert r["revisions"] == 1 and not r["remaining_failures"]
    assert "GiB" in r["xml"] and "CHANGED" not in r["xml"]
    assert any("unit GB" in f["problem"] for f in m.seen["tpu7x-specifications"][1])


def test_still_failing_after_two_revisions_becomes_a_question(env):
    bad = copy.deepcopy(GOOD["tpu7x-vm-types"])
    bad["body"]["vm-types"]["paras"][0]["text"] = "Each VM has 512 GB of host memory."
    r, _ = run(env, "vm-types", [bad])
    assert r["revisions"] == 2
    assert r["grammar"]["valid"] and "512 GB of host" not in r["xml"].split("<draft-comment")[0]
    assert any("512 GB" in q for q in r["questions"])


def test_safe_fixes_run_in_code(env):
    a = copy.deepcopy(GOOD["tpu7x-configurations"])
    a["title"] = "{product-tpu7x} Supported Configurations"
    a["body"]["slice-shapes"]["paras"][1]["text"] = "Slices larger than 2048 chips require contacting your account team."
    r, _ = run(env, "configurations", [a])
    assert r["revisions"] == 0
    assert any("sentence case" in x for x in r["fixes"]) and any("tpu7x-max-api-slice" in x for x in r["fixes"])
    assert 'keyref="tpu7x-max-api-slice"' in r["xml"]


def test_unknown_key_and_flag_fail_grounding(env):
    c, f = env
    p = plan_topic(c, f, "vm-types")
    a = copy.deepcopy(GOOD["tpu7x-vm-types"])
    a["body"]["vm-types"]["paras"][0]["text"] = "Use {tpu7x-topology} and `--version=v2-alpha-tpu7x`."
    probs = [x["problem"] for x in check_grounding(a, p, f, c).failures]
    assert any("not a key" in x for x in probs) and any("--version" in x for x in probs)


def test_new_concept_is_a_stub_with_questions(env):
    r, _ = run(env, "memory", good()["tpu7x-memory"])
    assert r["stub"] and r["grammar"]["valid"] and r["rules"]["pass"]
    assert "<p>" not in r["xml"] and len(r["questions"]) >= 4


def test_grammar_rejects_invalid_dita():
    bad = b'<reference id="x"><title>T</title><refbody><p>loose</p></refbody></reference>'
    assert not validate_grammar(bad)["valid"]


# ---- the set, the files and the gates ------------------------------------------------------------------------
def test_set_with_scripted_model(tmp_path):
    r = draft_set(FACTS, ScriptedModel(good()), parallel=1)
    m = r["metrics"]
    assert m["topics"] == 5 and m["grammar_failures_reaching_writer"] == 0
    assert m["values_sourced_pct"] == 100.0 and m["edit_ratio_met"]
    out = write_outputs(r, tmp_path / "o")
    assert len(list((out / "topics").glob("*.dita"))) == 5
    assert zip_bytes(r)[:2] == b"PK"
    assert main(["check", str(out / "topics" / "tpu7x-vm-types.dita"), "--merge"]) == 0
    assert main(["check", str(out / "topics" / "tpu7x-specifications.dita"), "--merge"]) == 1


def test_baseline_is_gated_too():
    r = draft_set(FACTS, BaselineModel(), parallel=1)
    assert r["metrics"]["grammar_failures_reaching_writer"] == 0
    spec = next(t for t in r["topics"] if t["slot"] == "specifications")
    assert any("unit GB" in f["problem"] for h in spec["history"] for f in h["failures"])


def test_edit_ratio():
    assert edit_ratio("a b c d", "a b c d") == 0
    assert edit_ratio("a b c d", "a b x d") == 0.25
    assert edit_ratio("", "a b") == 1


def test_generation_notes_belong_in_the_introduction(env):
    c, f = env
    def notes(slot, sid):
        s, _ = build_schema(plan_topic(c, f, slot), f, c)
        return s["properties"]["body"]["properties"][sid]["properties"]["notes"]["items"].get("enum", [])
    assert set(notes("intro", "body")) == {"note-tpu-api-tpu7x", "note-tensorflow-tpu7x"}
    assert notes("specifications", "system-architecture") == []
    assert notes("configurations", "slice-shapes") == ["note-slice-shape"]
