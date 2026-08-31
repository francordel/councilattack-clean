"""Unit tests for proposal JSON parsing (single-question and contextual
multi-turn template forms), including malformed-JSON repair."""

import json

from council_attack.proposals import ProposalGenerator


def test_parse_proposal_json_extracts_question():
    raw = json.dumps({"generated_question": "hello?", "rationale": "x"})
    assert ProposalGenerator._parse_proposal_json(raw) == "hello?"


def test_parse_proposal_json_strips_code_fence():
    raw = '```json\n{"generated_question": "q1"}\n```'
    assert ProposalGenerator._parse_proposal_json(raw) == "q1"


def test_parse_proposal_json_fallback_on_non_json():
    assert ProposalGenerator._parse_proposal_json("just text") == "just text"


def test_parse_contextual_proposal_wellformed():
    raw = json.dumps({
        "template": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
        "mutator": "roleplay",
        "rationale": "r",
    })
    out = ProposalGenerator._parse_contextual_proposal(raw)
    assert out["mutator"] == "roleplay"
    assert len(out["template"]) == 2
    assert out["template"][0] == {"role": "system", "content": "s"}


def test_parse_contextual_proposal_repairs_doubled_braces():
    raw = '{{"template": [{{"role": "user", "content": "hi"}}], "mutator": "m", "rationale": "r"}}'
    out = ProposalGenerator._parse_contextual_proposal(raw)
    assert out["template"] == [{"role": "user", "content": "hi"}]


def test_parse_contextual_proposal_repairs_truncated_json():
    # Missing closing brackets/braces (token-limit truncation).
    raw = '{"template": [{"role": "user", "content": "hi"}], "mutator": "m"'
    out = ProposalGenerator._parse_contextual_proposal(raw)
    assert out["template"] == [{"role": "user", "content": "hi"}]


def test_parse_contextual_proposal_empty_on_garbage():
    out = ProposalGenerator._parse_contextual_proposal("no json here")
    assert out["template"] == []


def test_parse_contextual_proposal_from_bracket_string_template():
    raw = json.dumps({
        "template": ["[SYSTEM] role", "[USER] ask"],
        "mutator": "m",
        "rationale": "r",
    })
    out = ProposalGenerator._parse_contextual_proposal(raw)
    assert out["template"] == [
        {"role": "system", "content": "role"},
        {"role": "user", "content": "ask"},
    ]
