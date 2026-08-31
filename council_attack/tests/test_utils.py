"""Unit tests for the pure text/JSON helpers in council_attack.utils."""

import json

import pytest

from council_attack.utils import (
    balance_brackets,
    collapse_doubled_braces,
    escape_inner_control_chars,
    extract_json,
    flatten_template,
    is_refusal,
    load_yaml_prompts,
    normalize_template,
    strip_thinking,
)


@pytest.mark.parametrize("text", [
    "I'm sorry, I can't help with that.",
    "I cannot assist with this.",
    "As an AI language model, I must decline.",
    "I’m sorry, I can’t help with that.",  # curly apostrophes inside words are normalised
])
def test_is_refusal_true(text):
    assert is_refusal(text) is True


@pytest.mark.parametrize("text", [
    "Sure, here is how you do it.",
    "Step 1: gather the materials.",
    "The answer is 42.",
])
def test_is_refusal_false(text):
    assert is_refusal(text) is False


def test_strip_thinking_removes_block():
    assert strip_thinking("<think>secret</think>visible") == "visible"


def test_strip_thinking_unclosed():
    assert strip_thinking("before<think>never closed") == "before"


def test_strip_thinking_harmony_final_channel():
    raw = "junk<|channel|>final<|message|>THE ANSWER<|return|>"
    assert strip_thinking(raw) == "THE ANSWER"


def test_extract_json_clean():
    assert extract_json('{"a": 1, "b": [1, 2]}') == {"a": 1, "b": [1, 2]}


def test_extract_json_with_surrounding_prose():
    assert extract_json('here you go: {"a": 1} thanks') == {"a": 1}


def test_extract_json_none_on_garbage():
    assert extract_json("no json here at all") is None


def test_collapse_doubled_braces():
    assert collapse_doubled_braces('{{"x": 1}}') == '{"x": 1}'


def test_escape_inner_control_chars_makes_valid_json():
    repaired = escape_inner_control_chars('{"a": "line1\nline2"}')
    assert json.loads(repaired) == {"a": "line1\nline2"}


def test_balance_brackets_closes_truncated():
    repaired = balance_brackets('{"a": [1, 2, {"b": 3')
    assert json.loads(repaired) == {"a": [1, 2, {"b": 3}]}


def test_flatten_template_roles_uppercased():
    out = flatten_template([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ])
    assert out == "[SYSTEM] sys\n[USER] hi"


def test_normalize_template_passthrough_dicts():
    tmpl = [{"role": "user", "content": "x"}]
    assert normalize_template(tmpl) == tmpl


def test_normalize_template_from_bracket_strings():
    out = normalize_template(["[SYSTEM] role text", "[USER] question text"])
    assert out == [
        {"role": "system", "content": "role text"},
        {"role": "user", "content": "question text"},
    ]


def test_load_yaml_prompts_fallback_when_no_path():
    got = load_yaml_prompts(None, ("a", "b"), ("da", "db"))
    assert got == ("da", "db")


def test_load_yaml_prompts_reads_file(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text("first: hello\nsecond: world\n", encoding="utf-8")
    got = load_yaml_prompts(str(p), ("first", "second"), ("da", "db"))
    assert got == ("hello", "world")


def test_load_yaml_prompts_missing_key_falls_back(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text("first: hello\n", encoding="utf-8")
    got = load_yaml_prompts(str(p), ("first", "second"), ("da", "db"))
    assert got == ("hello", "db")
