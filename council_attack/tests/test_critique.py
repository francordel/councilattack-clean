"""Unit tests for cross-critique JSON parsing and prompt construction."""

import json

from council_attack.critique import CrossCritic


def _critic():
    return CrossCritic(
        client=None,
        council_models=["m1", "m2", "m3"],
        critique_prompt_mode="hardened",
        hardened_template=CrossCritic.HARDENED_DEFAULT,
        soft_template=CrossCritic.SOFT_DEFAULT,
        log_raw_outputs=False,
    )


def test_parse_critique_primary_dict_schema():
    raw = json.dumps({
        "criterion": "attack effectiveness",
        "ranks": {"proposal_1": 2, "proposal_2": 1, "proposal_3": 3},
        "reasons": {"proposal_1": "a", "proposal_2": "b", "proposal_3": "c"},
    })
    out = CrossCritic.parse_critique_json(raw, 3)
    assert out["ranking"] == [2, 1, 3]
    assert out["reasons"] == ["a", "b", "c"]
    assert out["criterion"] == "attack effectiveness"


def test_parse_critique_legacy_list_schema():
    raw = json.dumps({"ranking": [2, 1, 3], "reasons": ["a", "b", "c"]})
    out = CrossCritic.parse_critique_json(raw, 3)
    assert out["ranking"] == [2, 1, 3]


def test_parse_critique_rejects_non_permutation():
    raw = json.dumps({"ranks": {"proposal_1": 1, "proposal_2": 1, "proposal_3": 2}})
    assert CrossCritic.parse_critique_json(raw, 3) is None


def test_parse_critique_none_on_garbage():
    assert CrossCritic.parse_critique_json("not json", 3) is None


def test_hardened_prompt_substitutes_placeholders():
    crit = _critic()
    system, user = crit.build_prompt_hardened(3, "PROPOSALS", "GOAL")
    for ph in ("{n}", "{ranks_example}", "{reasons_example}"):
        assert ph not in system
    assert "1..3" in system
    assert "proposal_1" in system and "proposal_3" in system
    assert "GOAL" in user and "PROPOSALS" in user


def test_soft_prompt_substitutes_placeholders():
    crit = _critic()
    system, _ = crit.build_prompt_soft(5, "P", "G")
    for ph in ("{n}", "{ranks_example}", "{reasons_example}"):
        assert ph not in system
    assert "proposal_5" in system


def test_hardened_prompt_preserves_json_object_braces():
    # The hardened template intentionally keeps doubled outer braces (rendered
    # as literal { } in the final prompt) - the model must see a JSON object.
    crit = _critic()
    system, _ = crit.build_prompt_hardened(2, "P", "G")
    assert '"criterion"' in system
    assert '"ranks"' in system
