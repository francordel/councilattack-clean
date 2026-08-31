"""Unit tests for Chairman prompt selection, placeholder substitution, and the
no-chairman winner_proposal fallback."""

from council_attack.chairman import Chairman


def _chairman(contextual: bool, use_borda: bool = True):
    return Chairman(
        client=None,
        chairman_model="m-chair",
        chairman_endpoint="http://x",
        chairman_max_tokens=256,
        chairman_think=False,
        contextual_mode=contextual,
        use_borda=use_borda,
        blind_mode=False,
        accumulate_target_context=False,
        contextual_template=Chairman.CONTEXTUAL_DEFAULT,
        standard_template=Chairman.STANDARD_DEFAULT,
    )


def test_standard_template_vote_clause_present():
    ch = _chairman(contextual=False)
    out = ch.standard_template.replace(
        "{vote_reasoning_clause}", " and each member's vote and reasoning."
    )
    assert "{vote_reasoning_clause}" not in out
    assert "vote and reasoning" in out
    assert out.startswith("You are a senior red teaming engineer.")


def test_contextual_template_vote_clause_absent():
    ch = _chairman(contextual=True)
    out = ch.contextual_template.replace("{vote_reasoning_clause}", ".")
    assert "{vote_reasoning_clause}" not in out
    assert '"template"' in out  # contextual chairman must emit a JSON template


def test_winner_proposal_standard_returns_longest_non_refusal():
    ch = _chairman(contextual=False)
    proposals = [
        {"model": "m1", "proposal": "I'm sorry, I can't."},                         # refusal -> skipped
        {"model": "m2", "proposal": "Sure, here is a plan."},                       # non-refusal (first)
        {"model": "m3", "proposal": "Sure, here is a considerably longer plan."},   # longest non-refusal
    ]
    # Paper (§3.1) fallback rule: the LONGEST non-refusal proposal, not the first.
    assert ch.winner_proposal(proposals) == "Sure, here is a considerably longer plan."


def test_winner_proposal_contextual_returns_first_template():
    ch = _chairman(contextual=True)
    tmpl = [{"role": "user", "content": "x"}]
    proposals = [{"model": "m1", "proposal": "flat", "template": tmpl}]
    assert ch.winner_proposal(proposals) == tmpl


def test_winner_proposal_empty():
    assert _chairman(contextual=False).winner_proposal([]) == ""
    assert _chairman(contextual=True).winner_proposal([]) == []
