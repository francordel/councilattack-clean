"""Unit tests for HarmJudge response parsing."""

import asyncio
import json

from council_attack.judge import JUDGE_UNSCORED, HarmJudge
from council_attack.llm_client import ModelClientError


class _RaisingClient:
    """A ModelClient stand-in whose query raises as a persistent endpoint outage."""

    async def query(self, *args, **kwargs):
        raise ModelClientError("judge endpoint down")


class _EmptyClient:
    async def query(self, *args, **kwargs):
        return ""


def test_judge_outage_is_unscored_not_zero():
    # A transport/HTTP outage at the judge endpoint must be UNSCORED (< 0), not
    # scored as a benign harm-0, which would deflate the reported ASR (#10).
    judge = HarmJudge(client=_RaisingClient(), judge_model="j", judge_max_tokens=100)
    score, reasoning, disclaimer, refusal = asyncio.run(judge.score("goal", "prompt", "response"))
    assert score == JUDGE_UNSCORED
    assert "unavailable" in reasoning.lower()


def test_judge_empty_or_unparseable_output_is_unscored():
    # An empty / unparseable judge completion is a judge FAILURE (no verdict), not a
    # genuine harm-0: it must be UNSCORED so it doesn't deflate the reported ASR.
    judge = HarmJudge(client=_EmptyClient(), judge_model="j", judge_max_tokens=100)
    score, *_ = asyncio.run(judge.score("goal", "prompt", "response"))
    assert score == JUDGE_UNSCORED


def test_judge_valid_zero_is_scored_zero():
    # A genuine harm-0 verdict (parseable JSON) is a real 0, NOT UNSCORED.
    class _ZeroClient:
        async def query(self, *args, **kwargs):
            return json.dumps({"reasoning": "full refusal", "harm": 0, "disclaimer": False, "refusal": True})

    score, _reason, _disc, refusal = asyncio.run(
        HarmJudge(client=_ZeroClient(), judge_model="j", judge_max_tokens=100).score("g", "p", "r")
    )
    assert score == 0 and refusal is True


def test_parse_response_valid_json():
    raw = json.dumps({"reasoning": "because", "harm": 3, "disclaimer": False, "refusal": False})
    score, reasoning, disclaimer, refusal = HarmJudge.parse_response(raw)
    assert score == 3
    assert reasoning == "because"
    assert disclaimer is False
    assert refusal is False


def test_parse_response_clamps_out_of_range():
    score, *_ = HarmJudge.parse_response(json.dumps({"harm": 9}))
    assert score == 5
    score2, *_ = HarmJudge.parse_response(json.dumps({"harm": -2}))
    assert score2 == 0


def test_parse_response_regex_fallback():
    score, reasoning, disclaimer, refusal = HarmJudge.parse_response('garble "harm": 2 more')
    assert score == 2
    assert disclaimer is False and refusal is False


def test_parse_response_empty():
    score, reasoning, *_ = HarmJudge.parse_response("")
    assert score == 0
    assert "empty" in reasoning.lower()


def test_parse_response_flags():
    raw = json.dumps({"harm": 4, "disclaimer": True, "refusal": True, "reasoning": "r"})
    score, _, disclaimer, refusal = HarmJudge.parse_response(raw)
    assert score == 4 and disclaimer is True and refusal is True
