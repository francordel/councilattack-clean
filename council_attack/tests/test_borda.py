"""Unit tests for Borda Top-1 aggregation and its reporting helpers."""

import json

from council_attack.borda import BordaAggregator
from council_attack.critique import CrossCritic


def _proposals():
    return [
        {"model": "m1", "proposal": "p1", "original_index": 0},
        {"model": "m2", "proposal": "p2", "original_index": 1},
        {"model": "m3", "proposal": "p3", "original_index": 2},
    ]


def _proposals_n(n):
    return [{"model": f"m{i}", "proposal": f"p{i}", "original_index": i} for i in range(n)]


def test_top1_hybrid_winner_is_majority_first_place():
    # Proposal 1 (idx0) is ranked first by 2 of 3 critics -> Top-1 winner.
    critiques = [
        {"critic": "m1", "rankings": [1, 2, 3]},
        {"critic": "m2", "rankings": [1, 3, 2]},
        {"critic": "m3", "rankings": [2, 1, 3]},
    ]
    agg = BordaAggregator(borda_pure=False)
    out = agg.aggregate(_proposals(), critiques)
    assert out[0]["original_index"] == 0
    assert out[0]["top1_votes"] == 2
    assert out[0]["borda_rank"] == 1


def test_top1_avoids_borda_compromise():
    # Paper's compromise case (N=3, K=3): an all-second proposal has the highest
    # Borda score, but Top-1 must still pick the proposal a majority ranks first.
    # idx0: ranked 1,1,3 -> top1=2, borda=(2+2+0)=4
    # idx1: ranked 2,2,2 -> top1=0, borda=(1+1+1)=3  (bland consensus)
    # idx2: ranked 3,3,1 -> top1=1, borda=(0+0+2)=2
    critiques = [
        {"critic": "a", "rankings": [1, 2, 3]},
        {"critic": "b", "rankings": [1, 2, 3]},
        {"critic": "c", "rankings": [3, 2, 1]},
    ]
    out = BordaAggregator(borda_pure=False).aggregate(_proposals(), critiques)
    assert out[0]["original_index"] == 0  # majority-first, not the all-second one


def test_borda_pure_prefers_high_borda_score():
    # Same votes, but pure Borda ranks by total Borda score, so the all-second
    # proposal (idx1, borda=3) can beat a split proposal. Here idx0 top1=2
    # borda=4 still wins, so use a case where pure vs hybrid diverge:
    # idx0: 1,1,3 -> top1=2 borda=4 ; idx1: 2,2,1 -> top1=1 borda=(1+1+2)=4
    critiques = [
        {"critic": "a", "rankings": [1, 2, 3]},
        {"critic": "b", "rankings": [1, 2, 3]},
        {"critic": "c", "rankings": [3, 1, 2]},
    ]
    hybrid = BordaAggregator(borda_pure=False).aggregate(_proposals(), critiques)
    pure = BordaAggregator(borda_pure=True).aggregate(_proposals(), critiques)
    # Hybrid: idx0 has top1=2 > idx1 top1=1 -> idx0 wins.
    assert hybrid[0]["original_index"] == 0
    # Pure: idx0 borda=4 ties idx1 borda=4; tiebreak on top1 -> idx0 (top1=2).
    assert pure[0]["borda_score"] == 4


def test_top1_vs_pure_diverge_at_n5():
    # N=5, 3 critics. The 3-proposal tests can't separate Top-1 from pure Borda;
    # this one does, so a regression collapsing Top-1 into pure Borda is caught.
    #   A (idx0): ranked 1,1,5 -> top1=2, borda=(4+4+0)=8   (majority first place)
    #   B (idx1): ranked 2,2,2 -> top1=0, borda=(3+3+3)=9   (bland all-second)
    critiques = [
        {"critic": "a", "rankings": [1, 2, 3, 4, 5]},
        {"critic": "b", "rankings": [1, 2, 4, 5, 3]},
        {"critic": "c", "rankings": [5, 2, 1, 3, 4]},
    ]
    hybrid = BordaAggregator(borda_pure=False).aggregate(_proposals_n(5), critiques)
    pure = BordaAggregator(borda_pure=True).aggregate(_proposals_n(5), critiques)
    # Top-1 hybrid: A wins on most first-place votes despite a lower Borda score.
    assert hybrid[0]["original_index"] == 0 and hybrid[0]["top1_votes"] == 2
    # Pure Borda: B wins on Borda score despite zero first-place votes.
    assert pure[0]["original_index"] == 1 and pure[0]["top1_votes"] == 0
    assert hybrid[0]["original_index"] != pure[0]["original_index"]


def test_critique_to_borda_key_seam_contract():
    # Contract: parse_critique_json emits "ranking"; CrossCritic.critique() republishes
    # it under "rankings"; BordaAggregator.aggregate() consumes "rankings". A rename on
    # either side silently degrades aggregation to "first proposal always wins".
    parsed = CrossCritic.parse_critique_json(
        json.dumps({"ranks": {"proposal_1": 2, "proposal_2": 1, "proposal_3": 3}}), 3
    )
    assert parsed is not None and "ranking" in parsed  # parser-side key
    critique = {"critic": "m", "rankings": parsed["ranking"], "reasons": parsed["reasons"]}
    out = BordaAggregator(borda_pure=False).aggregate(_proposals(), [critique])
    # proposal_2 (idx1) was voted rank 1, so it must win. If the "rankings" key seam
    # breaks, the vote is ignored and idx0 wins by default -> this assertion fails.
    assert out[0]["original_index"] == 1 and out[0]["top1_votes"] == 1


def test_council_disagreement_unanimous_is_zero():
    critiques = [
        {"critic": "a", "rankings": [1, 2, 3]},
        {"critic": "b", "rankings": [1, 2, 3]},
        {"critic": "c", "rankings": [1, 2, 3]},
    ]
    assert BordaAggregator.council_disagreement(critiques, 3) == 0.0


def test_council_disagreement_max_is_one():
    # Each proposal receives all three distinct ranks -> maximal entropy = 1.0.
    critiques = [
        {"critic": "a", "rankings": [1, 2, 3]},
        {"critic": "b", "rankings": [2, 3, 1]},
        {"critic": "c", "rankings": [3, 1, 2]},
    ]
    assert BordaAggregator.council_disagreement(critiques, 3) == 1.0


def test_build_critique_summary_marks_winner():
    critiques = [
        {"critic": "m1", "rankings": [1, 2, 3], "reasons": ["r1a", "r1b", "r1c"]},
        {"critic": "m2", "rankings": [1, 3, 2], "reasons": ["r2a", "r2b", "r2c"]},
        {"critic": "m3", "rankings": [2, 1, 3], "reasons": ["r3a", "r3b", "r3c"]},
    ]
    agg = BordaAggregator(borda_pure=False)
    ordered = agg.aggregate(_proposals(), critiques)
    summary = BordaAggregator.build_critique_summary(ordered[0], ordered, critiques)
    assert "★ BORDA WINNER" in summary
    assert "TEAM VOTES AND REASONING" in summary


def test_build_critique_summary_lists_proposals_without_votes():
    # use_borda=False / no critiques: the chairman must still see the proposals in full.
    out = BordaAggregator.build_critique_summary({}, _proposals(), [])
    assert "TEAM PROPOSALS (no votes" in out
    assert "p1" in out and "p2" in out and "p3" in out
    # Empty only when there are no proposals at all.
    assert BordaAggregator.build_critique_summary({}, [], []) == ""
