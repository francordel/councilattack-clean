"""Stage 2C of the Propose-Aggregate-Arbitrate pipeline: Borda Top-1
aggregation of council rankings (Section 3.2), plus the reporting helpers
that turn an aggregation result into text for the chairman / turn record."""

import math


class BordaAggregator:
    """Implements the Borda Top-1 rule: the winner is the proposal with the
    most rank-1 votes, with the accumulated Borda score as tiebreaker (or,
    in the ``borda_pure`` ablation, the classical Borda-count winner with
    top-1 votes as tiebreaker)."""

    def __init__(self, borda_pure: bool) -> None:
        self.borda_pure = borda_pure

    def aggregate(
        self, proposals: list[dict], critiques: list[dict]
    ) -> list[dict]:
        """
        Aggregate critic rankings using majority-vote for the winner (top-1),
        then Borda count for positions 2-N.

        Winner selection (Borda Top-1 hybrid, default): the proposal with the most
        rank-1 votes wins, with the accumulated Borda score as the sole tiebreaker.
        In the ``borda_pure`` ablation the order is reversed (highest Borda score
        wins, top-1 votes as tiebreaker). Positions 2-N: sorted by Borda score
        descending.

        Each proposal gets ``borda_score``, ``top1_votes``, and ``borda_rank``.
        """
        n = len(proposals)
        top1_votes   = [0] * n  # how many critics gave rank=1 to this proposal
        borda_scores = [0] * n  # tiebreaker + ordering for positions 2-N

        for critique in critiques:
            rankings = critique.get("rankings", [])
            if len(rankings) != n:
                continue
            for i, rank in enumerate(rankings):
                borda_scores[i] += (n - rank)  # n-1 pts for rank 1, 0 pts for rank n
                if rank == 1:
                    top1_votes[i] += 1

        # Winner selection. Borda Top-1 hybrid (default): most rank-1 votes,
        # Borda score as tiebreaker. Pure Borda (ablation, borda_pure=True):
        # highest Borda score, top-1 votes as tiebreaker.
        if self.borda_pure:
            winner_idx = max(range(n), key=lambda i: (borda_scores[i], top1_votes[i]))
        else:
            winner_idx = max(range(n), key=lambda i: (top1_votes[i], borda_scores[i]))
        rest_indices = sorted(
            [i for i in range(n) if i != winner_idx],
            key=lambda i: -borda_scores[i],
        )
        ordered = [winner_idx] + rest_indices
        sorted_proposals = [proposals[i] for i in ordered]

        for rank_pos, (p, orig_i) in enumerate(zip(sorted_proposals, ordered)):
            p["borda_score"] = borda_scores[orig_i]
            p["top1_votes"]  = top1_votes[orig_i]
            p["borda_rank"]  = rank_pos + 1  # 1 = winner

        return sorted_proposals

    @staticmethod
    def build_critique_summary(
        winner: dict, proposals: list[dict], critiques: list[dict]
    ) -> str:
        """
        Build a full critique summary for the chairman showing ALL proposals,
        each member's vote (rank) and reasoning for every proposal.

        Layout per proposal:
          Proposal N [model] - Borda winner / rank K
            • critic_A  voted rank=1  "reason..."
            • critic_B  voted rank=2  "reason..."
        """
        if not proposals:
            return ""

        # Always list the proposals so the chairman sees them even with no votes
        # (use_borda=False): the chairman injects proposals ONLY via this summary.
        has_votes = bool(critiques)
        header = (
            "TEAM VOTES AND REASONING (all proposals):" if has_votes
            else "TEAM PROPOSALS (no votes; not in Borda mode):"
        )
        lines: list[str] = [header]

        for p in proposals:
            orig_idx = p.get("original_index", 0)
            member_name = p["model"].split("/")[-1]
            if has_votes:
                borda_rank = p.get("borda_rank", "?")
                top1 = p.get("top1_votes", 0)
                borda_score = p.get("borda_score", 0)
                is_winner = borda_rank == 1
                label = " ★ BORDA WINNER" if is_winner else f" (rank {borda_rank})"
                header_line = (
                    f"\nProposal {orig_idx + 1} [{member_name}]{label} "
                    f" - top1_votes={top1}, borda_score={borda_score}:"
                )
            else:
                header_line = f"\nProposal {orig_idx + 1} [{member_name}]:"
            lines.append(header_line)
            # Full proposal text - the chairman sees every proposal in full.
            lines.append(f'  "{p["proposal"]}"')

            for critique in critiques:
                critic_name = critique["critic"].split("/")[-1]
                rankings = critique.get("rankings", [])
                reasons = critique.get("reasons", [])
                rank_given = rankings[orig_idx] if orig_idx < len(rankings) else "?"
                reason_given = reasons[orig_idx] if orig_idx < len(reasons) else ""
                lines.append(f"    • {critic_name}  voted rank={rank_given}  \"{reason_given}\"")

        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def council_disagreement(critiques: list[dict], n_proposals: int) -> float:
        """
        Compute the mean rank-entropy across proposals as a disagreement score.

        For each proposal, calculate the entropy of the distribution of ranks
        assigned by the critics (uniform distribution = maximum disagreement).
        Return the mean entropy across all proposals, normalised to [0, 1] by
        dividing by log2(min(n_proposals, n_critics)), the maximum achievable
        rank entropy (a proposal's ranks fall in at most min(N, K) non-empty
        bins) - or 0.0 if fewer than 2 valid critiques.

        A score near 0 means all critics agreed; near 1 means maximum disagreement.
        Useful as a paper signal: high disagreement → harder goal → lower success rate.
        """
        if len(critiques) < 2 or n_proposals < 2:
            return 0.0

        # rank_counts[proposal_idx][rank_value] = count
        rank_counts: list[dict[int, int]] = [{} for _ in range(n_proposals)]
        valid = 0
        for critique in critiques:
            rankings = critique.get("rankings", [])
            if len(rankings) != n_proposals:
                continue
            valid += 1
            for i, rank in enumerate(rankings):
                rank_counts[i][rank] = rank_counts[i].get(rank, 0) + 1

        if valid < 2:
            return 0.0

        entropies = []
        for counts in rank_counts:
            total = sum(counts.values())
            if total == 0:
                continue
            h = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
            entropies.append(h)

        if not entropies:
            return 0.0

        # Max achievable rank entropy: with K=valid critics and N=n_proposals
        # possible ranks, a proposal's ranks occupy at most min(N, K) non-empty
        # bins, so the uniform-distribution ceiling is log2(min(N, K)).
        max_entropy = math.log2(min(n_proposals, valid))
        return round(sum(entropies) / len(entropies) / max_entropy, 4) if max_entropy > 0 else 0.0
