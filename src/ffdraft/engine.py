"""`recommend_pick` - the only thing that decides anything.

The tool contract from brief section 2, unchanged:

    recommend_pick(league_id, drafted, my_roster, pick_number) -> [Recommendation]

There are no tuning parameters, and adding one is not a small change. If a
parameter encodes an opinion it belongs in the league config, written before
the draft, not in a call an LLM composes under a 60-second clock. In
particular:

  Risk tolerance is derived. Maximising P(title) in `simulate.py` buys variance
  when the roster is behind the field and sheds it when ahead - the same
  endogenous appetite section 3.1 describes, with nothing to set.

  Position need is derived. It falls out of `roster_marginal`: a third QB in a
  two-QB league improves no lineup, so he scores zero, and no flag had to say so.

`my_roster` is redundant with `drafted` plus the seat, and that is the point -
it is a cross-check. If the two disagree, something is wrong with the log and
the engine stops rather than recommending against a state that does not match
the room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audit import AuditLog, state_hash
from .board import Board
from .config import LeagueConfig, load_by_id
from .data import build_board_for, default_paths
from .draft import DraftState, DraftStateError, unfilled_mandatory_slots
from .ids import POSITIONS
from .opponents import DraftSimulator
from .simulate import SeasonSimulator, draw_season
from .vor import (
    candidate_shortlist,
    replacement_points,
    roster_marginal,
    tier_of,
    tiers_for_position,
    vor_array,
)


@dataclass
class Recommendation:
    """One candidate, fully priced. Ranked by delta P(title)."""

    rank: int
    player_id: str
    name: str
    pos: str
    team: str
    points: float
    vor: float
    marginal: float
    survival: float          # P(still available at my next pick)
    cost_of_waiting: float   # expected VOR lost by passing and hoping
    p_title: float
    p_playoffs: float
    p_weekly_win: float
    delta_p_title: float     # vs the top-ranked candidate; 0.0 for the leader
    delta_se: float          # paired standard error of that delta
    tier: int
    tier_remaining: int
    flags: list[str] = field(default_factory=list)

    @property
    def display(self) -> str:
        team = f", {self.team}" if self.team else ""
        return f"{self.name} ({self.pos}{team})"


@dataclass
class PickAdvice:
    """The full result: the ranked list plus what the LLM needs to narrate it."""

    league_id: str
    pick_number: int
    round_number: int
    state_hash: str
    recommendations: list[Recommendation]
    replacement: dict[str, float]
    notes: list[str] = field(default_factory=list)
    seed: int = 0
    n_sims: int = 0

    @property
    def top(self) -> Recommendation | None:
        return self.recommendations[0] if self.recommendations else None

    def edge_over_second(self) -> float:
        if len(self.recommendations) < 2:
            return 0.0
        return self.recommendations[0].vor - self.recommendations[1].vor

    def format_card(self) -> str:
        """The four lines you can actually read on a 60-second clock."""
        if not self.recommendations:
            return "PICK: (no legal candidates)"
        top = self.recommendations[0]
        lines = [f"PICK: {top.name} ({top.pos}, {top.team or '?'})"]
        if len(self.recommendations) > 1:
            second = self.recommendations[1]
            # VOR can be negative here and that is not a bug: the ranking is by
            # title odds, so the engine will take a lower-VOR player when the
            # simulation says he wins more often. Show the sign honestly rather
            # than printing "+-29.8".
            #
            # When the top two are inside simulation noise, the title-odds gap
            # between them is not a quantity - it is sampling error, and
            # printing "-1.1% title odds" next to the recommended pick reads as
            # an argument against it. Say they are tied and let the WHY line
            # carry the reason (which is the cost of waiting).
            tied = "statistical tie with the leader" in second.flags
            margin = (
                "title odds tied"
                if tied
                else f"{top.p_title - second.p_title:+.1%} title odds"
            )
            lines.append(
                f"EDGE: {top.vor - second.vor:+.1f} VOR over {second.name} ({margin})"
            )
        else:
            lines.append(f"EDGE: {top.vor:+.1f} VOR over replacement")
        lines.append(f"WHY:  {self.why_line()}")
        flags = [f for r in self.recommendations[:1] for f in r.flags] + self.notes
        if flags:
            lines.append(f"FLAG: {'; '.join(flags[:2])}")
        return "\n".join(lines)

    def why_line(self) -> str:
        top = self.recommendations[0]
        bits = []
        if top.tier_remaining <= 1:
            bits.append(f"last {top.pos} in his tier")
        elif top.tier_remaining <= 3:
            bits.append(f"{top.tier_remaining} left in his {top.pos} tier")
        if top.survival < 0.35:
            bits.append(f"{top.survival:.0%} to last to your next pick")
        if len(self.recommendations) > 1:
            gap = top.p_title - self.recommendations[1].p_title
            if abs(gap) >= 2 * max(self.recommendations[1].delta_se, 1e-9):
                bits.append(f"{gap * 100:+.1f}pp title odds over {self.recommendations[1].name}")
        if not bits:
            bits.append(f"{top.vor:.0f} points over replacement {top.pos}")
        return "; ".join(bits[:2])

    def decision_basis(self) -> str:
        """Which criterion actually picked the top candidate. For the log."""
        if not self.recommendations:
            return "no candidates"
        if any("statistical tie" in f for r in self.recommendations for f in r.flags):
            return "cost of waiting (title odds inside simulation noise)"
        return "P(title)"


# --- board registry ----------------------------------------------------------
# Boards are expensive to build and must not drift mid-draft. One board per
# league_id, built once, reused for every call.
_REGISTRY: dict[str, tuple[LeagueConfig, Board]] = {}


def register_league(config: LeagueConfig, board: Board) -> None:
    _REGISTRY[config.league_id] = (config, board)


def clear_registry() -> None:
    _REGISTRY.clear()


def get_league(league_id: str) -> tuple[LeagueConfig, Board]:
    if league_id not in _REGISTRY:
        config = load_by_id(league_id)
        proj, market = default_paths()
        if not Path(proj).exists():
            raise FileNotFoundError(
                f"no projections at {proj}. Run the ffanalytics pull "
                f"(R/pull_projections.R) or register a board explicitly with "
                f"ffdraft.engine.register_league()."
            )
        register_league(config, build_board_for(config, proj, market))
    return _REGISTRY[league_id]


# --- the engine --------------------------------------------------------------
def recommend_pick(
    league_id: str,
    drafted: list[str],
    my_roster: list[str],
    pick_number: int,
    *,
    audit: AuditLog | None = None,
) -> PickAdvice:
    """Rank the board for the pick now on the clock.

    Deterministic: the same four arguments against the same board always
    produce the same list, because the simulation seed is derived from a hash
    of exactly those inputs.
    """
    config, board = get_league(league_id)
    return _recommend(config, board, drafted, my_roster, pick_number, audit=audit)


def _recommend(
    config: LeagueConfig,
    board: Board,
    drafted: list[str],
    my_roster: list[str],
    pick_number: int,
    *,
    audit: AuditLog | None = None,
) -> PickAdvice:
    unknown = [p for p in list(drafted) + list(my_roster) if p not in board.index]
    if unknown:
        raise DraftStateError(
            f"not on the board: {unknown}. Resolve names to PlayerIDs before "
            f"recording them - a mis-keyed pick silently corrupts every number "
            f"that follows."
        )

    state = DraftState(config=config, drafted=list(drafted))
    state.cross_check(pick_number)

    seat = _infer_seat(state, my_roster, config)
    state.my_seat = seat
    if state.on_the_clock != seat:
        raise DraftStateError(
            f"pick {pick_number} belongs to seat {state.on_the_clock}, but you are "
            f"seat {seat}. The engine only ranks the board for your own pick - "
            f"ranking someone else's would answer a question nobody asked."
        )
    derived = state.my_roster
    if sorted(derived) != sorted(my_roster):
        raise DraftStateError(
            f"my_roster disagrees with the pick log. Seat {seat} owns "
            f"{derived}, but you passed {sorted(my_roster)}. One of the two is "
            f"wrong; reconcile before drafting."
        )

    sh = state_hash(config.fingerprint(), board.fingerprint(), list(drafted), seat, pick_number)
    seed = (config.sim.seed ^ int(sh[:8], 16)) % (2**32)
    rng = np.random.default_rng(seed)

    available = np.ones(len(board), dtype=bool)
    if drafted:
        available[np.array([board.idx(p) for p in drafted], dtype=np.int64)] = False

    vor = vor_array(board, available, config.vor_baseline)
    replacement = replacement_points(board, available, config.vor_baseline)

    notes: list[str] = []
    shortlist = candidate_shortlist(
        board, available, config, my_roster, config.sim.candidate_pool
    )
    shortlist, guard_notes = _apply_policy_guard(board, config, state, my_roster, shortlist, available)
    notes.extend(guard_notes)

    if not shortlist:
        advice = PickAdvice(
            league_id=config.league_id, pick_number=pick_number,
            round_number=state.current_round, state_hash=sh,
            recommendations=[], replacement=replacement,
            notes=notes + ["no legal candidates remain"], seed=seed,
            n_sims=config.sim.n_sims,
        )
        return advice

    # Common random numbers: opponent preferences and player outcomes are drawn
    # once and reused for every candidate, so the differences between
    # candidates are differences in the candidates.
    n_sims = config.sim.n_sims
    simulator = DraftSimulator(board, config, seat)
    base_rankings = simulator.base_rankings(rng, n_sims)
    draws = draw_season(board, config, n_sims, rng)
    season = SeasonSimulator(board, config)

    my_next = state.my_following_pick()
    snapshots = (my_next,) if my_next else ()

    results = {}
    availability = {}
    for idx in shortlist:
        rollout = simulator.rollout(
            drafted_idx=[board.idx(p) for p in drafted],
            my_value=vor,
            base_rankings=base_rankings,
            forced_pick=idx,
            forced_at_pick=pick_number,
            snapshot_picks=snapshots,
        )
        results[idx] = season.run(rollout.rosters, draws, seat)
        if my_next:
            availability[idx] = rollout.available_at[my_next]

    by_title = sorted(
        shortlist,
        key=lambda i: (-round(results[i].p_title, 9), -round(float(vor[i]), 6),
                       board.players[i].player_id),
    )
    survival = _survival_for(by_title, availability, results)
    waiting = _cost_of_waiting(board, vor, availability, survival, by_title)
    order, tie_group, tie_note = _rank(by_title, results, vor, waiting, board, n_sims)
    if tie_note:
        notes.append(tie_note)
    leader = order[0]

    tier_cache: dict[str, list] = {}
    recs: list[Recommendation] = []
    for rank, idx in enumerate(order, start=1):
        player = board.players[idx]
        result = results[idx]
        if player.pos not in tier_cache:
            tier_cache[player.pos] = tiers_for_position(
                board, player.pos, available, depth=max(24, config.vor_baseline.get(player.pos, 12) * 2)
            )
        tier = tier_of(tier_cache[player.pos], player.player_id)
        delta = result.p_title - results[leader].p_title
        paired = results[leader].title_indicator.astype(float) - result.title_indicator.astype(float)
        delta_se = float(paired.std(ddof=1) / np.sqrt(n_sims)) if n_sims > 1 else 0.0

        flags: list[str] = []
        if board.adp_imputed[idx]:
            flags.append("no market ADP; survival is an estimate")
        if idx in tie_group and idx != leader:
            flags.append("statistical tie with the leader")
        elif rank > 1 and abs(delta) < 2 * delta_se:
            flags.append("inside simulation noise of the leader")

        recs.append(
            Recommendation(
                rank=rank,
                player_id=player.player_id,
                name=player.name,
                pos=player.pos,
                team=player.team,
                points=float(board.points[idx]),
                vor=float(vor[idx]),
                marginal=roster_marginal(board, my_roster, player.player_id, config),
                survival=survival.get(idx, float("nan")),
                cost_of_waiting=waiting.get(idx, 0.0),
                p_title=result.p_title,
                p_playoffs=result.p_playoffs,
                p_weekly_win=result.p_weekly_win,
                delta_p_title=delta,
                delta_se=delta_se,
                tier=tier.number if tier else 0,
                tier_remaining=tier.size if tier else 0,
                flags=flags,
            )
        )

    advice = PickAdvice(
        league_id=config.league_id,
        pick_number=pick_number,
        round_number=state.current_round,
        state_hash=sh,
        recommendations=recs,
        replacement=replacement,
        notes=notes,
        seed=seed,
        n_sims=n_sims,
    )

    (audit or AuditLog()).record(
        state_hash=sh,
        league_id=config.league_id,
        pick_number=pick_number,
        kind="recommend_pick",
        payload={
            "seed": seed,
            "n_sims": n_sims,
            "drafted_count": len(drafted),
            "my_seat": seat,
            "recommendations": [
                {
                    "rank": r.rank, "player_id": r.player_id, "vor": round(r.vor, 4),
                    "survival": round(r.survival, 4) if r.survival == r.survival else None,
                    "p_title": round(r.p_title, 6),
                    "delta_p_title": round(r.delta_p_title, 6),
                }
                for r in recs
            ],
            "notes": notes,
        },
    )
    return advice


def _infer_seat(state: DraftState, my_roster: list[str], config: LeagueConfig) -> int:
    """Work out which seat is mine from the roster, or trust the config."""
    if config.my_seat is not None:
        return config.my_seat
    if not my_roster:
        raise DraftStateError(
            "cannot tell which seat is yours: draft.my_seat is unset and "
            "my_roster is empty. Set draft.my_seat in the league config."
        )
    candidates = [
        seat for seat in range(1, config.teams + 1)
        if sorted(state.roster_of(seat)) == sorted(my_roster)
    ]
    if len(candidates) != 1:
        raise DraftStateError(
            f"my_roster {sorted(my_roster)} matches {len(candidates)} seats. "
            f"Set draft.my_seat in the league config."
        )
    return candidates[0]


def _apply_policy_guard(
    board: Board,
    config: LeagueConfig,
    state: DraftState,
    my_roster: list[str],
    shortlist: list[int],
    available: np.ndarray,
) -> tuple[list[int], list[str]]:
    """Section 3.5: keep K and DST out of the shortlist until they are forced.

    The three most common roster builds are ~60% of all teams and all win about
    half their games; the builds that beat 50% carried more RB/WR at the
    expense of K and DST. The guard releases automatically the moment the picks
    you have left equal the mandatory slots you still have to fill, so the
    roster is always legal at the end.
    """
    notes: list[str] = []
    roster_positions = [board.player(p).pos for p in my_roster]
    unfilled = unfilled_mandatory_slots(roster_positions, config)
    picks_left = config.rounds - len(my_roster)
    must_fill_now = sum(unfilled.values()) >= picks_left

    if must_fill_now and unfilled:
        forced = [i for i in shortlist if board.pos_of(i) in unfilled]
        if not forced:
            live = np.flatnonzero(available)
            forced = [
                int(i) for i in live[np.argsort(-board.points[live], kind="stable")]
                if board.pos_of(int(i)) in unfilled
            ][: config.sim.candidate_pool]
        notes.append(
            f"roster must still fill {dict(unfilled)} with {picks_left} pick(s) left; "
            f"restricted to those positions"
        )
        return forced, notes

    floors = {"K": config.policy.min_round_k, "DST": config.policy.min_round_dst}
    kept = []
    blocked = set()
    for i in shortlist:
        pos = board.pos_of(i)
        floor = floors.get(pos)
        if floor is not None and state.current_round < floor:
            blocked.add(pos)
            continue
        kept.append(i)
    if blocked:
        notes.append(
            f"{'/'.join(sorted(blocked))} withheld until round "
            f"{min(floors[p] for p in blocked)} (brief 3.5: winning builds spend "
            f"those slots on RB/WR)"
        )
    return kept or shortlist, notes


def _rank(
    by_title: list[int],
    results: dict,
    vor: np.ndarray,
    waiting: dict[int, float],
    board: Board,
    n_sims: int,
) -> tuple[list[int], set[int], str]:
    """Order the shortlist, refusing to invent precision the simulation lacks.

    P(title) is the objective, but it is estimated from a binary outcome, and
    the gap between the top few candidates is routinely smaller than the
    standard error of that estimate. Sorting on it anyway would hand back a
    ranking that reshuffles on a different seed - the engine would look
    decisive and be arbitrary.

    So candidates the simulation cannot separate from the leader are collected
    into one tie group and ordered instead by cost of waiting: how much value
    you expect to lose by passing now and hoping he comes back. Among options
    that win the title equally often, take the one you are most likely to lose.
    That is the same logic as the dynamic program in section 3.2, applied where
    the Monte Carlo runs out of resolution.
    """
    if not by_title:
        return [], set(), ""
    leader = by_title[0]
    lead_ind = results[leader].title_indicator.astype(float)

    tie_group = {leader}
    for idx in by_title[1:]:
        paired = lead_ind - results[idx].title_indicator.astype(float)
        se = float(paired.std(ddof=1) / np.sqrt(n_sims)) if n_sims > 1 else 0.0
        gap = results[leader].p_title - results[idx].p_title
        if gap <= 2 * se:
            tie_group.add(idx)

    def tie_key(i: int) -> tuple:
        return (
            -round(waiting.get(i, 0.0), 6),
            -round(float(vor[i]), 6),
            board.players[i].player_id,
        )

    tied = sorted(tie_group, key=tie_key)
    rest = [i for i in by_title if i not in tie_group]
    note = ""
    if len(tied) > 1:
        note = (
            f"{len(tied)} candidates are within simulation noise on title odds "
            f"({n_sims} sims); ordered by cost of waiting instead"
        )
    return tied + rest, tie_group, note


def _cost_of_waiting(
    board: Board,
    vor: np.ndarray,
    availability: dict[int, np.ndarray],
    survival: dict[int, float],
    order: list[int],
) -> dict[int, float]:
    """Expected VOR given up by passing on a player and hoping he returns.

    (1 - P(survives)) x (his VOR - the VOR of the best player at his position
    you expect to still be there). Zero when the drop-off behind him is
    nothing, which is exactly when waiting is free.
    """
    if not availability or not order:
        return {}
    reference = availability.get(order[0])
    if reference is None:
        return {}

    best_by_pos: dict[str, float] = {}
    for pos in POSITIONS:
        mask = board.pos_mask(pos)
        if not mask.any():
            continue
        pos_vor = np.where(mask[None, :], vor[None, :], -np.inf)
        live = np.where(reference, pos_vor, -np.inf)
        best = live.max(axis=1)
        best = best[np.isfinite(best)]
        best_by_pos[pos] = float(best.mean()) if best.size else 0.0

    out: dict[int, float] = {}
    for idx in order:
        pos = board.pos_of(idx)
        surv = survival.get(idx)
        if surv is None or surv != surv:
            out[idx] = 0.0
            continue
        drop = max(0.0, float(vor[idx]) - best_by_pos.get(pos, 0.0))
        out[idx] = (1.0 - surv) * drop
    return out


def _survival_for(
    order: list[int], availability: dict[int, np.ndarray], results: dict
) -> dict[int, float]:
    """P(candidate is still there at my next pick), measured honestly.

    A candidate's survival has to be read from a world in which I did *not*
    take him - in his own rollout he is gone by construction. So each candidate
    is measured in the leader's rollout, and the leader is measured in the
    runner-up's.
    """
    if not availability:
        return {}
    leader = order[0]
    backup = order[1] if len(order) > 1 else leader
    out: dict[int, float] = {}
    for idx in order:
        reference = backup if idx == leader else leader
        mask = availability.get(reference)
        if mask is None:
            continue
        out[idx] = float(mask[:, idx].mean())
    return out
