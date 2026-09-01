"""Value over replacement, tiers, and team need.

Section 3.2: Fry, Lundberg & Ohlmann formalise the draft as a stochastic
dynamic program over value of the player, value of the others still available,
and team need, then reduce it to a tractable deterministic DP. This module is
the value half of that; `survival.py` is the availability half and
`simulate.py` is what turns both into an objective.

Two distinct numbers, and conflating them is the classic drafting error:

  vor       board-level scarcity. Points above the player who will be there
            anyway. Says nothing about *your* roster.
  marginal  team need. How much this player improves your best starting
            lineup. Falls straight out of the lineup optimiser, which is
            exactly why "position need" is never a parameter of the tool -
            it is a consequence of the roster you already hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import Board
from .config import LeagueConfig
from .ids import POSITIONS
from .lineup import best_lineup_points

# A tier break is a gap this many standard deviations above the mean gap
# between consecutive players at a position.
TIER_GAP_Z = 1.0


def replacement_points(
    board: Board,
    available: np.ndarray,
    vor_baseline: dict[str, int],
) -> dict[str, float]:
    """Points of the replacement-level player at each position, given the live board.

    The baseline rank counts from the *original* board, so as players come off
    the top the replacement slides down with them. Once a position is picked
    past its baseline, replacement is the best player still available - which is
    the honest answer: at that point the next man up really is your fallback.
    """
    out: dict[str, float] = {}
    for pos in POSITIONS:
        pos_mask = board.pos_mask(pos)
        drafted_at_pos = int(np.count_nonzero(pos_mask & ~available))
        remaining = np.sort(board.points[pos_mask & available])[::-1]
        if remaining.size == 0:
            out[pos] = 0.0
            continue
        idx = vor_baseline.get(pos, 12) - drafted_at_pos - 1
        idx = int(min(max(idx, 0), remaining.size - 1))
        out[pos] = float(remaining[idx])
    return out


def vor_array(
    board: Board,
    available: np.ndarray,
    vor_baseline: dict[str, int],
) -> np.ndarray:
    """VOR for every player on the board (drafted players included, for logging)."""
    repl = replacement_points(board, available, vor_baseline)
    repl_vec = np.array([repl[POSITIONS[c]] for c in board.pos_code], dtype=float)
    return board.points - repl_vec


def replacement_sd(
    board: Board, available: np.ndarray, vor_baseline: dict[str, int]
) -> dict[str, float]:
    """Outcome sd of the replacement-level player, per position."""
    out: dict[str, float] = {}
    for pos in POSITIONS:
        pos_mask = board.pos_mask(pos)
        live = pos_mask & available
        if not live.any():
            out[pos] = 0.0
            continue
        order = np.argsort(-board.points[live], kind="stable")
        sds = board.sd[live][order]
        drafted_at_pos = int(np.count_nonzero(pos_mask & ~available))
        idx = int(min(max(vor_baseline.get(pos, 12) - drafted_at_pos - 1, 0), sds.size - 1))
        out[pos] = float(sds[idx])
    return out


@dataclass(frozen=True)
class Tier:
    pos: str
    number: int
    player_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.player_ids)


def tiers_for_position(
    board: Board,
    pos: str,
    available: np.ndarray,
    depth: int,
) -> list[Tier]:
    """Split the live players at a position into tiers at the natural gaps.

    Deterministic and parameter-free apart from `TIER_GAP_Z`: a break goes
    wherever the gap to the next player is more than one standard deviation
    above the average gap in the pool being considered.
    """
    live = board.pos_mask(pos) & available
    idxs = np.flatnonzero(live)
    if idxs.size == 0:
        return []
    idxs = idxs[np.argsort(-board.points[idxs], kind="stable")][:depth]
    pts = board.points[idxs]
    if idxs.size < 3:
        return [Tier(pos=pos, number=1, player_ids=tuple(board.players[i].player_id for i in idxs))]

    gaps = pts[:-1] - pts[1:]
    threshold = float(gaps.mean() + TIER_GAP_Z * gaps.std())

    tiers: list[Tier] = []
    current: list[str] = [board.players[idxs[0]].player_id]
    for k, gap in enumerate(gaps, start=1):
        if gap > threshold and threshold > 0:
            tiers.append(Tier(pos=pos, number=len(tiers) + 1, player_ids=tuple(current)))
            current = []
        current.append(board.players[idxs[k]].player_id)
    if current:
        tiers.append(Tier(pos=pos, number=len(tiers) + 1, player_ids=tuple(current)))
    return tiers


def tier_of(tiers: list[Tier], player_id: str) -> Tier | None:
    for tier in tiers:
        if player_id in tier.player_ids:
            return tier
    return None


def roster_marginal(
    board: Board,
    my_roster: list[str],
    candidate_id: str,
    config: LeagueConfig,
) -> float:
    """Improvement in best-lineup season points from adding one player.

    This is "team need", derived. A third quarterback in a two-QB league scores
    zero here no matter how good he is, and no flag had to be passed to say so.
    """
    positions = {pid: board.player(pid).pos for pid in my_roster}
    scores = {pid: float(board.points[board.idx(pid)]) for pid in my_roster}
    before = best_lineup_points(scores, positions, config.roster)

    positions[candidate_id] = board.player(candidate_id).pos
    scores[candidate_id] = float(board.points[board.idx(candidate_id)])
    after = best_lineup_points(scores, positions, config.roster)
    return after - before


def candidate_shortlist(
    board: Board,
    available: np.ndarray,
    config: LeagueConfig,
    my_roster: list[str],
    size: int,
    selectable: np.ndarray | None = None,
) -> list[int]:
    """Board indices worth paying for a Monte Carlo evaluation of.

    Ranked by VOR blended with team need so a shortlist is never all one
    position when the roster already has three of them. This is a *filter*,
    not the decision - the ranking that matters comes out of the simulator.

    `available` is who is left on the board and sets replacement level.
    `selectable` is who *this* roster may legally take, which is a subset once
    the K/DST caps bite. Keeping them separate matters: replacement level has
    to reflect the real board even for a position you are barred from adding.
    """
    vor = vor_array(board, available, config.vor_baseline)
    live = np.flatnonzero(available if selectable is None else selectable)
    if live.size == 0:
        return []

    # Pre-trim by VOR so the lineup optimiser runs on a short list.
    prelim = live[np.argsort(-vor[live], kind="stable")][: max(size * 4, 24)]
    scored: list[tuple[float, int]] = []
    for i in prelim:
        pid = board.players[i].player_id
        marginal = roster_marginal(board, my_roster, pid, config)
        # Equal blend: VOR alone over-drafts positions you have filled,
        # marginal alone over-drafts whatever slot happens to be empty.
        scored.append((0.5 * float(vor[i]) + 0.5 * marginal, int(i)))
    scored.sort(key=lambda t: (-round(t[0], 6), board.players[t[1]].player_id))
    return [i for _, i in scored[:size]]
