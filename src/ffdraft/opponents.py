"""How the other seats behave, and the draft rollout built on top of it.

Section 3.4 gives two empirical findings from Lee & Liu's 1,350 Sleeper
leagues, and we implement exactly one of them:

  Herding is real and exploitable. For QBs in roughly the first 20 picks, and
  for K and DST throughout, P(position drafted) jumps when the previous team
  just took that position. This is the mechanism behind runs, and it is why the
  survival probability of the last good QB collapses the moment one goes.

  Handcuffing does not work. 793 teams with a handcuff pair won 51.04% against
  50.56% without, a Bayes factor of 4.2 *favouring no difference*. It is
  deliberately absent. Do not add it.

Each simulated team gets a persistent private ranking of the board - ADP plus
one draw of noise, fixed for the whole draft - rather than fresh noise at every
pick. Teams have consistent preferences; re-randomising each pick would model a
league of amnesiacs and would wash out exactly the runs we are trying to catch.

Determinism: the noise is drawn once from an explicitly seeded generator and
reused across every candidate evaluated at a given pick. That is common random
numbers, and it is the difference between resolving a 0.4-point edge and
sampling noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .board import Board
from .config import SLOT_ELIGIBILITY, LeagueConfig
from .draft import pick_owner
from .ids import POSITIONS

N_POS = len(POSITIONS)


def mandatory_slots(config: LeagueConfig) -> dict[str, int]:
    """Starting slots only one position can fill. These must be filled."""
    return {
        SLOT_ELIGIBILITY[slot][0]: count
        for slot, count in config.roster.slot_counts().items()
        if len(SLOT_ELIGIBILITY[slot]) == 1
    }


def roster_caps(config: LeagueConfig) -> dict[str, int]:
    """Most players at a position a rational team would ever draft.

    Starter slots the position can fill (fixed plus every flex it is eligible
    for) plus its share of the bench. Without this, simulated opponents draft
    four kickers and the survival model becomes fiction.
    """
    caps: dict[str, int] = {}
    counts = config.roster.slot_counts()
    bench_total = config.roster.bench
    share_total = sum(config.bench_shares.values()) or 1.0
    for pos in POSITIONS:
        starters = sum(n for slot, n in counts.items() if pos in SLOT_ELIGIBILITY[slot])
        bench = int(round(bench_total * config.bench_shares.get(pos, 0.0) / share_total))
        caps[pos] = max(starters + bench, 1 if starters else 0)
    return caps


@dataclass
class RolloutResult:
    """Completed drafts across all simulations."""

    rosters: np.ndarray                       # (n_sims, n_teams, rounds) int, -1 empty
    available_at: dict[int, np.ndarray] = field(default_factory=dict)
    n_sims: int = 0

    def survival(self, pick_number: int) -> np.ndarray:
        """P(each player is still on the board) at a pick number."""
        mask = self.available_at.get(pick_number)
        if mask is None:
            raise KeyError(f"no availability snapshot taken at pick {pick_number}")
        return mask.mean(axis=0)


class DraftSimulator:
    """Rolls the rest of the draft forward, for every seat including yours."""

    def __init__(self, board: Board, config: LeagueConfig, my_seat: int) -> None:
        self.board = board
        self.config = config
        self.my_seat = my_seat
        self.n_players = len(board)
        self.caps = np.array(
            [roster_caps(config).get(p, 0) for p in POSITIONS], dtype=np.int64
        )
        mand = mandatory_slots(config)
        self.mandatory = np.array([mand.get(p, 0) for p in POSITIONS], dtype=np.int64)

        # Herding expressed as a rank bonus rather than a probability
        # multiplier: under an ADP-plus-noise ranking, multiplying a position's
        # selection odds by m is equivalent to moving it log(m) * sd picks up
        # the board. With m = 2.5 and sd = 8 that is a shift of about 7 picks,
        # which is what a run actually looks like.
        opp = config.opponents
        self.herd_bonus = opp.adp_noise_picks * math.log(max(opp.herding_multiplier, 1.0))

    # -- private rankings -------------------------------------------------
    def base_rankings(self, rng: np.random.Generator, n_sims: int) -> np.ndarray:
        """(n_sims, n_teams, n_players) perceived draft position. Lower drafts sooner."""
        noise = rng.normal(
            0.0,
            self.config.opponents.adp_noise_picks,
            size=(n_sims, self.config.teams, self.n_players),
        )
        return self.board.adp[None, None, :] + noise

    def my_rankings(self, my_value: np.ndarray) -> np.ndarray:
        """My own continuation policy: best available by value, no noise.

        This is the rollout's stand-in for future versions of myself. It is
        deliberately simple - the real decision is made one pick at a time by
        the full engine, and a rollout policy that tried to be the engine would
        be both intractable and circular.
        """
        order = np.argsort(-my_value, kind="stable")
        rank = np.empty(self.n_players, dtype=float)
        rank[order] = np.arange(self.n_players, dtype=float)
        return rank

    # -- the rollout ------------------------------------------------------
    def rollout(
        self,
        drafted_idx: list[int],
        my_value: np.ndarray,
        base_rankings: np.ndarray,
        forced_pick: int | None = None,
        forced_at_pick: int | None = None,
        snapshot_picks: tuple[int, ...] = (),
    ) -> RolloutResult:
        """Complete the draft from the current state.

        drafted_idx    board indices already taken, in pick order
        my_value       value used by my continuation policy (typically VOR)
        base_rankings  (n_sims, n_teams, n_players) from `base_rankings`
        forced_pick    board index I am made to take
        forced_at_pick the overall pick number at which to force it. Must be
                       one of my seat's picks - passing the wrong number would
                       silently evaluate a candidate nobody ever drafted.
        snapshot_picks pick numbers at which to record who is still available
        """
        cfg = self.config
        n_sims = base_rankings.shape[0]
        teams, rounds = cfg.teams, cfg.rounds
        total = cfg.total_drafted

        available = np.ones((n_sims, self.n_players), dtype=bool)
        if drafted_idx:
            available[:, np.asarray(drafted_idx, dtype=np.int64)] = False

        pos_counts = np.zeros((n_sims, teams, N_POS), dtype=np.int64)
        rosters = np.full((n_sims, teams, rounds), -1, dtype=np.int64)
        fill = np.zeros((n_sims, teams), dtype=np.int64)

        # Seed counts and rosters from picks already made.
        for n, idx in enumerate(drafted_idx, start=1):
            seat = pick_owner(n, teams, cfg.draft_type) - 1
            slot = int(fill[0, seat])
            rosters[:, seat, slot] = idx
            fill[:, seat] += 1
            pos_counts[:, seat, int(self.board.pos_code[idx])] += 1

        if forced_pick is not None:
            if forced_at_pick is None:
                raise ValueError("forced_pick requires forced_at_pick")
            owner = pick_owner(forced_at_pick, teams, cfg.draft_type)
            if owner != self.my_seat:
                raise ValueError(
                    f"pick {forced_at_pick} belongs to seat {owner}, not seat "
                    f"{self.my_seat}; refusing to force a pick that is not mine"
                )

        my_rank = self.my_rankings(my_value)
        pos_code = self.board.pos_code
        snapshots: dict[int, np.ndarray] = {}
        last_pos = np.full(n_sims, -1, dtype=np.int64)

        sim_rows = np.arange(n_sims)
        for pick in range(len(drafted_idx) + 1, total + 1):
            if pick in snapshot_picks:
                snapshots[pick] = available.copy()

            seat = pick_owner(pick, teams, cfg.draft_type) - 1
            counts = pos_counts[:, seat, :]

            # Positions this team may still add.
            legal_pos = counts < self.caps[None, :]

            # Endgame: once picks remaining equal mandatory slots still empty,
            # nothing else may be taken. This is also what pulls K and DST off
            # the board in the last two rounds rather than never.
            picks_left = rounds - fill[:, seat]
            short = np.maximum(self.mandatory[None, :] - counts, 0)
            still_needed = short.sum(axis=1)
            must_fill = still_needed >= picks_left
            if must_fill.any():
                forced_mask = short > 0
                legal_pos = np.where(must_fill[:, None], forced_mask, legal_pos)

            playable = available & legal_pos[:, pos_code]

            if forced_pick is not None and pick == forced_at_pick:
                choice = np.full(n_sims, forced_pick, dtype=np.int64)
            elif seat + 1 == self.my_seat:
                score = np.where(playable, my_rank[None, :], np.inf)
                choice = np.argmin(score, axis=1)
            else:
                score = base_rankings[:, seat, :].copy()
                if self.herd_bonus > 0:
                    herd = self._herd_mask(last_pos, pick)
                    if herd is not None:
                        score = score - herd * self.herd_bonus
                score = np.where(playable, score, np.inf)
                choice = np.argmin(score, axis=1)

            # A sim with nothing playable (roster caps exhausted the board)
            # falls back to best available overall rather than stalling.
            dead = ~playable.any(axis=1)
            if dead.any():
                fallback = np.where(available, my_rank[None, :], np.inf)
                choice = np.where(dead, np.argmin(fallback, axis=1), choice)

            available[sim_rows, choice] = False
            slot = fill[:, seat]
            rosters[sim_rows, seat, slot] = choice
            fill[:, seat] += 1
            chosen_pos = pos_code[choice]
            np.add.at(pos_counts, (sim_rows, seat, chosen_pos), 1)
            last_pos = chosen_pos

        for pick in snapshot_picks:
            if pick > total:
                snapshots[pick] = available.copy()

        return RolloutResult(rosters=rosters, available_at=snapshots, n_sims=n_sims)

    def _herd_mask(self, last_pos: np.ndarray, pick: int) -> np.ndarray | None:
        """(n_sims, n_players) 1.0 where the previous pick makes this player hotter."""
        opp = self.config.opponents
        active: list[int] = []
        for pos in opp.herding_positions:
            if pos not in POSITIONS:
                continue
            code = POSITIONS.index(pos)
            if pos in opp.always_herd_positions or pick <= opp.herding_qb_pick_limit:
                active.append(code)
        if not active:
            return None
        active_arr = np.array(active, dtype=np.int64)
        # A run is on when the previous pick was one of the herding positions.
        herding_now = np.isin(last_pos, active_arr)
        if not herding_now.any():
            return None
        same_pos = self.board.pos_code[None, :] == last_pos[:, None]
        return (same_pos & herding_now[:, None]).astype(float)
