"""The draft board: every draftable player as parallel arrays plus an index.

Assembled once before the draft from calibrated projections and market data,
then held fixed. Holding it fixed is what makes a replay meaningful - if the
board can drift between calls, identical picks need not produce identical
recommendations, and the audit log proves nothing.

Late-breaking news (an inactive, a beat-writer report) enters through
`apply_update`, which returns a *new* board and records what changed. That is
brief section 2's job #3: the LLM feeds new information in through an explicit,
logged tool, never as unlogged narration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

import numpy as np

from .calibration import CalibratedProjection
from .ids import POSITIONS, Player, Resolver
from .projections import MarketData


@dataclass(frozen=True)
class ProjectionUpdate:
    """An explicit, logged adjustment to one player's outlook."""

    player_id: str
    points_multiplier: float = 1.0
    points_delta: float = 0.0
    sd_multiplier: float = 1.0
    out_for_season: bool = False
    reason: str = ""
    source: str = ""

    def describe(self) -> str:
        if self.out_for_season:
            what = "ruled out for the season"
        else:
            bits = []
            if self.points_multiplier != 1.0:
                bits.append(f"points x{self.points_multiplier:g}")
            if self.points_delta:
                bits.append(f"points {self.points_delta:+g}")
            if self.sd_multiplier != 1.0:
                bits.append(f"sd x{self.sd_multiplier:g}")
            what = ", ".join(bits) or "no change"
        src = f" [{self.source}]" if self.source else ""
        return f"{self.player_id}: {what} - {self.reason}{src}"


@dataclass
class Board:
    players: list[Player]
    points: np.ndarray
    sd: np.ndarray
    adp: np.ndarray
    pos_code: np.ndarray
    bye: np.ndarray
    raw_points: np.ndarray
    index: dict[str, int]
    adp_imputed: np.ndarray
    updates: tuple[ProjectionUpdate, ...] = ()
    _resolver: Resolver | None = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.players)

    @property
    def resolver(self) -> Resolver:
        if self._resolver is None:
            self._resolver = Resolver(self.players)
        return self._resolver

    def idx(self, player_id: str) -> int:
        try:
            return self.index[player_id]
        except KeyError:
            raise KeyError(f"{player_id!r} is not on the board") from None

    def player(self, player_id: str) -> Player:
        return self.players[self.idx(player_id)]

    def pos_of(self, i: int) -> str:
        return POSITIONS[int(self.pos_code[i])]

    def pos_mask(self, pos: str) -> np.ndarray:
        return self.pos_code == POSITIONS.index(pos)

    def fingerprint(self) -> str:
        """Hash of everything the engine reads off the board."""
        digest = hashlib.sha256()
        digest.update(b"ffdraft-board-v1")
        for arr in (self.points, self.sd, self.adp, self.pos_code, self.bye):
            digest.update(np.ascontiguousarray(arr, dtype=np.float64).round(6).tobytes())
        digest.update(json.dumps([p.player_id for p in self.players]).encode())
        digest.update(json.dumps([u.describe() for u in self.updates]).encode())
        return digest.hexdigest()[:16]

    def apply_update(self, update: ProjectionUpdate) -> "Board":
        """Return a new board with the update applied and recorded."""
        i = self.idx(update.player_id)
        points = self.points.copy()
        sd = self.sd.copy()
        if update.out_for_season:
            points[i] = 0.0
            sd[i] = max(1e-6, sd[i] * 0.01)
        else:
            points[i] = max(0.0, points[i] * update.points_multiplier + update.points_delta)
            sd[i] = max(1e-6, sd[i] * update.sd_multiplier)
        return replace(
            self,
            points=points,
            sd=sd,
            updates=self.updates + (update,),
            _resolver=None,
        )


def build_board(
    calibrated: list[CalibratedProjection],
    market: MarketData | None = None,
    pool_size: int | None = None,
    min_per_position: int = 0,
    impute_rank: dict[str, float] | None = None,
) -> Board:
    """Assemble a Board. `pool_size` trims to the top N by ADP-or-points.

    Trimming matters for the simulator's cost: nobody in a 136-pick draft is
    choosing among 900 players, and the arrays are re-materialised on every
    rollout step. `min_per_position` protects the positions the market ranks
    last from being trimmed out of existence.
    """
    market = market or MarketData()
    rows = list(calibrated)

    # Deterministic pre-sort so imputed ADP does not depend on input order.
    rows.sort(key=lambda c: (-round(c.points, 6), c.player_id))

    # Impute ADP for anyone the market file does not cover, offsetting past the
    # deepest real ADP so imputed players sit behind everyone the market
    # actually ranks.
    #
    # `impute_rank` should be VOR order where the caller can supply it. Raw
    # points order is a poor stand-in for draft order - it drafts every
    # quarterback far too early, because a QB outscores a running back without
    # being worth more than the next quarterback. VOR is the cheapest available
    # correction and needs no extra data.
    real_adp = [market.adp[c.player_id] for c in rows if c.player_id in market.adp]
    max_adp = max(real_adp) if real_adp else 0.0
    if impute_rank:
        order_key = sorted(
            range(len(rows)),
            key=lambda i: (-impute_rank.get(rows[i].player_id, float("-inf")),
                           rows[i].player_id),
        )
        fallback_rank = {rows[i].player_id: n for n, i in enumerate(order_key, start=1)}
    else:
        fallback_rank = {row.player_id: n for n, row in enumerate(rows, start=1)}

    adp_vals, imputed = [], []
    for rank, row in enumerate(rows, start=1):
        if row.player_id in market.adp:
            adp_vals.append(market.adp[row.player_id])
            imputed.append(False)
        elif row.player_id in market.ecr:
            adp_vals.append(market.ecr[row.player_id])
            imputed.append(False)
        else:
            adp_vals.append(max_adp + fallback_rank[row.player_id])
            imputed.append(True)

    order = np.argsort(np.array(adp_vals, dtype=float), kind="stable")
    if pool_size is not None:
        keep = list(order[:pool_size])
        # Trimming by ADP cuts kickers and defenses first - they have the
        # latest ADP by a wide margin. That is exactly backwards for a board
        # that must still be able to fill a mandatory K and DEF slot and
        # compute a replacement level at every position, so put back the top
        # `min_per_position` at each one regardless of where the market ranks
        # them. Without this the engine can run out of legal picks in the last
        # two rounds of a deep league.
        if min_per_position:
            kept = set(keep)
            by_pos: dict[str, list[int]] = {}
            for i in order:
                by_pos.setdefault(rows[i].player.pos, []).append(int(i))
            for pos, members in by_pos.items():
                members.sort(key=lambda i: -rows[i].points)
                for i in members[:min_per_position]:
                    if i not in kept:
                        kept.add(i)
                        keep.append(i)
        order = np.array(keep, dtype=np.int64)

    rows = [rows[i] for i in order]
    adp = np.array([adp_vals[i] for i in order], dtype=float)
    adp_imputed = np.array([imputed[i] for i in order], dtype=bool)

    players = [r.player for r in rows]
    return Board(
        players=players,
        points=np.array([r.points for r in rows], dtype=float),
        sd=np.array([r.sd for r in rows], dtype=float),
        adp=adp,
        pos_code=np.array([POSITIONS.index(r.player.pos) for r in rows], dtype=np.int64),
        bye=np.array([r.player.bye for r in rows], dtype=np.int64),
        raw_points=np.array([r.raw_points for r in rows], dtype=float),
        index={r.player_id: i for i, r in enumerate(rows)},
        adp_imputed=adp_imputed,
    )
