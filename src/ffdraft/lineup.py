"""Optimal starting-lineup selection.

Both the "team need" half of VOR and every week of the Monte Carlo need the
same primitive: given a roster and a set of scores, what does the best legal
starting lineup score?

The slots in both leagues form a *nested* (laminar) eligibility family -
QB is a subset of Q/W/R/T, WR and RB and TE are subsets of W/R/T which is a
subset of Q/W/R/T. For nested eligibility, filling slots from most restrictive
to least, each time taking the highest-scoring eligible player left, is exactly
optimal. `tests/test_lineup.py` holds us to that against brute force.

Ties are broken by roster order so the same inputs always produce the same
lineup, which matters because the audit log hashes the output.
"""

from __future__ import annotations

import numpy as np

from .config import SLOT_ELIGIBILITY, RosterSpec
from .ids import POSITIONS

NEG = -1.0e18



def slot_fill_order(starters: tuple[str, ...]) -> list[str]:
    """Most restrictive slot first; stable within a restrictiveness level."""
    return sorted(
        starters,
        key=lambda s: (len(SLOT_ELIGIBILITY[s]), s),
    )


def best_lineup_points(
    scores: dict[str, float],
    positions: dict[str, str],
    roster: RosterSpec,
) -> float:
    """Scalar version: total points of the optimal legal lineup."""
    used: set[str] = set()
    total = 0.0
    for slot in slot_fill_order(roster.starters):
        eligible = SLOT_ELIGIBILITY[slot]
        best_id, best_score = None, None
        for pid, pos in positions.items():
            if pid in used or pos not in eligible:
                continue
            score = scores.get(pid, 0.0)
            if best_score is None or score > best_score:
                best_id, best_score = pid, score
        if best_id is not None:
            used.add(best_id)
            total += best_score or 0.0
    return total


def lineup_slots(
    scores: dict[str, float],
    positions: dict[str, str],
    roster: RosterSpec,
) -> dict[str, str | None]:
    """Which player fills which slot. Slot keys are suffixed when repeated."""
    used: set[str] = set()
    out: dict[str, str | None] = {}
    counter: dict[str, int] = {}
    for slot in slot_fill_order(roster.starters):
        counter[slot] = counter.get(slot, 0) + 1
        label = slot if roster.starters.count(slot) == 1 else f"{slot}{counter[slot]}"
        eligible = SLOT_ELIGIBILITY[slot]
        best_id, best_score = None, None
        for pid, pos in positions.items():
            if pid in used or pos not in eligible:
                continue
            score = scores.get(pid, 0.0)
            if best_score is None or score > best_score:
                best_id, best_score = pid, score
        out[label] = best_id
        if best_id is not None:
            used.add(best_id)
    return out


class VectorLineup:
    """Vectorised lineup evaluation for the Monte Carlo.

    Built once per league shape, then applied to arrays of shape
    (..., n_roster). Each slot costs one masked argmax over the roster axis,
    and there are only ~11 slots, so a full season for a full league is a
    handful of milliseconds.
    """

    def __init__(self, roster: RosterSpec) -> None:
        self.roster = roster
        self.order = slot_fill_order(roster.starters)
        self.eligibility = [SLOT_ELIGIBILITY[s] for s in self.order]
        # Eligibility as a lookup table indexed by (slot, pos_code + 1), so an
        # empty roster spot (-1) lands in column 0 and is never eligible.
        # np.isin here instead costs more than the argmax it feeds.
        self._table = np.zeros((len(self.order), len(POSITIONS) + 1), dtype=bool)
        for k, eligible in enumerate(self.eligibility):
            for pos in eligible:
                self._table[k, POSITIONS.index(pos) + 1] = True

    def total(self, scores: np.ndarray, pos_codes: np.ndarray) -> np.ndarray:
        """Best-lineup total.

        scores:    (..., n_roster) float
        pos_codes: (..., n_roster) int, index into ffdraft.ids.POSITIONS
        returns:   (...) float, same dtype as `scores`
        """
        available = np.ones(scores.shape, dtype=bool)
        total = np.zeros(scores.shape[:-1], dtype=scores.dtype)
        slot_codes = np.asarray(pos_codes) + 1
        for k in range(len(self.order)):
            eligible_mask = self._table[k][slot_codes] & available
            masked = np.where(eligible_mask, scores, NEG)
            pick = np.argmax(masked, axis=-1)
            taken = np.take_along_axis(masked, pick[..., None], axis=-1)[..., 0]
            has_one = eligible_mask.any(axis=-1)
            total += np.where(has_one, taken, 0.0)
            # Retire the chosen player, but only where the slot was actually
            # filled. When it was not, `pick` is the argmax of an all -inf row
            # and points at an arbitrary player who must stay available.
            picked = pick[..., None]
            still = np.take_along_axis(available, picked, axis=-1)
            np.put_along_axis(available, picked, still & ~has_one[..., None], axis=-1)
        return total
