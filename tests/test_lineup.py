"""Greedy lineup filling must equal brute force, in scalar and vector form."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from ffdraft.config import SLOT_ELIGIBILITY, RosterSpec
from ffdraft.ids import POSITIONS
from ffdraft.lineup import VectorLineup, best_lineup_points, lineup_slots


def brute_force(scores, positions, roster: RosterSpec) -> float:
    """Every legal assignment of players to slots.

    Slots may be left empty and players may be benched, which matters: a roster
    with two players and five slots still has a best lineup. Only tractable for
    the tiny cases below.
    """
    slots = list(roster.starters)
    players = list(positions)
    best = 0.0
    # Each player goes to a distinct slot, or to the bench (-1).
    for assignment in itertools.product(range(-1, len(slots)), repeat=len(players)):
        used = [a for a in assignment if a >= 0]
        if len(used) != len(set(used)):
            continue
        total, ok = 0.0, True
        for pid, slot_idx in zip(players, assignment):
            if slot_idx < 0:
                continue
            if positions[pid] not in SLOT_ELIGIBILITY[slots[slot_idx]]:
                ok = False
                break
            total += scores[pid]
        if ok:
            best = max(best, total)
    return best


CASES = [
    # (starters, [(player, pos, score)])
    (("QB", "QWRT"), [("q1", "QB", 30), ("q2", "QB", 25), ("w1", "WR", 28)]),
    (("WR", "WRT"), [("w1", "WR", 20), ("r1", "RB", 25)]),
    (("QB", "RB", "RB", "WRT", "QWRT"),
     [("q1", "QB", 30), ("q2", "QB", 22), ("r1", "RB", 26), ("r2", "RB", 18),
      ("r3", "RB", 12), ("w1", "WR", 24), ("t1", "TE", 15)]),
    (("WRT", "WRT", "QWRT"),
     [("w1", "WR", 9), ("t1", "TE", 21), ("r1", "RB", 14), ("q1", "QB", 20)]),
    # Short roster: fewer players than slots.
    (("QB", "RB", "WR", "WRT"), [("q1", "QB", 18), ("w1", "WR", 11)]),
]


@pytest.mark.parametrize("starters,players", CASES)
def test_greedy_matches_brute_force(starters, players):
    roster = RosterSpec(starters=starters, bench=0)
    scores = {p: s for p, _, s in players}
    positions = {p: pos for p, pos, _ in players}
    assert best_lineup_points(scores, positions, roster) == pytest.approx(
        brute_force(scores, positions, roster)
    )


def test_vector_matches_scalar(seated_config):
    """The simulator's kernel and the VOR path must agree exactly."""
    rng = np.random.default_rng(4)
    roster = seated_config.roster
    vector = VectorLineup(roster)
    depth = roster.total_picks
    pos_choices = ["QB", "RB", "WR", "TE", "K", "DST"]

    for trial in range(25):
        picks = [pos_choices[rng.integers(len(pos_choices))] for _ in range(depth)]
        scores = rng.uniform(0, 300, size=depth)
        codes = np.array([POSITIONS.index(p) for p in picks], dtype=np.int64)
        expected = best_lineup_points(
            {str(i): scores[i] for i in range(depth)},
            {str(i): picks[i] for i in range(depth)},
            roster,
        )
        got = float(vector.total(scores[None, :], codes[None, :])[0])
        assert got == pytest.approx(expected, rel=1e-9), f"trial {trial}"


def test_vector_ignores_empty_roster_spots(seated_config):
    """A -1 position code is an unfilled roster spot and must never start."""
    vector = VectorLineup(seated_config.roster)
    depth = seated_config.roster.total_picks
    scores = np.full((1, depth), 100.0)
    codes = np.full((1, depth), -1, dtype=np.int64)
    assert float(vector.total(scores, codes)[0]) == 0.0

    codes[0, 0] = POSITIONS.index("QB")
    assert float(vector.total(scores, codes)[0]) == pytest.approx(100.0)


def test_third_qb_adds_nothing_in_a_two_qb_league(seated_config):
    """Position need, derived. No flag had to say so."""
    roster = seated_config.roster
    positions = {"q1": "QB", "q2": "QB"}
    scores = {"q1": 300.0, "q2": 280.0}
    before = best_lineup_points(scores, positions, roster)
    positions["q3"], scores["q3"] = "QB", 270.0
    after = best_lineup_points(scores, positions, roster)
    assert after == pytest.approx(before)


def test_lineup_slots_never_starts_a_player_twice(seated_config):
    positions = {"a": "RB", "b": "WR", "c": "QB"}
    scores = {"a": 200.0, "b": 190.0, "c": 250.0}
    filled = lineup_slots(scores, positions, seated_config.roster)
    used = [p for p in filled.values() if p is not None]
    assert len(used) == len(set(used))


def test_marginal_reduces_to_vor_on_an_empty_roster(seated_config, sample_board):
    """The bug this guards: marginal silently degenerating into raw points.

    On an empty roster every player improves the lineup by exactly his own
    projection, so an unadjusted "lineup improvement" measure IS raw projected
    points - the cross-position bias VOR exists to remove. In the superflex
    league that put four quarterbacks in the top six of the shortlist while
    running backs with higher VOR sat below them.
    """
    import numpy as np

    from ffdraft.vor import replacement_points, roster_marginal, vor_array

    available = np.ones(len(sample_board), dtype=bool)
    vor = vor_array(sample_board, available, seated_config.vor_baseline)
    replacement = replacement_points(sample_board, available, seated_config.vor_baseline)

    for i in range(0, len(sample_board), 17):
        pid = sample_board.players[i].player_id
        marginal = roster_marginal(sample_board, [], pid, seated_config, replacement)
        assert marginal == pytest.approx(float(vor[i]), abs=1e-6), pid


def test_a_quarterback_does_not_outrank_a_higher_vor_back(seated_config, sample_board):
    """The symptom, stated directly: superflex is not a licence to stack QBs."""
    import numpy as np

    from ffdraft.vor import candidate_shortlist

    available = np.ones(len(sample_board), dtype=bool)
    shortlist = candidate_shortlist(sample_board, available, seated_config, [], 6)
    positions = [sample_board.pos_of(i) for i in shortlist]
    assert positions.count("QB") <= 2, positions
    assert positions[0] in ("RB", "WR"), positions


def test_the_superflex_second_quarterback_still_counts(seated_config, sample_board):
    """...but the second one does fill a slot, and must not be zeroed.

    The QBs are taken in scoring order deliberately. Held out of order, a
    "third" QB who outscores the second is a genuine upgrade to the superflex
    slot and correctly scores above zero - which is right, and is not what this
    test is about.
    """
    import numpy as np

    from ffdraft.vor import replacement_points, roster_marginal

    available = np.ones(len(sample_board), dtype=bool)
    replacement = replacement_points(sample_board, available, seated_config.vor_baseline)
    qbs = sorted(
        (p.player_id for p in sample_board.players if p.pos == "QB"),
        key=lambda pid: -sample_board.points[sample_board.idx(pid)],
    )[:3]

    first = roster_marginal(sample_board, [], qbs[0], seated_config, replacement)
    second = roster_marginal(sample_board, [qbs[0]], qbs[1], seated_config, replacement)
    third = roster_marginal(sample_board, qbs[:2], qbs[2], seated_config, replacement)

    assert first > 0 and second > 0, "superflex starts two QBs"
    assert first > second, "the second QB fills a flex, not the QB slot"
    assert third == pytest.approx(0.0), "a worse third QB upgrades nothing"
