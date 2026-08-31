"""Snake order, and the consistency checks behind LLM job #2."""
from __future__ import annotations

import pytest

from ffdraft.draft import (
    DraftState,
    DraftStateError,
    next_pick_for,
    pick_owner,
    round_of,
    seat_picks,
    unfilled_mandatory_slots,
)


def test_snake_order_reverses_every_round():
    owners = [pick_owner(n, 8) for n in range(1, 25)]
    assert owners[:8] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert owners[8:16] == [8, 7, 6, 5, 4, 3, 2, 1]
    assert owners[16:24] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_linear_order_does_not_reverse():
    assert [pick_owner(n, 4, "linear") for n in range(1, 9)] == [1, 2, 3, 4, 1, 2, 3, 4]


def test_every_seat_gets_exactly_one_pick_per_round():
    for seat in range(1, 9):
        picks = seat_picks(seat, 8, 17)
        assert len(picks) == 17
        assert sorted({round_of(p, 8) for p in picks}) == list(range(1, 18))


def test_turn_ends_of_the_snake_pick_back_to_back():
    """Seat 8 in an 8-team draft picks at 8 and 9; that is the whole point."""
    assert next_pick_for(8, 8, 8, 17) == 9
    assert next_pick_for(1, 1, 8, 17) == 16


def test_wheel_seat_has_the_shortest_wait_and_the_turn_has_the_longest(cuomo_config):
    state = DraftState(config=cuomo_config, drafted=[], my_seat=1)
    assert state.my_next_pick() == 1
    assert state.picks_until_my_next() == 14   # picks 2..15 happen before pick 16


def test_duplicate_pick_is_refused(cuomo_config):
    state = DraftState(config=cuomo_config, drafted=["a|RB"])
    with pytest.raises(DraftStateError, match="already drafted"):
        state.record("a|RB")


def test_pick_count_mismatch_is_refused_with_a_direction(cuomo_config):
    state = DraftState(config=cuomo_config, drafted=["a|RB", "b|WR"])
    assert state.pick_number == 3
    with pytest.raises(DraftStateError, match="missing pick"):
        state.cross_check(5)
    with pytest.raises(DraftStateError, match="undo the extra"):
        state.cross_check(2)
    state.cross_check(3)


def test_overlong_draft_is_refused(cuomo_config):
    with pytest.raises(DraftStateError, match="only"):
        DraftState(config=cuomo_config, drafted=[f"p{i}|RB" for i in range(200)])


def test_roster_of_seat_follows_the_snake(cuomo_config):
    picks = [f"p{i}|RB" for i in range(1, 25)]
    state = DraftState(config=cuomo_config, drafted=picks, my_seat=8)
    # Seat 8 owns overall picks 8, 9, 24.
    assert state.roster_of(8) == ["p8|RB", "p9|RB", "p24|RB"]


def test_my_roster_without_a_seat_is_an_error_not_a_guess(cuomo_config):
    state = DraftState(config=cuomo_config, drafted=[], my_seat=None)
    with pytest.raises(DraftStateError, match="my_seat"):
        _ = state.my_roster


def test_unfilled_mandatory_slots_tracks_what_must_still_be_drafted(cuomo_config):
    empty = unfilled_mandatory_slots([], cuomo_config)
    assert empty == {"QB": 1, "RB": 2, "WR": 3, "K": 1, "DST": 1}

    partial = unfilled_mandatory_slots(["QB", "RB", "WR", "WR", "WR", "TE"], cuomo_config)
    assert partial == {"RB": 1, "K": 1, "DST": 1}

    # Extra bodies at a position do not create new obligations.
    assert "QB" not in unfilled_mandatory_slots(["QB", "QB", "QB"], cuomo_config)
