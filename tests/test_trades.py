"""Post-draft trade search.

The two things that must hold are: the fast filter agrees with the slow
optimiser it stands in for, and no idea is ever printed that only helps one
side. Everything else is presentation.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import make_board

from ffdraft.audit import AuditLog
from ffdraft.draft import DraftState
from ffdraft.lineup import VectorLineup, best_lineup_points
from ffdraft.trades import (
    TradeIdea,
    _lineup_totals,
    _pad,
    _trim,
    bye_week_profile,
    find_trades,
    format_report,
    format_surplus,
    positional_surplus,
    starter_demand,
    worst_bye_week,
)


@pytest.fixture(scope="module")
def drafted(seated_config, sample_board):
    """A full, legal draft: snake order down the board by projection."""
    order = sorted(
        range(len(sample_board)),
        key=lambda i: (-float(sample_board.points[i]), sample_board.players[i].player_id),
    )
    total = seated_config.teams * seated_config.rounds
    return [sample_board.players[i].player_id for i in order[:total]]


# --- the correctness anchor --------------------------------------------------
def test_the_fast_filter_agrees_with_the_lineup_optimiser(seated_config, sample_board, drafted):
    """Stage 1 is a vectorised stand-in for `best_lineup_points`.

    If the two ever disagree, every trade in the report is measured against a
    lineup nobody would actually start.
    """
    state = DraftState(config=seated_config, drafted=drafted, my_seat=3)
    vector_lineup = VectorLineup(seated_config.roster)
    for seat in range(1, seated_config.teams + 1):
        roster = state.roster_of(seat)
        idx = [sample_board.idx(p) for p in roster]
        scalar = best_lineup_points(
            {p: float(sample_board.points[sample_board.idx(p)]) for p in roster},
            {p: sample_board.player(p).pos for p in roster},
            seated_config.roster,
        )
        vector = float(_lineup_totals(vector_lineup, sample_board, _pad([idx], len(idx) + 2))[0])
        assert vector == pytest.approx(scalar, abs=1e-6), f"seat {seat}"


def test_padding_an_empty_slot_never_fills_a_lineup(seated_config, sample_board):
    """-1 means nobody. It must not be gathered as player 0."""
    vector_lineup = VectorLineup(seated_config.roster)
    empty = float(_lineup_totals(vector_lineup, sample_board, _pad([[]], 6))[0])
    assert empty == 0.0


# --- the constraint that makes it a trade ------------------------------------
def test_every_idea_helps_both_sides(seated_config, sample_board, drafted):
    report = find_trades(seated_config, sample_board, drafted, 3, top=6)
    assert report.ideas, "expected at least one idea on this board"
    for idea in report.ideas:
        assert idea.my_lineup_gain > 0, idea
        assert idea.their_lineup_gain > 0, idea


def test_raising_the_bar_for_the_other_side_only_removes_ideas(
    seated_config, sample_board, drafted
):
    loose = find_trades(seated_config, sample_board, drafted, 3, min_their_gain=1.0, top=20)
    strict = find_trades(seated_config, sample_board, drafted, 3, min_their_gain=40.0, top=20)
    assert len(strict.ideas) <= len(loose.ideas)
    for idea in strict.ideas:
        assert idea.their_lineup_gain >= 40.0


def test_it_only_trades_players_the_two_sides_actually_own(
    seated_config, sample_board, drafted
):
    state = DraftState(config=seated_config, drafted=drafted, my_seat=3)
    mine = set(state.roster_of(3))
    report = find_trades(seated_config, sample_board, drafted, 3, top=8)
    for idea in report.ideas:
        assert set(idea.give) <= mine, idea.give
        assert set(idea.get) <= set(state.roster_of(idea.partner_seat)), idea.get
        assert not set(idea.give) & set(idea.get)


def test_a_partner_is_never_yourself(seated_config, sample_board, drafted):
    report = find_trades(seated_config, sample_board, drafted, 3, top=8)
    assert all(idea.partner_seat != 3 for idea in report.ideas)


# --- bye weeks ---------------------------------------------------------------
def _board_with_byes(byes: list[int]):
    specs = [(f"P{i}", pos, pts, 30.0, float(i + 1)) for i, (pos, pts) in enumerate([
        ("QB", 300.0), ("RB", 250.0), ("RB", 240.0), ("WR", 230.0), ("WR", 220.0),
        ("WR", 210.0), ("TE", 150.0), ("K", 120.0), ("DST", 110.0), ("RB", 100.0),
        ("WR", 90.0),
    ])]
    board = make_board(specs)
    return dataclasses.replace(board, bye=np.array(byes, dtype=np.int64))


def test_the_bye_profile_finds_the_week_the_roster_is_thinnest(seated_config):
    """Three starting receivers off in the same week is the classic own goal."""
    byes = [0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0]
    board = _board_with_byes(byes)
    vector_lineup = VectorLineup(seated_config.roster)
    profile = bye_week_profile(
        seated_config, board, vector_lineup, list(range(len(board)))
    )
    week, drop = worst_bye_week(profile)
    assert week == 7
    assert drop > 0
    assert profile[6] == profile.min()


def test_a_roster_with_no_byes_has_a_flat_profile(seated_config):
    board = _board_with_byes([0] * 11)
    profile = bye_week_profile(
        seated_config, board, VectorLineup(seated_config.roster), list(range(len(board)))
    )
    assert np.allclose(profile, profile[0])
    assert worst_bye_week(profile)[1] == pytest.approx(0.0)


def test_a_bye_outside_the_fantasy_season_is_ignored(seated_config):
    """`draw_season` only zeroes byes inside the simulated weeks; so does this."""
    weeks = seated_config.regular_season_weeks + len(seated_config.playoff_weeks)
    board = _board_with_byes([weeks + 5] * 11)
    profile = bye_week_profile(
        seated_config, board, VectorLineup(seated_config.roster), list(range(len(board)))
    )
    assert np.allclose(profile, profile[0])


def test_ideas_report_the_bye_week_they_leave_you_with(
    seated_config, sample_board, drafted
):
    report = find_trades(seated_config, sample_board, drafted, 3, top=5)
    weeks = seated_config.regular_season_weeks + len(seated_config.playoff_weeks)
    for idea in report.ideas:
        assert 1 <= idea.bye_week <= weeks
        assert idea.bye_drop_after >= 0


# --- roster limits -----------------------------------------------------------
def test_an_uneven_trade_forces_a_cut(sample_board):
    """Receiving two for one is not a free extra player."""
    idx = list(range(5))
    kept, dropped = _trim(idx, sample_board, 3)
    assert len(kept) == 3 and len(dropped) == 2
    assert set(kept) | set(dropped) == set(idx)
    worst = min(idx, key=lambda i: float(sample_board.points[i]))
    assert worst in dropped


def test_trimming_leaves_a_legal_roster_alone(sample_board):
    kept, dropped = _trim([0, 1, 2], sample_board, 5)
    assert kept == [0, 1, 2] and dropped == []


def test_no_idea_leaves_either_roster_over_the_limit(
    seated_config, sample_board, drafted
):
    state = DraftState(config=seated_config, drafted=drafted, my_seat=3)
    limit = seated_config.roster.total_picks
    report = find_trades(seated_config, sample_board, drafted, 3, top=8)
    for idea in report.ideas:
        after = (len(state.roster_of(3)) - len(idea.give) + len(idea.get)
                 - len(idea.my_drops))
        assert after <= limit, idea


# --- determinism -------------------------------------------------------------
def test_the_same_draft_gives_the_same_report_twice(
    seated_config, sample_board, drafted
):
    """Same guarantee `ffdraft replay --check` holds the pick engine to."""
    a = find_trades(seated_config, sample_board, drafted, 3, top=5)
    b = find_trades(seated_config, sample_board, drafted, 3, top=5)
    assert a.state_hash == b.state_hash and a.seed == b.seed
    assert [(i.partner_seat, i.give, i.get) for i in a.ideas] == \
           [(i.partner_seat, i.give, i.get) for i in b.ideas]
    assert [i.my_delta_p_title for i in a.ideas] == [i.my_delta_p_title for i in b.ideas]


# --- scoping and refusals ----------------------------------------------------
def test_partner_restricts_the_search_to_one_seat(seated_config, sample_board, drafted):
    report = find_trades(seated_config, sample_board, drafted, 3, partners=[5], top=8)
    assert all(idea.partner_seat == 5 for idea in report.ideas)


def test_a_seat_that_does_not_exist_is_refused(seated_config, sample_board, drafted):
    with pytest.raises(ValueError, match="does not exist"):
        find_trades(seated_config, sample_board, drafted, 99)


def test_a_player_not_on_the_board_is_refused(seated_config, sample_board, drafted):
    with pytest.raises(ValueError, match="not on the board"):
        find_trades(seated_config, sample_board, drafted + ["ghost|RB"], 3)


def test_a_seat_with_no_players_is_refused(seated_config, sample_board):
    with pytest.raises(ValueError, match="no players"):
        find_trades(seated_config, sample_board, [], 3)


# --- presentation ------------------------------------------------------------
def test_the_report_does_not_show_one_idea_three_times(
    seated_config, sample_board, drafted
):
    """Variants that send the same package are one idea, not several."""
    report = find_trades(seated_config, sample_board, drafted, 3, top=10)
    keys = [(i.partner_seat, i.give) for i in report.ideas]
    assert len(keys) == len(set(keys))


def test_the_shortlist_reaches_more_than_one_partner(
    seated_config, sample_board, drafted
):
    """One mirror-image opponent used to crowd every other team out."""
    report = find_trades(seated_config, sample_board, drafted, 3, top=8)
    assert len({idea.partner_seat for idea in report.ideas}) > 1


def test_the_card_names_both_sides_of_the_deal(seated_config, sample_board, drafted):
    report = find_trades(seated_config, sample_board, drafted, 3, top=2)
    text = format_report(report, sample_board, seated_config)
    assert "SEND" in text and "GET" in text
    assert "YOU" in text and "THEM" in text
    assert "BYE" in text and "SELL" in text
    for idea in report.ideas:
        assert sample_board.player(idea.give[0]).display in text


def test_the_surplus_table_marks_your_seat(seated_config, sample_board, drafted):
    report = find_trades(seated_config, sample_board, drafted, 3, top=1)
    text = format_surplus(report, seated_config)
    assert "<- you" in text
    assert text.count("<- you") == 1


def test_surplus_counts_bodies_against_starting_slots(seated_config, sample_board):
    demand = starter_demand(seated_config)
    assert demand["QB"] > 0 and demand["RB"] > 0
    surplus = positional_surplus(seated_config, sample_board, [])
    assert all(v <= 0 for v in surplus.values())


def test_the_search_is_written_to_the_audit_log(
    seated_config, sample_board, drafted, tmp_path
):
    audit = AuditLog(tmp_path / "audit.jsonl")
    find_trades(seated_config, sample_board, drafted, 3, top=3, audit=audit)
    entries = AuditLog(tmp_path / "audit.jsonl").entries()
    assert any(e.kind == "find_trades" for e in entries)


def test_market_gap_says_which_way_the_names_are_moving():
    idea = TradeIdea(
        partner_seat=2, give=("a|RB",), get=("b|WR",),
        my_lineup_gain=10.0, their_lineup_gain=5.0, my_p_title=0.2,
        my_delta_p_title=0.01, delta_se=0.001, adp_sent=20.0, adp_received=60.0,
    )
    assert idea.market_gap == 40.0
    assert idea.joint_gain == 15.0


# --- the command line --------------------------------------------------------
SAMPLES = ["--projections", "data/samples/projections_synthetic.csv",
           "--market", "data/samples/market_synthetic.csv"]


@pytest.fixture
def log_path(seated_config, sample_board, drafted, tmp_path):
    from ffdraft.draft import pick_owner
    from ffdraft.replay import DraftLog

    log = DraftLog(league_id=seated_config.league_id, my_seat=3)
    for n, pid in enumerate(drafted, start=1):
        log.append(pid, seat=pick_owner(n, seated_config.teams, seated_config.draft_type))
    path = tmp_path / "draft.jsonl"
    log.save(path)
    return path


def _cli(capsys, log_path, extra=()):
    from ffdraft.cli import main

    main(["trades", "--league", "cuomo", "--seat", "3", "--sims", "60",
          *SAMPLES, "--log", str(log_path), *extra])
    return capsys.readouterr().out


def test_the_command_prints_ideas(capsys, log_path):
    out = _cli(capsys, log_path, ["--top", "2"])
    assert "SEND" in out and "GET" in out
    assert "P(title)" in out


def test_partner_narrows_the_command_to_one_seat(capsys, log_path):
    out = _cli(capsys, log_path, ["--partner", "5", "--top", "3"])
    partners = {line.split("seat ")[1].strip()
                for line in out.splitlines() if line.startswith(("1. WITH", "2. WITH", "3. WITH"))}
    assert partners <= {"5"}


def test_surplus_is_opt_in(capsys, log_path):
    assert "<- you" not in _cli(capsys, log_path, ["--top", "1"])
    assert "<- you" in _cli(capsys, log_path, ["--top", "1", "--surplus"])


def test_the_seat_comes_from_the_log_header_when_not_given(capsys, log_path):
    """Unlike `replay`, this reads `my_seat` out of the log it was handed."""
    from ffdraft.cli import main

    main(["trades", "--league", "cuomo", "--sims", "60", *SAMPLES,
          "--log", str(log_path), "--top", "1"])
    assert "seat 3" in capsys.readouterr().out
