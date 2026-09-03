"""Weekly lineups, and the freshness rules that gate them.

The failure these exist for is silent: last week's file parses cleanly,
produces a confident lineup, and starts a player who was ruled out on Friday.
Nothing in the output says so. So staleness is tested as a *refusal*, not as a
warning somebody has to notice.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from ffdraft.draft import DraftState
from ffdraft.weekly import (
    StaleDataError,
    WeeklyDataError,
    best_lineup,
    check_freshness,
    format_plan,
    format_sources,
    load_weekly,
    roster_for,
)

NOW = datetime(2026, 9, 14, 16, 0, tzinfo=timezone.utc)
HEADER = ["source", "fetched_at", "week", "player_id", "points", "status"]


def _write(path, rows, header=HEADER):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _stamp(hours_old):
    return (NOW - timedelta(hours=hours_old)).isoformat()


@pytest.fixture
def roster(seated_config, sample_board):
    order = sorted(range(len(sample_board)),
                   key=lambda i: -float(sample_board.points[i]))
    drafted = [sample_board.players[i].player_id
               for i in order[: seated_config.teams * seated_config.rounds]]
    state = DraftState(config=seated_config, drafted=drafted, my_seat=3)
    return drafted, state.roster_of(3)


@pytest.fixture
def weekly_file(tmp_path, sample_board, roster):
    _, mine = roster
    rows = []
    for source in ("CBS", "FantasyPros", "FFToday"):
        for i, player in enumerate(sample_board.players):
            rows.append([source, _stamp(1), 1, player.player_id,
                         f"{10.0 + (i % 7) + len(source) * 0.1:.2f}", "ACTIVE"])
    return _write(tmp_path / "wk1.csv", rows)


# --- aggregation -------------------------------------------------------------
def test_sources_are_averaged_equal_weight(tmp_path):
    path = _write(tmp_path / "w.csv", [
        ["A", _stamp(1), 1, "x|RB", "10", "ACTIVE"],
        ["B", _stamp(1), 1, "x|RB", "20", "ACTIVE"],
        ["C", _stamp(1), 1, "x|RB", "30", "ACTIVE"],
    ])
    data = load_weekly(path, now=NOW)
    assert data.points["x|RB"] == pytest.approx(20.0)
    assert data.n_sources["x|RB"] == 3


def test_disagreement_between_sources_is_kept(tmp_path):
    """The spread is the point of aggregating; a mean alone throws it away."""
    tight = _write(tmp_path / "tight.csv", [
        ["A", _stamp(1), 1, "x|RB", "20", ""], ["B", _stamp(1), 1, "x|RB", "20", ""],
    ])
    wide = _write(tmp_path / "wide.csv", [
        ["A", _stamp(1), 1, "x|RB", "5", ""], ["B", _stamp(1), 1, "x|RB", "35", ""],
    ])
    a, b = load_weekly(tight, now=NOW), load_weekly(wide, now=NOW)
    assert a.points["x|RB"] == b.points["x|RB"] == pytest.approx(20.0)
    assert a.source_sd.get("x|RB", 0.0) < b.source_sd["x|RB"]


def test_a_single_source_file_still_loads(tmp_path):
    path = _write(tmp_path / "w.csv", [["", _stamp(1), 1, "x|RB", "10", ""]])
    data = load_weekly(path, now=NOW)
    assert data.points["x|RB"] == 10.0
    assert data.n_sources["x|RB"] == 1


def test_the_worst_status_across_sources_wins(tmp_path):
    """One site saying ACTIVE does not cancel another saying OUT."""
    path = _write(tmp_path / "w.csv", [
        ["A", _stamp(1), 1, "x|RB", "10", "ACTIVE"],
        ["B", _stamp(1), 1, "x|RB", "10", "OUT"],
    ])
    assert load_weekly(path, now=NOW).is_out("x|RB")


# --- staleness: the refusals -------------------------------------------------
def test_a_stale_source_is_dropped_not_averaged_in(tmp_path):
    path = _write(tmp_path / "w.csv", [
        ["fresh", _stamp(1), 1, "x|RB", "10", ""],
        ["old", _stamp(72), 1, "x|RB", "30", ""],
    ])
    data = load_weekly(path, now=NOW, max_age_hours=24)
    assert data.points["x|RB"] == pytest.approx(10.0)
    assert [m.name for m in data.kept] == ["fresh"]
    assert any(m.dropped for m in data.sources)


def test_every_source_stale_is_refused(tmp_path):
    path = _write(tmp_path / "w.csv", [["A", _stamp(72), 1, "x|RB", "10", ""]])
    with pytest.raises(StaleDataError, match="every source is past"):
        load_weekly(path, now=NOW, max_age_hours=24)


def test_min_sources_is_enforced_after_dropping(tmp_path):
    path = _write(tmp_path / "w.csv", [
        ["fresh", _stamp(1), 1, "x|RB", "10", ""],
        ["old", _stamp(72), 1, "x|RB", "30", ""],
    ])
    with pytest.raises(StaleDataError, match="only 1 fresh source"):
        load_weekly(path, now=NOW, max_age_hours=24, min_sources=2)


def test_the_wrong_week_is_refused_before_anything_else(weekly_file, roster):
    _, mine = roster
    data = load_weekly(weekly_file, now=NOW)
    with pytest.raises(StaleDataError, match="week 1 and you asked for week 2"):
        check_freshness(data, 2, mine, now=NOW)


def test_old_data_is_refused(weekly_file, roster, tmp_path, sample_board):
    _, mine = roster
    rows = [["A", _stamp(20), 1, p, "10", "ACTIVE"] for p in mine]
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW, max_age_hours=30)
    with pytest.raises(StaleDataError, match="20.0h old"):
        check_freshness(data, 1, mine, now=NOW, max_age_hours=12)


def test_a_questionable_starter_tightens_the_bar(tmp_path, roster):
    """The tag resolves 90 minutes before kickoff; 5-hour-old data cannot see it."""
    _, mine = roster
    rows = [["A", _stamp(5), 1, p, "10", "ACTIVE"] for p in mine]
    rows[0][-1] = "QUESTIONABLE"
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW)
    with pytest.raises(StaleDataError, match="questionable or doubtful"):
        check_freshness(data, 1, mine, now=NOW, max_age_hours=24,
                        active_max_age_hours=3)


def test_the_same_data_passes_when_nobody_is_questionable(tmp_path, roster):
    _, mine = roster
    rows = [["A", _stamp(5), 1, p, "10", "ACTIVE"] for p in mine]
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW)
    assert check_freshness(data, 1, mine, now=NOW, max_age_hours=24,
                           active_max_age_hours=3) is not None


def test_allow_stale_overrides_but_says_so(weekly_file, roster):
    _, mine = roster
    data = load_weekly(weekly_file, now=NOW)
    notes = check_freshness(data, 2, mine, now=NOW, allow_stale=True)
    assert any("OVERRIDDEN" in n for n in notes)


def test_a_file_stitched_from_two_pulls_is_refused(tmp_path):
    """One source, two stamps, is a file whose age is not a single number."""
    path = _write(tmp_path / "w.csv", [
        ["A", _stamp(1), 1, "x|RB", "10", ""],
        ["A", _stamp(1), 2, "y|WR", "10", ""],
    ])
    with pytest.raises(WeeklyDataError, match="span several weeks"):
        load_weekly(path, now=NOW)


def test_a_file_with_no_week_and_no_flag_is_refused(tmp_path):
    path = _write(tmp_path / "w.csv", [["A", _stamp(1), "", "x|RB", "10", ""]])
    with pytest.raises(WeeklyDataError, match="no week column"):
        load_weekly(path, now=NOW)


def test_mtime_is_the_fallback_and_is_flagged(tmp_path, roster):
    _, mine = roster
    rows = [["A", "", 1, p, "10", ""] for p in mine]
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=None)
    assert data.from_mtime
    notes = check_freshness(data, 1, mine)
    assert any("modification time" in n for n in notes)


def test_unparseable_points_are_refused_not_zeroed(tmp_path):
    """A projection silently parsed as zero is a benched starter."""
    path = _write(tmp_path / "w.csv", [["A", _stamp(1), 1, "x|RB", "abc", ""]])
    with pytest.raises(WeeklyDataError, match="is not a number"):
        load_weekly(path, now=NOW)


# --- the lineup --------------------------------------------------------------
def test_an_out_player_is_never_started(tmp_path, seated_config, sample_board, roster):
    _, mine = roster
    best = max(mine, key=lambda p: float(sample_board.points[sample_board.idx(p)]))
    rows = [["A", _stamp(1), 1, p, "20" if p == best else "5", ""] for p in mine]
    rows[[r[3] for r in rows].index(best)][-1] = "OUT"
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW)
    plan = best_lineup(seated_config, sample_board, data, mine)
    assert best not in plan.starters
    assert any(pid == best for pid, _ in plan.benched_by_status)


def test_a_player_absent_from_the_file_is_not_started_as_a_zero(
    tmp_path, seated_config, sample_board, roster
):
    _, mine = roster
    rows = [["A", _stamp(1), 1, p, "10", ""] for p in mine[:-3]]
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW)
    plan = best_lineup(seated_config, sample_board, data, mine)
    assert not (set(plan.starters) & set(mine[-3:]))


def test_without_an_opponent_it_maximises_points(
    tmp_path, seated_config, sample_board, roster
):
    _, mine = roster
    rows = [["A", _stamp(1), 1, p,
             f"{float(sample_board.points[sample_board.idx(p)]) / 17:.2f}", ""]
            for p in mine]
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW)
    plan = best_lineup(seated_config, sample_board, data, mine)
    assert plan.p_win is None
    assert plan.expected_points == pytest.approx(
        sum(s.points for s in plan.slots), abs=1e-6
    )


def test_with_an_opponent_it_reports_a_win_probability(
    tmp_path, seated_config, sample_board, roster
):
    drafted, mine = roster
    theirs = roster_for(seated_config, drafted, 5)
    rows = [["A", _stamp(1), 1, p, "10", ""] for p in set(mine) | set(theirs)]
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW)
    plan = best_lineup(seated_config, sample_board, data, mine,
                       opponent_roster=theirs, opponent_seat=5, n_sims=2000)
    assert plan.p_win is not None and 0.0 <= plan.p_win <= 1.0
    assert plan.opponent_seat == 5
    assert plan.swaps_considered > 1


def test_the_lineup_is_deterministic(tmp_path, seated_config, sample_board, roster):
    drafted, mine = roster
    theirs = roster_for(seated_config, drafted, 5)
    rows = [["A", _stamp(1), 1, p, "10", ""] for p in set(mine) | set(theirs)]
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW)
    runs = [
        best_lineup(seated_config, sample_board, data, mine,
                    opponent_roster=theirs, opponent_seat=5, n_sims=1500)
        for _ in range(2)
    ]
    assert runs[0].starters == runs[1].starters
    assert runs[0].p_win == runs[1].p_win


def test_every_starting_slot_is_filled_when_the_roster_allows(
    tmp_path, seated_config, sample_board, roster
):
    _, mine = roster
    rows = [["A", _stamp(1), 1, p, "10", ""] for p in mine]
    data = load_weekly(_write(tmp_path / "w.csv", rows), now=NOW)
    plan = best_lineup(seated_config, sample_board, data, mine)
    assert len(plan.slots) == len(seated_config.roster.starters)
    assert all(s.player_id for s in plan.slots)
    assert len(set(plan.starters)) == len(plan.starters)


def test_the_card_shows_the_pull_time_and_the_source_table(
    weekly_file, seated_config, sample_board, roster
):
    _, mine = roster
    data = load_weekly(weekly_file, now=NOW)
    plan = best_lineup(seated_config, sample_board, data, mine)
    card = format_plan(plan, sample_board, data)
    assert "WEEK 1" in card and "data pulled" in card
    table = format_sources(data, now=NOW)
    assert "CBS" in table and "FantasyPros" in table


# --- stat-line files, straight from the R pull -------------------------------
def test_a_stat_line_file_is_scored_with_the_league_rules(tmp_path, seated_config):
    """`R/pull_projections.R --week N` writes stat lines, not points."""
    path = tmp_path / "stats.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "fetched_at", "week", "player", "pos", "rec",
                    "rec_yds", "rec_tds"])
        w.writerow(["CBS", _stamp(1), 1, "Some Guy", "WR", "5", "80", "1"])
        w.writerow(["FFToday", _stamp(1), 1, "Some Guy", "WR", "7", "100", "1"])
    data = load_weekly(path, now=NOW, rules=seated_config.scoring)
    pid = next(iter(data.points))
    assert data.points[pid] > 0
    assert data.n_sources[pid] == 2
    # Scored per source, then averaged - so the sources disagree on points.
    assert data.source_sd[pid] > 0


def test_a_stat_line_file_without_rules_is_refused(tmp_path):
    path = tmp_path / "stats.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "fetched_at", "week", "player", "pos", "rec_yds"])
        w.writerow(["CBS", _stamp(1), 1, "Some Guy", "WR", "80"])
    with pytest.raises(WeeklyDataError, match="no scoring rules"):
        load_weekly(path, now=NOW)


def test_scoring_happens_before_averaging(tmp_path, seated_config):
    """Averaging stat lines then scoring is a different number for DST, whose
    points-allowed bracket is a step function."""
    path = tmp_path / "dst.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "fetched_at", "week", "player", "pos",
                    "dst_pts_allowed", "dst_sacks"])
        w.writerow(["A", _stamp(1), 1, "Bears", "DST", "0", "3"])
        w.writerow(["B", _stamp(1), 1, "Bears", "DST", "28", "3"])
    data = load_weekly(path, now=NOW, rules=seated_config.scoring)
    pid = next(iter(data.points))
    scored_then_averaged = data.points[pid]

    mid = tmp_path / "mid.csv"
    with mid.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "fetched_at", "week", "player", "pos",
                    "dst_pts_allowed", "dst_sacks"])
        w.writerow(["A", _stamp(1), 1, "Bears", "DST", "14", "3"])
    averaged_then_scored = load_weekly(mid, now=NOW, rules=seated_config.scoring)
    assert scored_then_averaged != pytest.approx(
        averaged_then_scored.points[next(iter(averaged_then_scored.points))]
    )
