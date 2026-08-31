"""The replay harness - build order #1, and the determinism proof."""
from __future__ import annotations

import pytest

from ffdraft.replay import (
    DraftLog,
    assert_deterministic,
    backtest,
    load_actuals,
    replay,
    synthetic_log,
)


@pytest.fixture(scope="module")
def mock_draft(seated_config, sample_board):
    return synthetic_log(seated_config, sample_board, seed=11, seat=3)


def test_a_mock_draft_fills_every_pick_exactly_once(seated_config, mock_draft):
    assert len(mock_draft.picks) == seated_config.total_drafted
    ids = mock_draft.player_ids
    assert len(set(ids)) == len(ids)
    assert [p.pick for p in mock_draft.picks] == list(range(1, len(ids) + 1))


def test_draft_log_round_trips(tmp_path, mock_draft):
    path = tmp_path / "draft.jsonl"
    mock_draft.save(path)
    loaded = DraftLog.load(path)
    assert loaded.league_id == mock_draft.league_id
    assert loaded.my_seat == mock_draft.my_seat
    assert loaded.player_ids == mock_draft.player_ids


def test_a_hand_edited_log_is_refused(tmp_path, mock_draft):
    path = tmp_path / "bad.jsonl"
    mock_draft.save(path)
    lines = path.read_text().splitlines()
    del lines[3]                       # drop a pick, leaving a gap in the numbering
    path.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="must be 1..n in order"):
        DraftLog.load(path)


def test_replay_evaluates_every_pick_the_seat_owned(seated_config, sample_board, mock_draft):
    result = replay(seated_config, sample_board, mock_draft, seat=3, limit=4)
    assert result.picks_evaluated == 4
    assert [a.pick_number for a in result.advices] == [3, 14, 19, 30]
    for advice in result.advices:
        assert advice.recommendations


def test_replaying_the_same_log_twice_gives_identical_recommendations(
    seated_config, sample_board, mock_draft
):
    """The claim the whole design rests on. If this fails, nothing else counts."""
    assert_deterministic(seated_config, sample_board, mock_draft, seat=3, runs=2, limit=5)


def test_backtest_drafts_a_full_roster_and_scores_it(
    seated_config, sample_board, mock_draft
):
    actuals = {
        p.player_id: float(sample_board.points[i])
        for i, p in enumerate(sample_board.players)
    }
    result = backtest(seated_config, sample_board, mock_draft, actuals, seat=3)
    assert len(result.engine_roster) == seated_config.rounds
    assert len(result.actual_roster) == seated_config.rounds
    assert len(set(result.engine_roster)) == len(result.engine_roster)
    assert result.engine_points > 0
    assert not result.missing_actuals
    # The engine fills a legal lineup: every mandatory slot has a body in it.
    for slot in ("K", "DEF"):
        assert result.engine_lineup.get(slot) is not None


def test_the_re_simulated_draft_never_repeats_a_player(
    seated_config, sample_board, mock_draft
):
    """Opponents keep their recorded picks unless the engine took one first.

    When that happens they fall through to their next available recorded pick,
    so the room still behaves as it did - but nobody may be drafted twice.
    """
    actuals = {p.player_id: 1.0 for p in sample_board.players}
    result = backtest(seated_config, sample_board, mock_draft, actuals, seat=3)
    assert len(result.drafted) == len(set(result.drafted))
    assert set(result.engine_roster) <= set(result.drafted)
    # The engine drafted for seat 3 and nobody else.
    from ffdraft.draft import pick_owner

    for n, pid in enumerate(result.drafted, start=1):
        owner = pick_owner(n, seated_config.teams, seated_config.draft_type)
        assert (pid in result.engine_roster) == (owner == 3)


def test_load_actuals_accepts_names_or_ids(tmp_path):
    path = tmp_path / "actuals.csv"
    path.write_text("player,pos,points\nBijan Robinson,RB,301.4\n")
    assert load_actuals(path) == {"bijan-robinson|RB": 301.4}

    path2 = tmp_path / "actuals2.csv"
    path2.write_text("player_id,points\nbijan-robinson|RB,301.4\n")
    assert load_actuals(path2) == {"bijan-robinson|RB": 301.4}
