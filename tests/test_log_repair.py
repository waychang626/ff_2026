"""Repairing a log that drifted out of step with the room.

`fix` handles the wrong player at the right pick. These cover the other two
failures, which `fix` and `undo` between them cannot repair: a pick that never
got entered, and one that got entered twice. Both shift every later pick onto
the wrong seat, because the seat that owns a pick is derived from its number.
"""
from __future__ import annotations

import pytest

from ffdraft.cli import main
from ffdraft.draft import DraftState, DraftStateError, pick_owner
from ffdraft.replay import DraftLog

SAMPLES = ["--projections", "data/samples/projections_synthetic.csv",
           "--market", "data/samples/market_synthetic.csv"]


@pytest.fixture
def state(cuomo_config):
    return DraftState(
        config=cuomo_config,
        drafted=[f"p{i}|RB" for i in range(1, 7)],
        my_seat=3,
    )


# --- insert ------------------------------------------------------------------
def test_insert_shifts_the_tail_down_one(state):
    after = state.insert(3, "missed|WR")
    assert after.drafted[2] == "missed|WR"
    assert after.drafted[3:] == ["p3|RB", "p4|RB", "p5|RB", "p6|RB"]
    assert len(after.drafted) == len(state.drafted) + 1


def test_insert_re_attributes_every_later_pick(state, cuomo_config):
    """The reason this exists. Seat ownership is positional."""
    before = {n: pick_owner(n, cuomo_config.teams, cuomo_config.draft_type)
              for n in range(1, len(state.drafted) + 1)}
    after = state.insert(1, "missed|WR")
    for n, pid in enumerate(after.drafted, start=1):
        if pid == "missed|WR":
            continue
        was = int(pid[1]) if pid.startswith("p") else None
        if was:
            assert n == was + 1
    assert before[1] == pick_owner(1, cuomo_config.teams, cuomo_config.draft_type)


def test_insert_at_the_end_is_the_same_as_recording(state):
    end = len(state.drafted) + 1
    assert state.insert(end, "late|TE").drafted == state.drafted + ["late|TE"]


def test_insert_refuses_a_duplicate(state):
    with pytest.raises(DraftStateError, match="already recorded at pick 2"):
        state.insert(4, "p2|RB")


def test_insert_refuses_a_slot_off_the_end(state):
    with pytest.raises(DraftStateError, match="not a place to insert"):
        state.insert(99, "missed|WR")
    with pytest.raises(DraftStateError, match="not a place to insert"):
        state.insert(0, "missed|WR")


def test_insert_refuses_to_push_a_pick_past_the_end_of_the_draft(cuomo_config):
    full = DraftState(
        config=cuomo_config,
        drafted=[f"p{i}|RB" for i in range(cuomo_config.total_drafted)],
        my_seat=3,
    )
    with pytest.raises(DraftStateError, match="push a pick off the end"):
        full.insert(1, "missed|WR")


# --- drop --------------------------------------------------------------------
def test_drop_shifts_the_tail_up_one(state):
    after = state.remove(2)
    assert after.drafted == ["p1|RB", "p3|RB", "p4|RB", "p5|RB", "p6|RB"]


def test_drop_is_the_inverse_of_insert(state):
    assert state.insert(3, "missed|WR").remove(3).drafted == state.drafted


def test_drop_refuses_a_pick_that_is_not_in_the_log(state):
    with pytest.raises(DraftStateError, match="not in the log"):
        state.remove(99)


def test_a_repaired_state_still_validates(state):
    state.insert(3, "missed|WR").validate()
    state.remove(3).validate()


# --- through the console -----------------------------------------------------
def _run(monkeypatch, capsys, tmp_path, lines, seat=6):
    fed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda *_: next(fed))
    main(["draft", "--league", "cuomo", "--seat", str(seat), "--sims", "40",
          "--no-suggest", *SAMPLES, "--out", str(tmp_path / "log.jsonl")])
    return capsys.readouterr().out, DraftLog.load(tmp_path / "log.jsonl")


THREE = ["troy thomas", "roman harrison", "marvin daniels"]


def test_the_console_inserts_and_renumbers_the_saved_log(
    monkeypatch, capsys, tmp_path, cuomo_config
):
    out, log = _run(monkeypatch, capsys, tmp_path,
                    THREE + ["insert 2 marcus wilson", "quit"])
    assert "inserted at pick 2" in out
    assert [p.pick for p in log.picks] == [1, 2, 3, 4]
    assert [p.seat for p in log.picks] == [
        pick_owner(n, cuomo_config.teams, cuomo_config.draft_type) for n in (1, 2, 3, 4)
    ]
    assert log.picks[1].player_id == "marcus-wilson|RB"
    assert log.picks[2].player_id == "roman-harrison|RB"
    assert "inserted" in log.picks[1].note


def test_the_console_drops_and_renumbers_the_saved_log(monkeypatch, capsys, tmp_path):
    out, log = _run(monkeypatch, capsys, tmp_path, THREE + ["drop 2", "quit"])
    assert "removed pick 2" in out
    assert [p.pick for p in log.picks] == [1, 2]
    assert [p.player_id for p in log.picks] == ["troy-thomas|RB", "marvin-daniels|WR"]


def test_insert_then_drop_leaves_the_log_as_it_was(monkeypatch, capsys, tmp_path):
    _, log = _run(monkeypatch, capsys, tmp_path,
                  THREE + ["insert 2 marcus wilson", "drop 2", "quit"])
    assert [p.player_id for p in log.picks] == [
        "troy-thomas|RB", "roman-harrison|RB", "marvin-daniels|WR"
    ]
    assert [p.pick for p in log.picks] == [1, 2, 3]


def test_a_bad_insert_leaves_the_log_untouched(monkeypatch, capsys, tmp_path):
    out, log = _run(monkeypatch, capsys, tmp_path,
                    THREE + ["insert 2 troy thomas", "quit"])
    assert "already recorded at pick 1" in out
    assert len(log.picks) == 3


def test_insert_needs_a_name(monkeypatch, capsys, tmp_path):
    out, _ = _run(monkeypatch, capsys, tmp_path, THREE + ["insert 2", "quit"])
    assert "usage: insert" in out


def test_insert_refuses_an_ambiguous_name(monkeypatch, capsys, tmp_path, sample_board):
    first = {}
    for player in sample_board.players:
        first.setdefault(player.name.split()[0], []).append(player)
    shared = next(k for k, v in first.items() if len(v) > 1)
    out, log = _run(monkeypatch, capsys, tmp_path, THREE + [f"insert 2 {shared}", "quit"])
    assert "matches multiple players" in out
    assert len(log.picks) == 3


def test_drop_needs_a_number(monkeypatch, capsys, tmp_path):
    out, _ = _run(monkeypatch, capsys, tmp_path, THREE + ["drop", "quit"])
    assert "usage: drop" in out


def test_both_commands_are_in_the_help(monkeypatch, capsys, tmp_path):
    out, _ = _run(monkeypatch, capsys, tmp_path, ["help", "quit"])
    assert "insert <n> <name>" in out and "drop <n>" in out
