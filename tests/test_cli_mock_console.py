"""The mock console, and its parity with the live one.

`mock` exists to get the console into your fingers before a clock is running.
That only works if the commands are the same ones. It used to accept a strict
subset - no `log`, no `undo`, no `fix`, no `roster <seat>`, and a bare number
resolved as a player name instead of taking the engine's #2 - so the practice
taught muscle memory that fails on draft day. These pin the two consoles
together.
"""
from __future__ import annotations

import pytest

from ffdraft.audit import AuditLog
from ffdraft.cli import _shared_command, _word, main
from ffdraft.draft import DraftState
from ffdraft.replay import DraftLog

SAMPLES = ["--projections", "data/samples/projections_synthetic.csv",
           "--market", "data/samples/market_synthetic.csv"]

# One representative line per shared verb. Both consoles must accept every one.
SHARED_COMMANDS = [
    "help",
    "save",
    "log",
    "log 5",
    "roster",
    "roster 1",
    "board",
    "board RB",
]


def _mock(monkeypatch, capsys, tmp_path, lines, seat=3, extra=()):
    fed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda *_: next(fed))
    main([
        "mock", "--league", "cuomo", "--seat", str(seat), "--sims", "40",
        "--seed", "7", *SAMPLES, "--out", str(tmp_path / "mock.jsonl"),
        "--audit", str(tmp_path / "audit.jsonl"), *extra,
    ])
    return capsys.readouterr().out


def _draft(monkeypatch, capsys, tmp_path, lines, seat=3, extra=()):
    fed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda *_: next(fed))
    main([
        "draft", "--league", "cuomo", "--seat", str(seat), "--sims", "40",
        *SAMPLES, "--out", str(tmp_path / "draft.jsonl"),
        "--audit", str(tmp_path / "audit.jsonl"), *extra,
    ])
    return capsys.readouterr().out


# --- parity ------------------------------------------------------------------
@pytest.mark.parametrize("command", SHARED_COMMANDS)
def test_the_mock_accepts_every_command_the_live_console_does(
    monkeypatch, capsys, tmp_path, command
):
    out = _mock(monkeypatch, capsys, tmp_path, [command, "quit"])
    assert "! no board match" not in out
    assert "no match" not in out


@pytest.mark.parametrize("command", SHARED_COMMANDS)
def test_the_live_console_accepts_them_too(monkeypatch, capsys, tmp_path, command):
    out = _draft(monkeypatch, capsys, tmp_path, [command, "quit"])
    assert "! no board match" not in out
    assert "no match" not in out


def test_both_consoles_route_through_one_dispatcher(seated_config, sample_board,
                                                    tmp_path, capsys):
    """The parity is structural, not two lists that happen to agree today."""
    state = DraftState(config=seated_config, drafted=[], my_seat=3)
    log = DraftLog(league_id=seated_config.league_id, my_seat=3)
    for command in SHARED_COMMANDS:
        _, _, handled, _ = _shared_command(
            command, state, sample_board, seated_config, log,
            tmp_path / "log.jsonl", AuditLog(),
        )
        assert handled, f"{command!r} was not handled"
    capsys.readouterr()


def test_the_mock_help_lists_the_shared_commands(monkeypatch, capsys, tmp_path):
    out = _mock(monkeypatch, capsys, tmp_path, ["quit"])
    for verb in ("undo", "log", "fix", "roster", "board", "out", "bump", "save"):
        assert verb in out, verb


# --- checking rosters mid-draft ----------------------------------------------
def test_mock_roster_takes_a_seat_like_the_live_console(monkeypatch, capsys, tmp_path):
    """`roster 1` is how you check an opponent. The mock only had `roster`."""
    out = _mock(monkeypatch, capsys, tmp_path, ["roster 1", "quit"])
    assert "seat 1" in out


def test_mock_log_shows_the_opponent_picks_that_scrolled_past(
    monkeypatch, capsys, tmp_path
):
    out = _mock(monkeypatch, capsys, tmp_path, ["log 2", "quit"])
    assert "seat 1" in out and "seat 2" in out


def test_a_seat_that_does_not_exist_is_refused_not_shown_empty(
    monkeypatch, capsys, tmp_path
):
    out = _mock(monkeypatch, capsys, tmp_path, ["roster 99", "quit"])
    assert "does not exist" in out
    assert "seat 99: empty" not in out


def test_a_non_numeric_seat_does_not_crash_the_console(monkeypatch, capsys, tmp_path):
    out = _mock(monkeypatch, capsys, tmp_path, ["roster abc", "quit"])
    assert "not a seat number" in out


# --- picking -----------------------------------------------------------------
def test_a_bare_number_takes_that_ranked_candidate(monkeypatch, capsys, tmp_path):
    """In the live console `2` takes the engine's second choice. Here it used
    to be resolved as a player name and rejected."""
    out = _mock(monkeypatch, capsys, tmp_path, ["2", "quit"])
    second = [ln for ln in out.splitlines() if ln.strip().startswith("2  ")][0]
    name = second.split("2  ", 1)[1].split("  ")[0].strip()
    assert f"-> you take {name}" in out


def test_a_number_past_the_end_of_the_list_is_refused(monkeypatch, capsys, tmp_path):
    out = _mock(monkeypatch, capsys, tmp_path, ["9", "quit"])
    assert "pick a number from 1 to 3" in out


def test_show_controls_how_many_candidates_are_offered(monkeypatch, capsys, tmp_path):
    out = _mock(monkeypatch, capsys, tmp_path, ["quit"], extra=["--show", "5"])
    assert "type 1-5 to take one" in out


# --- state-changing commands replan ------------------------------------------
def _cards(out: str) -> int:
    return sum(1 for line in out.splitlines() if line.startswith("PICK: "))


def test_undo_rewinds_and_re_runs_the_engine(monkeypatch, capsys, tmp_path):
    """The advice on screen was computed against the state that just moved."""
    out = _mock(monkeypatch, capsys, tmp_path, ["undo", "quit"])
    assert "undid pick" in out
    assert _cards(out) == 2, out


def test_ruling_a_player_out_re_runs_the_recommendation(monkeypatch, capsys, tmp_path):
    out = _mock(monkeypatch, capsys, tmp_path,
                ["out troy thomas : ACL", "quit"])
    assert "ruled out for the season" in out
    assert _cards(out) == 2


def test_a_projection_update_reaches_the_audit_log(monkeypatch, capsys, tmp_path):
    _mock(monkeypatch, capsys, tmp_path, ["out troy thomas : ACL", "quit"])
    entries = AuditLog(tmp_path / "audit.jsonl").entries()
    assert any(e.kind == "projection_update" for e in entries)


def test_an_update_without_a_reason_is_refused(monkeypatch, capsys, tmp_path):
    out = _mock(monkeypatch, capsys, tmp_path, ["out troy thomas", "quit"])
    assert "a reason is required" in out


def test_fix_corrects_an_earlier_pick_and_replans(monkeypatch, capsys, tmp_path):
    out = _mock(monkeypatch, capsys, tmp_path, ["fix 1 marcus wilson", "quit"])
    assert "pick 1:" in out and "-> Marcus Wilson" in out
    assert _cards(out) == 2


def test_go_reprints_without_re_running_the_engine(monkeypatch, capsys, tmp_path):
    """`go` is a redraw. Re-simulating to show the same rows would cost a
    full recommendation every time the card scrolled off screen."""
    out = _mock(monkeypatch, capsys, tmp_path, ["go", "quit"])
    assert _cards(out) == 2
    assert out.count("type 1-3 to take one") == 2


# --- the word-boundary fix ---------------------------------------------------
def test_a_command_word_does_not_swallow_a_name_that_starts_with_it():
    """`startswith("log")` turned a pick for Logan into a log listing."""
    for name, word in [
        ("logan thomas", "log"), ("boardman jr", "board"),
        ("rosterman", "roster"), ("outlaw", "out"), ("fixon", "fix"),
    ]:
        assert not _word(name, word), f"{name!r} matched {word!r}"
    assert _word("log", "log")
    assert _word("log 20", "log")
