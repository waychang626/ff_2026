"""The live console's control flow.

Driven by feeding stdin, because the bug these cover was in *when* the console
decides to do something, not in what the engine computes.
"""
from __future__ import annotations

import pytest

from ffdraft.cli import main

SAMPLES = ["--projections", "data/samples/projections_synthetic.csv",
           "--market", "data/samples/market_synthetic.csv"]


def _run(monkeypatch, capsys, tmp_path, lines, seat, extra=()):
    fed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda *_: next(fed))
    main([
        "draft", "--league", "cuomo", "--seat", str(seat), "--sims", "60",
        *SAMPLES, "--out", str(tmp_path / "log.jsonl"),
        "--audit", str(tmp_path / "audit.jsonl"), *extra,
    ])
    return capsys.readouterr().out


def _cards(out: str) -> int:
    return sum(1 for line in out.splitlines() if line.startswith("PICK: "))


def test_the_first_seat_gets_a_recommendation_without_asking(
    monkeypatch, capsys, tmp_path
):
    """The reported bug: seat 1 opened the console and got nothing.

    Recommending only after a pick was recorded meant that when it was already
    your turn at startup - which is exactly the case for whoever picks 1.01 -
    nothing ever fired.
    """
    out = _run(monkeypatch, capsys, tmp_path, ["quit"], seat=1)
    assert _cards(out) == 1, out


def test_a_later_seat_gets_one_when_the_clock_reaches_them(
    monkeypatch, capsys, tmp_path
):
    out = _run(monkeypatch, capsys, tmp_path,
               ["marvin daniels", "quit"], seat=2)
    assert _cards(out) == 1


def test_it_does_not_recommend_on_someone_elses_pick(monkeypatch, capsys, tmp_path):
    out = _run(monkeypatch, capsys, tmp_path,
               ["marvin daniels", "troy thomas", "quit"], seat=8)
    assert _cards(out) == 0


def test_it_does_not_repeat_itself_on_every_command(monkeypatch, capsys, tmp_path):
    """`roster` and `board` must not re-run a 15-second simulation."""
    out = _run(monkeypatch, capsys, tmp_path,
               ["roster", "board", "board RB", "quit"], seat=1)
    assert _cards(out) == 1, "state did not change; should have recommended once"


def test_undo_re_issues_the_recommendation(monkeypatch, capsys, tmp_path):
    """Undoing your own pick lands on an identical state - show it anyway.

    The recommendation is not news, but the user just rewound deliberately and
    is looking at the console to find out where they now are.
    """
    out = _run(monkeypatch, capsys, tmp_path,
               ["me troy thomas", "undo", "quit"], seat=1)
    assert _cards(out) == 2


def test_ruling_a_player_out_re_runs_the_recommendation(monkeypatch, capsys, tmp_path):
    """A projection update changes the board, so the old answer is stale."""
    out = _run(monkeypatch, capsys, tmp_path,
               ["out troy thomas : ruled out, ACL", "quit"], seat=1)
    assert _cards(out) == 2
    assert "applied:" in out
    # The player just ruled out must not be the new recommendation.
    cards = [l for l in out.splitlines() if l.startswith("PICK: ")]
    assert "Troy Thomas" in cards[0]
    assert "Troy Thomas" not in cards[1]


def test_a_number_from_the_suggestion_list_records_that_player(
    monkeypatch, capsys, tmp_path
):
    out = _run(monkeypatch, capsys, tmp_path, ["1", "quit"], seat=8)
    assert "likely for seat 1" in out
    assert "1. seat 1:" in out


def test_no_suggest_turns_the_list_off(monkeypatch, capsys, tmp_path):
    out = _run(monkeypatch, capsys, tmp_path, ["quit"], seat=8,
               extra=["--no-suggest"])
    assert "likely for seat" not in out
