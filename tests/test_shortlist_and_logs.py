"""Two draft-day papercuts, both reported from a live practice draft.

The shortlist collapsed to one or two names in the later rounds - the rounds
where you least remember who is left - and every mock draft overwrote the last
one's log.
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from ffdraft.cli import main
from ffdraft.opponents import DraftSimulator

# Absolute: the log tests chdir into a tmp dir to control where logs land.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLES = [
    "--projections", str(_ROOT / "data/samples/projections_synthetic.csv"),
    "--market", str(_ROOT / "data/samples/market_synthetic.csv"),
]


@pytest.fixture
def simulator(seated_config, sample_board):
    return DraftSimulator(sample_board, seated_config, 3)


# --- the shortlist -----------------------------------------------------------
def test_a_dominant_faller_does_not_collapse_the_list(simulator, sample_board):
    """The reported bug.

    `np.unique(argmin)` only returns players who won a simulation. When one
    player's ADP sits far enough ahead of everyone left, he wins all 4,000
    draws and the list that exists to save typing had exactly one row.
    """
    order = np.argsort(sample_board.adp)
    drafted = [int(i) for i in order[1:80]]
    rows = simulator.likely_next_picks(drafted, seat=1, n=10, seed=1)
    assert len(rows) == 10
    assert rows[0][1] == pytest.approx(1.0, abs=0.01)


def test_the_list_is_as_long_as_asked_for_all_through_the_draft(simulator, sample_board):
    order = list(np.argsort(-sample_board.points))
    for taken in range(0, 120, 12):
        rows = simulator.likely_next_picks(
            [int(i) for i in order[:taken]], seat=1, n=10, seed=3
        )
        assert len(rows) == 10, f"{taken} drafted -> {len(rows)} rows"


def test_probabilities_stay_sorted_and_never_exceed_one(simulator, sample_board):
    order = list(np.argsort(-sample_board.points))
    rows = simulator.likely_next_picks([int(i) for i in order[:60]], seat=2, n=10, seed=5)
    probs = [p for _, p in rows]
    assert probs == sorted(probs, reverse=True)
    assert 0.0 <= min(probs) and max(probs) <= 1.0
    assert sum(probs) <= 1.0 + 1e-9


def test_the_backfill_is_in_adp_order(simulator, sample_board):
    """Below the simulated winners, the next names are the ones the model
    would reach for: earliest ADP first."""
    order = np.argsort(sample_board.adp)
    rows = simulator.likely_next_picks(
        [int(i) for i in order[1:80]], seat=1, n=8, seed=1
    )
    zero = [idx for idx, p in rows if p == 0.0]
    adps = [float(sample_board.adp[i]) for i in zero]
    assert adps == sorted(adps)


def test_it_never_offers_a_player_already_drafted(simulator, sample_board):
    order = list(np.argsort(-sample_board.points))
    drafted = [int(i) for i in order[:90]]
    rows = simulator.likely_next_picks(drafted, seat=4, n=10, seed=7)
    assert not ({idx for idx, _ in rows} & set(drafted))


def test_the_list_shrinks_only_when_the_board_does(seated_config, sample_board):
    """With fewer legal players left than asked for, a short list is correct."""
    simulator = DraftSimulator(sample_board, seated_config, 3)
    drafted = list(range(len(sample_board) - 3))
    rows = simulator.likely_next_picks(drafted, seat=1, n=10, seed=2)
    assert len(rows) <= 3


def test_the_console_never_prints_a_bare_zero_percent(monkeypatch, capsys, tmp_path):
    """A name the console is offering, labelled 0%, reads as a broken list."""
    fed = iter(["quit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(fed))
    main(["draft", "--league", "cuomo", "--seat", "8", "--sims", "40",
          *SAMPLES, "--out", str(tmp_path / "l.jsonl")])
    out = capsys.readouterr().out
    listed = [ln for ln in out.splitlines() if re.match(r"^\s+\d+\s{2}\S", ln)]
    assert listed, out
    assert not any(ln.rstrip().endswith(" 0%") for ln in listed), listed


# --- log files ---------------------------------------------------------------
def _mock(monkeypatch, capsys, extra=()):
    # One pick, then stop: `_finish` only reports the log path once the roster
    # has something in it.
    fed = iter(["", "quit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(fed))
    main(["mock", "--league", "cuomo", "--seat", "3", "--sims", "40", "--seed", "1",
          *SAMPLES, *extra])
    return capsys.readouterr().out


def test_each_mock_writes_its_own_log(monkeypatch, capsys, tmp_path):
    """The reported bug: practice runs overwrote each other."""
    monkeypatch.chdir(tmp_path)
    first = _mock(monkeypatch, capsys)
    second = _mock(monkeypatch, capsys)
    paths = [re.search(r"pick log: (\S+)", out).group(1) for out in (first, second)]
    assert paths[0] != paths[1], paths
    assert len(list((tmp_path / "logs").glob("mock_cuomo_*.jsonl"))) == 2


def test_an_explicit_out_path_is_used_verbatim(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = _mock(monkeypatch, capsys, ["--out", "mine.jsonl"])
    assert "pick log: mine.jsonl" in out
    assert (tmp_path / "mine.jsonl").exists()


def test_reopening_the_live_console_does_not_destroy_the_previous_log(
    monkeypatch, capsys, tmp_path
):
    """A pick log is the only record of the room and cannot be rebuilt."""
    monkeypatch.chdir(tmp_path)
    for lines in (["troy thomas", "quit"], ["quit"]):
        fed = iter(lines)
        monkeypatch.setattr("builtins.input", lambda *_: next(fed))
        main(["draft", "--league", "cuomo", "--seat", "8", "--sims", "40",
              "--no-suggest", *SAMPLES])
    out = capsys.readouterr().out
    assert "kept the previous log as" in out
    kept = list((tmp_path / "logs").glob("draft_cuomo.*.jsonl"))
    assert len(kept) == 1
    assert "troy-thomas|RB" in kept[0].read_text()


def test_an_empty_log_is_not_preserved(monkeypatch, capsys, tmp_path):
    """Nothing was lost, so there is nothing to keep and nothing to say."""
    monkeypatch.chdir(tmp_path)
    for _ in range(2):
        fed = iter(["quit"])
        monkeypatch.setattr("builtins.input", lambda *_: next(fed))
        main(["draft", "--league", "cuomo", "--seat", "8", "--sims", "40",
              "--no-suggest", *SAMPLES])
    assert not list((tmp_path / "logs").glob("draft_cuomo.2*.jsonl"))
