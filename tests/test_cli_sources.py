"""`ffdraft sources` - proving the pull actually got what it asked for."""
from __future__ import annotations

import pytest

from ffdraft.cli import main

HEADER = "source,player,pos,team,bye,rush_yds,rec,rec_yds\n"


def _write(path, rows):
    path.write_text(HEADER + "".join(rows))


def test_reports_every_source_and_position(tmp_path, capsys):
    path = tmp_path / "proj.csv"
    rows = [
        f"{src},Player {i},{pos},KC,7,100,10,120\n"
        for src in ("CBS", "ESPN", "FantasyPros")
        for pos in ("QB", "RB", "WR", "TE", "K", "DST")
        for i in range(12)
    ]
    _write(path, rows)

    assert main(["sources", "--projections", str(path)]) == 0
    out = capsys.readouterr().out
    for src in ("CBS", "ESPN", "FantasyPros"):
        assert src in out
    assert "3 sources" in out


def test_names_the_sources_that_returned_nothing(tmp_path, capsys):
    """A source that errors mid-scrape is skipped with a warning and vanishes."""
    path = tmp_path / "proj.csv"
    _write(path, [f"CBS,Player {i},RB,KC,7,100,10,120\n" for i in range(12)])

    main(["sources", "--projections", str(path)])
    out = capsys.readouterr().out
    assert "MISSING" in out
    for absent in ("ESPN", "FantasyPros", "NFL", "RTSports"):
        assert absent in out


def test_flags_a_source_that_is_present_but_thin(tmp_path, capsys):
    """The failure row counts hide: full coverage at QB, almost none at WR.

    The engine equal-weights whatever it finds, so a source covering a tenth of
    a position silently shifts the average there and nowhere else.
    """
    path = tmp_path / "proj.csv"
    rows = [
        f"{src},Player {i},WR,KC,7,0,10,120\n"
        for src in ("CBS", "ESPN")
        for i in range(40)
    ]
    rows += [f"FFToday,Player {i},WR,KC,7,0,10,120\n" for i in range(3)]
    _write(path, rows)

    main(["sources", "--projections", str(path)])
    out = capsys.readouterr().out
    assert "THIN" in out
    assert "FFToday/WR" in out


def test_warns_when_a_position_is_too_shallow_for_its_baseline(tmp_path, capsys):
    """Fewer players than the replacement rank means VOR cannot be computed."""
    path = tmp_path / "proj.csv"
    rows = [
        f"{src},Player {i},RB,KC,7,100,10,120\n"
        for src in ("CBS", "ESPN")
        for i in range(5)          # League 1 needs 23 RBs to reach replacement
    ]
    _write(path, rows)

    main(["sources", "--league", "cuomo", "--projections", str(path)])
    out = capsys.readouterr().out
    assert "LOW" in out
    assert "replacement rank 23" in out


def test_a_missing_file_is_reported_not_traced(tmp_path):
    with pytest.raises(SystemExit, match="no projections at"):
        main(["sources", "--projections", str(tmp_path / "nope.csv")])
