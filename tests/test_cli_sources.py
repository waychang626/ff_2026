"""`ffdraft sources` - proving the pull actually got what it asked for."""
from __future__ import annotations

import pytest

from ffdraft.cli import EXPECTED_SOURCES, main

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
    # Every source the pull asks for but did not get must be named. Read the
    # list from the module so trimming a dead source cannot silently rot this.
    for absent in EXPECTED_SOURCES:
        if absent != "CBS":
            assert absent in out, absent


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


def test_scale_check_flags_a_source_on_the_wrong_basis(tmp_path, capsys):
    """A per-game source among per-season peers is invisible to coverage checks.

    It shows full coverage at every position and contributes numbers ~17x too
    small, which equal weighting quietly absorbs into every player it touches.
    """
    path = tmp_path / "proj.csv"
    rows = [
        f"{src},Player {i},RB,KC,7,1200,50,400\n"
        for src in ("CBS", "ESPN", "FFToday")
        for i in range(20)
    ]
    # Same players, but per-game numbers.
    rows += [f"RTSports,Player {i},RB,KC,7,70,3,24\n" for i in range(20)]
    _write(path, rows)

    main(["sources", "--league", "cuomo", "--projections", str(path)])
    out = capsys.readouterr().out
    assert "OFF SCALE" in out
    assert "RTSports is on a different scale" in out


def test_scale_check_does_not_punish_a_source_for_covering_only_stars(
    tmp_path, capsys
):
    """The bug this replaced: comparing medians measures depth, not scale.

    A source carrying only the top ten at a position has a median around twice
    that of one carrying a hundred, purely because of who it covers. Comparing
    on shared players cancels that out.
    """
    path = tmp_path / "proj.csv"
    rows = []
    for i in range(100):
        yards = 1600 - i * 12          # a realistic decline down the board
        for src in ("CBS", "ESPN"):
            rows.append(f"{src},Player {i},RB,KC,7,{yards},40,300\n")
    # A shallow source, agreeing exactly, but only on the top ten.
    for i in range(10):
        rows.append(f"FantasyPros,Player {i},RB,KC,7,{1600 - i * 12},40,300\n")
    _write(path, rows)

    main(["sources", "--league", "cuomo", "--projections", str(path)])
    out = capsys.readouterr().out
    assert "OFF SCALE" not in out, out
    assert "all sources agree on scale" in out


def test_scale_check_is_quiet_when_sources_agree(tmp_path, capsys):
    path = tmp_path / "proj.csv"
    rows = [
        f"{src},Player {i},RB,KC,7,{1200 + n * 40},50,400\n"
        for n, src in enumerate(("CBS", "ESPN", "FFToday"))
        for i in range(20)
    ]
    _write(path, rows)

    main(["sources", "--league", "cuomo", "--projections", str(path)])
    out = capsys.readouterr().out
    assert "scale check" in out
    assert "OFF SCALE" not in out
