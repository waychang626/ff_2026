"""`ffdraft rules` - the config read back in a form you can check."""
from __future__ import annotations

import pytest

from ffdraft.cli import main


@pytest.fixture
def cuomo(capsys):
    main(["rules", "--league", "cuomo"])
    return capsys.readouterr().out


def test_prints_the_shape_and_the_starting_lineup(cuomo):
    assert "8 teams" in cuomo
    assert "17 rounds, 136 picks" in cuomo
    assert "QB WR WR WR RB RB TE WRT QWRT K DEF" in cuomo
    assert "4 of 8" in cuomo


def test_yardage_rules_read_like_a_settings_page(cuomo):
    """0.04 is not a number anyone recognises; '1 pt per 25 yds' is."""
    assert "1 pt per 25 yds" in cuomo      # passing
    assert "1 pt per 10 yds" in cuomo      # rushing and receiving


def test_half_ppr_and_the_four_point_passing_td_are_visible(cuomo):
    assert "reception" in cuomo and "+0.5" in cuomo
    assert "passing TD" in cuomo and "+4 pts" in cuomo


def test_the_points_allowed_bracket_is_shown_as_ranges(cuomo):
    assert "POINTS ALLOWED" in cuomo
    assert "1-6" in cuomo
    assert "7-13" in cuomo


def test_known_compromises_are_surfaced(cuomo):
    """League 1 cannot express 'only penalise misses under 20 yards'."""
    assert "missed FG" in cuomo
    assert "Kickers are slightly over-valued" in cuomo


def test_a_rule_the_config_expresses_exactly_is_not_flagged(capsys):
    """League 2 penalises every missed FG at -1, which the field says exactly."""
    main(["rules", "--league", "league2"])
    out = capsys.readouterr().out
    assert "missed FG" in out
    assert "over-valued" not in out


def test_diff_names_only_what_actually_differs(capsys):
    main(["rules", "--league", "league2", "--diff", "cuomo"])
    out = capsys.readouterr().out

    # All the differences between these two leagues are in K and DST.
    assert "FG 60+" in out and "DIFFERS" in out
    assert "forced fumble" in out
    assert "skill-position scoring is IDENTICAL" in out
    assert "one projection set serves both leagues" in out

    # No skill-position line should be marked as differing.
    for line in out.splitlines():
        if "DIFFERS" in line:
            assert any(k in line for k in ("FG", "extra point", "forced fumble")), line


def test_replacement_levels_are_shown_with_the_other_league(capsys):
    main(["rules", "--league", "league2", "--diff", "cuomo"])
    out = capsys.readouterr().out
    assert "REPLACEMENT LEVEL" in out
    assert "RB 30" in out and "vs 20" in out      # much deeper in the 12-team league
