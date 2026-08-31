"""Scoring: the stat line -> points function, per league."""
from __future__ import annotations

import pytest

from ffdraft.scoring import (
    bracket_points,
    score_stats,
    to_r_scoring_rules,
    unknown_stats,
    unscored_stats,
)


def test_half_ppr_skill_line(cuomo_config):
    """A hand-checked WR line. 100 rec, 1400 yds, 10 TD, 1 fumble."""
    stats = {"rec": 100, "rec_yds": 1400, "rec_tds": 10, "fumbles_lost": 1}
    # 100*0.5 + 1400*0.1 + 10*6 - 2 = 50 + 140 + 60 - 2
    assert score_stats(stats, cuomo_config.scoring) == pytest.approx(248.0)


def test_four_point_passing_td_and_minus_one_int(cuomo_config):
    stats = {"pass_yds": 4500, "pass_tds": 35, "pass_int": 12}
    # 4500*0.04 + 35*4 - 12 = 180 + 140 - 12
    assert score_stats(stats, cuomo_config.scoring) == pytest.approx(308.0)


def test_rushing_qb_premium_is_real(cuomo_config):
    """Section 4: 4-pt passing TDs make rushing QBs disproportionately valuable."""
    pocket = {"pass_yds": 4600, "pass_tds": 34, "pass_int": 11, "rush_yds": 80}
    runner = {"pass_yds": 3900, "pass_tds": 26, "pass_int": 9, "rush_yds": 750,
              "rush_tds": 7}
    assert score_stats(runner, cuomo_config.scoring) > score_stats(pocket, cuomo_config.scoring)


def test_points_allowed_bracket_boundaries(cuomo_config):
    bracket = cuomo_config.scoring.pts_bracket
    assert bracket_points(0, bracket) == 10      # shutout
    assert bracket_points(6, bracket) == 7       # 1-6
    assert bracket_points(7, bracket) == 4       # 7-13
    assert bracket_points(13, bracket) == 4
    assert bracket_points(14, bracket) == 1
    assert bracket_points(35, bracket) == -4     # catch-all floor
    assert bracket_points(99, bracket) == -4
    assert bracket_points(120, bracket) == -4


def test_bracket_is_applied_once_not_as_a_multiplier(cuomo_config):
    stats = {"dst_sacks": 40, "dst_int": 12, "dst_pts_allowed": 18}
    # 40*1 + 12*2 + bracket(18)=1
    assert score_stats(stats, cuomo_config.scoring) == pytest.approx(65.0)


def test_missing_stats_score_zero_not_error(cuomo_config):
    assert score_stats({}, cuomo_config.scoring) == 0.0


def test_unknown_and_unscored_stats_are_surfaced(cuomo_config):
    assert unknown_stats({"rec": 1, "vibes": 3}) == ["vibes"]
    # League 1 pays nothing for a missed FG; that should be visible, not silent.
    assert "fg_miss" in unscored_stats({"fg_miss": 4, "rec": 10}, cuomo_config.scoring)


def test_r_export_round_trips_the_league_rules(cuomo_config):
    """The YAML is the source of truth; the R block is generated from it."""
    code = to_r_scoring_rules(cuomo_config.scoring, "cuomo_scoring")
    assert code.startswith("cuomo_scoring <- list(")
    assert "pass_tds = 4" in code
    assert "pass_int = -1" in code       # overrides ffanalytics' -3
    assert "rec = 0.5" in code           # half PPR; ffanalytics defaults to 0
    assert "fumbles_lost = -2" in code   # overrides ffanalytics' -3
    assert "all_pos = TRUE" in code
    assert "list(threshold = 0, points = 10)" in code
    assert code.count("pts_bracket") == 1
