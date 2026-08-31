"""Season simulation mechanics."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import make_board
from ffdraft.simulate import SeasonSimulator, draw_season, lognormal_params, round_robin


def test_schedule_is_symmetric_and_nobody_plays_themselves():
    for teams in (4, 8, 10, 12):
        schedule = round_robin(teams, 14)
        for week in range(14):
            for team in range(teams):
                opponent = schedule[week, team]
                if opponent >= 0:
                    assert opponent != team
                    assert schedule[week, opponent] == team


def test_odd_team_counts_leave_exactly_one_team_idle_per_week():
    schedule = round_robin(7, 14)
    for week in range(14):
        assert int((schedule[week] < 0).sum()) == 1


def test_lognormal_params_reproduce_the_target_mean_and_sd():
    mean = np.array([100.0, 250.0])
    sd = np.array([30.0, 120.0])
    mu, sigma = lognormal_params(mean, sd)
    draws = np.exp(np.random.default_rng(0).normal(mu, sigma, size=(400_000, 2)))
    assert draws.mean(axis=0) == pytest.approx(mean, rel=0.02)
    assert draws.std(axis=0) == pytest.approx(sd, rel=0.05)


def _uniform_league(config, n_players_per_team=17):
    specs, rows = [], []
    pos_cycle = ["QB", "QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE",
                 "K", "DST", "RB", "WR", "TE", "RB", "WR"]
    for team in range(config.teams):
        row = []
        for pos in pos_cycle[:n_players_per_team]:
            specs.append((f"t{team}_{len(specs)}", pos, 150.0, 40.0, float(len(specs) + 1)))
            row.append(len(specs) - 1)
        rows.append(row)
    return make_board(specs), np.array(rows, dtype=np.int64)


def test_identical_teams_split_titles_evenly(cuomo_config):
    """A sanity floor: with every team identical, P(title) must be 1/teams."""
    config = dataclasses.replace(cuomo_config, my_seat=1)
    board, rows = _uniform_league(config)
    simulator = SeasonSimulator(board, config)
    n = 4000
    rosters = rows[None, :, :].repeat(n, axis=0)
    draws = draw_season(board, config, n, np.random.default_rng(2))
    result = simulator.run(rosters, draws, my_seat=1)

    expected = 1.0 / config.teams
    se = np.sqrt(expected * (1 - expected) / n)
    assert result.p_title == pytest.approx(expected, abs=4 * se)
    assert result.p_playoffs == pytest.approx(config.playoff_teams / config.teams, abs=0.05)
    assert result.p_weekly_win == pytest.approx(0.5, abs=0.03)


def test_a_stronger_team_wins_more_of_everything(cuomo_config):
    config = dataclasses.replace(cuomo_config, my_seat=1)
    board, rows = _uniform_league(config)
    boosted = board.points.copy()
    boosted[rows[0]] *= 1.4
    board = dataclasses.replace(board, points=boosted, _resolver=None)

    simulator = SeasonSimulator(board, config)
    n = 1500
    draws = draw_season(board, config, n, np.random.default_rng(3))
    result = simulator.run(rows[None, :, :].repeat(n, axis=0), draws, my_seat=1)
    assert result.p_weekly_win > 0.6
    assert result.p_playoffs > 0.8
    assert result.p_title > 1.0 / config.teams


def test_a_playoff_field_needing_more_rounds_than_weeks_is_refused(cuomo_config):
    """Six teams need three rounds; two playoff weeks cannot host them."""
    config = dataclasses.replace(
        cuomo_config, my_seat=1, playoff_teams=6, playoff_weeks=(16, 17)
    )
    board, rows = _uniform_league(config)
    simulator = SeasonSimulator(board, config)
    draws = draw_season(board, config, 50, np.random.default_rng(1))
    with pytest.raises(ValueError, match="needs 3 rounds"):
        simulator.run(rows[None, :, :].repeat(50, axis=0), draws, my_seat=1)


def test_six_team_bracket_fits_three_weeks(cuomo_config):
    config = dataclasses.replace(
        cuomo_config, my_seat=1, playoff_teams=6, playoff_weeks=(15, 16, 17),
        regular_season_weeks=14,
    )
    board, rows = _uniform_league(config)
    simulator = SeasonSimulator(board, config)
    n = 800
    draws = draw_season(board, config, n, np.random.default_rng(4))
    result = simulator.run(rows[None, :, :].repeat(n, axis=0), draws, my_seat=1)
    # 6 of 8 identical teams qualify, so any one of them does 3/4 of the time.
    assert result.p_playoffs == pytest.approx(6 / 8, abs=0.05)
    assert 0.0 < result.p_title < 0.35


def test_byes_remove_exactly_one_week_of_scoring(cuomo_config):
    config = dataclasses.replace(cuomo_config, my_seat=1)
    board, _ = _uniform_league(config)
    with_bye = dataclasses.replace(
        board, bye=np.full(len(board), 5, dtype=np.int64), _resolver=None
    )
    draws = draw_season(with_bye, config, 200, np.random.default_rng(6))
    assert float(draws.weekly[:, 4, :].max()) == 0.0     # week 5 is the bye
    assert float(draws.weekly[:, 3, :].max()) > 0.0
