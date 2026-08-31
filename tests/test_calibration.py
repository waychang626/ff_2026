"""Section 3.3 corrections, as a separate testable transform stage."""
from __future__ import annotations

import math
import statistics

import pytest

from ffdraft.calibration import calibrate, residual_sd, shrink
from ffdraft.ids import Player
from ffdraft.projections import PlayerProjection


def _proj(name, pos, points, sd=6.0, n=6):
    return PlayerProjection(
        player=Player(f"{name}|{pos}", name, pos),
        points=points, source_sd=sd, n_sources=n,
    )


def test_shrink_and_optimism_commute():
    """Order of operations must not change the answer, and it doesn't.

    Shrinking toward the raw mean then subtracting a constant equals
    subtracting first and shrinking toward the shifted mean. The docs claim it;
    this holds us to it.
    """
    value, anchor, slope, optimism = 240.0, 150.0, 0.79, 21.6
    shrink_then_subtract = shrink(value, anchor, slope) - optimism
    subtract_then_shrink = shrink(value - optimism, anchor - optimism, slope)
    assert shrink_then_subtract == pytest.approx(subtract_then_shrink)


def test_shrink_pulls_the_top_down_and_lifts_the_bottom():
    """Top-projected underperform, bottom-projected meet or beat (brief 3.3)."""
    anchor, slope = 150.0, 0.79
    assert shrink(300.0, anchor, slope) < 300.0
    assert shrink(50.0, anchor, slope) > 50.0
    assert shrink(anchor, anchor, slope) == pytest.approx(anchor)


def test_residual_sd_follows_from_slope_and_r_squared():
    """sd(eps) = slope * sd(proj) * sqrt((1-rho)/rho). At rho=0.2 the factor is 2."""
    assert residual_sd(1.0, 50.0, 0.20) == pytest.approx(100.0)
    assert residual_sd(0.67, 50.0, 0.20) == pytest.approx(67.0)
    # Lower R^2 means a wider outcome distribution.
    assert residual_sd(1.0, 50.0, 0.14) > residual_sd(1.0, 50.0, 0.26)


def test_qb_gets_the_largest_optimism_haircut(cuomo_config):
    """QB +46.5 vs +21.6 elsewhere (brief 3.3)."""
    params = cuomo_config.calibration
    projections = [_proj(f"qb{i}", "QB", 300 - 5 * i) for i in range(30)] + [
        _proj(f"wr{i}", "WR", 300 - 5 * i) for i in range(60)
    ]
    out = {c.player.name: c for c in calibrate(projections, params, cuomo_config.vor_baseline)}
    qb_drop = out["qb0"].raw_points - out["qb0"].points
    wr_drop = out["wr0"].raw_points - out["wr0"].points
    assert qb_drop > wr_drop


def test_variance_multipliers_average_to_one_over_the_anchor_pool(cuomo_config):
    """Heteroskedasticity is allowed; inflating total variance is not.

    If the level and disagreement factors averaged above 1 they would quietly
    widen every distribution past what the published R^2 supports, and every
    probability the engine reports would drift toward a coin flip.
    """
    params = cuomo_config.calibration
    projections = [
        _proj(f"rb{i}", "RB", 260 - 4.5 * i, sd=4 + 0.4 * i) for i in range(50)
    ]
    out = calibrate(projections, params, cuomo_config.vor_baseline)
    n_anchor = math.ceil(params.anchor_multiple * cuomo_config.vor_baseline["RB"])
    pool = sorted(out, key=lambda c: -c.points)[:n_anchor]
    spread = statistics.stdev([p.raw_points for p in pool])
    expected = residual_sd(params.slopes["RB"], spread, params.r_squared["RB"])
    assert statistics.fmean([p.sd for p in pool]) == pytest.approx(expected, rel=0.02)


def test_high_disagreement_widens_a_player_relative_to_his_peers(cuomo_config):
    """Disagreement is measured relative to the projection, not in raw points.

    Absolute source spread scales with projected points, so comparing it
    directly would re-label every high-scoring player as uncertain.
    """
    params = cuomo_config.calibration
    projections = [_proj(f"wr{i}", "WR", 200 - 2 * i, sd=8.0) for i in range(60)]
    projections.append(_proj("contested", "WR", 200.0, sd=40.0))
    projections.append(_proj("agreed", "WR", 200.0, sd=2.0))
    out = {c.player.name: c for c in calibrate(projections, params, cuomo_config.vor_baseline)}
    assert out["contested"].points == pytest.approx(out["agreed"].points)
    assert out["contested"].sd > out["agreed"].sd


def test_calibration_never_produces_negative_points_or_zero_sd(cuomo_config):
    projections = [_proj(f"te{i}", "TE", max(1.0, 120 - 6 * i)) for i in range(40)]
    for c in calibrate(projections, cuomo_config.calibration, cuomo_config.vor_baseline):
        assert c.points >= 0.0
        assert c.sd > 0.0


def test_calibration_is_order_independent(cuomo_config):
    projections = [_proj(f"rb{i}", "RB", 250 - 3 * i) for i in range(40)]
    forward = {c.player_id: (c.points, c.sd) for c in
               calibrate(projections, cuomo_config.calibration, cuomo_config.vor_baseline)}
    backward = {c.player_id: (c.points, c.sd) for c in
                calibrate(list(reversed(projections)), cuomo_config.calibration,
                          cuomo_config.vor_baseline)}
    assert forward == backward
