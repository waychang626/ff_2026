"""Section 3.3: correct third-party projections before using them.

Three published corrections, applied in this order:

  1. Aggregate equal-weight            (done upstream in projections.aggregate)
  2. Shrink the spread                 slope < 1.0 at every position
  3. Subtract optimism                 mean error +21.6 overall, QB +46.5

Steps 2 and 3 commute - shrinking toward the raw mean then subtracting a
constant gives exactly the same answer as subtracting first and shrinking
toward the shifted mean - so the order is a matter of exposition, not result.
There is a test that holds us to that.

The fourth correction is the one that shapes the whole engine: projections
explain only ~14-26% of within-position variance. That is not a caveat to note
and move past, it is a *number we can use*. Under the linear calibration model

    outcome = a + slope * projection + eps,   R^2 = rho

the residual scale follows directly:

    sd(eps) = slope * sd(projection within position) * sqrt((1 - rho) / rho)

At rho = 0.20 that factor is 2.0, so the true outcome spread is roughly twice
the spread of the projections themselves. This is why cross-source disagreement
alone is a poor variance estimate: sources agreeing with each other says nothing
about them agreeing with reality. Disagreement enters only as a per-player
modulation of the position-level scale.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .config import CalibrationParams
from .ids import POSITIONS, Player
from .projections import PlayerProjection


@dataclass
class CalibratedProjection:
    """A player's season outlook: a mean *and* a spread.

    Point estimates are insufficient (brief 3.1). Everything downstream - the
    Monte Carlo, P(weekly win), P(title) - reads `sd` as seriously as `points`.
    """

    player: Player
    points: float          # corrected season-total mean
    sd: float              # season-total outcome sd
    raw_points: float      # before correction, for the audit trail
    source_sd: float
    n_sources: int

    @property
    def player_id(self) -> str:
        return self.player.player_id

    @property
    def pos(self) -> str:
        return self.player.pos


def positional_anchor(
    projections: list[PlayerProjection],
    pos: str,
    baseline_rank: int,
    anchor_multiple: float,
) -> float:
    """Mean projection over the relevant pool at a position.

    Shrinkage regresses toward this. The pool is the top
    `anchor_multiple x baseline_rank` players: including deep third-stringers
    would drag the anchor down and inflate every real contributor after
    shrinkage, which is the opposite of the correction's intent.
    """
    pool = sorted(
        (p.points for p in projections if p.player.pos == pos), reverse=True
    )
    if not pool:
        return 0.0
    n = max(1, int(math.ceil(anchor_multiple * max(baseline_rank, 1))))
    return statistics.fmean(pool[:n])


def calibrate(
    projections: list[PlayerProjection],
    params: CalibrationParams,
    vor_baseline: dict[str, int],
    disagreement_weight: float = 0.5,
) -> list[CalibratedProjection]:
    """Apply the section 3.3 corrections and attach an outcome sd.

    The sd of every player at a position starts from one number - the residual
    scale the published slope and R^2 imply - and is then modulated by two
    unit-mean factors:

      level         a 300-point projection and a 40-point projection do not
                    carry the same absolute uncertainty.
      disagreement  sources disagreeing about a player is weak evidence that he
                    is genuinely harder to forecast.

    Both factors are normalised to average 1.0 over the anchor pool. That is
    what keeps them honest: heteroskedasticity is real, but if the multipliers
    averaged above 1 they would quietly inflate total variance past what the
    published R^2 actually supports, and every downstream probability would
    drift toward a coin flip.

    Disagreement is measured *relative* to a player's own projection. Absolute
    source spread is roughly proportional to projected points, so using it
    directly would re-label every high-scoring player as uncertain and
    double-count the level factor.
    """
    by_pos: dict[str, list[PlayerProjection]] = {p: [] for p in POSITIONS}
    for proj in projections:
        by_pos.setdefault(proj.player.pos, []).append(proj)

    out: list[CalibratedProjection] = []
    for pos, group in by_pos.items():
        if not group:
            continue
        slope = params.slopes.get(pos, 1.0)
        optimism = params.optimism.get(pos, 0.0)
        rho = min(max(params.r_squared.get(pos, 0.20), 1e-6), 0.999999)
        baseline_rank = vor_baseline.get(pos, 12)
        anchor = positional_anchor(projections, pos, baseline_rank, params.anchor_multiple)

        n_anchor = max(1, int(math.ceil(params.anchor_multiple * max(baseline_rank, 1))))
        pool = sorted((p.points for p in group), reverse=True)[:n_anchor]
        spread = statistics.stdev(pool) if len(pool) > 1 else 0.0
        pos_sd = residual_sd(slope, spread, rho)

        anchor_corrected = max(1.0, anchor - optimism)
        ranked = sorted(group, key=lambda p: (-round(p.points, 6), p.player_id))
        anchor_pool = ranked[:n_anchor]

        corrected = {
            p.player_id: max(0.0, anchor + slope * (p.points - anchor) - optimism)
            for p in group
        }

        def level_factor(proj: PlayerProjection) -> float:
            level = corrected[proj.player_id] / anchor_corrected
            return min(2.0, max(0.25, 0.5 + 0.5 * level))

        def relative_disagreement(proj: PlayerProjection) -> float:
            if proj.n_sources < 2 or proj.points <= 0:
                return 0.0
            return proj.source_sd / proj.points

        rel = [relative_disagreement(p) for p in anchor_pool]
        rel_nonzero = [r for r in rel if r > 0]
        mean_rel = statistics.fmean(rel_nonzero) if rel_nonzero else 0.0

        def disagreement_factor(proj: PlayerProjection) -> float:
            if mean_rel <= 0:
                return 1.0
            r = relative_disagreement(proj)
            if r <= 0:
                return 1.0
            return min(1.8, max(0.6, 1.0 + disagreement_weight * (r / mean_rel - 1.0)))

        def combined(proj: PlayerProjection) -> float:
            return level_factor(proj) * disagreement_factor(proj)

        # Normalise the *product*, not each factor separately. The two are
        # correlated in general - a position where disagreement grows down the
        # board has level and disagreement pulling opposite ways - and
        # E[XY] != E[X]E[Y], so normalising them independently leaves the
        # combined multiplier off by a few percent in either direction.
        norm = statistics.fmean([combined(p) for p in anchor_pool]) or 1.0

        for proj in group:
            value = corrected[proj.player_id]
            sd = pos_sd * combined(proj) / norm
            # A projection carries irreducible uncertainty even at the very
            # bottom of the board; never hand the simulator a point mass.
            sd = max(sd, 0.05 * max(value, 1.0))

            out.append(
                CalibratedProjection(
                    player=proj.player,
                    points=value,
                    sd=sd,
                    raw_points=proj.points,
                    source_sd=proj.source_sd,
                    n_sources=proj.n_sources,
                )
            )

    out.sort(key=lambda c: (-round(c.points, 6), c.player_id))
    return out


def shrink(value: float, anchor: float, slope: float) -> float:
    """Regress one projection toward the positional anchor."""
    return anchor + slope * (value - anchor)


def residual_sd(slope: float, spread: float, r_squared: float) -> float:
    """sd of the outcome residual implied by a calibration slope and R^2."""
    rho = min(max(r_squared, 1e-6), 0.999999)
    return slope * spread * math.sqrt((1.0 - rho) / rho)
