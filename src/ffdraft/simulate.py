"""Monte Carlo season simulation.

Section 3.1 is the reason this module exists. Becker & Sun maximise the
likelihood of *winning matchups*, not total points; Lee & Liu confirm it
empirically across 1,350 leagues. So point estimates are insufficient, every
player carries a distribution, and P(weekly win), P(playoffs) and P(title) are
three readouts of one simulation rather than three separate models.

It is also where variance appetite comes from. Nothing in here reads a risk
setting, because there isn't one: maximising P(title) buys variance when your
roster is behind the field and sheds it when you are ahead, automatically and
for the right reason. `tests/test_variance_appetite.py` pins that behaviour
down in both directions. If you ever find yourself adding a `risk_tolerance`
parameter, this module is the argument against it.

Distributions are lognormal. Fantasy outcomes are right-skewed, and with half
the field making the playoffs in both leagues, the ceiling is worth more than
the symmetric normal would price it at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import Board
from .config import LeagueConfig
from .ids import POSITIONS
from .lineup import VectorLineup


def lognormal_params(mean: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a target mean/sd into lognormal mu/sigma, elementwise."""
    mean = np.maximum(mean, 1e-9)
    sd = np.maximum(sd, 1e-9)
    sigma_sq = np.log1p((sd / mean) ** 2)
    mu = np.log(mean) - 0.5 * sigma_sq
    return mu, np.sqrt(sigma_sq)


def round_robin(n_teams: int, n_weeks: int) -> np.ndarray:
    """(n_weeks, n_teams) array giving each team's opponent index.

    Circle method, repeated once the rotation exhausts. Fixed, not random - the
    schedule is a property of the league, and re-randomising it every call
    would add noise to a comparison that is trying to isolate one pick.
    """
    teams = list(range(n_teams))
    if n_teams % 2 == 1:
        teams.append(-1)  # bye placeholder
    size = len(teams)
    schedule = np.full((n_weeks, n_teams), -1, dtype=np.int64)
    rotation = teams[:]
    for week in range(n_weeks):
        for i in range(size // 2):
            a, b = rotation[i], rotation[size - 1 - i]
            if a >= 0 and b >= 0:
                schedule[week, a] = b
                schedule[week, b] = a
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]
    return schedule


@dataclass
class SeasonDraws:
    """Player-level randomness, drawn once and shared by every candidate.

    Sharing these across candidates is common random numbers. Two candidates
    are then compared in the *same* simulated seasons, so the difference
    between them is a real difference rather than a difference in luck.
    """

    weekly: np.ndarray      # (n_sims, n_weeks, n_players) float32
    n_sims: int
    n_weeks: int


def draw_season(
    board: Board,
    config: LeagueConfig,
    n_sims: int,
    rng: np.random.Generator,
) -> SeasonDraws:
    """Simulate every player's weekly scoring for a whole season."""
    n_weeks = config.regular_season_weeks + len(config.playoff_weeks)
    n_players = len(board)

    # 1. Season-long outcome: talent, role and health uncertainty. This is the
    #    dominant term and is what section 3.3's R^2 sizes.
    mu, sigma = lognormal_params(board.points, board.sd)
    season = np.exp(rng.normal(mu[None, :], sigma[None, :], size=(n_sims, n_players)))

    # 2. Spread it over the weeks the player is actually active. A bye inside
    #    the fantasy season means the same season total lands in fewer weeks.
    bye = board.bye
    has_bye = (bye >= 1) & (bye <= n_weeks)
    active_weeks = np.where(has_bye, n_weeks - 1, n_weeks).astype(np.float64)
    per_week = season / active_weeks[None, :]

    # 3. Week-to-week noise around that rate.
    cv = np.array(
        [config.sim.weekly_cv.get(POSITIONS[c], 0.5) for c in board.pos_code],
        dtype=np.float64,
    )
    w_mu, w_sigma = lognormal_params(np.ones_like(cv), cv)
    weekly_mult = np.exp(
        rng.normal(
            w_mu[None, None, :], w_sigma[None, None, :], size=(n_sims, n_weeks, n_players)
        )
    )
    weekly = per_week[:, None, :] * weekly_mult

    # 4. Byes.
    week_idx = np.arange(1, n_weeks + 1)[None, :, None]
    weekly = np.where((week_idx == bye[None, None, :]) & has_bye[None, None, :], 0.0, weekly)

    # 5. Injury. Modelled as a season-ending event at a uniformly random week
    #    rather than independent weekly absences: losing a back for the year is
    #    the tail that actually decides seasons, and independent weekly noise
    #    would average it away. Expected games missed is calibrated to the
    #    config value via E[missed | hurt] ~ n_weeks / 2.
    loss = np.array(
        [config.sim.injury_game_loss.get(POSITIONS[c], 0.0) for c in board.pos_code],
        dtype=np.float64,
    )
    p_hurt = np.clip(2.0 * loss / max(n_weeks, 1), 0.0, 0.95)
    hurt = rng.random((n_sims, n_players)) < p_hurt[None, :]
    hurt_week = rng.integers(1, n_weeks + 1, size=(n_sims, n_players))
    out = hurt[:, None, :] & (week_idx >= hurt_week[:, None, :])
    weekly = np.where(out, 0.0, weekly)

    return SeasonDraws(
        weekly=weekly.astype(np.float32), n_sims=n_sims, n_weeks=n_weeks
    )


@dataclass
class SeasonResult:
    p_weekly_win: float
    p_playoffs: float
    p_title: float
    mean_points: float
    title_indicator: np.ndarray   # (n_sims,) bool - kept for paired comparisons
    playoff_indicator: np.ndarray
    weekly_win_rate: np.ndarray   # (n_sims,) float


class SeasonSimulator:
    def __init__(self, board: Board, config: LeagueConfig) -> None:
        self.board = board
        self.config = config
        self.lineup = VectorLineup(config.roster)
        self.schedule = round_robin(config.teams, config.regular_season_weeks)
        self.n_playoff_rounds = len(config.playoff_weeks)

    def team_weekly_totals(
        self, rosters: np.ndarray, draws: SeasonDraws
    ) -> np.ndarray:
        """(n_sims, n_weeks, n_teams) best-lineup points for every team."""
        n_sims, n_teams, depth = rosters.shape
        totals = np.empty((n_sims, draws.n_weeks, n_teams), dtype=np.float32)
        pos_code = self.board.pos_code
        weeks = draws.n_weeks
        for team in range(n_teams):
            idx = rosters[:, team, :]                       # (n_sims, depth)
            filled = idx >= 0
            safe = np.where(filled, idx, 0)
            # broadcast_to rather than repeat: the week axis is a view, not a
            # copy, and this runs once per team per candidate.
            gather = np.broadcast_to(safe[:, None, :], (n_sims, weeks, depth))
            scores = np.take_along_axis(draws.weekly, gather, axis=2)
            scores = np.where(filled[:, None, :], scores, np.float32(0.0))
            codes = np.where(filled, pos_code[safe], -1)
            totals[:, :, team] = self.lineup.total(
                scores, np.broadcast_to(codes[:, None, :], (n_sims, weeks, depth))
            )
        return totals

    def run(
        self, rosters: np.ndarray, draws: SeasonDraws, my_seat: int
    ) -> SeasonResult:
        cfg = self.config
        totals = self.team_weekly_totals(rosters, draws)
        n_sims, _, n_teams = totals.shape
        me = my_seat - 1

        reg = totals[:, : cfg.regular_season_weeks, :]
        opponents = self.schedule                                   # (weeks, teams)
        opp_scores = np.take_along_axis(
            reg, np.broadcast_to(opponents[None, :, :], reg.shape), axis=2
        )
        played = np.broadcast_to(opponents[None, :, :] >= 0, reg.shape)
        wins = ((reg > opp_scores) & played).sum(axis=1).astype(np.float64)
        ties = ((reg == opp_scores) & played).sum(axis=1).astype(np.float64)
        points_for = reg.sum(axis=1).astype(np.float64)
        record = wins + 0.5 * ties

        # Seeding: record first, total points as the tiebreak. Scaling points
        # into the fractional part keeps it a strict tiebreak rather than
        # letting a high-scoring 6-8 team leapfrog an 8-6 one.
        rank_key = record + points_for / (points_for.max() + 1.0)
        order = np.argsort(-rank_key, axis=1, kind="stable")        # (n_sims, teams)
        seeds = order[:, : cfg.playoff_teams]
        made_playoffs = (seeds == me).any(axis=1)

        champion = self._run_playoffs(totals, seeds)

        # The schedule is fixed, so games played is the same in every sim.
        games = max(int((opponents[:, me] >= 0).sum()), 1)
        weekly_win_rate = (wins[:, me] + 0.5 * ties[:, me]) / games

        return SeasonResult(
            p_weekly_win=float(weekly_win_rate.mean()),
            p_playoffs=float(made_playoffs.mean()),
            p_title=float((champion == me).mean()),
            mean_points=float(points_for[:, me].mean()),
            title_indicator=(champion == me),
            playoff_indicator=made_playoffs,
            weekly_win_rate=weekly_win_rate,
        )

    def _run_playoffs(self, totals: np.ndarray, seeds: np.ndarray) -> np.ndarray:
        """Single-elimination bracket. Higher seed advances on a tie.

        The field is padded up to a power of two with empty slots, which is
        what gives the top seeds their first-round byes: a 6-team field in an
        8-slot bracket means seeds 1 and 2 are paired against nothing and
        advance. Slot i always meets slot (size - 1 - i), so re-seeding is
        implicit and the best remaining seed always draws the worst.
        """
        cfg = self.config
        n_sims, n_playoff = seeds.shape
        if n_playoff <= 1 or not cfg.playoff_weeks:
            return seeds[:, 0] if n_playoff else np.full(n_sims, -1, dtype=np.int64)

        bracket_size = 1
        while bracket_size < n_playoff:
            bracket_size *= 2
        rounds_needed = bracket_size.bit_length() - 1
        if rounds_needed > len(cfg.playoff_weeks):
            raise ValueError(
                f"{cfg.league_id}: a {n_playoff}-team playoff needs {rounds_needed} "
                f"rounds but only {len(cfg.playoff_weeks)} playoff weeks are "
                f"configured ({list(cfg.playoff_weeks)})."
            )

        alive = np.full((n_sims, bracket_size), -1, dtype=np.int64)
        alive[:, :n_playoff] = seeds

        # Playoff weeks sit after the regular season in `totals`. If the window
        # is longer than the bracket needs, the extra weeks are at the front.
        first_round_col = cfg.regular_season_weeks + (len(cfg.playoff_weeks) - rounds_needed)

        size = bracket_size
        for rnd in range(rounds_needed):
            scores = totals[:, first_round_col + rnd, :]
            half = size // 2
            high = alive[:, :half]
            low = alive[:, ::-1][:, :half]      # slot i meets slot size-1-i
            high_score = np.where(
                high >= 0, np.take_along_axis(scores, np.maximum(high, 0), axis=1), -np.inf
            )
            low_score = np.where(
                low >= 0, np.take_along_axis(scores, np.maximum(low, 0), axis=1), -np.inf
            )
            alive = np.where((low < 0) | (high_score >= low_score), high, low)
            size = half
        return alive[:, 0]
