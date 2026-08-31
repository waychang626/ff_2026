"""Section 3.1: variance appetite is endogenous, not a config knob.

"When projected above the threshold you need, reduce variance; when below it,
buy variance." Nothing in the engine reads a risk setting - this behaviour has
to *emerge* from maximising P(title) through the simulator, or the claim in
`simulate.py`'s docstring is decoration.

The experiment holds everything fixed except one starting flex spot, which is
filled by one of two players with identical projected points and very different
spreads. Then it varies how strong the rest of the roster is relative to the
field and asks which player the simulator prefers.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import make_board
from ffdraft.simulate import SeasonSimulator, draw_season

TEAM_POS = [
    "QB", "QB", "RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "WR",
    "TE", "K", "DST", "WR", "RB", "TE",
]
BASE = {"QB": 260.0, "RB": 150.0, "WR": 140.0, "TE": 110.0, "K": 120.0, "DST": 100.0}
CANDIDATE_SLOT = 6      # one of my starting WR spots
N_SIMS = 3000
SEED = 5


def _preference(config, my_factor: float) -> tuple[float, float]:
    """(P(playoffs) with the volatile player) - (with the steady one), and 2 SE."""
    specs, rows = [], []
    for team in range(config.teams):
        row = []
        for pos in TEAM_POS:
            factor = my_factor if team == 0 else 1.0
            specs.append(
                (f"t{team}_{pos}_{len(specs)}", pos, BASE[pos] * factor,
                 BASE[pos] * 0.35, float(len(specs) + 1))
            )
            row.append(len(specs) - 1)
        rows.append(row)

    steady = len(specs)
    specs.append(("cand_steady", "WR", 140.0, 25.0, 999.0))
    volatile = len(specs)
    specs.append(("cand_volatile", "WR", 140.0, 130.0, 999.0))

    board = make_board(specs)
    simulator = SeasonSimulator(board, config)

    outcomes = {}
    for label, candidate in (("steady", steady), ("volatile", volatile)):
        rosters = np.array(rows, dtype=np.int64)[None, :, :].repeat(N_SIMS, axis=0)
        rosters[:, 0, CANDIDATE_SLOT] = candidate
        draws = draw_season(board, config, N_SIMS, np.random.default_rng(SEED))
        outcomes[label] = simulator.run(rosters, draws, my_seat=1)

    steady_p = outcomes["steady"].p_playoffs
    diff = outcomes["volatile"].p_playoffs - steady_p
    se2 = 2 * float(np.sqrt(steady_p * (1 - steady_p) / N_SIMS))
    return diff, se2


@pytest.fixture(scope="module")
def sweep(cuomo_config):
    config = dataclasses.replace(cuomo_config, my_seat=1)
    return {f: _preference(config, f) for f in (0.75, 1.00, 1.30)}


def test_a_roster_behind_the_field_buys_variance(sweep):
    diff, se2 = sweep[0.75]
    assert diff > se2, (
        f"a roster at 0.75x the field should prefer the volatile player, "
        f"got {diff:+.4f} against 2SE {se2:.4f}"
    )


def test_variance_appetite_falls_monotonically_as_the_roster_improves(sweep):
    """The shape is the claim, and it is much sharper than any single level.

    Above the threshold the preference does turn negative, but the magnitude is
    small - one flex spot is a modest share of a team's total variance - so the
    monotone trend is what this asserts rather than significance at each point.
    """
    weak, par, strong = sweep[0.75][0], sweep[1.00][0], sweep[1.30][0]
    assert weak > par > strong, f"expected monotone decline, got {weak:+.4f} {par:+.4f} {strong:+.4f}"


def test_the_preference_changes_sign_around_the_playoff_threshold(sweep):
    assert sweep[0.75][0] > 0, "behind the field: should buy variance"
    assert sweep[1.30][0] < 0, "ahead of the field: should shed variance"


def test_nothing_in_the_config_sets_risk_tolerance(cuomo_config):
    """The structural guarantee: there is no knob to get this wrong with."""
    import dataclasses as dc

    names = set()
    for obj in (cuomo_config, cuomo_config.sim, cuomo_config.opponents,
                cuomo_config.policy, cuomo_config.calibration):
        if dc.is_dataclass(obj):
            names |= {f.name for f in dc.fields(obj)}
    for banned in ("risk", "risk_tolerance", "variance_appetite", "aggression",
                   "position_need", "upside"):
        assert banned not in names, f"{banned} must not be configurable"
