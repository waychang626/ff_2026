"""Section 3.4: model opponents, herding included - and handcuffing excluded."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from ffdraft.ids import POSITIONS
from ffdraft.opponents import DraftSimulator, mandatory_slots, roster_caps


@pytest.fixture(scope="module")
def sim(seated_config, sample_board):
    return DraftSimulator(sample_board, seated_config, my_seat=3)


def test_roster_caps_stop_a_team_hoarding_kickers(seated_config):
    caps = roster_caps(seated_config)
    assert caps["K"] == 1
    assert caps["DST"] == 1
    assert caps["QB"] >= 2        # superflex: two QBs start
    assert caps["RB"] > caps["K"]
    assert caps["WR"] > caps["K"]


def test_mandatory_slots_are_the_single_position_starters(seated_config):
    assert mandatory_slots(seated_config) == {"QB": 1, "RB": 2, "WR": 3, "K": 1, "DST": 1}


def test_completed_drafts_respect_caps_and_fill_mandatory_slots(sim, sample_board, seated_config):
    rng = np.random.default_rng(3)
    rankings = sim.base_rankings(rng, 40)
    result = sim.rollout([], -np.argsort(np.argsort(sample_board.adp)).astype(float), rankings)

    caps = roster_caps(seated_config)
    mandatory = mandatory_slots(seated_config)
    codes = sample_board.pos_code

    for team in range(seated_config.teams):
        picks = result.rosters[:, team, :]
        assert (picks >= 0).all(), "every roster spot must be filled"
        for pos_i, pos in enumerate(POSITIONS):
            counts = (codes[picks] == pos_i).sum(axis=1)
            assert counts.max() <= caps[pos], f"{pos} exceeded cap on team {team}"
            if pos in mandatory:
                assert counts.min() >= mandatory[pos], (
                    f"team {team} finished without enough {pos}"
                )


def test_no_player_is_drafted_twice(sim, sample_board):
    rng = np.random.default_rng(8)
    result = sim.rollout([], np.zeros(len(sample_board)), sim.base_rankings(rng, 20))
    for s in range(result.rosters.shape[0]):
        picks = result.rosters[s].ravel()
        picks = picks[picks >= 0]
        assert len(picks) == len(set(picks.tolist()))


def test_herding_creates_position_runs(seated_config, sample_board):
    """P(position) should rise after a team just took that position.

    Measured as back-to-back same-position picks over a whole draft, with the
    herding multiplier on versus off. Everything else, seed included, is held
    fixed.
    """
    def consecutive_same_position(multiplier: float) -> float:
        config = dataclasses.replace(
            seated_config,
            opponents=dataclasses.replace(
                seated_config.opponents, herding_multiplier=multiplier
            ),
        )
        simulator = DraftSimulator(sample_board, config, my_seat=3)
        rng = np.random.default_rng(21)
        rankings = simulator.base_rankings(rng, 60)
        result = simulator.rollout([], np.zeros(len(sample_board)), rankings)

        # Rebuild the pick order from the rosters to read runs off it.
        from ffdraft.draft import pick_owner

        fill = {t: 0 for t in range(config.teams)}
        order = []
        for pick in range(1, config.total_drafted + 1):
            team = pick_owner(pick, config.teams, config.draft_type) - 1
            order.append(result.rosters[:, team, fill[team]])
            fill[team] += 1
        seq = np.stack(order, axis=1)                    # (sims, picks)
        pos = sample_board.pos_code[seq]
        return float((pos[:, 1:] == pos[:, :-1]).mean())

    assert consecutive_same_position(2.5) > consecutive_same_position(1.0)


def test_handcuffing_is_not_implemented_anywhere():
    """Brief 3.4: Bayes factor 4.2 *favouring no difference*. Do not add it.

    Checked against the parsed syntax tree rather than the raw text, so the
    modules stay free to explain in prose why the feature is absent.
    """
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "ffdraft"
    offenders = []
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # String *values* can carry it (a config key would); docstrings
                # are statements and are excluded below.
                names = []
            for name in names:
                if "handcuff" in name.lower():
                    offenders.append(f"{path.name}:{node.lineno} {name}")
    assert not offenders, f"handcuff logic found at {offenders}"


def test_forcing_a_pick_that_is_not_mine_is_refused(sim, sample_board):
    rng = np.random.default_rng(1)
    rankings = sim.base_rankings(rng, 5)
    with pytest.raises(ValueError, match="not mine"):
        sim.rollout([], np.zeros(len(sample_board)), rankings,
                    forced_pick=0, forced_at_pick=1)   # pick 1 belongs to seat 1


def test_rollout_is_deterministic_given_the_same_rankings(sim, sample_board):
    rng = np.random.default_rng(12)
    rankings = sim.base_rankings(rng, 15)
    value = np.zeros(len(sample_board))
    a = sim.rollout([], value, rankings)
    b = sim.rollout([], value, rankings)
    assert np.array_equal(a.rosters, b.rosters)
