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
    assert mandatory_slots(seated_config) == {
        "QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1
    }


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


def test_next_selection_drives_a_mock_draft_without_repeats(sim, sample_board):
    """The mock console picks one at a time; it must agree with a full rollout."""
    rng = np.random.default_rng(5)
    rankings = sim.base_rankings(rng, 1)
    value = -np.argsort(np.argsort(sample_board.adp)).astype(float)

    drafted: list[int] = []
    for _ in range(20):
        idx = sim.next_selection(drafted, value, rankings)
        assert idx not in drafted, "next_selection returned an already-drafted player"
        drafted.append(idx)

    # Same rankings and same state must give the same pick every time.
    assert sim.next_selection(drafted, value, rankings) == sim.next_selection(
        drafted, value, rankings
    )


def test_likely_next_picks_is_ranked_deterministic_and_covers_most_outcomes(
    sim, sample_board
):
    """The numbered list is only useful if the real pick is usually on it."""
    rows = sim.likely_next_picks([], seat=1, n=10, seed=4)
    assert len(rows) == 10

    probs = [p for _, p in rows]
    assert probs == sorted(probs, reverse=True), "must be ranked by likelihood"
    assert 0.5 < sum(probs) <= 1.0, f"top 10 should cover most outcomes, got {sum(probs)}"

    again = sim.likely_next_picks([], seat=1, n=10, seed=4)
    assert rows == again, "same seed and state must give the same list"


def test_likely_next_picks_never_suggests_someone_already_drafted(sim, sample_board):
    drafted = [0, 1, 2, 3, 4, 5]
    rows = sim.likely_next_picks(drafted, seat=7, n=10, seed=2)
    assert not {idx for idx, _ in rows} & set(drafted)


def test_likely_next_picks_respects_the_seat_roster_caps(seated_config, sample_board):
    """A team that already has its kicker should not be offered another."""
    from ffdraft.draft import seat_picks

    simulator = DraftSimulator(sample_board, seated_config, my_seat=3)
    kicker = next(
        i for i, pl in enumerate(sample_board.players) if pl.pos == "K"
    )
    # Hand seat 1 the kicker by putting him at one of seat 1's pick slots.
    picks = seat_picks(1, seated_config.teams, seated_config.rounds)
    drafted = []
    filler = (i for i, pl in enumerate(sample_board.players) if pl.pos != "K")
    for n in range(1, picks[1] + 1):
        drafted.append(kicker if n == picks[0] else next(filler))

    rows = simulator.likely_next_picks(drafted, seat=1, n=10, seed=1)
    assert not any(sample_board.pos_of(i) == "K" for i, _ in rows)


def test_a_run_pulls_that_position_up_the_list(seated_config, sample_board):
    """Herding (brief 3.4) has to be visible in what the list predicts.

    Measured late in the draft, which is the only place it can show: the best
    defense sits around ADP 185, and the herd bonus is worth about seven picks,
    so in round 2 no bonus could bring a DST near the top. By round 15 they are
    genuinely in contention and the effect is large.
    """
    simulator = DraftSimulator(sample_board, seated_config, my_seat=3)
    drafted = list(range(120))
    dst_code = POSITIONS.index("DST")

    def dst_share(rows):
        return sum(p for i, p in rows if sample_board.pos_of(i) == "DST")

    calm = simulator.likely_next_picks(drafted, seat=5, n=12, seed=9)
    run = simulator.likely_next_picks(
        drafted, seat=5, n=12, seed=9, last_pos_code=dst_code, pick_number=121
    )
    assert dst_share(run) > 2 * dst_share(calm), (
        f"a defense just went; the next one should be much likelier. "
        f"calm={dst_share(calm):.1%} after={dst_share(run):.1%}"
    )
