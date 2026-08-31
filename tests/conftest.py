from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ffdraft.board import Board  # noqa: E402
from ffdraft.config import load_league  # noqa: E402
from ffdraft.data import build_board_for  # noqa: E402
from ffdraft.ids import POSITIONS, Player  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
CUOMO = ROOT / "configs" / "leagues" / "cuomo.yaml"


@pytest.fixture(scope="session")
def cuomo_config():
    return load_league(CUOMO)


@pytest.fixture(scope="session")
def seated_config(cuomo_config):
    """League 1 with a seat and a small sim budget, so tests stay quick."""
    return dataclasses.replace(
        cuomo_config,
        my_seat=3,
        sim=dataclasses.replace(cuomo_config.sim, n_sims=180, candidate_pool=6),
    )


@pytest.fixture(scope="session")
def sample_board(seated_config):
    return build_board_for(
        seated_config,
        SAMPLES / "projections_synthetic.csv",
        SAMPLES / "market_synthetic.csv",
    )


def make_board(specs: list[tuple[str, str, float, float, float]]) -> Board:
    """Build a Board directly from (name, pos, points, sd, adp) tuples."""
    players = [
        Player(player_id=f"{name.lower().replace(' ', '-')}|{pos}", name=name, pos=pos)
        for name, pos, _, _, _ in specs
    ]
    return Board(
        players=players,
        points=np.array([s[2] for s in specs], dtype=float),
        sd=np.array([s[3] for s in specs], dtype=float),
        adp=np.array([s[4] for s in specs], dtype=float),
        pos_code=np.array([POSITIONS.index(s[1]) for s in specs], dtype=np.int64),
        bye=np.zeros(len(specs), dtype=np.int64),
        raw_points=np.array([s[2] for s in specs], dtype=float),
        index={p.player_id: i for i, p in enumerate(players)},
        adp_imputed=np.zeros(len(specs), dtype=bool),
    )
