"""Assemble a Board for a league from projection and market files.

Skill-position scoring is effectively identical across both leagues, so one
projection file serves both (brief section 4). The board still differs per
league, because scoring, calibration anchors and replacement baselines all do.
"""

from __future__ import annotations

from pathlib import Path

from .board import Board, build_board
from .calibration import calibrate
from .config import LeagueConfig
from .projections import MarketData, aggregate, load_market, load_stat_lines

DEFAULT_POOL = 260


def data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def default_paths(season: int = 2026) -> tuple[Path, Path]:
    base = data_dir()
    return base / f"projections_{season}.csv", base / f"market_{season}.csv"


def build_board_for(
    config: LeagueConfig,
    projections_path: str | Path,
    market_path: str | Path | None = None,
    pool_size: int | None = DEFAULT_POOL,
) -> Board:
    rows = load_stat_lines(projections_path)
    aggregated = aggregate(rows, config.scoring)
    calibrated = calibrate(aggregated, config.calibration, config.vor_baseline)
    market: MarketData | None = None
    if market_path is not None and Path(market_path).exists():
        market = load_market(market_path)
    # Always keep enough depth at every position to reach replacement level and
    # still fill a mandatory slot at the end of the draft.
    min_per_position = (
        max((v + 6 for v in config.vor_baseline.values()), default=0)
        if pool_size
        else 0
    )
    return build_board(
        calibrated,
        market,
        pool_size=pool_size,
        min_per_position=min_per_position,
        impute_rank=_vor_rank(calibrated, config.vor_baseline),
    )


def _vor_rank(calibrated, vor_baseline: dict[str, int]) -> dict[str, float]:
    """VOR per player, used as the stand-in draft order when ADP is missing."""
    by_pos: dict[str, list] = {}
    for row in calibrated:
        by_pos.setdefault(row.pos, []).append(row)

    replacement: dict[str, float] = {}
    for pos, group in by_pos.items():
        points = sorted((r.points for r in group), reverse=True)
        idx = min(max(vor_baseline.get(pos, 12) - 1, 0), len(points) - 1)
        replacement[pos] = points[idx]

    return {r.player_id: r.points - replacement[r.pos] for r in calibrated}
