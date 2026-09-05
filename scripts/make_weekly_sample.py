#!/usr/bin/env python3
"""Write a synthetic multi-source weekly file, for exercising `ffdraft lineup`.

Real weekly data comes from `R/pull_projections.R --week N`, which scrapes
several sources. This produces a file of the same shape - one row per
(source, player), each source carrying its own `fetched_at` - so the aggregation
and the freshness rules can be exercised without a network.

The numbers are not predictions of anything.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffdraft.config import load_by_id  # noqa: E402
from ffdraft.data import build_board_for  # noqa: E402

SOURCES = ("CBS", "FantasyPros", "FFToday", "NumberFire", "FantasySharks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="cuomo")
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--projections", default="data/samples/projections_synthetic.csv")
    ap.add_argument("--market", default="data/samples/market_synthetic.csv")
    ap.add_argument("--out", default="data/samples/weekly_synthetic.csv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sources", type=int, default=len(SOURCES))
    ap.add_argument("--stale-source", help="give this source an old fetched_at")
    ap.add_argument("--stale-hours", type=float, default=72.0)
    ap.add_argument("--age-hours", type=float, default=1.0,
                    help="how old the fresh sources should look")
    ap.add_argument("--questionable", type=int, default=6,
                    help="how many players to tag QUESTIONABLE")
    ap.add_argument("--out-players", type=int, default=4,
                    help="how many players to rule OUT")
    args = ap.parse_args()

    config = load_by_id(args.league)
    board = build_board_for(config, args.projections, args.market)
    rng = np.random.default_rng(args.seed)
    now = datetime.now(timezone.utc)

    weeks = config.regular_season_weeks + len(config.playoff_weeks)
    # A season total spread over the weeks a player is actually active.
    active = np.where((board.bye >= 1) & (board.bye <= weeks), weeks - 1, weeks)
    per_week = board.points / active

    statuses: dict[int, str] = {}
    pool = rng.permutation(len(board))
    for i in pool[: args.out_players]:
        statuses[int(i)] = "OUT"
    for i in pool[args.out_players : args.out_players + args.questionable]:
        statuses[int(i)] = "QUESTIONABLE"
    for i in range(len(board)):
        if board.bye[i] == args.week:
            statuses[i] = "BYE"

    used = SOURCES[: max(1, args.sources)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "fetched_at", "week", "player_id", "player",
                         "pos", "team", "points", "status", "opponent"])
        for source in used:
            hours = (
                args.stale_hours if source == args.stale_source else args.age_hours
            )
            stamp = (now - timedelta(hours=hours)).isoformat()
            # Each source has its own house view: a per-source bias plus noise.
            bias = rng.normal(1.0, 0.05)
            for i, player in enumerate(board.players):
                status = statuses.get(i, "ACTIVE")
                points = 0.0 if status in ("OUT", "BYE") else max(
                    0.0, float(per_week[i] * bias * rng.normal(1.0, 0.18))
                )
                writer.writerow([
                    source, stamp, args.week, player.player_id, player.name,
                    player.pos, player.team, f"{points:.2f}", status, "",
                ])
    print(f"wrote {out} - week {args.week}, {len(used)} sources, "
          f"{len(board)} players each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
